import argparse
import os
import uuid
from datetime import datetime
from pathlib import Path
import yaml

import taxonomist as src
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = src.add_dataset_args(parser)
    parser = src.add_dataloader_args(parser)
    parser = src.add_model_args(parser)
    parser = src.add_train_args(parser)
    parser = src.add_program_args(parser)

    args = parser.parse_args()

    gpu_count = torch.cuda.device_count()

    # Run name parsing
    basename = f"{args.out_prefix}_{args.model}_{args.criterion}_b{args.batch_size}"
    args.basename = basename

    if not args.ckpt_path:
        uid = datetime.now().strftime("%y%m%d-%H%M") + f"-{str(uuid.uuid4())[:4]}"
    else:
        # Parse the uid from filename
        print(f"Using checkpoint from {args.ckpt_path}")
        ckpt_name = Path(args.ckpt_path).stem
        uid = ckpt_name.split("_")[-3]
        assert basename == "_".join(ckpt_name.split("_")[:-4])

    outname = f"{basename}_f{args.fold}_{uid}"

    out_folder = (
        Path(args.out_folder) / Path(args.dataset_name) / basename / f"f{args.fold}"
    )
    out_folder.mkdir(exist_ok=True, parents=True)

    # Class / label map loading
    if args.class_map != "none":
        class_map = src.load_class_map(args.class_map)
    else:
        class_map = {"fwd": None, "inv": None}

    if args.deterministic:
        pl.seed_everything(seed=args.global_seed)

    # Data and model
    dm = src.LitDataModule(
        data_folder=args.data_folder,
        dataset_name=args.dataset_name,
        csv_path=args.csv_path,
        fold=args.fold,
        label=args.label,
        label_transform=class_map["fwd"],
        imsize=args.imsize,
        batch_size=args.batch_size,
        aug=args.aug,
        load_to_memory=args.load_to_memory,
    )

    opt_args = {"name": args.opt}

    model = src.LitModule(
        model=args.model,
        freeze_base=args.freeze_base,
        pretrained=args.pretrained,
        criterion=args.criterion,
        opt=opt_args,
        n_classes=args.n_classes,
        lr=args.lr,
        label_transform=class_map["inv"],
    )

    # Training callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=out_folder,
        filename=f"{outname}_" + "epoch{epoch:02d}_val-loss{val/loss:.2f}",
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback]

    if args.early_stopping:
        callbacks.append(
            EarlyStopping(monitor="val/loss", patience=args.early_stopping_patience)
        )

    if not args.debug:
        wandb_resume = True if args.ckpt_path else None
        print(wandb_resume)
        logger = WandbLogger(
            project=args.log_dir,
            name=outname,
            id=uid,
            resume=wandb_resume,
            allow_val_change=wandb_resume,
        )

        logger.watch(model)
        wandb.config.update(args, allow_val_change=True)
        # logger = TensorBoardLogger(args.log_dir,
        #                            name=basename,
        #                            version=uid)
        # logger.log_hyperparams(vars(args))
        # logger.log_graph(model)
    else:
        logger = True

    if args.smoke_test:
        dm.setup()
        limit_train_batches = (args.batch_size * 2) / len(dm.trainset)
        limit_val_batches = (args.batch_size * 2) / len(dm.valset)
        limit_test_batches = (args.batch_size * 2) / len(dm.testset)
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0

    # Training
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        logger=logger,
        log_every_n_steps=10,
        auto_lr_find=args.auto_lr,
        gpus=gpu_count,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        callbacks=callbacks,
        precision=args.precision,
        deterministic=args.deterministic,
    )

    if args.auto_lr:
        trainer.tune(model, dm)
        print(f"New lr: {model.hparams.lr}")
        wandb.config.update({"new_lr": model.hparams.lr}, allow_val_change=True)

    if not args.debug:  # In debug because we can't access wandb.config
        with open(out_folder / f"config_{uid}.yml", "w") as f:
            f.write(yaml.dump(vars(wandb.config)["_items"]))

    trainer.fit(model, dm, ckpt_path=args.ckpt_path)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    dm.visualize_datasets(out_folder / f"aug-{args.aug}-{uid}")

    print(
        f"Best model: {checkpoint_callback.best_model_path} | score: {checkpoint_callback.best_model_score}"
    )
