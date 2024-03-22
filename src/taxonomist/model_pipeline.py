
from pathlib import Path
import uuid
from datetime import datetime
import yaml
import sys
from typing import Dict, Optional
from dataclasses import dataclass, replace
import pandas as pd

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
import wandb

from .lightning_data_wrapper import Dataset, LitDataModule
from .lightning_model_wrapper import Model, LitModule
from .utils import load_class_map

@dataclass(frozen=True)
class LightningModelArguments():
    timm_model_name:str
    data_folder:str          
    log_dir:str
    out_prefix:str # ''
    class_map_name:str
    imsize:int
    csv_path:Optional[str]
    label_column:Optional[str]
    out_folder:str
    min_epochs:Optional[int]=None
    max_epochs:Optional[int]=None
    fold:int=0
    batch_size:int=256
    criterion:str='mse'
    lr:float=1e-4
    auto_lr:bool=False
    early_stopping:bool=False
    dataset_name:str='imagefolder'
    pretrained:bool=True
    freeze_base:bool=False
    ckpt_path:Optional[str]=None #required if resume=True
    tta:bool=False
    tta_n:int=5
    aug:str='only_flips'
    debug:bool=False
    global_seed:int=42
    smoke_test:bool=False
    log_every_n_steps:Optional[int]=10
    opt:str='adam'
    load_to_memory:bool=False
    early_stopping_patience:int=5 #used if early_stopping=True
    precision:int=32
    deterministic:bool=False
    resume:bool=False


class LightningModelWrapper:
    def __init__(self, args: LightningModelArguments):
        self.args = args
        self.basename = f"{args.out_prefix}_{args.timm_model_name}"
        self.uid = self._parse_uid()
        self.outname = f"{self.basename}_f{args.fold}_{self.uid}"

        if args.deterministic:
            pl.seed_everything(seed=args.global_seed)

    def _parse_uid(self):
        # It is possible to resume to an existing run that was cancelled/stopped if argument ckpt_path is provided that contains the weights of when the run was stopped/cancelled
        if not self.args.resume:
            uid = datetime.now().strftime("%y%m%d-%H%M") + f"-{str(uuid.uuid4())[:4]}"
        else:
            if not self.args.ckpt_path:
                raise ValueError("When resuming, a ckpt_path must be set")
            # Parse the uid from filename
            print(f"Using checkpoint from {self.args.ckpt_path}")
            ckpt_name = Path(self.args.ckpt_path).stem
            uid = ckpt_name.split("_")[-3]
            assert self.basename == "_".join(ckpt_name.split("_")[:-4])
        return uid

    def _create_out_folder(self, training=True):
        if training:
            out_folder = (
                Path(self.args.out_folder) / Path(self.args.dataset_name) / self.basename / f"f{self.args.fold}"
            )
        else:
            tag = f"{self.args.aug}"
            if self.args.tta:
                tag += "_tta"
            out_folder = Path(self.args.ckpt_path).parents[0] / "predictions" / tag

        out_folder.mkdir(exist_ok=True, parents=True)
        return out_folder
    
    def _create_prediction_out_folder(self):
        tag = f"{self.args.aug}"
        if self.args.tta:
            tag += "_tta"
        out_folder = Path(self.args.ckpt_path).parents[0] / "predictions" / tag
        out_folder.mkdir(exist_ok=True, parents=True)

    def _load_class_map(self):
        # Class / label map loading
        if self.args.class_map_name != None:
            class_map = load_class_map(self.args.class_map_name)
            n_classes = len(class_map["fwd_dict"])
        else:
            class_map = {"fwd": None, "inv": None, "fwd_dict": None, "inv_dict": None}
            n_classes = 1
        return class_map, n_classes

    def _create_data_module(self, class_map):
        # https://lightning.ai/docs/pytorch/stable/data/datamodule.html 
        # A datamodule encapsulates the five steps involved in data processing in PyTorch: 1) Download / tokenize / process. 2) Clean and (maybe) save to disk. 3) Load inside Dataset. 4) Apply transforms (rotate, tokenize, etc…). 5) Wrap inside a DataLoader.
        dm = LitDataModule(
            data_folder=self.args.data_folder,
            dataset_name=self.args.dataset_name,
            csv_path=self.args.csv_path,
            fold=self.args.fold,
            label=self.args.label_column,
            label_transform=class_map["fwd"],
            imsize=self.args.imsize,
            batch_size=self.args.batch_size,
            aug=self.args.aug,
            load_to_memory=self.args.load_to_memory,
            tta_n=self.args.tta_n
        )
        return dm

    def _create_model(self, n_classes, class_map, ckpt=None, training=True):
        if training:
            # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
            # A LightningModule organizes your PyTorch code into 6 sections: 1) Initialization (__init__ and setup()). 2) Train Loop (training_step()) 3) Validation Loop (validation_step()) 4) Test Loop (test_step()) 5) Prediction Loop (predict_step()) 6) Optimizers and LR Schedulers (configure_optimizers())
            model = LitModule(
                model=self.args.timm_model_name,
                freeze_base=self.args.freeze_base,
                pretrained=self.args.pretrained,
                criterion=self.args.criterion,
                opt={"name": self.args.opt},
                n_classes=n_classes,
                lr=self.args.lr,
                label_transform=class_map["inv"],
            )
            return model
        else:
            model = LitModule(**ckpt["hyper_parameters"])

            model.load_state_dict(ckpt["state_dict"])
            model.label_transform = class_map["inv"]
            model.freeze()
            return model
    
    def _load_model(self, ckpt):
        return LitModule(**ckpt["hyper_parameters"])

    def _load_checkpoint(self, model):
        ckpt = torch.load(
            self.args.ckpt_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        model.load_state_dict(ckpt["state_dict"])

    def _create_callbacks(self, out_folder):
        # Training callbacks
        checkpoint_callback_best = ModelCheckpoint(
            monitor="val/loss",
            dirpath=out_folder,
            filename=f"{self.outname}_" + "epoch{epoch:02d}_val-loss{val/loss:.2f}",
            auto_insert_metric_name=False,
        )
        checkpoint_callback_last = ModelCheckpoint(
            monitor="epoch",
            mode="max",
            dirpath=out_folder,
            filename=f"{self.outname}_" + "epoch{epoch:02d}_val-loss{val/loss:.2f}_last",
            auto_insert_metric_name=False,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks = [checkpoint_callback_best, checkpoint_callback_last, lr_monitor]
        if self.args.early_stopping:
            callbacks.append(
                EarlyStopping(monitor="val/loss", patience=self.args.early_stopping_patience)
            )
        return callbacks

    def _create_logger(self, model):
        wandb_resume = True if self.args.resume else None
        print(wandb_resume)
        logger = WandbLogger(
            project=self.args.log_dir,
            name=self.outname,
            id=self.uid,
            resume=wandb_resume,
            allow_val_change=wandb_resume,
        )

        logger.watch(model)
        wandb.config.update(self.args, allow_val_change=True)
        # logger = TensorBoardLogger(args.log_dir,
        #                            name=basename,
        #                            version=uid)
        # logger.log_hyperparams(vars(args))
        # logger.log_graph(model)
        return logger

    def _create_trainer(self, callbacks, logger, training=True):
        if training:
            if self.args.smoke_test:
                limit_train_batches = 4
                limit_val_batches = 4
                limit_test_batches = 4
            else:
                limit_train_batches = 1.0
                limit_val_batches = 1.0
                limit_test_batches = 1.0

            # Training
            #  https://lightning.ai/docs/pytorch/stable/common/trainer.html
            # The Lightning Trainer does much more than just “training”. Under the hood, it handles all loop details for you, some examples include: 1) Automatically enabling/disabling grads 2) Running the training, validation and test dataloaders 3) Calling the Callbacks at the appropriate times 4) Putting batches and computations on the correct devices
            trainer = pl.Trainer(
                max_epochs=self.args.max_epochs,
                min_epochs=self.args.min_epochs,
                logger=logger,
                log_every_n_steps=10,
                devices="auto",
                accelerator="auto",
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
                limit_test_batches=limit_test_batches,
                callbacks=callbacks,
                precision=self.args.precision,
                deterministic=self.args.deterministic,
            )
            return trainer
        else:
            trainer = pl.Trainer(
                devices="auto",
                accelerator="auto",
                fast_dev_run=2 if self.args.smoke_test else False,
                logger=False,
            )
            return trainer

    def _perform_training(self, trainer, model, dm, resume_ckpt):
        trainer.fit(model, dm, ckpt_path=resume_ckpt)
        trainer.test(model, datamodule=dm, ckpt_path="best")

    def _tune_lr(self, trainer, model, dm):
        tuner = Tuner(trainer)
        tuner.lr_find(model, dm)
        print(f"New lr: {model.hparams.lr}")
        wandb.config.update({"new_lr": model.hparams.lr}, allow_val_change=True)

    def _save_config(self, out_folder, uid):
        with open(out_folder / f"config_{uid}.yml", "w") as f:
            f.write(yaml.dump(vars(wandb.config)["_items"]))

    def _predict(self, trainer, model, dm, class_map, n_classes, out_folder = None):
        if not self.args.tta:
            trainer.test(model, dm)
            y_true, y_pred = model.y_true, model.y_pred

        else:
            dm.setup()
            trainer.test(model, dataloaders=dm.tta_dataloader())
            y_true = dm.tta_process(model.y_true)
            y_pred = dm.tta_process(model.y_pred)

        df_pred = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        if n_classes > 1:
            softmax = model.softmax

            if self.args.tta:
                softmax = dm.tta_process_softmax(softmax)
            n_classes = softmax.shape[1]
            classes = class_map["inv"](list(range(n_classes)))
            df_prob = pd.DataFrame(data=softmax, columns=classes)
            df = pd.concat((df_pred, df_prob), axis=1)
        else:
            df = df_pred

        if out_folder:
            model_stem = Path(self.args.ckpt_path).stem
            out_stem = f"{self.args.out_prefix}_{model_stem}_{self.args.aug}"
            if self.args.tta:
                out_stem += "_tta"
            outname = out_stem + ".csv"
            df.to_csv(out_folder / outname, index=False)

        return df


    def train_model(self):
        # initialize and get folder where the parameters are saved
        out_folder = self._create_out_folder()

        # get class mapping
        class_map, n_classes = self._load_class_map()

        # get
        dm = self._create_data_module(class_map)

        model = self._create_model(n_classes, class_map)

        if (not self.args.resume) and self.args.ckpt_path:
            self._load_checkpoint(model)
            resume_ckpt = None
        else:
            resume_ckpt = self.args.ckpt_path

        callbacks = self._create_callbacks(out_folder)

        if not self.args.debug:
            logger = self._create_logger(model)
        else:
            logger = True

        trainer = self._create_trainer(callbacks, logger)

        if self.args.auto_lr:
            self._tune_lr(trainer, model, dm)

        if not self.args.debug: # In debug because we can't access wandb.config
            self._save_config(out_folder, self.uid)

        self._perform_training(trainer, model, dm, resume_ckpt)

        dm.visualize_datasets(out_folder / f"aug-{self.args.aug}-{self.uid}")

        print(f"Best model: {callbacks[0].best_model_path} | score: {callbacks[0].best_model_score}")
        return trainer
    
    def predict(self):
        gpu_count = torch.cuda.device_count()

        out_folder = self._create_out_folder(training=False)

        ckpt = torch.load(
            self.args.ckpt_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        class_map, n_classes = self._load_class_map()

        dm = self._create_data_module(class_map)

        model = self._create_model(None, class_map, ckpt, training=False)

        trainer = self._create_trainer(None, None, training=False)

        df_pred = self._predict(trainer, model, dm, class_map, n_classes, out_folder=out_folder)
        
        dm.visualize_datasets(out_folder)




        # Low memory architectures: MobileNetV2, EfficientNet, MnasNet
        # Relatively low memory architectures: RegNet, ResNet, DenseNet, SqueezeNet, ShuffleNetV2
        # More Mixer-B/16, Mixer-L/16
        # def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)