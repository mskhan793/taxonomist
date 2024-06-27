import os
from datetime import datetime
from pathlib import Path

import albumentations as A
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import PIL.Image as Image
import torchvision.transforms.functional as TF
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from tqdm import tqdm   

from .utils import load_module_from_path, read_image, visualize_dataset, load_class_map, encode_labels, calculate_class_counts


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset for reading image/target pairs from a filepath list

    Args:
        filenames: list of filepaths
        y: list of targets
        preload_transform: transform to apply to the PIL image after loading and before
                            loading into memory
        transform: transform to apply to the image after loading  
        minority_transform: transform to apply to minority class samples
        minority_classes: list of minority class labels
        load_to_memory: if True, the images are loaded into memory
    """

    def __init__(
        self,
        filenames: list,
        y: list,
        label_map=None,
        preload_transform=None,
        transform=None,
        minority_transform=None,
        minority_classes=None,
        load_to_memory=True,
    ):
        self.filenames = filenames
        self.y = y
        self.label_map = label_map
        self.preload_transform = preload_transform
        self.transform = transform
        self.mem_dataset = None
        self.minority_transform = minority_transform
        self.minority_classes = minority_classes
        self.mem_dataset = None
        if load_to_memory:
            self.mem_dataset = []
            print("Loading dataset to memory...")
            for i in tqdm(range(len(filenames))):
                self.mem_dataset.append(self.__readfile(i))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Reads item either from memory or from disk"""
        if self.mem_dataset:
            X = self.mem_dataset[index]
        else:
            X = self.__readfile(index)

        if self.transform:
            if self.minority_classes is not None and self.y[index] in self.minority_classes:
                X = self._apply_transform(X, self.minority_transform)
            else:
                X = self._apply_transform(X, self.transform)

        if self.y is not None:
            y = torch.as_tensor(self.y[index], dtype=torch.float32)
        else:
            y = None
        batch = {"x": X, "y": y, "fname": str(self.filenames[index])}
        return batch
        #return X, y

    def _apply_transform(self, img, transform):
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        img = transform(img)
        if isinstance(img, Image.Image):
            img = TF.to_tensor(img)
        return img

    def __readfile(self, index):
        """Actual loading of the item"""
        fname = self.filenames[index]
        img = read_image(fname)
        if self.preload_transform:
            img = self.preload_transform(img)
        return img


class LitDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for an arbitary dataset

    Args:

        csv_path: path to the csv file containing the filenames

        data_folder: path to the folder containing the images

        fold: cross-validation fold to use

        label: column to use as the label

        label_transform: function to apply to the label list

        batch_size: batch size

        imsize: size of the images

        load_to_memory: whether to load the images to memory

        tta_n: The number of test-time-augmentation rounds
    """

    def __init__(
        self,
        data_folder: str,
        dataset_config_path: str = None,
        dataset_name: str = None,
        csv_path: str = None,
        fold: int = None,
        label: str = None,
        aug: str = "none",
        batch_size: int = 128,
        imsize: int = 224,
        label_transform=None,
        load_to_memory: bool = False,
        tta_n: int = 5,
        class_map_path: str = None,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.dataset_config_path = dataset_config_path
        self.dataset_name = dataset_name

        self.csv_path = csv_path
        self.fold = fold
        self.label = label

        self.aug = aug
        self.batch_size = batch_size
        self.imsize = imsize

        self.label_transform = label_transform
        self.load_to_memory = load_to_memory
        self.tta_n = tta_n

        self.class_map_path = class_map_path

        self.labels = {} # Initialize the labels attribute

        self.cpu_count = int(
            os.getenv("SLURM_CPUS_PER_TASK") or torch.multiprocessing.cpu_count()
        )
        self.drop_last = lambda x: True if len(x) % batch_size == 1 else False

        self.aug_args = {"imsize": imsize}
        #self.tf_test, self.tf_train = choose_aug(self.aug, self.aug_args)
        self.class_counts = None  # Add this line

    def setup(self, stage=None):
        dataset_config_module = load_module_from_path(self.dataset_config_path)

        fnames, labels = dataset_config_module.preprocess_dataset(
            data_folder=self.data_folder,
            dataset_name=self.dataset_name,
            csv_path=self.csv_path,
            fold=self.fold,
            label=self.label,
        )
        # Ensure class_map_path is correctly assigned before checking it
        class_map_path = self.class_map_path
        if class_map_path is None:
            raise ValueError("class_map_path is None. Please provide a valid path.")

        print(f"Loading class map from {class_map_path}")
        class_map = load_class_map(class_map_path)

        # Forward map is used to encode labels
        label_to_index = class_map['fwd_dict']

        # Encode the training labels using the forward map
        labels_encoded = {k: encode_labels(v, label_to_index) for k, v in labels.items()}
        self.labels_encoded = labels_encoded
            
        # Number of classes
        num_classes = len(label_to_index)
            
        # Store the class counts
        self.class_counts = calculate_class_counts(labels_encoded['train'], num_classes)

        if stage == 'fit' or stage is None:
            # Load the class map
            class_map_path = self.class_map_path
            class_map = load_class_map(class_map_path)

            # Forward map is used to encode labels
            label_to_index = class_map['fwd_dict']

            # Encode the training labels using the forward map
            labels_encoded = encode_labels(labels["train"], label_to_index)
            self.labels_encoded = {"train": labels_encoded}
            
            # Number of classes
            num_classes = len(label_to_index)
            
            # Store the class counts
            self.class_counts = calculate_class_counts(labels_encoded, num_classes)
            # #   print(self.class_counts)

            if self.aug.startswith("up-sampling"):
                self.tf_train, tf_aug_02, self.tf_test, minority_classes = choose_aug(self.aug, self.aug_args, class_counts=self.class_counts)

                # Determine the upsampling factor from the augmentation string
                upsample_factor = int(self.aug.split("up-sampling")[-1]) if self.aug[-1].isdigit() else 4  # Default to 4 if no digit is found

                # Upsample minority classes
                upsampled_fnames = []
                upsampled_labels = []
                upsampled_original_labels = []
                for fname, encoded_label, original_label in zip(fnames["train"], labels_encoded["train"], labels["train"]):
                    upsampled_fnames.append(fname)
                    upsampled_labels.append(encoded_label)
                    upsampled_original_labels.append(original_label)
                    if encoded_label in minority_classes:
                        for _ in range(upsample_factor):  # Adjust this number to control the amount of upsampling
                            upsampled_fnames.append(fname)
                            upsampled_labels.append(encoded_label)
                            upsampled_original_labels.append(original_label)
                fnames["train"] = upsampled_fnames
                labels_encoded["train"] = upsampled_labels
                labels["train"] = upsampled_original_labels

                if self.label_transform:
                    labels["train"] = self.label_transform(labels["train"])
                #     labels["val"] = self.label_transform(labels["val"])
                #     labels["test"] = self.label_transform(labels["test"])

                self.trainset = Dataset(
                    fnames["train"],
                    labels["train"],
                    preload_transform=None,
                    transform=self.tf_train,
                    minority_transform=tf_aug_02,
                    minority_classes=minority_classes,
                    load_to_memory=self.load_to_memory,
                )
                
                #self.tf_test = tf_test
            else:
                self.tf_test, self.tf_train = choose_aug(self.aug, self.aug_args)[:2]

                if self.label_transform:
                    labels["train"] = self.label_transform(labels["train"])
                #     labels["val"] = self.label_transform(labels["val"])
                #     labels["test"] = self.label_transform(labels["test"])

                self.trainset = Dataset(
                    fnames["train"],
                    labels["train"],
                    preload_transform=None,
                    transform=self.tf_train,
                    load_to_memory=self.load_to_memory,
                )
        else:
            self.tf_test, self.tf_train = choose_aug(self.aug, self.aug_args)[:2]
            if self.label_transform:
                labels["train"] = self.label_transform(labels["train"])
            self.trainset = Dataset(
                    fnames["train"],
                    labels["train"],
                    preload_transform=None,
                    transform=self.tf_train,
                    load_to_memory=self.load_to_memory,
                )

        if self.label_transform:
            #labels["train"] = self.label_transform(labels["train"])
            labels["val"] = self.label_transform(labels["val"])
            labels["test"] = self.label_transform(labels["test"])

        # self.trainset = Dataset(
        #     fnames["train"],
        #     labels["train"],
        #     preload_transform=None,
        #     transform=self.tf_train,
        #     load_to_memory=self.load_to_memory,
        # )

        self.valset = Dataset(
            fnames["val"],
            labels["val"],
            preload_transform=None,
            transform=self.tf_test,
            load_to_memory=self.load_to_memory,
        )

        self.testset = Dataset(
            fnames["test"],
            labels["test"],
            preload_transform=None,
            transform=self.tf_test,
            load_to_memory=self.load_to_memory,
        )

        tta_list = [self.testset] + [
            Dataset(
                fnames["test"],
                labels["test"],
                preload_transform=None,
                transform=self.tf_train,
                load_to_memory=self.load_to_memory,
            )
            for _ in range(self.tta_n - 1)
        ]

        self.ttaset = torch.utils.data.ConcatDataset(tta_list)

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last(self.trainset),
            num_workers=self.cpu_count,
        )

        return trainloader

    def val_dataloader(self):
        valloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size,
            drop_last=self.drop_last(self.valset),
            num_workers=self.cpu_count,
        )

        return valloader

    def test_dataloader(self):
        testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.cpu_count,
        )

        return testloader

    def tta_dataloader(self):
        ttaloader = torch.utils.data.DataLoader(
            self.ttaset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.cpu_count,
        )
        return ttaloader

    def tta_process(self, y):
        A = y.reshape(self.tta_n, len(self.testset))
        return pd.DataFrame(A).T.mode(axis=1).iloc[:, 0].values

    def tta_process_output(self, output):
        A = output.T.reshape(output.shape[1], self.tta_n, len(self.testset))
        return A.mean(axis=1).T

    def visualize_datasets(self, folder):
        _now = datetime.now().strftime("%H%M%S")
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        visualize_dataset(self.trainset, v=False, name=folder / f"{_now}-train.jpg")
        visualize_dataset(self.valset, v=False, name=folder / f"{_now}-val.jpg")
        visualize_dataset(self.testset, v=False, name=folder / f"{_now}-test.jpg")


class ConvertToUint8:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.tensor(img)
        elif isinstance(img, Image.Image):
            img = TF.to_tensor(img)
        elif not isinstance(img, torch.Tensor):
            raise TypeError(f"Input img should be PIL Image, ndarray, or tensor. Got {type(img)} instead.")
        img = TF.convert_image_dtype(img, dtype=torch.uint8)
        return TF.to_pil_image(img)

def choose_aug(aug, args, class_counts=None):
    imsize = args["imsize"]
    a_end_tf = A.Compose(
        [
            A.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    keep_aspect_resize = A.Compose(
        [
            A.LongestMaxSize(max_size=imsize),
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=0),
        ],
        p=1.0,
    )
    if aug == "none":
        tf_test = transforms.Compose(
            [
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        tf_train = tf_test


    
    elif aug == "torch-only-flips":
        tf_test = transforms.Compose(
            [
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        tf_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif aug == "aug-01":
        tf_test = transforms.Compose(
            [
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        tf_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomChoice(
                    [
                        transforms.GaussianBlur(kernel_size=(3, 3)),
                        transforms.ColorJitter(brightness=0.5, hue=0.1),
                    ]
                ),
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(degrees=(0, 360)),
                        transforms.RandomPerspective(distortion_scale=0.1),
                        transforms.RandomAffine(
                            degrees=(30, 70), translate=(0.1, 0.3), scale=(0.75, 0.9)
                        ),
                        transforms.RandomResizedCrop(size=(imsize, imsize)),
                    ]
                ),
                transforms.RandomChoice(
                    [
                        transforms.RandomAdjustSharpness(sharpness_factor=2),
                        transforms.RandomAutocontrast(),
                        transforms.RandomEqualize(),
                    ]
                ),
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif aug == "color-jitter":
        tf_test = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        tf_train = tf_test

    elif aug == "keep-aspect":
        transform_test = A.Compose([keep_aspect_resize, a_end_tf])
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = tf_test
    
    elif aug == "flips-cont-rotate":
        transform_test = A.Compose(
            [
                A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                A.Resize(imsize, imsize, p=1.0),
                A.Flip(),
                A.RandomRotate90(p=1.0),
                A.Rotate(p=0.5, border_mode=0),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]

    elif aug.startswith("flips"):
        keep_aspect = "keep-aspect" in aug
        rotate = "rotate" in aug
        transform_test = A.Compose(
            [
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                A.Flip(),
                A.RandomRotate90(p=0.5) if rotate else A.NoOp(),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]

    elif aug.startswith("color"):
        keep_aspect = "keep-aspect" in aug
        transform_test = A.Compose(
            [
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                A.OneOf(
                    [
                        A.ColorJitter(),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20
                        ),
                        A.ToGray(p=0.2),
                    ],
                    p=0.8,
                ),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]

    elif aug.startswith("geometric"):
        border = 0
        keep_aspect = "keep-aspect" in aug
        transform_test = A.Compose(
            [
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                A.OneOf(
                    [
                        A.Rotate(limit=360, border_mode=border),
                        A.Perspective(pad_mode=border),
                        A.Affine(
                            scale=(0.5, 0.9),
                            translate_percent=0.1,
                            shear=(-30, 30),
                            rotate=360,
                        ),
                    ],
                    p=0.8,
                ),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]

    elif aug.startswith("aug-02"):
        apply_eq = "EQ" in aug
        apply_bw = "BW" in aug
        keep_aspect = "keep-aspect" in aug
        border = 0
        transform_test = A.Compose(
            [
                A.ToGray(p=1.0) if apply_bw else A.NoOp(),
                A.Equalize(p=1.0) if apply_eq else A.NoOp(),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                # Possible equalization
                A.ToGray(p=1.0) if apply_bw else A.NoOp(),
                A.Equalize(p=1.0) if apply_eq else A.NoOp(),
                # Slow pixel tf
                A.Posterize(p=0.1),
                A.NoOp() if apply_eq else A.Equalize(p=0.2),
                A.CLAHE(0.2),
                A.OneOf(
                    [
                        A.GaussianBlur(),
                        A.Sharpen(),
                    ],
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Solarize(p=0.2),
                # Colors
                A.OneOf(
                    [
                        A.ColorJitter(),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2
                        ),
                        A.NoOp() if apply_bw else A.ToGray(p=0.5),
                    ],
                    p=0.2,
                ),
                # Slow geometrical tf
                A.OneOf(
                    [
                        A.Rotate(limit=360, border_mode=border),
                        A.Perspective(pad_mode=border),
                        A.Affine(
                            scale=(0.5, 0.9),
                            translate_percent=0.1,
                            shear=(-30, 30),
                            rotate=360,
                        ),
                    ],
                    p=0.8,
                ),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                A.CoarseDropout(
                    max_holes=30,
                    max_height=15,
                    max_width=15,
                    min_holes=1,
                    min_height=2,
                    min_width=2,
                ),
                A.RandomSizedCrop(
                    min_max_height=(int(0.5 * imsize), int(0.8 * imsize)),
                    height=imsize,
                    width=imsize,
                    p=0.3,
                ),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]
    
    elif aug.startswith("up-sampling"):
        if class_counts is None:
            raise ValueError("class_counts must be provided for up-sampling augmentation")
        
        # Get the indices of the 5 minority classes
        minority_classes = sorted(range(len(class_counts)), key=lambda i: class_counts[i])[:1]
        
        tf_aug_02 = transforms.Compose(
            [
                #transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                ConvertToUint8(),  # Add conversion to uint8 before equalize
                # Apply aug-02 specific transformations here...
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomChoice(
                    [
                        transforms.GaussianBlur(kernel_size=(3, 3)),
                        #transforms.ColorJitter(brightness=0.5, hue=0.1), #changes brightness and hue.
                    ]
                ),
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(degrees=(0, 360)),
                        transforms.RandomPerspective(distortion_scale=0.1),
                        transforms.RandomAffine(
                            degrees=(30, 70), translate=(0.1, 0.3), scale=(0.75, 0.9)
                        ),
                        transforms.RandomResizedCrop(size=(imsize, imsize)),
                    ]
                ),
                transforms.RandomChoice(
                    [
                        #transforms.RandomAdjustSharpness(sharpness_factor=2), #modify the image sharpness
                        #transforms.RandomAutocontrast(),      #modify the image contrast.
                        transforms.RandomEqualize(),
                    ]
                ),
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        
        tf_none = transforms.Compose(
            [
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        
        # def tf_train(image, label):
        #     if label in minority_classes:
        #         return tf_aug_02(image)
        #     else:
        #         return tf_none(image)

        # tf_test = tf_none  # Use 'none' augmentation for test as per the specification

        return tf_none, tf_aug_02, tf_none, minority_classes
    #self.tf_train, tf_aug_02, self.tf_test, minority_classes
    else:
        raise ValueError(f"Invalid augmentation value {aug}")
    
    return tf_test, tf_train
