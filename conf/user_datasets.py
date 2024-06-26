from pathlib import Path

import pandas as pd

"""
Defines custom functions for reading dataset data from train-test-splitted csv-files
"""


def preprocess_dataset(data_folder, dataset_name, csv_path=None, fold=None, label=None):
    data_folder = Path(data_folder)
    fnames = {}
    labels = {}
    for set_ in ["train", "val", "test"]:
        if dataset_name == "rodi":
            fnames[set_], labels[set_] = process_split_csv_rodi(
                data_folder, csv_path, set_, fold, label
            )
        elif dataset_name == "finbenthic2":
            fnames[set_], labels[set_] = process_split_csv_finbenthic2(
                data_folder, csv_path, set_, fold, label
            )

        elif dataset_name == "finbenthic1":
            fnames[set_], labels[set_] = process_split_csv_finbenthic1(
                data_folder, csv_path, set_, fold, label
            )

        elif dataset_name == "biodiscover":
            """
            You should change the name of the dataset to match the true name 
            of your BioDiscover dataset
            """
            fnames[set_], labels[set_] = process_split_csv_biodiscover(
                data_folder, csv_path, set_, fold, label
            )

        elif dataset_name == "cifar10":
            fnames[set_], labels[set_] = process_split_csv_cifar10(
                data_folder, csv_path, set_, fold, label
            )

        elif dataset_name == "moss":
            fnames[set_], labels[set_] = process_split_csv_mos(
                data_folder, csv_path, set_, fold, label
            )

        elif dataset_name == "my_dataset":
            """

            YOUR CODE GOES HERE

            """
            fnames[set_], labels[set_] = None, None

        else:
            raise Exception("Unknown dataset name")

    return fnames, labels


def process_split_csv_biodiscover(data_folder, csv_path, set_, fold, label):
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df.apply(
        lambda x: Path(
            data_folder,
            x["Species Name"],
            x["individual"],
            x["Sample Name/Number"],
            x["Image File Name"],
        ).resolve(),
        axis=1,
    ).values

    for fname in fnames:
        assert fname.exists()

    labels = df[label].values

    return fnames, labels


def process_split_csv_finbenthic1(data_folder, csv_path, set_, fold, label):
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df.apply(
        lambda x: Path(data_folder, "Cropped images", x["taxon"], x["img"]).resolve(),
        axis=1,
    ).values

    for fname in fnames:
        assert fname.exists()

    labels = df[label].values

    return fnames, labels


def process_split_csv_finbenthic2(data_folder, csv_path, set_, fold, label):
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df.apply(
        lambda x: Path(data_folder, "Images", x["individual"], x["img"]).resolve(),
        axis=1,
    ).values

    for fname in fnames:
        assert fname.exists()

    labels = df[label].values

    return fnames, labels


def process_split_csv_rodi(data_folder, csv_path, set_, fold, label):
    "RODI -specific function for reading train-test-split csvs"
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df["image"].apply(lambda x: data_folder.resolve() / x).values

    for fname in fnames:
        assert fname.exists()

    labels = df[label].values

    return fnames, labels


def process_split_csv_cifar10(data_folder, csv_path, set_, fold, label):
    "CIFAR10 -specific function for reading train-test-split csvs"
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]

    fnames = df["Filename"].apply(lambda x: data_folder.resolve() / x).values

    # # Print file paths for debugging
    # print("File paths:", fnames)

    # Check if the files exist
    for fname in fnames:
        assert fname.exists(), f"File '{fname}' does not exist"  # Add informative message

    labels = df[label].values

    return fnames, labels


# def process_split_csv_mos(data_folder, csv_path, set_, fold, label):
#     "CIFAR10 -specific function for reading train-test-split csvs"
#     df0 = pd.read_csv(csv_path)
#     df = df0[df0[str(fold)] == set_]
   
#     # fnames = df.apply(
#     #     lambda x: Path(data_folder, "images", x["Label"], x["ImageName"]).resolve(),
#     #     axis=1,
#     # ).values


#     fnames = []
#     for _, row in df.iterrows():
#         path = Path(data_folder, "images", row["Label"], row["ImageName"]).resolve()
#         print(path)
#         fnames.append(path)

#     #Print file paths for debugging
#     #print("The path is: ", pp)
#     print("File paths:", fnames)

#     for fname in fnames:
#         assert fname.exists()

#     labels = df[label].values

#     return fnames, labels

def process_split_csv_mos(data_folder, csv_path, set_, fold, label):
    "MOS-specific function for reading train-test-split csvs"
    df0 = pd.read_csv(csv_path)
    df = df0[df0[str(fold)] == set_]
   
    fnames = []
    for _, row in df.iterrows():
        path = Path(data_folder, row["Label"], row["ImageName"]).resolve()
        #print(f"Checking path: {path}")  # Debugging statement
        if not path.exists():
            print(f"Path does not exist: {path}")  # Error identification
        fnames.append(path)

    #print("File paths:", fnames)

    for fname in fnames:
        assert fname.exists(), f"File '{fname}' does not exist"  # Detailed error message

    labels = df[label].values

    return fnames, labels
