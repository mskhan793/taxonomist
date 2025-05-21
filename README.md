# Taxonomist - a species classification pipeline

Taxonomist is a pipeline for classifying images of species, with a focus on scientific applications in natural sciences. It describes a simple framework that is easy to extend an modify for different needs. Taxonomist takes care of most parts of the classification pipeline (training, cross-validation, logging, evaluation) and lets you focus on designing the experiments and analyzing the results of different classification approaches. 

Features:
- Image classification and regression with state-of-the-art Deep Learning models from the [PyTorch Image Models (`timm`)](https://timm.fast.ai/) library.
- Transparent and easy to modify. Operates around simple python scripts and `.csv`-files without opaque modules and functions with side-effects.
- Opinionated folder structure designed for scientific, reproducible experiments.
- Easy result comparisons between experiments and across datasets.
- Produces results in commonly used `.csv` format that can be further analyzed with other tools
- Implements best practices for classifier evaluation, such as bootstrap confidence intervals and cross-validation.

In essence, Taxonomist is a framework around [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), providing an opinionated project structure for scientific experiments using supervised learning on hierarchical data.

Note that Taxonomist is still under heavy development and large changes can be introduced!

# Installation

Clone the repository

```bash
git clone https://github.com/mikkoim/taxonomist.git
cd taxonomist
```

Install anaconda or miniconda, for example by running the commands:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

For windows, see the installation instructions for Miniconda.

Installing [Mamba](https://github.com/mamba-org/mamba) is recommended as it makes the installation much faster than with conda. In the ```(base)``` environment, run
```bash
conda install mamba -n base -c conda-forge
```
Now you can replace the ```conda``` commands with ```mamba``` when installing packages.

Next, Create the environment

```bash
mamba env create -f environment.yml
conda activate taxonomist
```

And install the library package:

```bash
pip install -e .
```

# Getting started

The workflow in [docs/workflows/00_workflow_rodi.md](docs/workflows/00_workflow_rodi.md) is a good place to start. It walks through all the features, and the model training takes around 30 minutes with a decent GPU. 

# Overview

To use Taxonomist with your own data, you have to produce data loading functions to make your dataset compatible with the pipeline.

1. **Load and get to know your dataset**

    Analyze your dataset structure and find a way to represent the dataset in a table format. The table should contain the at least the following columns:

    - filename
    - label

    If the dataset has a nested folder structure, columns that specify the location in the folder structure are needed. Also, if there is a grouping among the images, for example if there are several images from a specimen, a grouping identifier is needed.

    | label | folder | individual | filename |
    | --- | --- | --- | --- |
    | cat | felines | A | 01.png |
    | cat | felines | A | 02.png |
    | dog | canines | B | 03.png |
    | dog | canines | B | 04.png |
    | wolf | canines | C | 05.png |

    The columns can also contain any other metadata that is seemed useful.

    It is useful to create a separate `data` folder, with two subfolders:
    - `raw`: Contains raw data that should be immutable
    - `processed`: Contains files that are processed from the raw data using scripts, like the preprocessing scripts.

1. **Preprocessing**
    - Create a preprocessing script that reads the filenames in your dataset and produces a table like above (examples in `scripts/preprocessing/`). The preprocessing script should also create a list of all the labels in the dataset into a text file. This label list is used as a label mapping, ensuring that all labels get a proper index even for folds where all labels are not present.
    - Add data loading functions to the library (examples in `src/taxonomist/datasets.py`)
    
    The data loading function should be able resolve into a full path when given a root directory. The root directory can be specified during training so the data location can change without changing the dataset table.


When these steps are complete, Taxonomist can automate the rest of the classification pipeline:

3. **Train-test-val -splits**: Handles splitting the dataset into train, test and validation splits, handling stratification and possible groups where data leakage could occur. The test sets are mutually exclusive, together becoming the full dataset.
3. **Training**: Trains a deep neural network of your choice from the architectures supported by `timm`. 
3. **Prediction**: 
    - Prediction with test-time augmentation
    - Prediction can be done easily across datasets, with previously trained models and mixed label sets.
3. **Evaluation**:
    - Cross-validation
    - Grouping: If the unit of classification is a group, such as an individual specimen, all classifications from this group can be aggregated to produce better estimates.
3. **Comparison**: Comparison between models, datasets and experiments is easy with the flexible comparison script. It produces a csv file with all the results for easy analysis. Experiments, models and datasets can be tagged with arbitary tags that appear in the final result table.

Each step produces intermediary files in csv format, making custom analysis and modifications easy.

# File system
Taxonomist is based on an opinionated file system that produces following output files:

- Model checkpoints (weights, hyperparameters)
- Train-time augmentation visualizations
- Predict-time augmentation visualizations
- Prediction outputs as csv, with softmax scores for all classes
- Grouped prediction outputs
- Metrics for several models in csv format

# Loss Functions

Taxonomist provides three different loss functions to handle various classification scenarios, especially for imbalanced datasets common in species classification:

## Cross Entropy Loss

The standard cross-entropy loss function is the default choice for classification tasks. It works well when classes are balanced.

```python
# Example usage in your configuration:
criterion: "cross-entropy"
```

## Focal Loss

Focal Loss is designed to address class imbalance by down-weighting the loss assigned to well-classified examples, focusing more on hard, misclassified examples. It's particularly useful for datasets with a high imbalance ratio.

```python
# Example usage in your configuration:
criterion: "focal"
params:
  gamma: 2.0  # Adjusts the down-weighting of well-classified examples (default is 2.0)
```

The Focal Loss automatically adjusts class weights based on the class distribution in your dataset. The `gamma` parameter controls how much to down-weight easy examples - higher values increase focus on hard examples.

## Class Imbalance Loss (CILoss)

This is a specialized loss function for handling severe class imbalance. It applies a logarithmic weight based on class frequencies and includes an exponential term to adjust the loss based on the model's confidence.

```python
# Example usage in your configuration:
criterion: "class-imbalance"
params:
  k: 0.3     # Controls the exponential term (default is 0.3)
  theta: 3.0  # Offset for the logarithmic weighting (default is 3.0)
```

### When to use which loss function?

- **Cross Entropy**: Use for balanced datasets or as a baseline
- **Focal Loss**: Use when you have moderate class imbalance and want to focus on hard examples
- **Class Imbalance Loss**: Use for severe class imbalance where some classes have very few examples

You can specify the loss function in your training configuration file or command line arguments when running the training script.

# Data Sampling Techniques

When working with imbalanced datasets, which is common in species classification, you can apply various sampling techniques to improve model performance. Taxonomist supports several approaches to handle class imbalance at the data level:

## Upsampling

Upsampling increases the number of samples in minority classes by randomly duplicating existing samples. This helps the model learn better from underrepresented classes.

```python
# General example of upsampling minority classes
import pandas as pd
from sklearn.utils import resample

# Example with a dataset table
dataset = pd.read_csv('your_dataset.csv')
train_data = dataset[dataset['split'] == 'train']  # Only modify training data

# Set target count for each class
target_samples = 1000  # Choose appropriate value for your dataset

upsampled_data = pd.DataFrame()
# Process each class separately
for label, group in train_data.groupby('label'):
    if len(group) < target_samples:
        # Only upsample if below target
        upsampled_group = resample(
            group,
            replace=True,  # Sample with replacement
            n_samples=target_samples,
            random_state=42,
            # Optionally stratify by another column to preserve structure
            stratify=group['group_id'] if 'group_id' in group.columns else None
        )
        upsampled_data = pd.concat([upsampled_data, upsampled_group])
    else:
        # Keep larger classes as they are
        upsampled_data = pd.concat([upsampled_data, group])

# Combine with validation/test data which remains unchanged
final_data = pd.concat([upsampled_data, dataset[dataset['split'] != 'train']])
```

## Downsampling

Downsampling reduces the number of samples in majority classes to balance the dataset. This can help prevent the model from being biased toward majority classes.

```python
# General example of downsampling majority classes
import pandas as pd
from sklearn.utils import resample

# Example with a dataset table
dataset = pd.read_csv('your_dataset.csv')
train_data = dataset[dataset['split'] == 'train']

# Set maximum samples per class
max_samples = 500  # Choose appropriate value for your dataset

downsampled_data = pd.DataFrame()
# Process each class separately
for label, group in train_data.groupby('label'):
    if len(group) > max_samples:
        # Only downsample if above target
        downsampled_group = resample(
            group,
            replace=False,  # Sample without replacement
            n_samples=max_samples,
            random_state=42,
            # Optionally stratify by another column to preserve structure
            stratify=group['group_id'] if 'group_id' in group.columns else None
        )
        downsampled_data = pd.concat([downsampled_data, downsampled_group])
    else:
        # Keep smaller classes as they are
        downsampled_data = pd.concat([downsampled_data, group])

# Combine with validation/test data which remains unchanged
final_data = pd.concat([downsampled_data, dataset[dataset['split'] != 'train']])
```

## Hybrid Approach (Up-Down Sampling)

A hybrid approach combines both upsampling and downsampling to achieve balanced classes. This targets a specific number of samples per class by either upsampling minority classes or downsampling majority classes.

```python
# General example of hybrid up-down sampling
import pandas as pd
from sklearn.utils import resample

# Example with a dataset table
dataset = pd.read_csv('your_dataset.csv')
train_data = dataset[dataset['split'] == 'train']

# Set target samples for each class
target_samples = 800  # Choose appropriate value for your dataset

balanced_data = pd.DataFrame()
# Process each class separately
for label, group in train_data.groupby('label'):
    if len(group) < target_samples:
        # Upsample minority class
        resampled_group = resample(
            group,
            replace=True,  # Sample with replacement
            n_samples=target_samples,
            random_state=42,
            stratify=group['group_id'] if 'group_id' in group.columns else None
        )
    elif len(group) > target_samples:
        # Downsample majority class
        resampled_group = resample(
            group,
            replace=False,  # Sample without replacement
            n_samples=target_samples,
            random_state=42,
            stratify=group['group_id'] if 'group_id' in group.columns else None
        )
    else:
        # Keep as is if already at target size
        resampled_group = group
        
    balanced_data = pd.concat([balanced_data, resampled_group])

# Combine with validation/test data which remains unchanged
final_data = pd.concat([balanced_data, dataset[dataset['split'] != 'train']])
```

## Combining with Augmentation

Sampling techniques can be combined with image augmentation for even better results. Taxonomist supports various augmentation strategies through the Albumentations library.

When upsampling, consider applying different augmentations to the duplicated samples to increase diversity:

```python
# Example of upsampling with augmentation
import pandas as pd
import numpy as np
import cv2
from sklearn.utils import resample
import albumentations as A

# Define augmentation pipeline
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(p=0.2),
    # Add more transformations as needed
])

# Function to apply augmentation
def augment_images(image_paths, augmentation, n_augmentations=1):
    augmented_images = []
    augmented_paths = []
    
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for i in range(n_augmentations):
            augmented = augmentation(image=image)['image']
            # Save augmented image or keep in memory
            aug_path = f"{path.split('.')[0]}_aug_{i}.{path.split('.')[1]}"
            cv2.imwrite(aug_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            augmented_paths.append(aug_path)
    
    return augmented_paths

# Apply in upsampling workflow
# Instead of just duplicating samples, create new augmented versions
```

### Tips for Effective Sampling

1. **Only modify training data** - Always keep your validation and test sets with their original distribution
2. **Consider stratification** - When sampling, stratify by relevant groups (e.g., specimen ID) to maintain data structure
3. **Experiment with target values** - The optimal number of samples per class depends on your specific dataset
4. **Combine with loss functions** - Use appropriate loss functions alongside sampling techniques for best results
5. **Monitor performance** - Track metrics on the validation set to ensure sampling improves model generalization