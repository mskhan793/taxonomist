import argparse
import pandas as pd
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import entropy
from PIL import Image
from collections import OrderedDict
import timm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Custom Dataset Class
class Data(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None, split='train', label_encoder=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # Filter the dataframe based on the split
        self.data_frame = self.data_frame[self.data_frame['0'] == split]

        # Initialize label encoder if necessary
        if label_encoder is None and split == 'train':
            self.label_encoder = LabelEncoder()
            self.data_frame['taxon'] = self.label_encoder.fit_transform(self.data_frame['taxon'])
        elif label_encoder is not None:
            self.label_encoder = label_encoder
            self.data_frame['taxon'] = self.label_encoder.transform(self.data_frame['taxon'])
        else:
            raise ValueError("Label encoder must be provided for non-training splits.")
    
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        taxon = self.data_frame.iloc[idx]['individual']
        image_name = self.data_frame.iloc[idx]['img']
        img_path = os.path.join(self.root_dir, taxon, image_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # Label is already encoded as integer
        label = self.data_frame.iloc[idx]['taxon']
        
        return image, label


# Pretrained Feature Extractor using timm library with checkpoint loading
class PretrainedModelFeatureExtractor(pl.LightningModule):
    def __init__(self, model_name='resnet50', checkpoint_path=None):
        super().__init__()
        # Load the model from timm with pretrained weights and remove the last fully connected layer
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)  # num_classes=0 removes the final classification layer
        
        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

        # Freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)


# Ensemble Classifier Module
class ClassifierEnsemble:
    def __init__(self):
        self.svc = SVC(probability=True)  # Support vector classifier with probability output
        self.rf = RandomForestClassifier()  # Random Forest Classifier
        self.lr = LogisticRegression()  # Logistic Regression

    def fit(self, X_train, y_train):
        # Train classifiers in parallel
        self.svc.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)
        self.lr.fit(X_train, y_train)

    def predict_proba(self, X_test):
        # Predict probabilities (softmax outputs) from all classifiers
        svc_probs = self.svc.predict_proba(X_test)
        rf_probs = self.rf.predict_proba(X_test)
        lr_probs = self.lr.predict_proba(X_test)
        return svc_probs, rf_probs, lr_probs

    def entropy_selection(self, svc_probs, rf_probs, lr_probs):
        # Compute entropy for each sample from each classifier
        svc_entropy = entropy(svc_probs, axis=1)
        rf_entropy = entropy(rf_probs, axis=1)
        lr_entropy = entropy(lr_probs, axis=1)

        # Stack all probabilities for comparison
        all_probs = np.stack([svc_probs, rf_probs, lr_probs], axis=1)
        all_entropy = np.stack([svc_entropy, rf_entropy, lr_entropy], axis=1)

        # For each sample, select the classifier with the lowest entropy (most confident)
        selected_probs = []
        for i in range(len(svc_probs)):
            min_entropy_index = np.argmin(all_entropy[i])
            selected_probs.append(all_probs[i, min_entropy_index])

        return np.array(selected_probs)


# Main Function
def main(data_dir, csv_file, checkpoint_paths, num_classes):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Load datasets
    label_encoder = LabelEncoder()

    # Train, validation, and test datasets
    train_dataset = Data(csv_file=csv_file, root_dir=data_dir, transform=transform, split='train', label_encoder=label_encoder)
    val_dataset = Data(csv_file=csv_file, root_dir=data_dir, transform=transform, split='val', label_encoder=label_encoder)
    test_dataset = Data(csv_file=csv_file, root_dir=data_dir, transform=transform, split='test', label_encoder=label_encoder)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the three pretrained feature extractors using the provided checkpoints
    feature_extractors = [PretrainedModelFeatureExtractor(model_name='resnet50', checkpoint_path=ckpt) for ckpt in checkpoint_paths]

    # Extract features for training, validation, and test sets
    def extract_features(loader, feature_extractor):
        features_list, labels_list = [], []
        for batch in loader:
            images, labels = batch
            features = feature_extractor(images)
            features_list.append(features.detach().numpy())
            labels_list.append(labels.numpy())
        return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

    # Extract features from all models and concatenate them
    def extract_features_from_all_models(loader):
        all_features = []
        for extractor in feature_extractors:
            features, labels = extract_features(loader, extractor)
            all_features.append(features)
        return np.concatenate(all_features, axis=1), labels  # Concatenate features from all models

    # Extract features for training, validation, and test sets
    train_features, train_labels = extract_features_from_all_models(train_loader)
    val_features, val_labels = extract_features_from_all_models(val_loader)
    test_features, test_labels = extract_features_from_all_models(test_loader)

    # Initialize ensemble classifiers
    ensemble = ClassifierEnsemble()
    ensemble.fit(train_features, train_labels)

    # Predict on validation set and select based on entropy
    svc_probs_val, rf_probs_val, lr_probs_val = ensemble.predict_proba(val_features)
    selected_probs_val = ensemble.entropy_selection(svc_probs_val, rf_probs_val, lr_probs_val)

    # Get validation predicted labels and evaluate performance
    y_pred_val = np.argmax(selected_probs_val, axis=1)
    val_accuracy = accuracy_score(val_labels, y_pred_val)
    print(f"Validation accuracy: {val_accuracy}")

    # Predict on test set and select based on entropy
    svc_probs_test, rf_probs_test, lr_probs_test = ensemble.predict_proba(test_features)
    selected_probs_test = ensemble.entropy_selection(svc_probs_test, rf_probs_test, lr_probs_test)

    # Get test predicted labels and evaluate performance
    y_pred_test = np.argmax(selected_probs_test, axis=1)
    test_accuracy = accuracy_score(test_labels, y_pred_test)
    print(f"Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory where data is located')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file containing data information')
    parser.add_argument('--checkpoint1', type=str, required=True, help='Path to first pre-trained model checkpoint')
    parser.add_argument('--checkpoint2', type=str, required=True, help='Path to second pre-trained model checkpoint')
    parser.add_argument('--checkpoint3', type=str, required=True, help='Path to third pre-trained model checkpoint')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the dataset')
    args = parser.parse_args()

    checkpoint_paths = [args.checkpoint1, args.checkpoint2, args.checkpoint3]
    main(args.data_dir, args.csv_file, checkpoint_paths, args.num_classes)
