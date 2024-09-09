import argparse
import pandas as pd
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np
from scipy.stats import entropy
from PIL import Image
import timm
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed

# Custom Dataset Class
class Data(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None, split='train', label_encoder=None, split_column='0'):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # Filter the dataframe based on the split
        self.data_frame = self.data_frame[self.data_frame[split_column] == split]

        # Initialize or use the provided label encoder
        if split == 'train':
            self.label_encoder = LabelEncoder() if label_encoder is None else label_encoder
            self.data_frame['taxon'] = self.label_encoder.fit_transform(self.data_frame['taxon'])
        else:
            if label_encoder is None:
                raise ValueError("Label encoder must be provided for non-training splits.")
            self.label_encoder = label_encoder
            self.data_frame['taxon'] = self.label_encoder.transform(self.data_frame['taxon'])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        taxon = self.data_frame.iloc[idx]['individual']
        image_name = self.data_frame.iloc[idx]['img']
        img_path = os.path.join(self.root_dir, taxon, image_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        label = self.data_frame.iloc[idx]['taxon']
        
        return image, label

# Pretrained Feature Extractor using timm library
class PretrainedModelFeatureExtractor(pl.LightningModule):
    def __init__(self, model_name='resnet50', checkpoint_path=None):
        super().__init__()
        # Load the model from timm without final classification layer
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)

        # Load checkpoint if provided
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint path {checkpoint_path} not found.")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Adjust the keys to remove any unwanted prefixes
            new_state_dict = {}
            for key in state_dict:
                if key.startswith('model.base_model.'):
                    new_key = key.replace('model.base_model.', '')
                elif key.startswith('model.proj_head.') or key.startswith('criterion.'):
                    continue
                else:
                    new_key = key
                new_state_dict[new_key] = state_dict[key]

            self.model.load_state_dict(new_state_dict, strict=False)

        # Freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

# Helper function to fit and score a single classifier with given params
def evaluate_model(clf, param_grid, clf_name, X_train, y_train, X_val, y_val):
    best_score = 0
    best_param = None

    for params in ParameterGrid(param_grid):
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        score = balanced_accuracy_score(y_val, y_pred)

        print(f"{clf_name} - Params: {params}, Score: {score}")

        if score > best_score:
            best_score = score
            best_param = params

    return best_param, best_score

# Ensemble Classifier with joblib for parallel execution
class ClassifierEnsemble:
    def __init__(self, n_jobs=-1):
        self.svc = SVC(probability=True)
        self.rf = RandomForestClassifier()
        self.lr = LogisticRegression()

        self.param_grid_svc = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        self.param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
        self.param_grid_lr = {'C': [0.1, 1, 10]}

        self.n_jobs = n_jobs
        self.best_params = {}  # Store best parameters for each classifier

    def fit(self, X_train, y_train, X_val, y_val):
        print("Starting parameter search...")

        best_scores = {}

        # Parallelize the grid search for all classifiers using joblib
        results = Parallel(n_jobs=self.n_jobs)(delayed(evaluate_model)(
            clf, param_grid, clf_name, X_train, y_train, X_val, y_val
        ) for clf, param_grid, clf_name in [
            (self.svc, self.param_grid_svc, 'SVC'),
            (self.rf, self.param_grid_rf, 'RandomForest'),
            (self.lr, self.param_grid_lr, 'LogisticRegression')
        ])

        # Assign best params and scores from parallelized grid search
        for (clf_name, clf), (best_param, best_score) in zip([('SVC', self.svc), ('RandomForest', self.rf), ('LogisticRegression', self.lr)], results):
            self.best_params[clf_name] = best_param
            best_scores[clf_name] = best_score
            
            # Train with the best params
            clf.set_params(**best_param)
            clf.fit(X_train, y_train)
            
            print(f"Best params for {clf_name}: {best_param}")
            print(f"Best validation score for {clf_name}: {best_score}")

        print("Finished parameter search.")
        print("\nBest parameters for each classifier:")
        for clf_name, params in self.best_params.items():
            print(f"{clf_name}: {params}")

    def predict_proba(self, X_test):
        svc_probs = self.svc.predict_proba(X_test)
        rf_probs = self.rf.predict_proba(X_test)
        lr_probs = self.lr.predict_proba(X_test)
        return svc_probs, rf_probs, lr_probs

    def entropy_selection(self, svc_probs, rf_probs, lr_probs):
        svc_entropy = entropy(svc_probs, axis=1)
        rf_entropy = entropy(rf_probs, axis=1)
        lr_entropy = entropy(lr_probs, axis=1)

        all_probs = np.stack([svc_probs, rf_probs, lr_probs], axis=1)
        all_entropy = np.stack([svc_entropy, rf_entropy, lr_entropy], axis=1)

        selected_probs = []
        for i in range(len(svc_probs)):
            min_entropy_index = np.argmin(all_entropy[i])
            selected_probs.append(all_probs[i, min_entropy_index])

        return np.array(selected_probs)

# GPU usage and Efficient Feature Extraction
# def extract_features(loader, feature_extractor):
#     feature_extractor.to('cuda' if torch.cuda.is_available() else 'cpu')
#     features_list, labels_list = [], []
#     print(f"Extracting features from {len(loader.dataset)} samples...")
    
#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(tqdm(loader)):
#             images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
#             features = feature_extractor(images)
#             features_list.append(features.cpu())
#             labels_list.append(labels)

#     features = torch.cat(features_list).numpy()
#     labels = torch.cat(labels_list).numpy()
#     return features, labels

def extract_features(loader, feature_extractor):
    feature_extractor.to('cuda' if torch.cuda.is_available() else 'cpu')
    features_list, labels_list = [], []
    print(f"Extracting features from {len(loader.dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader)):
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            features = feature_extractor(images)
            features_list.append(features.cpu().numpy())  # Convert to numpy immediately
            labels_list.append(labels.numpy())
            
            if (batch_idx + 1) % 10 == 0:  # Process in smaller chunks
                yield np.concatenate(features_list), np.concatenate(labels_list)
                features_list, labels_list = [], []
    
    if features_list:
        yield np.concatenate(features_list), np.concatenate(labels_list)

# def extract_features_from_all_models(loader, feature_extractors):
#     all_features = []
#     for extractor in feature_extractors:
#         features, labels = extract_features(loader, extractor)
#         all_features.append(features)
#     return np.concatenate(all_features, axis=1), labels

def extract_features_from_all_models(loader, feature_extractors):
    all_features = [[] for _ in range(len(feature_extractors))]
    all_labels = []
    
    for i, extractor in enumerate(feature_extractors):
        for features, labels in extract_features(loader, extractor):
            all_features[i].append(features)
            if i == 0:  # Only append labels once
                all_labels.append(labels)
    
    concatenated_features = np.concatenate([np.concatenate(f) for f in all_features], axis=1)
    concatenated_labels = np.concatenate(all_labels)
    return concatenated_features, concatenated_labels

# Main Function
def main(data_dir, csv_file, checkpoint_paths, num_classes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    label_encoder = LabelEncoder()
    train_dataset = Data(csv_file=csv_file, root_dir=data_dir, transform=transform, split='train', label_encoder=label_encoder)
    val_dataset = Data(csv_file=csv_file, root_dir=data_dir, transform=transform, split='val', label_encoder=label_encoder)
    test_dataset = Data(csv_file=csv_file, root_dir=data_dir, transform=transform, split='test', label_encoder=label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=32)

    feature_extractors = [PretrainedModelFeatureExtractor(model_name='resnet50', checkpoint_path=ckpt) for ckpt in checkpoint_paths]

    print("Extracting features for training data...")
    train_features, train_labels = extract_features_from_all_models(train_loader, feature_extractors)

    print("Extracting features for validation data...")
    val_features, val_labels = extract_features_from_all_models(val_loader, feature_extractors)

    print("Extracting features for test data...")
    test_features, test_labels = extract_features_from_all_models(test_loader, feature_extractors)

    ensemble = ClassifierEnsemble(n_jobs=-1)
    ensemble.fit(train_features, train_labels, val_features, val_labels)

    # Create SummaryWriter after ensemble fitting
    writer = SummaryWriter()

    print("Evaluating on validation set...")
    svc_probs_val, rf_probs_val, lr_probs_val = ensemble.predict_proba(val_features)
    selected_probs_val = ensemble.entropy_selection(svc_probs_val, rf_probs_val, lr_probs_val)
    y_pred_val = np.argmax(selected_probs_val, axis=1)
    val_accuracy = accuracy_score(val_labels, y_pred_val)
    val_balanced_accuracy = balanced_accuracy_score(val_labels, y_pred_val)
    print(f"Validation accuracy: {val_accuracy}")
    print(f"Validation balanced accuracy: {val_balanced_accuracy}")
    writer.add_scalar('Validation/Accuracy', val_accuracy)
    writer.add_scalar('Validation/BalancedAccuracy', val_balanced_accuracy)

    print("Evaluating on test set...")
    svc_probs_test, rf_probs_test, lr_probs_test = ensemble.predict_proba(test_features)
    selected_probs_test = ensemble.entropy_selection(svc_probs_test, rf_probs_test, lr_probs_test)
    y_pred_test = np.argmax(selected_probs_test, axis=1)
    test_accuracy = accuracy_score(test_labels, y_pred_test)
    test_balanced_accuracy = balanced_accuracy_score(test_labels, y_pred_test)
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test balanced accuracy: {test_balanced_accuracy}")
    writer.add_scalar('Test/Accuracy', test_accuracy)
    writer.add_scalar('Test/BalancedAccuracy', test_balanced_accuracy)

    writer.close()

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