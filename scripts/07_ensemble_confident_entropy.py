import os
import sys
import pandas as pd
from collections import Counter
import argparse
import numpy as np
from scipy.stats import entropy

def read_csv_from_folder(base_folder, fold):
    fold_folder = os.path.join(base_folder, fold)
    if not os.path.exists(fold_folder):
        print(f"{fold} folder not found in {base_folder}")
        return None

    predictions_folder = os.path.join(fold_folder, 'predictions')
    if not os.path.exists(predictions_folder):
        print(f"predictions folder not found in {fold_folder}")
        return None

    # Look for the subfolder inside predictions
    subfolders = [f.path for f in os.scandir(predictions_folder) if f.is_dir()]
    if not subfolders:
        print(f"No subfolders found in {predictions_folder}")
        return None

    # Assuming there is only one subfolder
    final_folder = subfolders[0]
    
    # Look for the csv file inside the final folder
    csv_files = [f for f in os.listdir(final_folder) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {final_folder}")
        return None

    # Assuming there is only one CSV file
    csv_file_path = os.path.join(final_folder, csv_files[0])
    df = pd.read_csv(csv_file_path)
    return df

def create_directory_structure(new_folder_path, data_aug):
    os.makedirs(new_folder_path, exist_ok=True)
    folds = ['f0', 'f1', 'f2', 'f3', 'f4']
    for fold in folds:
        fold_path = os.path.join(new_folder_path, fold, 'predictions', data_aug)
        os.makedirs(fold_path, exist_ok=True)
    return new_folder_path

def safe_convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert {value} to float. Using 0.0 instead.")
        return 0.0


def calculate_confidence_metrics(probabilities):
    probabilities = np.array([safe_convert_to_float(p) for p in probabilities])
    probabilities = np.clip(probabilities, 1e-10, 1)
    sum_probs = probabilities.sum()
    if sum_probs == 0:
        print("Warning: All probabilities are zero. Using uniform distribution.")
        probabilities = np.ones_like(probabilities) / len(probabilities)
    else:
        probabilities /= sum_probs

    entropy_score = entropy(probabilities)
    entropy_confidence = 1 - (entropy_score / np.log(len(probabilities)))
    
    return entropy_confidence


def ensemble_predictions(dfs, class_columns):
    ensemble_df = dfs[0][['fname', 'y_true', 'y_pred']].copy()
    ensemble_df['y_pred_ensemble'] = None
    
    for col in class_columns:
        ensemble_df[col] = 0.0
    
    for i in range(len(ensemble_df)):
        weighted_probs = np.zeros(len(class_columns))
        total_weight = 0
        
        print(f"Processing row {i}")
        
        for df_index, df in enumerate(dfs):
            probs = df.iloc[i][class_columns].values
            probs = np.array([safe_convert_to_float(p) for p in probs])
            print(f"Model {df_index} probabilities: {probs}")
            
            if np.all(probs == 0):
                print(f"Warning: All probabilities are zero for model {df_index} in row {i}. Skipping this model.")
                continue
            
            entropy_conf = calculate_confidence_metrics(probs)
            confidence = entropy_conf
            
            print(f"Model {df_index} confidence: {confidence}")
            
            weighted_probs += probs * confidence
            total_weight += confidence
        
        print(f"Total weight: {total_weight}")
        
        if total_weight > 0:
            final_probs = weighted_probs / total_weight
            print(f"Final probabilities: {final_probs}")
            
            for j, col in enumerate(class_columns):
                ensemble_df.at[i, col] = final_probs[j]
            
            ensemble_df.at[i, 'y_pred_ensemble'] = class_columns[np.argmax(final_probs)]
        else:
            print(f"Warning: No valid predictions for row {i}. Using average probabilities.")
            avg_probs = np.mean([np.array([safe_convert_to_float(p) for p in df.iloc[i][class_columns].values]) for df in dfs], axis=0)
            for j, col in enumerate(class_columns):
                ensemble_df.at[i, col] = avg_probs[j]
            ensemble_df.at[i, 'y_pred_ensemble'] = class_columns[np.argmax(avg_probs)]
    
    return ensemble_df


def main():
    parser = argparse.ArgumentParser(description='Ensemble model predictions.')
    parser.add_argument('--model1', required=True, help='Path to the first model folder')
    parser.add_argument('--model2', required=True, help='Path to the second model folder')
    parser.add_argument('--model3', required=True, help='Path to the third model folder')
    parser.add_argument('--destination', required=True, help='Destination folder path')
    parser.add_argument('--data', required=True, help='Data argument to replace subfolder')
    parser.add_argument('--aug', required=True, help='Augmentation argument to replace subfolder')
    
    args = parser.parse_args()
    
    base_folder1 = args.model1
    base_folder2 = args.model2
    base_folder3 = args.model3
    new_folder_path = args.destination
    data_aug = f"{args.data}_{args.aug}"

    final_folder_name = os.path.basename(new_folder_path.rstrip('/'))

    base_folder_paths = [base_folder1, base_folder2, base_folder3]
    new_base_folder = create_directory_structure(new_folder_path, data_aug)

    folds = ['f0', 'f1', 'f2', 'f3', 'f4']

    for fold in folds:
        dfs = []
        for path in base_folder_paths:
            df = read_csv_from_folder(path, fold)
            if df is not None:
                dfs.append(df)

        if len(dfs) != 3:
            print(f"Could not read CSV files from all folders for {fold}.")
            continue

        class_columns = dfs[0].columns[3:]  # Assuming class columns start from index 3

        # Ensure all values are numeric
        for df in dfs:
            for col in class_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        ensemble_df = ensemble_predictions(dfs, class_columns)

        print("Ensemble DataFrame Head:")
        print(ensemble_df.head())
        print("\nEnsemble DataFrame Info:")
        print(ensemble_df.info())
        print("\nEnsemble DataFrame Description:")
        print(ensemble_df.describe())

        # Rename columns for consistency
        ensemble_df.rename(columns={'y_pred': 'y_pred_old', 'y_pred_ensemble': 'y_pred'}, inplace=True)

        # Save the updated ensemble results
        output_filename = os.path.join(new_base_folder, fold, 'predictions', data_aug, f'{final_folder_name}_{fold}.csv')
        ensemble_df.to_csv(output_filename, index=False)
        print(f"Updated ensemble predictions saved to {output_filename}")

if __name__ == "__main__":
    main()