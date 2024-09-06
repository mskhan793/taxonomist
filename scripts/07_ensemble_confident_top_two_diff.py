import os
import sys
import pandas as pd
from collections import Counter
import argparse
import numpy as np

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

def get_top_two_probabilities(row):
    sorted_probs = sorted(row, reverse=True)
    return sorted_probs[0], sorted_probs[1]

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

        ensemble_df = dfs[0][['fname', 'y_true', 'y_pred']].copy()
        ensemble_df['y_pred_ensemble'] = None

        class_columns = dfs[0].columns[3:]

        for col in class_columns:
            ensemble_df[col] = None

        for df in dfs:
            for col in class_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        for i in range(len(ensemble_df)):
            confidence_scores = []
            for df in dfs:
                row_probs = df.iloc[i][class_columns].values
                top_prob, second_prob = get_top_two_probabilities(row_probs)
                confidence_score = top_prob - second_prob
                confidence_scores.append(confidence_score)
            
            most_confident_model = np.argmax(confidence_scores)
            
            for class_name in class_columns:
                ensemble_df.at[i, class_name] = dfs[most_confident_model].at[i, class_name]
            
            ensemble_df.at[i, 'y_pred_ensemble'] = dfs[most_confident_model].at[i, 'y_pred']

        ensemble_df.rename(columns={'y_pred': 'y_pred_old', 'y_pred_ensemble': 'y_pred'}, inplace=True)

        output_filename = os.path.join(new_base_folder, fold, 'predictions', data_aug, f'{final_folder_name}_{fold}.csv')
        ensemble_df.to_csv(output_filename, index=False)
        print(f"Updated ensemble predictions saved to {output_filename}")

if __name__ == "__main__":
    main()