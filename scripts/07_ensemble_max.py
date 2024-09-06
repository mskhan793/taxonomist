import os
import sys
import pandas as pd
from collections import Counter
import argparse

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

    # Get the final folder name from the destination path
    final_folder_name = os.path.basename(new_folder_path.rstrip('/'))

    base_folder_paths = [base_folder1, base_folder2, base_folder3]
    new_base_folder = create_directory_structure(new_folder_path, data_aug)

    # Define all folds
    folds = ['f0', 'f1', 'f2', 'f3', 'f4']

    # Iterate over each fold
    for fold in folds:
        # Read CSV files from each base folder for the current fold
        dfs = []
        for path in base_folder_paths:
            df = read_csv_from_folder(path, fold)
            if df is not None:
                dfs.append(df)

        # Ensure we have read all three dataframes
        if len(dfs) != 3:
            print(f"Could not read CSV files from all folders for {fold}.")
            continue

        # Create a new DataFrame for the ensemble results
        ensemble_df = dfs[0][['fname', 'y_true', 'y_pred']].copy()
        ensemble_df['y_pred_ensemble'] = None

        # Create the same 39 column names that are similar in other 3 csv files
        class_columns = dfs[0].columns[3:]  # Assuming the last 39 columns are from index 3 onwards

        # Initialize these columns in ensemble_df with None values
        for col in class_columns:
            ensemble_df[col] = None

        # Replace non-numeric values with 0 in the original dataframes
        for df in dfs:
            for col in class_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Compute the maximum of each class score and update ensemble_df
        for i in range(len(ensemble_df)):
            for class_name in class_columns:
                # Extract the class scores from each dataframe for the current row
                scores = [df.at[i, class_name] for df in dfs]
                # Compute the maximum score
                max_score = max(scores)
                # Update the ensemble_df with the maximum score
                ensemble_df.at[i, class_name] = max_score

        # Find the maximum value in the last 39 columns for each row and update y_pred_ensemble
        for i in range(len(ensemble_df)):
            # Extract the last 39 columns for the current row
            row_values = ensemble_df.iloc[i][class_columns]
            
            # Find the maximum value in these columns
            max_value = row_values.max()
            
            # Find the corresponding column name
            max_class = row_values.idxmax()
            
            # Update the y_pred_ensemble column with this column name
            ensemble_df.at[i, 'y_pred_ensemble'] = max_class

        # Rename y_pred to y_pred_old and y_pred_ensemble to y_pred
        ensemble_df.rename(columns={'y_pred': 'y_pred_old', 'y_pred_ensemble': 'y_pred'}, inplace=True)

        # Save the updated ensemble results to a CSV file
        output_filename = os.path.join(new_base_folder, fold, 'predictions', data_aug, f'{final_folder_name}_{fold}.csv')
        ensemble_df.to_csv(output_filename, index=False)
        print(f"Updated ensemble predictions saved to {output_filename}")

if __name__ == "__main__":
    main()
