from pathlib import Path
import pandas as pd
import argparse
import numpy as np

def load_class_map(class_map_file):
    """
    Loads class map from a text file.
    Args:
        class_map_file (Path): Path to the class map file.
    Returns:
        list: A list of class labels.
    """
    with open(class_map_file, 'r') as file:
        labels = [line.strip() for line in file]
    return labels

def calculate_median_sample_size(df, label_column):
    """
    Calculates the median sample size across classes in the dataframe.
    Args:
        df (DataFrame): The DataFrame to analyze.
        label_column (str): Column name containing class labels.
    Returns:
        int: The median sample size.
    """
    return int(df[label_column].value_counts().median())

def downsample_df(df, label_column, individual_column, target_sample_size):
    """
    Downsamples the dataframe to a specified number of samples per class, considering multiple specimens.
    Args:
        df (DataFrame): The original dataframe.
        label_column (str): The column name containing the class labels.
        individual_column (str): The column name containing the specimen identifiers.
        target_sample_size (int): Target number of samples per class.
    Returns:
        DataFrame: A new dataframe with downsampled classes.
    """
    grouped = df.groupby(label_column)
    downsampled_frames = []
    for name, group in grouped:
        # Group further by individual specimen
        individual_groups = group.groupby(individual_column)
        samples_per_individual = np.ceil(target_sample_size / len(individual_groups)).astype(int)

        # Collect samples from each specimen
        for _, ind_group in individual_groups:
            downsampled = ind_group.sample(min(samples_per_individual, len(ind_group)), random_state=42)
            downsampled_frames.append(downsampled)

    return pd.concat(downsampled_frames, ignore_index=True)

def main(csv_path, out_folder, class_map_file):
    """
    Main function to load data, downsample, and save the output.
    Args:
        csv_path (Path): Path to the CSV file with data.
        out_folder (Path): Output directory path.
        class_map_file (Path): Path to the class map file.
    """
    df = pd.read_csv(csv_path)
    labels = load_class_map(class_map_file)
    
    # Filter the dataframe to include only the classes in the class map
    df = df[df['taxon'].isin(labels)]
    
    # Calculate the median sample size for downsampling
    median_sample_size = calculate_median_sample_size(df, 'taxon')
    
    # Downsample the dataframe
    downsampled_df = downsample_df(df, 'taxon', 'individual', median_sample_size)
    
    # Saving the downsampled dataframe
    output_path = out_folder / "01_finbenthic2_downsampled.csv"
    downsampled_df.to_csv(output_path, index=False)
    print(f"Data downsampled and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample FinBenthic2 data based on taxon column.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing the data.")
    parser.add_argument("--out_folder", type=str, default=".", help="Output directory for the downsampled data.")
    parser.add_argument("--class_map_file", type=str, required=True, help="Path to the class map text file.")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    out_folder = Path(args.out_folder)
    class_map_file = Path(args.class_map_file)
    
    # Ensure output directory exists
    out_folder.mkdir(exist_ok=True, parents=True)
    
    main(csv_path, out_folder, class_map_file)