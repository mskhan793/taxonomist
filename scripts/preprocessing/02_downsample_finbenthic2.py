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

def downsample_individuals(group, individual_column, downsample_percentage, random_state):
    """
    Downsamples the dataframe group to a specified percentage of samples per individual.
    Args:
        group (DataFrame): The DataFrame group to downsample.
        individual_column (str): The column name containing the specimen identifiers.
        downsample_percentage (float): The percentage of samples to keep per individual.
        random_state (int): The random state for random sampling.
    Returns:
        DataFrame: A new dataframe with downsampled individuals.
    """
    individual_groups = group.groupby(individual_column)
    downsampled_frames = []
    for _, ind_group in individual_groups:
        num_samples = int(len(ind_group) * downsample_percentage)
        downsampled = ind_group.sample(num_samples, random_state=random_state)
        downsampled_frames.append(downsampled)

    return pd.concat(downsampled_frames, ignore_index=True)

def downsample_df(df, label_column, individual_column, median_sample_size, downsample_percentage, random_state):
    """
    Downsamples the dataframe to a specified percentage of samples per class, considering multiple specimens.
    Args:
        df (DataFrame): The original dataframe.
        label_column (str): The column name containing the class labels.
        individual_column (str): The column name containing the specimen identifiers.
        median_sample_size (int): Median number of samples per class.
        downsample_percentage (float): The percentage of samples to keep per class.
        random_state (int): The random state for random sampling.
    Returns:
        DataFrame: A new dataframe with downsampled classes.
    """
    downsampled_frames = []
    class_counts = df[label_column].value_counts()

    for label, count in class_counts.items():
        if count > median_sample_size:
            group = df[df[label_column] == label]
            downsampled_group = downsample_individuals(group, individual_column, downsample_percentage, random_state)
            downsampled_frames.append(downsampled_group)
        else:
            downsampled_frames.append(df[df[label_column] == label])

    return pd.concat(downsampled_frames, ignore_index=True)

def main(csv_path, out_folder, class_map_file, downsample_percentage, random_state):
    """
    Main function to load data, downsample, and save the output.
    Args:
        csv_path (Path): Path to the CSV file with data.
        out_folder (Path): Output directory path.
        class_map_file (Path): Path to the class map file.
        downsample_percentage (float): The percentage of samples to keep.
        random_state (int): The random state for random sampling.
    """
    df = pd.read_csv(csv_path)
    labels = load_class_map(class_map_file)
    
    # Filter the dataframe to include only the classes in the class map
    df = df[df['taxon'].isin(labels)]
    
    # Calculate the median sample size for downsampling based on taxon column
    median_sample_size = calculate_median_sample_size(df, 'taxon')
    
    # Downsample the dataframe
    downsampled_df = downsample_df(df, 'taxon', 'individual', median_sample_size, downsample_percentage, random_state)
    
    # Save the downsampled dataframe
    output_path = out_folder / "01_finbenthic2_downsampled.csv"
    downsampled_df.to_csv(output_path, index=False)
    print(f"Data downsampled and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample FinBenthic2 data based on taxon column.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing the data.")
    parser.add_argument("--out_folder", type=str, default=".", help="Output directory for the downsampled data.")
    parser.add_argument("--class_map_file", type=str, required=True, help="Path to the class map text file.")
    parser.add_argument("--downsample_percentage", type=float, required=True, help="Percentage of samples to keep (0.0 to 1.0).")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for random sampling.")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    out_folder = Path(args.out_folder)
    class_map_file = Path(args.class_map_file)
    
    # Ensure output directory exists
    out_folder.mkdir(exist_ok=True, parents=True)
    
    main(csv_path, out_folder, class_map_file, args.downsample_percentage, args.random_state)
