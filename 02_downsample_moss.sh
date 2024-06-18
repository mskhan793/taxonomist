#!/bin/bash
#SBATCH --job-name=downsample_finbenthic2
#SBATCH --account=Project_2009950
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o "downsample_finbenthic2_output.txt"
#SBATCH -e "downsample_finbenthic2_error.txt"

echo "Starting data downsampling..."
source tykky
# Assuming the Python script is saved as 'downsample_finbenthic2.py' and is in the 'scripts' folder
python scripts/preprocessing/process_imbalance_moss.py \
    --csv_path="data/processed/moss/01_moss_processed.csv" \
    --out_folder="data/moss/downsampled" \

echo "Data downsampling completed."


# Instructions to run:
# 1. chmod +x 02_downsample_moss.sh  # Give execution permission
# 2. sbatch 02_downsample_moss.sh         # Execute the script
