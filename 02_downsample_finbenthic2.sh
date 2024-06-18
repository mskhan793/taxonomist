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
python scripts/preprocessing/02_downsample_finbenthic2.py \
    --csv_path="data/processed/finbenthic2/01_finbenthic2_processed.csv" \
    --out_folder="data/downsampled" \
    --class_map_file="data/processed/finbenthic2/label_map_01_taxon.txt" \
    --downsample_percentage 0.5 \
    --random_state 123

echo "Data downsampling completed."
