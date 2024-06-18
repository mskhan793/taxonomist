#!/bin/bash
#SBATCH --job-name=eb0
#SBATCH --account=Project_2009950
#SBATCH --partition=test
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH -o "o_eb0.txt"
#SBATCH -e "e_eb0.txt"
echo "Extracting data..."
unzip -q /scratch/project_2009950/FIN-Benthic2.zip -d $TMPDIR
source tykky

python scripts/01_train_test_split_original.py \
    --csv_path "data/processed/finbenthic2/01_finbenthic2_processed.csv" \
    --target_col "taxon" \
    --group_col "individual" \
    --n_splits 5 \
    --out_folder "data/processed/finbenthic2" \
    --random_state 123