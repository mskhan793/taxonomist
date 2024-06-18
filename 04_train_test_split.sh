#!/bin/bash
#SBATCH --job-name=eb0
#SBATCH --account=Project_2009950
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH -o "o_eb0.txt"
#SBATCH -e "e_eb0.txt"

# This batchjob resumes a previously trained model with a lower learning rate

echo "Extracting data..."
unzip -q /scratch/project_2009950/moss.zip -d $TMPDIR
echo "Done!"
source tykky        

python scripts/01_train_test_split_original.py \
    --csv_path "data/moss/downsampled/01_moss_processed_imbalanced.csv" \
    --target_col "Label" \
    --group_col "ID" \
    --n_splits 3 \
    --out_folder "data/moss/downsampled"\
    --random_state 123


#unzip -q /scratch/project_2009950/cifar10.zip -d $TMPDIR




#Finbenthic2
# unzip -q /scratch/project_2009950/FIN-Benthic2.zip -d $TMPDIR
# echo "Done!"
# source tykky        

# python scripts/01_train_test_split_original.py \
#     --csv_path "data/downsampled/01_finbenthic2_downsampled.csv" \
#     --target_col "taxon" \
#     --group_col "individual" \
#     --n_splits 5 \
#     --out_folder "data/processed/finbenthic2/downsampled"\
#     --random_state 123



#ciar10

# echo "Extracting data..."
# unzip -q /scratch/project_2009950/cifar10.zip -d $TMPDIR
# echo "Done!"
# source tykky        

# python scripts/01_train_test_split_original.py \
#     --csv_path "data/downsampled/01_cifar10_processed_imbalanced.csv" \
#     --target_col "Label" \
#     --group_col "ID" \
#     --n_splits 3 \
#     --out_folder "data/processed/cifar10/downsampled"\
#     --random_state 123

#sbatch 04_train_test_split.sh