#!/bin/bash
#SBATCH --job-name=eb0_cross_f2
#SBATCH --account=Project_2009950
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_eb0.txt"
#SBATCH -e "e_eb0.txt"

# This batchjob resumes a previously trained model with a lower learning rate

echo "Extracting data..."
unzip -q /scratch/project_2009950/FIN-Benthic2.zip -d $TMPDIR
source tykky

python scripts/03_predict.py \
    --data_folder "$TMPDIR/IDA/" \
    --dataset_name "finbenthic2" \
    --csv_path "data/processed/finbenthic2/01_finbenthic2_processed_5splits_taxon.csv" \
    --label "taxon" \
    --class_map "data/processed/finbenthic2/label_map_01_taxon.txt" \
    --fold 2 \
    --imsize 224 \
    --batch_size 128 \
    --aug 'up-sampling' \
    --load_to_memory 'False' \
    --out_folder 'outputs' \
    --tta 'False' \
    --out_prefix 'finbenthic2_cross_UPsample-test-updated' \
    --ckpt_path "outputs/finbenthic2/finbenthic2_cross_UPsample-test-updated_efficientnet_b0/f2/finbenthic2_cross_UPsample-test-updated_efficientnet_b0_f2_epoch33_epoch49_val-loss0.30_last.ckpt"


#sbatch 06_predict_all.sh