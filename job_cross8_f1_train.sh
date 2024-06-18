#!/bin/bash
#SBATCH --job-name=cross8_f1
#SBATCH --account=Project_2009950
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_eb_cross8_f1.txt"
#SBATCH -e "e_eb_cross8_f1.txt"

echo "Extracting data for fold 1..."
Disable zip bomb detection and unzip the file
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q /scratch/project_2009950/moss.zip -d $TMPDIR
echo $(ls $TMPDIR/moss)
echo "Done!"
source tykky

python scripts/02_train.py \
                --data_folder "$TMPDIR/moss/images" \
                --dataset_name "moss" \
                --csv_path "data/moss/downsampled/01_moss_processed_imbalanced_3splits_Label.csv" \
                --label "Label" \
                --fold 1 \
                --class_map "data/moss/downsampled/moss_label_map.txt" \
                --imsize 224 \
                --batch_size 256 \
                --aug 'up-sampling8' \
                --load_to_memory 'False' \
                --model 'efficientnet_b0' \
                --freeze_base 'False' \
                --pretrained 'True' \
                --opt 'adamw' \
                --max_epochs 50 \
                --min_epochs 50 \
                --early_stopping 'False' \
                --early_stopping_patience 50 \
                --criterion 'cross-entropy' \
                --lr 0.001 \
                --auto_lr 'False' \
                --log_dir 'test_moss-models-sampling' \
                --out_folder 'outputs' \
                --out_prefix 'test_moss_cross_UP_8_sample_simple_aug2' \
                --deterministic 'True' \
                --resume 'False'
