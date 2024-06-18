#!/bin/bash
#SBATCH --job-name=cross-entropy-aug-02
#SBATCH --account=Project_2009950
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G   
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_eb0.txt"
#SBATCH -e "e_eb0.txt"

# This batchjob resumes a previously trained model with a lower learning rate

echo "Extracting data..."
#unzip -q data/raw/finbenthic2/FIN-Benthic2.zip -d $TMPDIR
unzip -q /scratch/project_2009950/cifar10.zip -d $TMPDIR
chmod -R 755 $TMPDIR
#export TMPDIR="data/raw/cifar10"  # On CSC this should be the normal nvme TMPDIR
echo $(ls $TMPDIR/cifar10)
echo "Done!"
source tykky

#for i in {3..4}
#do
python scripts/02_train.py \
    --data_folder "$TMPDIR/cifar10/Images" \
    --dataset_name "cifar10" \
    --csv_path "data/processed/cifar10/01_cifar10_processed_3splits_Label.csv" \
    --label "Label" \
    --fold 0 \
    --class_map "data/processed/cifar10/cifar10_label_map.txt" \
    --imsize 224 \
    --batch_size 256 \
    --aug 'none' \
    --load_to_memory 'False' \
    --model 'mobilenetv3_large_100' \
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
    --log_dir 'cifar10-models_cross' \
    --out_folder 'outputs' \
    --out_prefix 'cifar10-cross-loss-fix-el' \
    --deterministic 'True' \
    --resume 'False' 
    #--ckpt_path "outputs/finbenthic2/finbenthic2-focal-loss_mobilenetv3_large_100/f4/finbenthic2-focal-loss_mobilenetv3_large_100_f4_epoch11_epoch30_val-loss0.00_last.ckpt"
#done

#Models
#mobilenetv3_large_100
#efficientnet_b0