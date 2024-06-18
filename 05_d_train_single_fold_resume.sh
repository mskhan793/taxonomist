#!/bin/bash

# Define the base directory
# base_folder="outputs/finbenthic2/finbenthic2-cross-entropy-aug-02_mobilenetv3_large_100"

# Define the fold number to train
fold=4

# Create a batch script for the specified fold
cat <<EOF > "job_ci_f${fold}_train.sh"
#!/bin/bash
#SBATCH --job-name=cir_f${fold}
#SBATCH --account=Project_2009950
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_eb_ci_f${fold}.txt"
#SBATCH -e "e_eb_ci_f${fold}.txt"

echo "Extracting data for fold ${fold}..."
unzip -q /scratch/project_2009950/FIN-Benthic2.zip -d \$TMPDIR

# Change permissions of extracted files
chmod -R 755 \$TMPDIR
echo \$(ls \$TMPDIR/IDA/)
echo "Done!"
source tykky

python scripts/02_train.py \\
                --data_folder "\$TMPDIR/IDA/" \\
                --dataset_name "finbenthic2" \\
                --csv_path "data/processed/finbenthic2/downsampled/01_finbenthic2_downsampled_5splits_taxon.csv" \\
                --label "taxon" \\
                --fold ${fold} \\
                --class_map "data/processed/finbenthic2/label_map_01_taxon.txt" \\
                --imsize 224 \\
                --batch_size 256 \\
                --aug 'none' \\
                --load_to_memory 'False' \\
                --model 'efficientnet_b0' \\
                --freeze_base 'False' \\
                --pretrained 'True' \\
                --opt 'adamw' \\
                --max_epochs 50 \\
                --min_epochs 50 \\
                --early_stopping 'False' \\
                --early_stopping_patience 50 \\
                --criterion 'class-imbalance' \\
                --lr 0.001 \\
                --auto_lr 'False' \\
                --log_dir 'finbenthic2-models' \\
                --out_folder 'outputs' \\
                --out_prefix 'finbenthic2_ci_UP_Down_sample' \\
                --deterministic 'True' \\
                --resume 'True' \\
                --ckpt_path "outputs/finbenthic2/finbenthic2_ci_UP_Down_sample_efficientnet_b0/f4/finbenthic2_ci_UP_Down_sample_efficientnet_b0_f4_240519-1218-5036_epoch39_val-loss0.70_last.ckpt"
EOF

# Submit the job
sbatch "job_ci_f${fold}_train.sh"


##for data folder
#cifar10 = $TMPDIR/cifar10/Images/
#FinBenthic2 = $TMPDIR/IDA/

#Models
#mobilenetv3_large_100
#efficientnet_b0

# Instructions to run:
# 1. chmod +x ./05_d_train_single_fold_resume.sh  # Give execution permission
# 2. ./05_d_train_single_fold_resume.sh         # Execute the script

#chmod -R 755 $TMPDIR/cifar10/Images
#unzip -q /scratch/project_2009950/cifar10.zip -d $TMPDIR