#!/bin/bash

# Define the base directory
base_folder="outputs/cifar10/Cifar10_ci_15_UP_sample_efficientnet_b0"

for fold in {1..2}
do
    # Define the checkpoint path
    ckpt_path=$(ls ${base_folder}/f${fold}/*_last.ckpt)

    # Create a batch script for each fold
    cat <<EOF > "job_ci_f${fold}_train.sh"
#!/bin/bash
#SBATCH --job-name=ci_f${fold}
#SBATCH --account=Project_2009950
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_eb_ci_f${fold}.txt"
#SBATCH -e "e_eb_ci_f${fold}.txt"

echo "Extracting data for fold ${fold}..."
unzip -q /scratch/project_2009950/cifar10.zip -d \$TMPDIR

# Change permissions of extracted files
chmod -R 755 \$TMPDIR
echo "Done!"
source tykky

python scripts/02_train.py \\
                --data_folder "\$TMPDIR/cifar10/Images/" \\
                --dataset_name "cifar10" \\
                --csv_path "data/processed/cifar10/01_cifar10_processed_imbalanced_3splits_Label.csv" \\
                --label "Label" \\
                --fold ${fold} \\
                --class_map "data/processed/cifar10/cifar10_label_map.txt" \\
                --imsize 32 \\
                --batch_size 256 \\
                --aug 'up-sampling' \\
                --load_to_memory 'False' \\
                --model 'efficientnet_b0' \\
                --freeze_base 'False' \\
                --pretrained 'True' \\
                --opt 'adamw' \\
                --max_epochs 30 \\
                --min_epochs 30 \\
                --early_stopping 'False' \\
                --early_stopping_patience 30 \\
                --criterion 'class-imbalance' \\
                --lr 0.0001 \\
                --auto_lr 'False' \\
                --log_dir 'Cifar10-models' \\
                --out_folder 'outputs' \\
                --out_prefix 'Cifar10_ci_15_UP_sample' \\
                --deterministic 'True' \\
                --resume 'True' \\
                --ckpt_path "${ckpt_path}"
EOF

    # Submit the job
    sbatch "job_ci_f${fold}_train.sh"
done


##for data folder
#cifar10 = \$TMPDIR/cifar10/Images/
#FinBenthic2 = \$TMPDIR/IDA/

#Models
#mobilenetv3_large_100
#efficientnet_b0

# Instructions to run:
# 1. chmod +x 05__b_train_all_auto_resume.sh  # Give execution permission
# 2. ./05_b_train_all_auto_resume.sh         # Execute the script

#chmod -R 755 \$TMPDIR/cifar10/Images
#unzip -q /scratch/project_2009950/cifar10.zip -d \$TMPDIR
