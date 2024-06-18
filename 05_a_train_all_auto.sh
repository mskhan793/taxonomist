#!/bin/bash

for fold in {0..1}
do
    # Create a batch script for each fold
    cat <<EOF > "job_cross8_f${fold}_train.sh"
#!/bin/bash
#SBATCH --job-name=cross8_f${fold}
#SBATCH --account=Project_2009950
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_eb_cross8_f${fold}.txt"
#SBATCH -e "e_eb_cross8_f${fold}.txt"

echo "Extracting data for fold ${fold}..."
Disable zip bomb detection and unzip the file
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q /scratch/project_2009950/moss.zip -d \$TMPDIR
echo \$(ls \$TMPDIR/moss)
echo "Done!"
source tykky

python scripts/02_train.py \\
                --data_folder "\$TMPDIR/moss/images" \\
                --dataset_name "moss" \\
                --csv_path "data/moss/downsampled/01_moss_processed_imbalanced_3splits_Label.csv" \\
                --label "Label" \\
                --fold ${fold} \\
                --class_map "data/moss/downsampled/moss_label_map.txt" \\
                --imsize 224 \\
                --batch_size 256 \\
                --aug 'up-sampling8' \\
                --load_to_memory 'False' \\
                --model 'efficientnet_b0' \\
                --freeze_base 'False' \\
                --pretrained 'True' \\
                --opt 'adamw' \\
                --max_epochs 50 \\
                --min_epochs 50 \\
                --early_stopping 'False' \\
                --early_stopping_patience 50 \\
                --criterion 'cross-entropy' \\
                --lr 0.001 \\
                --auto_lr 'False' \\
                --log_dir 'test_moss-models-sampling' \\
                --out_folder 'outputs' \\
                --out_prefix 'test_moss_cross_UP_8_sample_simple_aug2' \\
                --deterministic 'True' \\
                --resume 'False'
EOF

    # Submit the job
    sbatch "job_cross8_f${fold}_train.sh"
done


##for data folder
#cifar10 = \$TMPDIR/cifar10/Images/
#FinBenthic2 = \$TMPDIR/IDA/

#Models
#mobilenetv3_large_100
#efficientnet_b0

# Instructions to run:
# 1. chmod +x 05_a_train_all_auto.sh  # Give execution permission
# 2. ./05_a_train_all_auto.sh         # Execute the script

#chmod -R 755 \$TMPDIR/cifar10/Images

##CIFAR-10
#unzip -q /scratch/project_2009950/cifar10.zip -d \$TMPDIR

##FIN-Benthic2
#unzip -q /scratch/project_2009950/FIN-Benthic2.zip -d \$TMPDIR

# Change permissions of extracted files
# chmod -R 755 \$TMPDIR
# echo \$(ls \$TMPDIR/IDA/)
# echo "Done!"
# source tykky


## FINBENTHIC2 training script

# unzip -q /scratch/project_2009950/FIN-Benthic2.zip -d \$TMPDIR

# echo "Done!"
# source tykky

# python scripts/02_train.py \\
#                 --data_folder "\$TMPDIR/IDA/" \\
#                 --dataset_name "finbenthic2" \\
#                 --csv_path "data/processed/finbenthic2/downsampled/01_finbenthic2_downsampled_5splits_taxon.csv" \\
#                 --label "taxon" \\
#                 --fold ${fold} \\
#                 --class_map "data/processed/finbenthic2/label_map_01_taxon.txt" \\
#                 --imsize 224 \\
#                 --batch_size 256 \\
#                 --aug 'up-sampling16' \\
#                 --load_to_memory 'False' \\
#                 --model 'efficientnet_b0' \\
#                 --freeze_base 'False' \\
#                 --pretrained 'True' \\
#                 --opt 'adamw' \\
#                 --max_epochs 50 \\
#                 --min_epochs 50 \\
#                 --early_stopping 'False' \\
#                 --early_stopping_patience 50 \\
#                 --criterion 'focal' \\
#                 --lr 0.001 \\
#                 --auto_lr 'False' \\
#                 --log_dir 'finbenthic2-models-sampling' \\
#                 --out_folder 'outputs' \\
#                 --out_prefix 'finbenthic2_focal_UP_Down_16_sample_simple_aug2' \\
#                 --deterministic 'True' \\
#                 --resume 'False'
# EOF

#     # Submit the job
#     sbatch "job_focal16_f${fold}_train.sh"


##for cifar10 data
# unzip -q /scratch/project_2009950/cifar10.zip -d \$TMPDIR

# # Change permissions of extracted files
# chmod -R 755 \$TMPDIR
# source tykky

# unzip -q /scratch/project_2009950/cifar10.zip -d \$TMPDIR
# echo "Done!"
# source tykky

# python scripts/02_train.py \\
#                 --data_folder "\$TMPDIR/cifar10/Images/" \\
#                 --dataset_name "cifar10" \\
#                 --csv_path "data/processed/cifar10/downsampled/01_cifar10_processed_imbalanced_3splits_Label.csv" \\
#                 --label "Label" \\
#                 --fold ${fold} \\
#                 --class_map "data/processed/cifar10/cifar10_label_map.txt" \\
#                 --imsize 32 \\
#                 --batch_size 256 \\
#                 --aug 'up-sampling8' \\
#                 --load_to_memory 'False' \\
#                 --model 'efficientnet_b0' \\
#                 --freeze_base 'False' \\
#                 --pretrained 'True' \\
#                 --opt 'adamw' \\
#                 --max_epochs 50 \\
#                 --min_epochs 50 \\
#                 --early_stopping 'False' \\
#                 --early_stopping_patience 50 \\
#                 --criterion 'focal' \\
#                 --lr 0.001 \\
#                 --auto_lr 'False' \\
#                 --log_dir 'Cifar10-models-sampling' \\
#                 --out_folder 'outputs' \\
#                 --out_prefix 'Cifar10_focal_up_sample_290524' \\
#                 --deterministic 'True' \\
#                 --resume 'False'




# # Disable zip bomb detection and unzip the file
# Disable zip bomb detection and unzip the file
# UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q /scratch/project_2009950/moss.zip -d \$TMPDIR
# echo \$(ls \$TMPDIR/moss)
# echo "Done!"
# source tykky

# python scripts/02_train.py \\
#                 --data_folder "\$TMPDIR/moss/images" \\
#                 --dataset_name "moss" \\
#                 --csv_path "data/moss/downsampled/01_moss_processed_imbalanced_3splits_Label.csv" \\
#                 --label "Label" \\
#                 --fold ${fold} \\
#                 --class_map "data/moss/downsampled/moss_label_map.txt" \\
#                 --imsize 224 \\
#                 --batch_size 256 \\
#                 --aug 'up-sampling32' \\
#                 --load_to_memory 'False' \\
#                 --model 'efficientnet_b0' \\
#                 --freeze_base 'False' \\
#                 --pretrained 'True' \\
#                 --opt 'adamw' \\
#                 --max_epochs 50 \\
#                 --min_epochs 50 \\
#                 --early_stopping 'False' \\
#                 --early_stopping_patience 50 \\
#                 --criterion 'cross-entropy' \\
#                 --lr 0.001 \\
#                 --auto_lr 'False' \\
#                 --log_dir 'moss-models-sampling' \\
#                 --out_folder 'outputs' \\
#                 --out_prefix 'moss_cross_UP_32_sample_simple_aug2' \\
#                 --deterministic 'True' \\
#                 --resume 'False'
# EOF

#     # Submit the job
#     sbatch "job_cross32_f${fold}_train.sh"