#!/bin/bash

# Define the base directory
base_folder="outputs/moss/moss_cross_UP_32_sample_simple_aug2_efficientnet_b0"

for fold in {0..2}
do
    # Define the checkpoint path
    ckpt_path=$(ls ${base_folder}/f${fold}/*_last.ckpt)

    # Create a batch script for each fold
    cat <<EOF > "job_predict_f${fold}.sh"
#!/bin/bash
#SBATCH --job-name=predict_f${fold}
#SBATCH --account=Project_2009950
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o "o_predict_f${fold}.txt"
#SBATCH -e "e_predict_f${fold}.txt"

# This batchjob resumes a previously trained model with a lower learning rate

echo "Extracting data for fold ${fold}..."
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q /scratch/project_2009950/moss.zip -d \$TMPDIR
source tykky

python scripts/03_predict.py \\
    --data_folder "\$TMPDIR/moss/images" \\
    --dataset_name "moss" \\
    --csv_path "data/moss/downsampled/01_moss_processed_imbalanced_3splits_Label.csv" \\
    --label "Label" \\
    --class_map "data/moss/downsampled/moss_label_map.txt" \\
    --fold ${fold} \\
    --imsize 224 \\
    --batch_size 256 \\
    --aug 'none' \\
    --load_to_memory 'False' \\
    --out_folder 'outputs' \\
    --tta 'False' \\
    --out_prefix 'moss_cross_UP_32_sample_simple_aug02' \\
    --ckpt_path "${ckpt_path}"
EOF

    # Submit the job
    sbatch "job_predict_f${fold}.sh"
done


##for data folder
#cifar10 = \$TMPDIR/cifar10/Images/
#FinBenthic2 = \$TMPDIR/IDA/

#Models
#mobilenetv3_large_100
#efficientnet_b0

# Instructions to run:
# 1. chmod +x 06_predict_all.sh  # Give execution permission
# 2. ./06_predict_all.sh         # Execute the script

#chmod -R 755 \$TMPDIR/cifar10/Images
#unzip -q /scratch/project_2009950/cifar10.zip -d \$TMPDIR

##for finbenthic2 data
# echo "Extracting data for fold ${fold}..."


# unzip -q /scratch/project_2009950/FIN-Benthic2.zip -d \$TMPDIR
# source tykky

# python scripts/03_predict.py \\
#     --data_folder "\$TMPDIR/IDA/" \\
#     --dataset_name "finbenthic2" \\
#     --csv_path "data/processed/finbenthic2/downsampled/01_finbenthic2_downsampled_5splits_taxon.csv" \\
#     --label "taxon" \\
#     --class_map "data/processed/finbenthic2/label_map_01_taxon.txt" \\
#     --fold ${fold} \\
#     --imsize 224 \\
#     --batch_size 256 \\
#     --aug 'none' \\
#     --load_to_memory 'False' \\
#     --out_folder 'outputs' \\
#     --tta 'False' \\
#     --out_prefix 'finbenthic2_focal_UP_Down_32_sample_simple_aug2' \\
#     --ckpt_path "${ckpt_path}"

###for cifar10 data
# unzip -q /scratch/project_2009950/cifar10.zip -d \$TMPDIR
# source tykky

# python scripts/03_predict.py \\
#     --data_folder "\$TMPDIR/cifar10/Images/" \\
#     --dataset_name "cifar10" \\
#     --csv_path "data/processed/cifar10/01_cifar10_processed_imbalanced_3splits_Label.csv" \\
#     --label "Label" \\
#     --class_map "data/processed/cifar10/cifar10_label_map.txt" \\
#     --fold ${fold} \\
#     --imsize 32 \\
#     --batch_size 256 \\
#     --aug 'up-sampling' \\
#     --load_to_memory 'False' \\
#     --out_folder 'outputs' \\
#     --tta 'False' \\
#     --out_prefix 'Cifar10_cross_UP_sample' \\
#     --ckpt_path "${ckpt_path}"




## For MOSS DATa
# UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q /scratch/project_2009950/moss.zip -d \$TMPDIR
# source tykky

# python scripts/03_predict.py \\
#     --data_folder "\$TMPDIR/moss/images" \\
#     --dataset_name "moss" \\
#     --csv_path "data/moss/downsampled/01_moss_processed_imbalanced_3splits_Label.csv" \\
#     --label "Label" \\
#     --class_map "data/moss/downsampled/moss_label_map.txt" \\
#     --fold ${fold} \\
#     --imsize 224 \\
#     --batch_size 256 \\
#     --aug 'none' \\
#     --load_to_memory 'False' \\
#     --out_folder 'outputs' \\
#     --tta 'False' \\
#     --out_prefix 'moss_cross_UP_32_sample_simple_aug02' \\
#     --ckpt_path "${ckpt_path}"
# EOF

#     # Submit the job
#     sbatch "job_predict_f${fold}.sh"