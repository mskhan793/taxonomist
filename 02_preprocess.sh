#!/bin/bash
#SBATCH --job-name=eb0
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
unzip -q /scratch/project_2009950/moss.zip -d $TMPDIR
ls $TMPDIR
echo "Done!"
source tykky        

#export TMPDIR = "data/raw/rodi/" # on CSC this should be the normal nvme TMPDIR
python scripts/preprocessing/process_moss.py \
    --csv_path="$TMPDIR/moss/mosquito_images.csv" \
    --out_folder="data/processed/moss"


#cifar 10
#unzip -q /scratch/project_2009950/cifar10.zip -d $TMPDIR
#unzip -q data/raw/cifar10.zip -d $TMPDIR

# python scripts/preprocessing/process_moss.py \
#     --csv_path="$TMPDIR/cifar10/image_labels.csv" \
#     --out_folder="data/processed/cifar10"