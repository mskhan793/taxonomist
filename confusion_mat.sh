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

# Assuming 'source tykky' sets up your Python environment
python scripts/confusion_mat.py \
    --predictions "outputs/moss/moss_focal_UP_32_sample_aug2_efficientnet_b0/predictions/moss_focal_UP_32_sample_aug2_efficientnet_b0_moss_none_grouped.csv"


#sbatch confusion_mat.sh
