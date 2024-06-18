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
unzip -q data/raw/finbenthic2/FIN-Benthic2.zip -d $TMPDIR
source tykky
python scripts/07_calculate_metrics.py\
        --predictions "outputs/finbenthic2/finbenthic2-base-200_efficientnet_b0/f2/predictions/finbenthic2_none/finbenthic2_finbenthic2-base-200_efficientnet_b0_f2_240325-0535-2a2e_epoch02_val-loss0.02_none.csv"\
        --out_folder "outputs/finbenthic2/finbenthic2-base-200_efficientnet_b0/f2/predictions"