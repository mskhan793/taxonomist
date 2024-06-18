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

source tykky

python scripts/05_evaluate.py \
    --predictions "outputs/finbenthic2/finbenthic2_cross_UP_Down_32_sample_simple_aug2_efficientnet_b0/predictions/finbenthic2_cross_UP_Down_32_sample_simple_aug2_efficientnet_b0_finbenthic2_none.csv" \
    --metric_config conf/eval.yaml \
    --around 4 \
    --no_bootstrap

#chmod +x 09_evaluation.sh  # Give execution permission
#sbatch 09_evaluation.sh