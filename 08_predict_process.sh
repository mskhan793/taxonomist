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
python scripts/04_group_predictions.py \
    --predictions "outputs/finbenthic2/finbenthic2-ci-loss_mobilenetv3_large_100/f4/predictions/finbenthic2_none/finbenthic2_finbenthic2-ci-loss_mobilenetv3_large_100_f4_240330-1714-ade6_epoch00_val-loss2.49_none.csv" \
    --reference_csv "data/processed/finbenthic2/01_finbenthic2_processed_5splits_taxon.csv" \
    --reference_target "taxon" \
    --fold 4 \
    --reference_group "individual" \
    --agg_func "mode"