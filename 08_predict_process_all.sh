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

#echo "Extracting data..."
#unzip -q /scratch/project_2009950/FIN-Benthic2.zip -d $TMPDIR
source tykky

python scripts/04_group_predictions.py \
    --predictions "outputs/finbenthic2/finbenthic2_focal_UP_Down_64_sample_simple_aug2_efficientnet_b0/predictions/finbenthic2_focal_UP_Down_64_sample_simple_aug2_efficientnet_b0_finbenthic2_none.csv" \
    --reference_csv "data/processed/finbenthic2/downsampled/01_finbenthic2_downsampled_5splits_taxon.csv" \
    --reference_target "taxon" \
    --reference_group "individual" \
    --agg_func "mode"


#sbatch 08_predict_process_all.sh
##CIFAR!0
# python scripts/04_group_predictions.py \
#     --predictions "outputs/cifar10/cifar10_testIM-cross-loss-fix-el_efficientnet_b0/predictions/cifar10_testIM-cross-loss-fix-el_efficientnet_b0_cifar10_none.csv" \
#     --reference_csv "data/processed/cifar10/01_cifar10_processed_imbalanced_3splits_Label.csv" \
#     --reference_target "Label" \
#     --reference_group "ID" \
#     --agg_func "mode"


#FinBenthic2
# python scripts/04_group_predictions.py \
#     --predictions "outputs/moss/moss_focal_none_sample_efficientnet_b0/predictions/moss_focal_none_sample_efficientnet_b0_moss_none.csv" \
#     --reference_csv "data/processed/finbenthic2/downsampled/01_finbenthic2_downsampled_5splits_taxon.csv" \
#     --reference_target "taxon" \
#     --reference_group "individual" \
#     --agg_func "mode"


#moss
# python scripts/04_group_predictions.py \
#     --predictions "outputs/moss/moss_focal_UP_32_sample_simple_aug2_efficientnet_b0/predictions/moss_focal_UP_32_sample_simple_aug2_efficientnet_b0_moss_none.csv" \
#     --reference_csv "data/moss/downsampled/01_moss_processed_imbalanced_3splits_Label.csv" \
#     --reference_target "Label" \
#     --reference_group "ID" \
#     --agg_func "mode"