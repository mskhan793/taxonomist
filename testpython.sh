#!/bin/bash
#SBATCH --job-name=eb0
#SBATCH --account=Project_2009950
#SBATCH --partition=test
#SBATCH --time=0:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH -o "o_eb0.txt"
#SBATCH -e "e_eb0.txt"


#echo "Extracting data..."
#unzip -q /scratch/project_2009950/cifar10.zip -d \$TMPDIR
source tykky

python testpython.py

#sbatch 09_evaluation.sh