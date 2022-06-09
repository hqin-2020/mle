#!/bin/bash

#SBATCH --job-name=mle
#SBATCH --output=mle.out
#SBATCH --ntasks=4
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

# Load Python modules
module load python/anaconda-2020.02


python mle.py 