#!/bin/bash

#SBATCH --job-name=mle
#SBATCH --output=mle.out
#SBATCH --error=mle.err
#SBATCH --time=24:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=2000
#SBATCH --constraint=fdr

# Load Python modules
module load python/anaconda-2020.02

python datapoints.py 
