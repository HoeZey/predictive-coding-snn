#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --job-name=INSTALL_ENV
#SBATCH --ntasks=1
#SBATCH --time=00:00:10
# SBATCH --output=jobs/output/install_env%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

conda env create -f environment.yaml