#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=ENV
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --output=outputs/create_env/output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

conda remove -n nddl --all
conda env create -f environment.yaml