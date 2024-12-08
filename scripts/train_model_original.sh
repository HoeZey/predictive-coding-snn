#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=ORIGINAL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=outputs/train_model/train_model_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

cd $HOME/nddl/

source activate nddl
srun python main_train_model.py -f configs/train_original_config.toml
conda deactivate