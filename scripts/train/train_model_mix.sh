#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=TRAIN_MIX
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:30:00
#SBATCH --output=outputs/train_model/mix/output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

cd $HOME/nddl/

source activate nddl
srun python main_train_model.py -f configs/train_mix.toml
conda deactivate