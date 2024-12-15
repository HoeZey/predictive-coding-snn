#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=TRAIN_BOTTLENECK
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=outputs/train_model/bottleneck/output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

cd $HOME/nddl/

source activate nddl
srun python main_train_bottleneck.py -f configs/train_bottleneck_energy.toml
conda deactivate