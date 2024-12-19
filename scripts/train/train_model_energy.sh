#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=TR_ENRG
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=outputs/train_model/energy/output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

cd $HOME/nddl/

source activate nddl
srun python main_train_original.py -f configs/train_energy.toml
conda deactivate