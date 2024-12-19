#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=TR_S_NE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=outputs/train_model/ssl_NE/output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

cd $HOME/nddl/

source activate nddl
srun python main_train_ssl.py -f configs/train_ssl_no_energy.toml
conda deactivate