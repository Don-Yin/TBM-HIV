#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH -p small

# -------- functions --------
conda activate densenet
source jade/utils.sh

# -------- main --------
python -m analyse || exit 1
