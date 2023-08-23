#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH -p big

# -------- functions --------
source jade/utils.sh

# -------- main --------
if [ $# -eq 0 ]; then
    echo "No arguments supplied. Please provide an index as an argument."
fi

index=$1
python -m grid_search --run_with $index || exit 1
