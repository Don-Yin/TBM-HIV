# submitting a batch to JADE:
set -e
trap 'echo "Error encountered! Pausing..."; read -p "Press enter to continue..."' ERR

conda activate densenet

# -------- functions --------
source jade/utils.sh

# -------- main --------
sbatch "jade/analyse_small.sh"
