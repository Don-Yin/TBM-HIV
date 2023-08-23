# submitting a batch to JADE:
set -e
trap 'echo "Error encountered! Pausing..."; read -p "Press enter to continue..."' ERR

conda activate densenet

# -------- functions --------
source jade/utils.sh

# -------- main --------
print_centered " SUBMITTING HYPERPARAM TUNNING JOBs "

echo "initializing - checking the number of total combinations of parameters ..."

total_params=$(python -m grid_search --len)

print_blue "Total parameters to train: $total_params"

for index in $(seq 0 $total_params); do
    print_blue "submitting grid search with index $index / $total_params"

    idx_node_type=$(python -m grid_search --check_node_type $index)

    job_script="jade/job_small.sh"

    if [ "$idx_node_type" = "BIG" ]; then
        print_blue "using big node for $index / $total_params"
        job_script="jade/job_big.sh"
    fi

    if sbatch $job_script $index; then
        python -m grid_search --mark_submitted $index
    else
        echo "An error occurred while submitting grid search with index $index. Press any key to continue or Ctrl+C to exit."
        read -n 1
    fi
done

print_centered " SUBMITTING HYPERPARAM TUNNING JOBs ENDS "
