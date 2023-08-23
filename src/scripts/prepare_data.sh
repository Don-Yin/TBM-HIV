#!/bin/bash -e
# this script prepares the clinical data for the analysis
# no need to run this on JADE script if the data is already prepared

source jade/utils.sh
# Store the current directory path
initial_directory=$(pwd)

# navigate to script directory
cd src/scripts

# change permissions for all scripts to make them executable
for script in *; do
    chmod +x "$script"
done

# navigate back to the initial directory
cd $initial_directory

print_centered "-"
echo "Conda environment activated; All scripts in src/scripts are now executable."
print_centered "-"

source activate.sh

python -m src.scripts.prepare.merge
python -m src.scripts.prepare.keep_only_selected
python -m src.scripts.prepare.check_missing_random
python -m src.scripts.prepare.imputation
# python -m src.scripts.train_test_split  # not needed for now as the data is already split with a good ks stat

echo "All scripts ran successfully."
