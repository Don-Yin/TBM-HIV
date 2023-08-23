#!/bin/bash -e

conda activate densenet

source jade/utils.sh
initial_directory=$(pwd)

# navigate to script directory
cd src/scripts

# change permissions for all scripts to make them executable
for script in *; do
    chmod +x "$script"
done

# navigate back to the initial directory
cd $initial_directory
print_centered "Scripts in src/scripts are now executable."

# -------- missing values --------
python -m src.clinical.analyse.inspect_missing

# -------- demographics --------
python -m src.scripts.demographics.hiv_group_severity
python -m src.scripts.demographics.variable_selection_flow
python -m src.scripts.demographics.variable_selection_bar
python -m src.scripts.demographics.lesion_distribution

# -------- models --------
python -m src.scripts.models.model_structure

# -------- analysis / interepreation on test set --------
python -m src.analysis.inference_interpret
python -m src.analysis.tbm_lesion_contingency

# -------- send to overleaf on dropbox --------
python -m src.scripts.demographics.send_to_overleaf

print_centered "Done"
