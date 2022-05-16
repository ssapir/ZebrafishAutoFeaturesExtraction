#!/bin/bash
set -e

# run me with: ./scripts/run_feeding_assay_fish_features_parallel.sh <fish-name>

################# Parameters ############################

if [[ "$#" -gt 1 ]]; then
   dir=$2
else
   dir=`dirname $(realpath $0)` # script path
fi
echo "dir: $dir"
CONFIG_PATH=$dir/parameters_feeding_assay.cfg

################# Genetic code  #############################

# (read cfg file after validating content)
# commented lines, empty lines and lines key='Value' are valid
CONFIG_SYNTAX="^\s*#|^\s*$|^[a-zA-Z_]+='[^']*'\s*$"

# check if the file contains something we don't want
if egrep -q -v "${CONFIG_SYNTAX}" "$CONFIG_PATH"; then
   echo "Error parsing config file ${CONFIG_PATH}." >&2
   echo "The following lines in the configfile do not fit the syntax:" >&2
   egrep -vn "${CONFIG_SYNTAX}" "$CONFIG_PATH"
   exit 5
fi
source $CONFIG_PATH

echo "Input params: Dataset_path $dataset_path, args $args, rerun $rerun"
# todo validate additional parameters?

# For convenience. Can for loop from outside over all missing fish
if [[ "$#" -gt 0 ]]; then
   fish=$1
else
   fish='20200720-f2'
fi

input_args=$@ # save if needed
shift $# # remove arguments - this is preventing bug in source usage below

sbatch --job-name=features-$fish $dir/slurm_files/run_features_per_fish.sh $dataset_path $fish $repository_relative_to_script_path $opencv_conda_env
