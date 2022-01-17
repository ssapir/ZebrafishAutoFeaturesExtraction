#!/bin/bash
set -e

################# Parameters ############################

dir=`dirname $(realpath $0)` # script path
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

# read dirname and basename from input parameter to match script
data_folder=$(basename $dataset_path) 
dataset_path=$(dirname $dataset_path)/ 

echo "Input params: Dataset_path $dataset_path, data_folder $data_folder"

if [[ "$#" -gt 0 ]]; then
   fish=$1
else
   fish='20200720-f2'
fi

dir=`dirname $(realpath $0)`

input_args=$@ # save if needed
shift $# # remove arguments - this is preventing bug in source usage below

sbatch --job-name=cut-$fish $dir/slurm_files/run_cut_events_from_raw.sh $dataset_path $fish $data_folder
