#!/bin/bash
# Write output as following (%j is JOB_ID)
#SBATCH -o cluster_behavior_features_zebrafish-%j.out
#SBATCH -e cluster_behavior_features_zebrafish-%j.err

#SBATCH --mem 35G

dataset_path=$1
fish=$2
repository_relative_to_script_path=$3
opencv_conda_env=$4

input_args=$@ # save if needed
shift $# # remove arguments - this is preventing bug in source usage below

curr_date=$(date +%F_time_%H-%M-%S)

# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
# `if [ -n $SLURM_JOB_ID ]` checks if $SLURM_JOB_ID is not an empty string
if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}' | cut -f1 -d " ")
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi
echo $fish $SCRIPT_PATH
path=$(dirname $SCRIPT_PATH)

# get script's path to allow running from any folder without errors
source ~/anaconda3/bin/activate
conda init
conda activate $opencv_conda_env

echo "feature_analysis/fish_environment/main.py $dataset_path --fish_name $fish --override"
export PYTHONPATH=$PYTHONPATH:$path/../$repository_relative_to_script_path
python $path/../$repository_relative_to_script_path/feature_analysis/fish_environment/main.py $dataset_path --fish_name $fish --override
end_time=$(date)
echo "Stop $end_time"

conda deactivate

