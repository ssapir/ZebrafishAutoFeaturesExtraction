#!/bin/bash
# run me with: sbatch example.sh -p gpu.q --gres:gpu:1 

# Write output as following (%j is JOB_ID)
#SBATCH -o cluster_behavior_agg_features_zebrafish-%j.out

# Ask one CPU and limit run to 2 days
#SBATCH --mem 10G

# should be args
dataset_path='/ems/elsc-labs/avitan-l/Lab-Shared/Analysis/FeedingAssaySapir/'
repository_relative_to_script_path='../'
opencv_conda_env=opencv_contrib_for_behavior
with_plots=1

if [[ "$#" -gt 0 ]]; then
   global_args=$1
else
   global_args="--is_combine_age --outcome_map_type=strike_abort"
fi

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
start_time=$(date)
echo "Start $start_time"

# global sets combine + outcome (metadata names)
export PYTHONPATH=$PYTHONPATH:$path/../$repository_relative_to_script_path
common="--gaussian --is_bounding_box"
echo "features_for_mat.py $dataset_path "*" $common $global_args:"
python $path/../$repository_relative_to_script_path/feature_analysis/fish_environment/features_for_mat.py $dataset_path "*" $common $global_args 
python $path/../$repository_relative_to_script_path/feature_analysis/fish_environment/feature_utils.py $dataset_path "*" $common $global_args 

if [[ $with_plots == 1 ]]; then
echo "plots.py $dataset_path "*" $common $global_args:"
python $path/../$repository_relative_to_script_path/feature_analysis/fish_environment/presentation_plots.py $dataset_path "*" $common $global_args
fi

end_time=$(date)
echo "Stop $end_time"

conda deactivate

#chmod 770 -R $dataset_path/dataset_plots/

