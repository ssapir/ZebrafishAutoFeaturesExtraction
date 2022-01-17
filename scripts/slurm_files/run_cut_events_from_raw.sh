#!/bin/bash

# Write output as following (%j is JOB_ID)
#SBATCH -o cluster_behavior_cut_events-%j.out
#SBATCH -e cluster_behavior_cut_events-%j.err

# Ask one CPU and limit run to 2 days
#SBATCH -t 3:00:00
#SBATCH --mem 10G

# parameters!
dataset_path=$1
fish=$2
data_folder=$3

curr_date=$(date +%F_time_%H-%M-%S)

input_args=$@ # save if needed
shift $# # remove arguments - this is preventing bug in source usage below

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
echo $SCRIPT_PATH

# get script's path to allow running from any folder without errors
path=$(dirname $SCRIPT_PATH)

# activate anaconda installed on your user (Default: /ems/..../<lab>/<user>/anaconda3
source ~/anaconda3/bin/activate
conda init
# Use predefined opencv env
conda activate opencv_contrib_for_behavior

python $path/../automatic_events_detection/cut_events_from_raw.py $dataset_path $fish --data_folder $data_folder

conda deactivate

# quickfix permissions on output
chmod 770 -R $dataset_path/$data_folder/$fish/events/
chmod 770 -R $dataset_path/$data_folder/$fish/frames_for_noise/
chmod 770 -R $dataset_path/$data_folder/$fish/control_events/
