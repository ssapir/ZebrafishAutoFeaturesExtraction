#!/bin/bash

################## Slurm part  ##############################################

# Write output as following (%j is JOB_ID)
#SBATCH -o cluster_behavior_zebrafish_behavior-combine-%j.out
#SBATCH -e cluster_behavior_zebrafish_behavior-combine-%j.err

#SBATCH -t 1:00:00

################# Parameters (events + full run) ############################

dataset_path=$1
fish=$2
args=''
suf=''
if [[ $# -ge 3 ]]; then
  if [[ "$3" == "--full" ]] ; then
     args='--full'
     suf='_whole_movie'
  elif [[ "$3" == "--control"  ]]; then
     args='--control_data'
     suf='_control'
  else
     echo "error: unknown 3rd argument $3" >&2; exit 1
  fi
fi

echo "combine $args (suf $suf)"
curr_date=$(date +%F_time_%H-%M-%S)

input_args=$@ # save if needed
shift $# # remove arguments - this is preventing bug in source usage below

# check if script is started via SLURM or bash
if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}' | cut -f1 -d " ")
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi

# get script's path 
path=$(dirname $SCRIPT_PATH)

################# Main (same code can be used manually) #############################

# activate anaconda installed on your user (Default: /ems/..../<lab>/<user>/anaconda3
source ~/anaconda3/bin/activate
conda init

# Use predefined opencv - this should be created from env file
conda activate opencv_for_behavior 

# Export working directory to allow pipeline_scripts to use main modules - path is relative to script path
export PYTHONPATH=$PYTHONPATH:$path/../../../ZebrafishBehaviorTracking/
python $path/../../scripts/python_scripts/main_combine_parallel.py $dataset_path $fish $args
python $path/../../scripts/python_scripts/main_metadata.py $dataset_path $fish $args

conda deactivate

# quickfix permissions on output
chmod 770 -R $dataset_path/$fish/processed_data$suf/
chmod 770 -R $dataset_path/$fish/debug_movies$suf/
