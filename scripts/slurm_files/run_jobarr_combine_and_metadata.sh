#!/bin/bash

################## Slurm part  ##############################################

# Write output as following (%j is JOB_ID)
#SBATCH -o cluster_behavior_zebrafish_behavior-combine-%j.out
#SBATCH -e cluster_behavior_zebrafish_behavior-combine-%j.err

#SBATCH -t 1:00:00

################# Parameters (events + full run) ############################

dataset_path=$1
fish=$2
repository_relative_to_script_path=$3
opencv_conda_env=$4
args=''
suf=''
is_metadata=0

# this supports both 
# inside the github's repo, I use a config file for similar purpose
if [[ $# -ge 5 ]]; then
  if [[ "$5" == "--full" ]] ; then
     args='--full'
     suf='_whole_movie'
  elif [[ "$5" == "--control"  ]]; then
     args='--control_data'
     suf='_control'
  elif [[ "$5" == "--metadata" ]] ; then
     is_metadata=1
  elif [[ $# -eq 5 ]]; then  # no 6th
     echo "error: unknown 5rd argument $@" >&2; exit 1
  fi
  if [[ $# -ge 6 ]]; then
      if [[ "$6" == "--metadata" ]] ; then
        is_metadata=1
      fi
  fi
fi

echo "combine $args (suf $suf)"
curr_date=$(date +%F_time_%H-%M-%S)

input_args=$@ # save if needed
shift $# # remove arguments - this is preventing bug in source usage below

################# Main (same code can be used manually) #############################

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


# activate anaconda installed on your user (Default: /ems/..../<lab>/<user>/anaconda3
source ~/anaconda3/bin/activate
conda init

# Use predefined opencv env
conda activate $opencv_conda_env

# Export working directory to allow pipeline_scripts to use main modules
export PYTHONPATH=$PYTHONPATH:$path/../$repository_relative_to_script_path
python $path/../$repository_relative_to_script_path/scripts/python_scripts/main_combine_parallel.py $dataset_path $fish $args

if [[ $is_metadata == 1 ]]; then
    python $path/../$repository_relative_to_script_path/scripts/python_scripts/main_metadata.py $dataset_path $fish $args
fi

conda deactivate

# quickfix permissions on output
chmod 770 -R $dataset_path/$fish/processed_data$suf/
chmod 770 -R $dataset_path/$fish/debug_movies$suf/
