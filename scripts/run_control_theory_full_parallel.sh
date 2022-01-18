#!/bin/bash
set -e

################# Parameters ############################

dir=`dirname $(realpath $0)` # script path
CONFIG_PATH=$dir/parameters_control_theory.cfg

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

# For convenience. Can for loop from outside over all missing fish
if [[ "$#" -gt 0 ]]; then
   fish=$1
else
   fish='20210715-f3'
fi

# create folders if doesn't exist (before parallel run!)
f1="$dataset_path/$fish/processed_data_whole_movie$suf/"
f2="$dataset_path/$fish/debug_movies_whole_movie$suf/"

input_args=$@ # save if needed
shift $# # remove arguments - this is preventing bug in source usage below

################# Main (pipeline build, run 'squeue --me' to follow jobs) ####################################

################# Part 1: read how many jobs needed (split full to parallel pieces)###########################
source ~/anaconda3/bin/activate
conda init
conda activate $opencv_conda_env

args="--vid_type .raw --full --parallel"
export PYTHONPATH=$PYTHONPATH:$dir/$repository_relative_to_script_path/  # allow using the module's imports from inner folders
value=$(python $dir/$repository_relative_to_script_path/scripts/python_scripts/main_calc_videos_n_frames.py "$dataset_path" $fish $args)
conda deactivate

n_frames=$(echo $value | rev| cut -d ' ' -f1 | rev)
re='^[0-9]+([.][0-9]+)?$'  # number regex
if ! [[ $n_frames =~ $re ]] ; then
   echo "error: Not a number" >&2; exit 1
fi

# calculate how parallel from size and length
num=$(( $n_frames / $size - 1))
if [ $(($n_frames % $size )) -gt 0 ]; then num=$(( $num + 1)); fi # ceil

echo "Total frames: $n_frames, n_jobs: 0-$num for size $size"

################# Part 2: create pipeline in slurm  ################################

# Create multiple jobs for analysis using jobarray feature
jobarrout=$(sbatch --job-name=$fish-full --array 0-$num $dir/slurm_files/run_jobarr_full_main.sh $dataset_path $fish $size $repository_relative_to_script_path $opencv_conda_env)
jobarrid=$(echo $jobarrout | rev| cut -d ' ' -f1 | rev)
echo "id $jobarrid from $jobarrout "

# depend on successfull finish of jobarray => call metadata
combineout=$(sbatch --job-name=combine-$fish --depend=afterany:$jobarrid $dir/slurm_files/run_jobarr_combine_and_metadata.sh $dataset_path $fish $repository_relative_to_script_path $opencv_conda_env --full)
combineid=$(echo $combineout | rev| cut -d ' ' -f1 | rev)

# wait for all jobs finish - chmod even if failure
sbatch --job-name=fix_perm-$fish  --depend=afterany:$jobarrid:$combineid --wrap "chmod 770 -R $f1; chmod 770 -R $f2" --output=/dev/null

# todo should this have rerun? currently, if segment fails, it can be fixed manually by running the python scripts
