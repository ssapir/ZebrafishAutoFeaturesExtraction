#!/bin/bash
set -e

################# Parameters ############################

dir=`dirname $(realpath $0)` # script path
CONFIG_PATH=$dir/parameters_events.cfg

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

f1="$dataset_path/$fish/processed_data$suf/"
f2="$dataset_path/$fish/debug_movies$suf/"

n_events=$(ls "$dataset_path/$fish/$pref"events/*.raw | wc -l)
echo "$dataset_path/$fish/$pref""events/ has $n_events events"

# validate number
re='^[0-9]+([.][0-9]+)?$'  # number regex
if ! [[ $n_events =~ $re ]] ; then
   echo "error: Not a number" >&2; exit 1
fi

# create folders if doesn't exist (before parallel run) - needed?
#mkdir -p $f1
#mkdir -p $f2

num=$(( $n_events - 1))
echo "# events: $n_events, n_jobs: 0-$num"

################# Main (pipeline build, run 'squeue --me' to follow jobs) #############################

# Create multiple jobs for analysis using jobarray feature
jobarrout=$(sbatch --job-name=events-$fish$name --array 0-$num $dir/slurm_files/run_jobarr_events_main.sh $dataset_path $fish $args)
jobarrid=$(echo $jobarrout | rev | cut -d ' ' -f1 | rev)
echo "id $jobarrid from $jobarrout "

# depend on finishing jobarray => call metadata
combineout=$(sbatch --job-name=combine-$fish$name --depend=afterany:$jobarrid $dir/slurm_files/run_jobarr_combine_and_metadata.sh $dataset_path $fish $args)
combineid=$(echo $combineout | rev | cut -d ' ' -f1 | rev)

# wait for all jobs finish - chmod even if failed
chout=$(sbatch --job-name=fix-perm-$fish$name  --depend=afterany:$jobarrid:$combineid --wrap "chmod 770 -R $f1; chmod 770 -R $f2" --output=/dev/null)
chid=$(echo $chout | rev | cut -d ' ' -f1 | rev)

# todo fixme before add
if $rerun; then
  sbatch --job-name=rerun-$fish  --depend=afterany:$jobarrid:$combineid:$chid --wrap "$dir/rerun_events.sh $CONFIG_PATH $fish 10"
fi

# todo add presentation movies?
