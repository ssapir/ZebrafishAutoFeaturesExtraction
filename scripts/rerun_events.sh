#!/bin/bash

################# Parameters ############################

dir=`dirname $(realpath $0)`/slurm_files/

ids=""
mem="10G"
max_mem="50G"
check=0 # 1 for only check, anything else for not

if [[ "$#" -gt 1 ]]; then
   CONFIG_PATH=$1
   fish=$2
   mem="$3G"
   echo "$fish and mem $mem" 
else
   CONFIG_PATH=$dir/parameters_events.cfg
   fish='20200720-f2'
   #ids="1,2,5,9"
   mem="10G"
fi

if [[ "$#" -gt 3 ]]; then
   check=1
fi

################# Genetic code  #############################
CONFIG_SYNTAX="^\s*#|^\s*$|^[a-zA-Z_]+='[^']*'\s*$"
if egrep -q -v "${CONFIG_SYNTAX}" "$CONFIG_PATH"; then
   echo "Error parsing config file ${CONFIG_PATH}." >&2
   exit 5
fi
source $CONFIG_PATH

# folders
proc_dir="$dataset_path/$fish/processed_data$suf/"
movies_dir="$dataset_path/$fish/debug_movies$suf/"
events_dir="$dataset_path/$fish/$pref"events/

echo "Input params: Dataset_path $dataset_path, args $args, events-data: $events_dir - $proc_dir"

if [[ -z "$ids" ]] ; then
   echo "search missing ids"
   event_ids=$(ls $events_dir/*.raw | rev | cut -d/ -f1 | rev | cut -d. -f1 | sort | cut -d- -f3)
   mat_ids=$(ls $proc_dir/parallel_parts/*frame*to*.mat | rev | cut -d/ -f1 | rev | cut -d. -f1 | cut -d_ -f1-2 | sed 's@_@-@g' | sort | cut -d- -f3)
   echo ev $(echo $event_ids | wc -w): $event_ids
   echo mat $(echo $mat_ids | wc -w): $mat_ids
   # compute diff
   ids=$(comm -23 <(tr ' ' $'\n' <<< "$event_ids") <(tr ' ' $'\n' <<< "$mat_ids"))
   # reduce 1 from each number since joarray works from 0 to num
   ids=$(echo $ids | (tr ' ' $'\n') | xargs -n1 expr -1 +)
   #ids are separated by comma
   ids=$(echo $ids | sed 's@ @,@g')
fi

echo $fish missing ids \(event-1\): $ids

if [[ "$check" -eq 1 ]] ; then
   echo only checking
   exit
fi

if [[ -z "$ids" ]]; then
  echo no jobs to run
  exit
fi
echo run jobarray 

jobs=""
jobarrout=$(sbatch --mem=$mem --job-name=re-events-$fish$name --array $ids $dir/run_jobarr_events_main.sh $dataset_path $fish $args)
jobarrid=$(echo $jobarrout | rev| cut -d ' ' -f1 | rev)
jobs=$(echo $jobs:$jobarrid)

echo jobarray $jobs

# depend on successfull finish of jobarray => call metadata
combineout=$(sbatch --job-name=re-combine-$fish$name --depend=afterany$jobs $dir/run_jobarr_combine_and_metadata.sh $dataset_path $fish $args)
combineid=$(echo $combineout | rev| cut -d ' ' -f1 | rev)

# wait for all jobs finish - chmod even if failure
sbatch --job-name=fix_perm-$fish  --depend=afterany$jobs:$combineid --wrap "chmod 770 -R $proc_dir; chmod 770 -R $movies_dir" --output=/dev/null

next_mem=$(( ${mem%"G"} + 10 ))
if (( ${max_mem%"G"} >= ${mem%"G"} )) ; then
  # not working well
  echo "Re-run with memory $next_mem"
  sbatch --job-name=rerun-$fish  --depend=afterany:$jobarrid:$combineid --wrap "$dir/../rerun_events.sh $CONFIG_PATH $fish $next_mem"
fi
