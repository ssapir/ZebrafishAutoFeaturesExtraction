#!/bin/bash
set -e

dir=`dirname $(realpath $0)`

################# Parameters ############################

CONFIG_PATH=$dir/parameters_feeding_assay.cfg
fish_pref='202'  # 2021/2020 prefix

# (read cfg file after validating content)
# commented lines, empty lines and lines key='Value' are valid
CONFIG_SYNTAX="^\s*#|^\s*$|^[a-zA-Z_]+='[^']*'\s*$"

# check if the file contains something we don't want (todo function)
if egrep -q -v "${CONFIG_SYNTAX}" "$CONFIG_PATH"; then
   echo "Error parsing config file ${CONFIG_PATH}." >&2
   echo "The following lines in the configfile do not fit the syntax:" >&2
   egrep -vn "${CONFIG_SYNTAX}" "$CONFIG_PATH"
   exit 5
fi

source $CONFIG_PATH
events_dir=$pref'events'
proc_dir='processed_data'$suf
echo "Input params: Dataset_path $dataset_path, events dir $events_dir, data dir $proc_dir"
################# Main ############################

if [[ "$#" -gt 0 ]]; then
   all=false
else
   all=true
fi


fish_list=$(ls -d "$dataset_path/$fish_pref"*/$events_dir/ | awk -F"$dataset_path/" '{print $NF}' | awk -F"$events_dir" '{print $1}' | sed 's@/@@g')
echo "All fish with event files:"; echo $fish_list
echo "Total #events: $(ls "$dataset_path/$fish_pref"*/$events_dir/*.raw | wc -l)"

for fish in $fish_list; do
   if $all; then
      echo "$fish has $(ls $dataset_path/$fish/$events_dir/*.raw | wc -l) events"
      if [ -d $dataset_path/$fish/$proc_dir ]; then
         c_date=$(stat -c %y $dataset_path/$fish/$proc_dir | awk '{print $1}' | sed 's@-@_@g')
         is_combined=false
         if [ -f "$dataset_path/$fish/$proc_dir/$fish"_preprocessed.mat ]; then is_combined=true; fi
         echo "$fish has $(ls $dataset_path/$fish/$proc_dir/*frame*to*.mat | wc -l) mat files (from date $c_date). Combined? $is_combined"
      fi
   else
      n_events=$(ls $dataset_path/$fish/$events_dir/*.raw | wc -l)
      n_mat=$(ls $dataset_path/$fish/$proc_dir/*frame*to*.mat | wc -l)
      is_combined=false
      if [ -f "$dataset_path/$fish/$proc_dir/$fish"_preprocessed.mat ]; then is_combined=true; fi
      if [ $n_events -ne $n_mat ]; then echo "$fish: $n_events events, $n_mat data"  ; fi
      if ! $is_combined; then echo "$fish combined? $is_combined"; fi
   fi
done
