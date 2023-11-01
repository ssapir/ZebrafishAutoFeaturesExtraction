#!/bin/bash
set -e

dataset_path='/ems/Lab-Shared/Data/'
dataset_path_features='/ems/Lab-Shared/Analysis/'
dir=`dirname $(realpath $0)`
prefix="202"
postfix="_sb"
sub="test_target"

input_args=$@ # save if needed
shift $# # remove arguments - this is preventing bug in source usage below

fish_list=$(ls $dataset_path/$prefix*/processed_data$postfix -d | rev | cut -d/ -f2 | rev)

# Create multiple jobs for analysis using jobarray feature
for fish in $fish_list; do
    # uncomment to skip prev analysed
    #if [[ ! -f "$dataset_path_features/data_set_features/$sub/inter_bout_interval/"$fish"_ibi_processed.mat" ]]; then
       $dir/scripts/run_feeding_assay_fish_features_parallel.sh $fish $dir/scripts/
    #fi
done

