#!/bin/bash
set -e

dir=`dirname $(realpath $0)`

for h in "--heatmap_type=residuals" "--heatmap_type=target_only" "--heatmap_type=all_para" ; do
    for o in "--outcome_map_type=hit_miss_abort" ; do
        for a in "--is_combine_age"; do
            for n in "--heatmap_n_paramecia_type=n30" "--heatmap_n_paramecia_type=n50" "--heatmap_n_paramecia_type=all"; do
               for f in "--feeding_type=all_feeding" "--feeding_type=before_feeding" "--feeding_type=after_feeding"; do
                if [[ "$a" == "--is_combine_age" ]]; then
                   for g in "--age_groups=v2" ; do
		      args="--no_metadata $h $o $a $n $g $f"
                      sbatch $dir/slurm_files/run_parallel_agg_features.sh "$args"
		   done
		# else - not supported now
		fi
	      done
	    done
	done
    done
done


