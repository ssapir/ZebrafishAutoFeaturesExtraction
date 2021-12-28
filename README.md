# ZebrafishAutoFeatureExtraction
An (open-cv based) infrastructure for tracking and features extraction of freely-moving zebrafish.

The original project is a collaboration of programmers, and tracking algorithms. Current repository only contains 
the code I've written, for infrastructure (& supporting scripts), pre-processing cv detection and further features calculation, 
without other programmers' algorithms (which is their property).

This code supports the following use cases:
- Semi-automatic segmentation of hunting events.
- Extraction of various features for describing both the fish and its environment.
- Parallel run.

Outputs:
- .mat files for matlab usage, with a class wrapping python API.
- presentation videos.  
  

## Quick-start and analysis steps
(Pre-processing) Analyse a specific fish: ``` python main.py <data_path> <fish_folder_name>.```

(Processing) Extract features from the dataset: ``` python feature_analysis/fish_environment/main.py <data_path> --fish_name <fish_folder_name> --override.```
For each output (mat) file of pre-processed data, this main creates extended output (based on feature_analysis/fish_environment/fish_processed_data.py)
with additional features calculated. 

The output is in 2 forms: "all_fish" is a folder containing the entire data, "inter_bout_intervals" (IBIs) 
is a subset of the data during beginning and end of each IBI. 

(Post-Processing) Aggregate features: 
After calculating features, this stage reduce the dataset to specific questions asked about the data, for example 
aggregate all events based on age and outcome, and extract statistical measures in specific FOV.

``` python feature_analysis/fish_environment/features_utils.py <data_path> --outcome_map_type hit_miss_abort --is_combine_age --heatmap_type=target_only.```
``` python feature_analysis/fish_environment/features_for_mat.py <data_path> --outcome_map_type hit_miss_abort --is_combine_age --heatmap_type=target_only.```

For each output (mat) file

Read results: see example_run_over_analysis_result_mat_files.py

### (Parallel) analysis on slurm server
See 'scripts' folder for an updated files list. 

Note: these scripts use predefined conda environment for dependencies (see install below).

Two commonly used scripts:
1. ```run_events_parallel.sh <fish_name>```: run main.py on all event files, as pipeline. 
The path and folder names are based on parameters file (written within the script 1st lines).
This script can be run in parallel on several different fish.

2. ```show_available_fish.sh```: show data-set status (available fish for main.py analysis, meaning marked events).   
2.1. ```show_available_fish.sh 1```: show data-set differences (for tracking errors).   


* Use ```squeue --me -o "%.18i %30j %.2t %.10M %.6D %R"``` to see list of your running jobs. 

* Use sacct, seff and sstat to monitor resources (see [slurm documentation](https://slurm.schedmd.com/sacct.html)).

### Quick environment install
Use [conda](https://www.anaconda.com/distribution/) or [pip](https://pypi.org/project/pip/) by following commands:
- conda env create -f environment.yaml
- pip install -r requirements.txt
