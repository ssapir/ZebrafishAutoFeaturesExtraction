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
  

## Quick-start
(Pre-processing) Analyse a specific fish (local): ``` python main.py <data_path> <fish_folder_name>.```

(Processing) Analyse a specific fish (local): ``` python feature_analuysis/fish_environment/main.py <data_path> --fish_name <fish_folder_name> --override.```

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
