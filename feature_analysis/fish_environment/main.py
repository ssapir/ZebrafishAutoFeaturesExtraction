import argparse
import glob
import logging
import os
import sys
import traceback

from tqdm import tqdm
import numpy as np

from feature_analysis.fish_environment.fish_processed_data import FishAndEnvDataset, SingleFishAndEnvData, ExpandedEvent
from fish_preprocessed_data import FishPreprocessedData
from utils.main_utils import create_dirs_if_missing


def inter_bout_interval_data(fish: SingleFishAndEnvData):
    """Reduce data to IBI only

    :param fish:
    :return:
    """
    event: ExpandedEvent
    for event in tqdm(fish.events, desc="ibi current event"):
        event.change_to_ibi_only_data()


def main(full_files_list, fullfile, processed_path, ibi_processed_path, is_saving_whole_dataset=False, override=True):
    """Create env_processed.mat files.

    :param full_files_list:
    :param fullfile:
    :param processed_path:
    :param is_saving_whole_dataset: if true, will aggregate to single 1 mat. Due to size limits,
    better to have false for large files
    :return:
    """
    all_fish = []
    for curr_full_name in tqdm(full_files_list, desc="current fish"):
        curr_fish_name = os.path.basename(curr_full_name).split("_")[0]
        output_path = os.path.join(processed_path, curr_fish_name + "_env_processed.mat")
        error_events = os.path.join(processed_path, curr_fish_name + "_thrown_events.csv")
        if not os.path.exists(output_path) or override:
            try:
                print(curr_full_name)
                exp_f, thrown_events = \
                    SingleFishAndEnvData.from_preprocessed(FishPreprocessedData.import_from_matlab(curr_full_name))
                np.savetxt(error_events, np.array(thrown_events, dtype=str), delimiter=',', fmt='%s')
                exp_f.export_to_matlab(output_path)
            except Exception as e:
                print(e)
                traceback.print_tb(e.__traceback__)
            print("Saved ", output_path)
        else:
            exp_f = SingleFishAndEnvData.import_from_matlab(output_path)

        ibi_output_path = os.path.join(ibi_processed_path, exp_f.name + "_ibi_processed.mat")
        inter_bout_interval_data(exp_f)
        exp_f.export_to_matlab(ibi_output_path)
        print("Saved ", ibi_output_path)

        if is_saving_whole_dataset:
            all_fish.append(exp_f)  # note - ibis only

    if is_saving_whole_dataset:
        print("Found ", len(all_fish), " fish with names: ", [fish.name for fish in all_fish])
        dataset = FishAndEnvDataset(all_fish)
        dataset.export_to_matlab(fullfile)
        print("Saved to ", fullfile)


def parse_input_from_command():
    parser = argparse.ArgumentParser(description='Analyse fish data.')
    parser.add_argument('data_path', type=str, help='Full path to data folder (events)')
    parser.add_argument('--fish_name', default="*",
                        type=str, help='Fish folder name (inside events). If empty- run all')
    parser.add_argument('--override', default=False, action='store_true',
                        help='Override if previous features exist (default: false, skip if already exists)')
    parser.add_argument('--analysis_folder', default=None, type=str,
                        help='A different path for the analysis outputs')

    # print(parser.print_help())
    args = parser.parse_args(sys.argv[1:])
    return args.data_path, args.fish_name, args.override, args.analysis_folder


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) >= 1:  # argv[0] is script name
        data_path, fish_name, override, analysis_folder = parse_input_from_command()
    else:
        data_path = "E:\\Lab-Shared\\Data\\FeedingAssay2020\\"
        fish_name = "*"
        override = True
        analysis_folder = None

    output_folder = analysis_folder if analysis_folder is not None else data_path

    d2 = data_path.replace("Data/FeedingAssay2020", "Analysis/FeedingAssaySapir")
    add_i, add_o = "", "_23.8"
    if fish_name == "*":  # wildcard fish name + take combined fish data
        full_files_list_ = glob.glob(os.path.join(data_path, "*", "processed_data" + add_o, "*_preprocessed.mat"))
    else:
        full_files_list_ = glob.glob(os.path.join(data_path, fish_name, "processed_data" + add_o, "*_preprocessed.mat"))
    full_files_list_ = [name for name in full_files_list_ if "_frame_" not in name.lower()]  # todo use regex?

    logging.info("Fish {0} found ({1}) files: {2}".format(fish_name, len(full_files_list_), full_files_list_))
    #logging.info("Output folder {0}, input folder {1}".format(output_folder, data_path))
    logging.info("Output folder {0}, input folder {1}".format(d2, data_path))

    #fullfile_path = os.path.join(output_folder, "data_set_features")
    fullfile_path = os.path.join(d2, "data_set_features")
    processed_path = os.path.join(fullfile_path, "all_fish")
    ibi_processed_path = os.path.join(fullfile_path, "inter_bout_interval")
    create_dirs_if_missing([fullfile_path, processed_path, ibi_processed_path])

    fullfile = os.path.join(fullfile_path, "fish_env_dataset.mat")
    main(full_files_list_, fullfile, processed_path, ibi_processed_path, override=override)
