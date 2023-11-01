import argparse
import glob
import sys
import os
from tqdm import tqdm

# hand made fun
from fish_preprocessed_data import FishPreprocessedData


def main(full_files_list):
    for curr_full_name in tqdm(full_files_list, desc="current fish"):
        fish = FishPreprocessedData.import_from_matlab(curr_full_name)
        # todo your code :)


def parse_input_from_command():
    """Read input argv, with default

    :return:
    """
    parser = argparse.ArgumentParser(description='Analyse fish data.')
    parser.add_argument('data_path', type=str, help='Full path to data folder (FA2020 for example)')
    parser.add_argument('--fish_name', default="*",
                        type=str, help='Fish folder name (inside events). Empty or * will run over all fish')

    # todo this is the very minimum. Add as much as needed
    args = parser.parse_args(sys.argv[1:])
    return args.data_path, args.fish_name


# run me with data path, and an optional fish name (for debug/parallel run purpose)
if __name__ == '__main__':
    if len(sys.argv) >= 1:  # argv[0] is script name. If any args given, use them
        data_path, fish_name = parse_input_from_command()
    else:
        data_path = "E:\\Lab-Shared\\Data\\"
        fish_name = "*"  # wildcard fish name

    full_files_list_ = glob.glob(os.path.join(data_path, fish_name, "processed_data", "*_preprocessed.mat"))
    full_files_list_ = [name for name in full_files_list_ if "_frame_" not in name.lower()]  # only final outputs

    main(full_files_list_)
