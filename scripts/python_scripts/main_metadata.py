import os
import sys
from fish_preprocessed_data import FishPreprocessedData
import pandas as pd
import numpy as np

from utils.main_utils import get_parameters, get_mat_file_name_parts


def read_excel(file_full_path, names=None):
    if names is None:
        return pd.read_excel(file_full_path)
    return pd.read_excel(file_full_path, header=None, names=names)


def read_csv(file_full_path, names=None):
    if names is None:
        return pd.read_csv(file_full_path, warn_bad_lines=True, error_bad_lines=False)
    return pd.read_csv(file_full_path, warn_bad_lines=True, error_bad_lines=False, names=names, header=None)


def read_file(file_full_path, names=None):
    if not os.path.isfile(file_full_path):
        print("Missing file {0}.\nFound files: {1}".format(file_full_path, os.listdir(os.path.dirname(file_full_path))))
        sys.exit(1)

    if file_full_path.endswith(".csv"):
        return read_csv(file_full_path, names=names)
    elif file_full_path.endswith(".xlsx") or file_full_path.endswith(".xls"):
        return read_excel(file_full_path, names=names)
    print("Error. File format not supported for file ", file_full_path)
    return None  # this is an error


def convert_feeding(feeding_str: str):
    if "before" in feeding_str:
        return 0
    elif "after" in feeding_str:
        return 1
    return np.nan


def convert_outcome(outcome: int):
    to_str = {0: 'abort,escape', 1: 'miss', 2: 'spit', 3: 'hit', 4: 'abort,no-escape', 5: 'miss,no-target',
              6: 'abort,no-target'}
    if outcome in to_str.keys():
        return to_str[outcome]
    return str(outcome)


def convert_complex(complex: ''):
    if complex == 'yes':
        return True
    return False


def convert_comment(comment):
    if pd.isnull(comment) or pd.isna(comment):
        return ""
    return comment


def convert_to_int(data):
    if isinstance(data, float):
        return round(data)
    elif isinstance(data, int):
        return data
    else:
        return np.nan


def add_metadata_to_fish(fish_full_path, fish_name, excel_full_path):
    fish = FishPreprocessedData.import_from_matlab(fish_full_path)
    metadata_df = read_file(excel_full_path)
    metadata_df = metadata_df[metadata_df.fishName == fish_name]  # only current fish
    # validate and fix types (for what can be fixed in excel errors)
    metadata_df.outcome = metadata_df.outcome.map(convert_to_int)
    metadata_df.age = metadata_df.age.map(convert_to_int)
    metadata_df.NumberOfParamecia = metadata_df.NumberOfParamecia.map(convert_to_int)
    metadata_df.AcclimationTime = metadata_df.AcclimationTime.map(convert_to_int)
    # add converted fields
    metadata_df['is_complex_hunt'] = metadata_df.complexhunt.map(convert_complex)
    metadata_df['outcome_str'] = metadata_df.outcome.map(convert_outcome)
    metadata_df['feeding_int'] = metadata_df.Feeding.map(convert_feeding)
    # per event metadata
    for event in fish.events:
        curr = metadata_df.iloc[event.event_id - 1]  # iloc index start from 0
        event.set_metadata(is_complex_hunt=curr.is_complex_hunt,
                           outcome=curr.outcome,
                           outcome_str=curr.outcome_str,
                           comments="{0}. {1}".format(convert_comment(curr.AcquisitionComments),
                                                      convert_comment(curr.CuttingComments)),
                           event_frame_ind=(curr['endFrame'] - curr['startFrame']))  # starts from index 1
    # first row only for fish metadata
    metadata_df = metadata_df.iloc[0]
    fish.set_metadata(age_dpf=metadata_df.age,
                      num_of_paramecia_in_plate=metadata_df.NumberOfParamecia,
                      acclimation_time_min=metadata_df.AcclimationTime,
                      feeding_str=metadata_df.Feeding,
                      feeding=metadata_df.feeding_int)
    fish.export_to_matlab(fish_full_path)


# run me as: python main_metadata.py <data_path> <fish_folder_name>.
# Example: python main_metadata.py /ems/data/FA2020 20200720-f3
if __name__ == '__main__':
    INPUT_NAME = "{0}-frames.csv"
    _, mat_inputs_folder, _, _, data_path, _ = get_parameters()

    mat_files = []
    for name in [f for f in os.listdir(mat_inputs_folder) if f.lower().endswith("_preprocessed.mat")]:
        event_number, fish_name, _, _ = get_mat_file_name_parts(name)
        if name.lower() == (fish_name + "_preprocessed.mat").lower():  # combined
            mat_files.append((name, fish_name))

    excel_full_path = os.path.join(data_path, fish_name, INPUT_NAME.format(fish_name)) # todo change
    if not os.path.exists(excel_full_path):
        raise Exception("Error: ", excel_full_path, " not found (Error in usage!).")

    for (name, fish_name) in mat_files:
        fish_full_path = os.path.join(mat_inputs_folder, name)
        if not os.path.exists(fish_full_path):
            print("Error: ", fish_full_path, " not found (create fish data before running this script).")
        else:
            print("Metadata for fish ", fish_full_path)
            add_metadata_to_fish(fish_full_path, fish_name, excel_full_path)
