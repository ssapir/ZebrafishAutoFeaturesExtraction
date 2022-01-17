import logging
import os.path
import time
import csv
import traceback

import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
import cv2
import argparse

DISABLE_FRAMES_PROGRESS_BAR = False
FRAME_ROWS = 896
FRAME_COLS = 900
FPS = 30
AVERAGE_EVENT_LENGTH = 500


# =============================================================================
# Funcs
# =============================================================================

def create_video_with_cv2_exe(name, video_frames, fps):
    first_frame = np.array(video_frames[0])
    ny, nx = (first_frame.shape[1], first_frame.shape[0])
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"MJPG"), fps, (ny, nx))
    for frame in video_frames:
        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_GRAY2BGR).astype('uint8'))
    out.release()
    del out


def read_file(file_full_path):
    if not os.path.isfile(file_full_path):
        print("Missing file {0}.\nFound files: {1}".format(file_full_path, os.listdir(os.path.dirname(file_full_path))))
        sys.exit(1)
    if file_full_path.endswith(".csv"):
        return pd.read_csv(file_full_path, warn_bad_lines=True, error_bad_lines=False)
    elif file_full_path.endswith(".xlsx") or file_full_path.endswith(".xls"):
        return pd.read_excel(file_full_path, engine='openpyxl')
    print("Error. File format not supported for file ", file_full_path)
    return None  # this is an error


def write_file(df, file_full_path):
    if file_full_path.endswith(".csv"):
        df.to_csv(file_full_path, index=False, sep=',', header=True)  # False for no header
    elif file_full_path.endswith(".xlsx") or file_full_path.endswith(".xls"):
        df.to_excel(file_full_path, index=False)
    else:
        print("Error. File format not supported for file ", file_full_path)
        return  # this is an error. Don't throw exception to allow events to be created


def create_dirs_and_validate_overriding(output_folders_path, output_csv, subdirs: list):
    # Script cannot override a folder and will throw error (mkdir have modes but better to manually fix)
    if not os.path.exists(output_folders_path):
        os.mkdir(output_folders_path)
    print("Try to create sub-dirs (should be overridden therefore must not exists)")
    for curr in subdirs:
        if os.path.exists(curr):
            raise Exception('Error: Folder already exists: {0}'.format(curr))
        os.mkdir(curr)
    if os.path.exists(output_csv):
        raise Exception('Error: File already exists: {0}'.format(output_csv))

    print("Output will be saved in folder: {0}".format(output_folders_path))


def save_raw(output_fullname, frameStart, frameEnd, input_data_frames):
    output_data = np.memmap(output_fullname, dtype=np.uint8, mode='w+',
                            shape=(frameEnd - frameStart, FRAME_COLS, FRAME_ROWS))
    output_data[:] = input_data_frames[frameStart - 1: frameEnd - 1, :, :].astype(np.uint8)
    del output_data  # flush and release memmap


# =============================================================================
# Main
# =============================================================================

def parse_input_from_command(vid_type, allowed_vid_types, data_folder):
    parser = argparse.ArgumentParser(description='Analyse fish data.')
    parser.add_argument('data_path', type=str, help='Full path to lab\'s data folder (/.../Lab-Shared/Data)')
    parser.add_argument('fish_name', type=str, help='Fish folder name (inside data_folder)')
    parser.add_argument('--data_folder', type=str, help='Data folder name (default: {0})'.format(data_folder))
    parser.add_argument('--input_csv_name', type=str, help='Events CSV name (within data folder {1}) (default: {0})'.format("fish_name", data_folder))
    parser.add_argument('--vid_type', type=str,
                        help='Video type to work on (from: ' + str(allowed_vid_types))

    # print(parser.print_help())
    args = parser.parse_args(sys.argv[1:])
    fish_name = args.fish_name
    if args.vid_type is not None and args.vid_type.lower() in allowed_vid_types:
        vid_type = args.vid_type.lower()
    if args.data_folder is not None:
        data_folder = args.data_folder

    input_csv_folder = os.path.join(args.data_path, data_folder)
    if args.input_csv_name is not None:
        input_csv_name = args.input_csv_name
    else:
        input_csv_name = "FA-{0}.xlsx".format(fish_name)
        input_csv_folder = os.path.join(args.data_path, data_folder, fish_name)

    return args.data_path, data_folder, fish_name, vid_type, input_csv_name, input_csv_folder


def main():
    # defaultf. can be overridden by args
    vid_type = ".raw"
    allowed_vid_types = [".avi", ".raw"]
    data_folder = "FeedingAssay2020"

    if len(sys.argv) >= 2:  # argv[0] is script name. Override default params if given
        lab_path, data_folder, fish_name, vid_type, input_csv_name, input_csv_folder = \
            parse_input_from_command(vid_type=vid_type, allowed_vid_types=allowed_vid_types, data_folder=data_folder,
                                     )
        input_csv_file_path = os.path.join(input_csv_folder, input_csv_name)  # automatic script output
    else:
        lab_path = os.path.join("\\\\ems.elsc.huji.ac.il", "avitan-lab", "Lab-Shared", "Data")
        fish_name = "20201231-f1"
        input_csv_name = "FA-{0}.xlsx".format(fish_name)
        input_csv_file_path = os.path.join(lab_path, data_folder, input_csv_name)  # automatic script output

    # inputs
    input_folder = os.path.join(lab_path, data_folder, fish_name, "raw_whole_movie")

    # outputs
    output_folders_path = os.path.join(lab_path, data_folder, fish_name)
    output_events_folder = os.path.join(output_folders_path, "events")
    output_noise_folder = os.path.join(output_folders_path, "frames_for_noise")
    output_ctrl_events_folder = os.path.join(output_folders_path, "control_events")
    output_csv = os.path.join(output_folders_path, fish_name + "-frames.csv")
    
    print("input csv:", input_csv_file_path, " input folder", input_folder)
    print("output folders: ", output_folders_path, " output csv", output_csv)

    # make sure 1 raw exists
    video = [f for f in os.listdir(input_folder) if f.lower().endswith(vid_type)]
    if len(video) != 1:
        if len(video) == 0:
            raise Exception("Error. Missing file *.{0} in {1}".format(vid_type, input_folder))
        raise Exception("Error. Working on 1 video only (Found {0} videos: {1})".format(len(video), video))

    # create needed dirs - make sure the output sub-folder doesn't already exists!
    create_dirs_and_validate_overriding(output_folders_path, output_csv,
                                        [output_events_folder, output_noise_folder, output_ctrl_events_folder])

    # Validate that can read inputs
    input_data_fullname = os.path.join(input_folder, video[0])
    input_data_frames = np.memmap(input_data_fullname, dtype=np.uint8, mode='r').reshape([-1, FRAME_COLS, FRAME_ROWS])
    n_frames = input_data_frames.shape[0]
    print("Read Input Video: {0}, shape {1}, n_frames {2}".format(input_data_fullname, input_data_frames.shape, n_frames))

    print("Read Input csv: {0}".format(input_csv_file_path))
    all_fish_events_csv_df = read_file(input_csv_file_path)
    curr_fish_df = all_fish_events_csv_df[all_fish_events_csv_df.fishName == fish_name].copy()
    if curr_fish_df.empty:
        raise Exception("Empty dataframe for fish {0} in file {1}".format(fish_name, input_csv_file_path))
    curr_fish_df.index = range(1, 1 + len(curr_fish_df.index))
    print(curr_fish_df)
    print("Output csv: {0}".format(output_csv))
    write_file(curr_fish_df, output_csv)

    start = time.process_time()
    eventStartsEnds = []
    for index, row in tqdm(curr_fish_df.iterrows(), disable=DISABLE_FRAMES_PROGRESS_BAR, desc="current event"):
        frameStart = int(row['startFrame'])
        frameEnd = int(row['endFrame-cutPoint']) + 1
        if frameStart > frameEnd:
            print("Error. Row #{0} has start > end index".format(index))
        eventStartsEnds.append([frameStart, frameEnd])

        print("{2}: Frame start: {0}, frame end: {1}".format(frameStart, frameEnd, index))
        output_fullname = os.path.join(output_events_folder, fish_name + "-" + str(index) + ".raw")
        save_raw(output_fullname, frameStart, frameEnd, input_data_frames)
        print("Output was saved to:", output_fullname)
        output_fullname = os.path.join(output_events_folder, fish_name + "-" + str(index) + ".avi")
        create_video_with_cv2_exe(output_fullname, input_data_frames[frameStart - 1: frameEnd - 1, :, :], FPS)
        print("Output was saved to:", output_fullname)
        if index > 0 and index % 10 == 0:
            print("re-read input memmap")
            del input_data_frames 
            input_data_frames = np.memmap(input_data_fullname, dtype=np.uint8, mode='r').reshape([-1, FRAME_COLS, FRAME_ROWS])

    # Add noise frames
    noise_frames = range(1, n_frames, 1000)
    for index, frameStart in tqdm(zip(range(1, 1 + len(noise_frames)), noise_frames),
                                  disable=DISABLE_FRAMES_PROGRESS_BAR, desc="current noise"):
        frameEnd = frameStart + 1
        print("{2}: Frame start: {0}, frame end: {1}".format(frameStart, frameEnd, index))
        output_fullname = os.path.join(output_noise_folder, fish_name + "-" + str(index) + ".raw")
        save_raw(output_fullname, frameStart, frameEnd, input_data_frames)
        print("Output was saved to:", output_fullname)
        output_fullname = os.path.join(output_noise_folder, fish_name + "-" + str(index) + ".jpg")
        noise_frame = input_data_frames[frameStart - 1: frameEnd - 1, :, :].squeeze()
        print(noise_frame.shape)
        cv2.imwrite(output_fullname, noise_frame)
        print("Output was saved to:", output_fullname)

    # Add ctrl events - important! need to be after reading csv
    try:
        ctrlEventStarts = []
        for start in range(1, n_frames, AVERAGE_EVENT_LENGTH):
            good = True
            for event_start_end in eventStartsEnds:
                if (start < event_start_end[0] < start + AVERAGE_EVENT_LENGTH) or \
                        (event_start_end[0] < start < event_start_end[1]) or \
                        (start < event_start_end[1] < start + AVERAGE_EVENT_LENGTH):
                    good = False
            if good:
                ctrlEventStarts.append(start)
        # Size should be the same as curr_fish_df, since ctrlEventStarts should be much longer
        size = min(len(ctrlEventStarts), curr_fish_df.shape[0])
        control_events_indices = np.random.permutation(np.arange(len(ctrlEventStarts)))[0:size]
        print("ctrl events indices: {0}.".format(control_events_indices))
        control_events = np.array(ctrlEventStarts)[control_events_indices.astype(int)]
        print("Found {0} ctrl events. Taking first {1} of them (df size={2}).".format(len(ctrlEventStarts),
                                                                                      len(control_events),
                                                                                      curr_fish_df.shape[0]))
        for index, frameStart in tqdm(zip(range(1, 1 + len(control_events)), control_events),
                                      disable=DISABLE_FRAMES_PROGRESS_BAR, desc="current control"):
            frameEnd = frameStart + AVERAGE_EVENT_LENGTH
            print("{2}: Frame start: {0}, frame end: {1}".format(frameStart, frameEnd, index))
            output_fullname = os.path.join(output_ctrl_events_folder, fish_name + "-" + str(index) + ".raw")
            save_raw(output_fullname, frameStart, frameEnd, input_data_frames)
            print("Output was saved to:", output_fullname)
            output_fullname = os.path.join(output_ctrl_events_folder, fish_name + "-" + str(index) + ".avi")
            create_video_with_cv2_exe(output_fullname, input_data_frames[frameStart - 1: frameEnd - 1, :, :], FPS)
            print("Output was saved to:", output_fullname)
            if index > 0 and index % 10 == 0:
                print("re-read input memmap")
                del input_data_frames
                input_data_frames = np.memmap(input_data_fullname, dtype=np.uint8, mode='r').reshape([-1, FRAME_COLS, FRAME_ROWS])
    except Exception as err:
        logging.exception(err)
        print(traceback.format_exc())
        traceback.print_tb(err.__traceback__)

    print("Main loop time: ", time.process_time() - start)


main()
