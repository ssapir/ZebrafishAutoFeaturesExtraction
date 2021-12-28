import argparse
import glob
import gzip
import os
import pickle
import re
import sys

import numpy as np

from fish_preprocessed_data import Paramecium, Event, FishPreprocessedData
from utils import video_utils


class FishOutput:
    """
    Struct holds subset of fish analysis result, extracted via main loop to fish struct
    """

    def __init__(self):
        self.origin_head_points_list = []
        self.destination_head_points_list = []
        self.eyes_abs_angle_list = []  # angle of the eyes
        # diff from direction - saved out due to rounding mistakes if calculating diff frp, direction - eyes
        self.eyes_head_dir_diff_angle_list = []
        self.eyes_areas_pixels_list = []  # angle of the eyes
        self.fish_status_list = []
        self.is_head_prediction_list = []
        self.tail_tip_point_list = []
        self.tail_path_list = []
        self.tail_tip_status_list = []
        self.is_bout_frame_list = []
        self.velocity_norms = []

    def reset(self, frame_start, n_frames):
        for val in self.__dict__.values():
            del val

        total_number_of_frames = n_frames - frame_start + 1
        nan_pair_list = np.full([total_number_of_frames, 2], fill_value=np.nan)
        self.origin_head_points_list = nan_pair_list.copy()
        self.destination_head_points_list = nan_pair_list.copy()
        self.eyes_abs_angle_list = nan_pair_list.copy()
        self.eyes_head_dir_diff_angle_list = nan_pair_list.copy()
        self.eyes_areas_pixels_list = nan_pair_list.copy()
        self.fish_status_list = [False] * total_number_of_frames
        self.is_head_prediction_list = [False] * total_number_of_frames
        self.tail_tip_status_list = [False] * total_number_of_frames
        # todo If nan is found when doing a convolution the entire array turns to nan, we need to solve this somehow
        self.tail_tip_point_list = np.zeros([total_number_of_frames, 2])
        self.tail_path_list = [[]]*total_number_of_frames # Since tail_path_list is a list of lists, at the length of the total number of frames.
        self.is_bout_frame_list = [False] * total_number_of_frames
        self.velocity_norms = np.array([np.nan] * total_number_of_frames)

    def __str__(self):
        return "Lengths: {0}".format({k: len(v) for k, v in self.__dict__.items()})


class FishContoursAnnotationOutput:
    def __init__(self):
        self.fish_contour = []
        self.eyes_contour = []
        self.ellipse_centers = []
        self.ellipse_angles = []
        self.ellipse_axes = []

    def __str__(self):
        return "Lengths: {0}".format({k: len(v) for k, v in self.__dict__.items()})

    def reset(self, frame_start, n_frames):
        for val in self.__dict__.values():
            del val
        
        total_number_of_frames = n_frames - frame_start + 1
        nan_pair_list = np.full([total_number_of_frames, 2], fill_value=np.nan)
        nan_4_list = np.full([total_number_of_frames, 2, 2], fill_value=np.nan)
        self.fish_contour = np.array([np.nan] * total_number_of_frames, dtype=object)
        self.eyes_contour = np.array([np.nan] * total_number_of_frames, dtype=object)
        self.ellipse_centers = nan_4_list.copy()
        self.ellipse_angles = nan_pair_list.copy()
        self.ellipse_axes = nan_4_list.copy()


def save_middle_mat_file(mat_output_folder, filename, frame_start, frame_end, event_id, fish_output: FishOutput,
                         paramecium_list, fish_name):
    curr_event_data, ok = tracker_outputs_to_event(event_id, filename, fish_output, paramecium_list)
    if not ok:
        print("Error, couldn't create struct for " + filename)
        return

    # create and save fish
    current_fish = FishPreprocessedData(fish_name, [curr_event_data])
    name = os.path.join(mat_output_folder,
                        "{0}_{1}_frame_{2}_to_{3}_preprocessed.mat".format(fish_name, event_id,
                                                                           frame_start, frame_end).lower())
    print("End. Saving fish...", name)
    current_fish.export_to_matlab(name)


def tracker_outputs_to_event(event_id, filename, fish_output: FishOutput, paramecium_list):
    # Create event instance
    paranicium_data = Paramecium.from_tracker_output(paramecium_list)
    curr_event_data, ok = \
        Event.from_tracker_output(filename,
                                  event_id=event_id,
                                  origin_head_points_list=fish_output.origin_head_points_list,
                                  destination_head_points_list=fish_output.destination_head_points_list,
                                  fish_tracking_status_list=fish_output.fish_status_list,
                                  is_head_prediction_list=fish_output.is_head_prediction_list,
                                  eyes_abs_angle_list=fish_output.eyes_abs_angle_list,
                                  eyes_head_dir_diff_ang_list=fish_output.eyes_head_dir_diff_angle_list,
                                  eyes_areas_pixels_list=fish_output.eyes_areas_pixels_list,
                                  tail_tip_point_list=fish_output.tail_tip_point_list,
                                  tail_path_list=fish_output.tail_path_list,
                                  tail_tip_status_list=fish_output.tail_tip_status_list,
                                  is_bout_frame_list=fish_output.is_bout_frame_list,
                                  velocity_norms=fish_output.velocity_norms,
                                  paramecium_tracker_output=paranicium_data)
    return curr_event_data, ok


def save_debug_avi(video_output_folder, filename, fish_tracker_name, frame_start, frame_end, video_frames, fps,
                   IS_CV2_VID_WRITE=True):
    # save video - todo very slow right now
    name = os.path.join(video_output_folder,
                        "{0}_preprocessed_by_{1}_frame_{2}_to_{3}.avi".format(
                            filename, fish_tracker_name, frame_start, frame_end).lower())
    print("End. Saving video output...", name)
    if IS_CV2_VID_WRITE:
        video_utils.create_video_with_cv2_exe(name, video_frames, fps)
    else:
        video_utils.create_video_with_local_ffmpeg_exe(name, video_frames, fps)
    print("End. Saving output - done.")


def save_annotation_data(video_output_folder, filename, fish_tracker_name, frame_start, frame_end,
                         fish_contours_output: FishContoursAnnotationOutput, fish_output: FishOutput):
    name = os.path.join(video_output_folder,
                        "{0}_preprocessed_by_{1}_frame_{2}_to_{3}".format(
                            filename, fish_tracker_name, frame_start, frame_end).lower())
    print("End. Saving video annotation output...", name)
    varying_len_keys = ['eyes_contour', 'fish_contour']  # save as object and not multi-dim matrix
    np.savez_compressed(name + ".npz",
                        **{k: np.array(v, dtype=object) for (k, v) in fish_contours_output.__dict__.items()
                           if k in varying_len_keys},
                        **{k: v for (k, v) in fish_contours_output.__dict__.items() if k not in varying_len_keys},
                        **fish_output.__dict__)  # This generates a VisibleDeprecationWarning as of now.
    with gzip.open(name + ".pklz", 'wb') as f:
        pickle.dump(fish_contours_output, f)
        pickle.dump(fish_output, f)
    print("End. Saving output - done.")


def load_annotation_data(full_file_name):
    if full_file_name.endswith(".npz"):
        loaded = np.load(full_file_name, allow_pickle=True)
        fish_contours_output = FishContoursAnnotationOutput()
        fish_output = FishOutput()
        for curr_key in loaded.files:
            if curr_key in fish_contours_output.__dict__.keys():
                fish_contours_output.__setattr__(curr_key, loaded[curr_key])
            elif curr_key in fish_output.__dict__.keys():
                fish_output.__setattr__(curr_key, loaded[curr_key])
            else:
                print("Error. Key {0} in file .npz does not match expected class".format(curr_key))
        return fish_output, fish_contours_output
    elif full_file_name.endswith(".pkl") or full_file_name.endswith(".pklz"):
        open_file_func = open
        if full_file_name.endswith(".pklz"):
            open_file_func = gzip.open
        with open_file_func(full_file_name, 'rb') as f:
            fish_contours_output = pickle.load(f)
            fish_output = pickle.load(f)
            return fish_output, fish_contours_output
    else:
        print("Error. Unsupported file type ", full_file_name)
    return None, None


def create_dirs_if_missing(dirs_list):
    for curr_dif in dirs_list:
        if not os.path.exists(curr_dif):
            os.makedirs(curr_dif)


def get_info_from_event_name(name):
    """Search for pattern: <date-digits>-f<digits>-<digits><smt>.avi/mp4 (- or _ as separators)
    to extract fish_name, event_number
    If not found, split to find name

    :param name:
    :return:
    """
    pattern = re.match(r'^(?=(\d+(?:-|_)f\d+)(?:-|_)(\d+)).*\.(avi|mp4|raw|AVI|MP4|RAW)$', name)
    # Match: <date-digits>-f<digits>-<digits><smt>.avi/mp4
    if pattern is None or len(pattern.groups()) != 2:  # hard coded if the above didnt work
        event_name = name.split(".")[0]
        fish_name = "-".join(event_name.split("-")[0:2])  # todo this wont work for _
        event_number = -1
        if len(event_name.split("-")) > 2:
            event_number = event_name.split("-")[-1]
            if not event_number.isnumeric():
                print("Error. Event {0} can't find a valid event number.".format(event_name))
                event_number = -1
    else:
        fish_name = pattern.group(1)
        event_number = pattern.group(2)  # this is matching \d+ therefore 100% number
    return fish_name, int(event_number)


def parse_video_name(name):
    pattern_parts = re.match(
        r'^(?=(\d+(?:-|_)f\d+)(?:-|_)(-?\d+).*(?:-|_)frame(?:-|_)(\d+)(?:-|_)to(?:-|_)(-?\d+)).*\.(?:avi|npz|pkl|NPZ|PKL|AVI)$',
        name)
    if pattern_parts is None:
        return None, None, None, None
    else:
        fish_name, event_number, frame_start, frame_end = pattern_parts.groups()
        return fish_name, int(event_number), int(frame_start), int(frame_end)


def get_mat_file_name_parts(name):
    pattern_parts = re.match(
        r'^(?=(\d+(?:-|_)f\d+)(?:-|_)(-?\d+)(?:-|_)frame(?:-|_)(-?\d+)(?:-|_)to(?:-|_)(-?\d+)).*\.(?:mat|MAT)$',
        name)
    pattern_combined = re.match(r'^(?=(\d+(?:-|_)f\d+)).*\.(?:mat|MAT)$', name)
    # Match: <date-digits>-f<digits>-<digits>_frame_<digits>_to_<digits><smt>.mat
    # Groups: 0-fish name (date + number), 1- event id, 2- frame start, 3-frame end
    if pattern_parts is None and pattern_combined is None:  # hard coded if the above didnt work
        print("Error. no match")
        event_name = name.split(".")[0]
        if 0 < len(event_name.split("_")):
            fish_name = event_name.split("_")[0]  # todo this wont work for all cases
            event_number = event_name.split("_")[1]
            if not event_number.isnumeric():
                event_number = -1
        if len(event_name.split("_")) >= 5:
            frame_start = int(event_name.split("_")[3])
            frame_end = int(event_name.split("_")[5])
        else:
            frame_start = 1
            frame_end = 1
    elif pattern_parts is not None:
        fish_name, event_number, frame_start, frame_end = pattern_parts.groups()
    elif pattern_combined is not None:
        fish_name, = pattern_combined.groups()
        event_number = -1
        frame_start = 1
        frame_end = 1
    return int(event_number), fish_name, int(frame_start), int(frame_end)


def parse_input_from_command(vid_type, allowed_vid_types, event_number=None, start_frame=None, end_frame=None,
                             scale_area=None, max_num_of_events=None, is_parallel=False):
    parser = argparse.ArgumentParser(description='Analyse fish data.')
    parser.add_argument('data_path', type=str, help='Full path to data folder (events)')
    parser.add_argument('fish_name', type=str, help='Fish folder name (inside events)')
    parser.add_argument('--vid_type', type=str,
                        help='Video type to work on (from: ' + str(allowed_vid_types) + ')')
    parser.add_argument('--full', default=False, action='store_true',
                        help='Run on full video (change only output dir name)')
    parser.add_argument('--parallel', default=is_parallel, action='store_true',
                        help='Is this run part of parallel jobs? (True=don\'t create final fish output)')
    parser.add_argument('--start_frame', type=int, help='Frame to start from (int)')
    parser.add_argument('--end_frame', type=int, help='Frame to end at (int)')
    parser.add_argument('--event_number', type=int, help='Number of the event to run (int), when --full not given')
    parser.add_argument('--max_num_of_events', type=int, help='How many events to run (int), when --full not given')
    parser.add_argument('--scale_area', type=int, help='Scale area to match zoom (int, for fish tracker usage)')
    parser.add_argument('--input_video_has_no_plate', default=False, action='store_true',
                        help='Should plate be removed?')  # default is True, therefore this flag is negative
    parser.add_argument('--fast_run', default=False, action='store_true',
                        help='Should run only fast analysis? (Default: false)')
    parser.add_argument('--visualize_movie', default=False, action='store_true',
                        help='Should visualize movie? (Default: false)')
    parser.add_argument('--control_data', default=False, action='store_true',
                        help='Should run over control events? (Default: false). Note: Won\'t work with --full')
    parser.add_argument('--fish_only', default=False, action='store_true',
                        help='Should enable only fish-tracker? (i.e. no background. Default: false). ')

    # print(parser.print_help())
    args = parser.parse_args(sys.argv[1:])

    if args.vid_type is not None and args.vid_type.lower() in allowed_vid_types:
        vid_type = args.vid_type.lower()

    # Read start or end. If both exist- validate the order make sense
    if args.start_frame is not None and args.end_frame is not None:
        if args.start_frame <= args.end_frame:
            start_frame, end_frame = args.start_frame, args.end_frame
    elif args.start_frame is not None:
        start_frame = args.start_frame
    elif args.end_frame is not None:
        end_frame = args.end_frame

    if args.scale_area is not None:
        scale_area = args.scale_area

    if not args.full:
        if args.max_num_of_events is not None:
            max_num_of_events = args.max_num_of_events
        if args.event_number is not None:
            event_number = args.event_number
    return args.data_path, args.fish_name, max_num_of_events, args.full, vid_type, start_frame, end_frame, event_number, \
        args.parallel, scale_area, not args.input_video_has_no_plate, args.fast_run, args.visualize_movie, \
        args.control_data, args.fish_only


class RunParameters:
    start_frame = None  # starts counting from 1, up to number of frames
    end_frame = None  # same
    event_number = None  # all
    scale_area = None  # zoom relative to normal camera
    fish_name = ""
    vid_names = []

    def __init__(self, is_full_video=False, is_parallel=False, input_video_has_plate=True, is_fast_run=False,
                 visualize_movie=False, is_control_data=False, is_fish_only=False):
        self.is_full_video = is_full_video
        self.is_parallel = is_parallel
        self.input_video_has_plate = input_video_has_plate
        self.is_fast_run = is_fast_run
        self.visualize_movie = visualize_movie
        self.is_control_data = is_control_data
        self.is_fish_only = is_fish_only

    def __str__(self):
        return ' full? {0}, parallel? {1}, plate? {2}, fast? {3}, visualize? {4}, control? {5}, fish-only? {6}, event {7}, Frames: {8}-{9}'.format(
            self.is_full_video, self.is_parallel, self.input_video_has_plate, self.is_fast_run, self.visualize_movie,
            self.is_control_data, self.is_fish_only, self.event_number, self.start_frame, self.end_frame)


def get_parameters(vid_type=".raw", allowed_vid_types=[".avi", ".raw"], is_parallel=False, is_vid_names=False):
    """From default variables or input from script (in future can read conf file)
    :return: input_folder, mat_output_folder, video_output_folder, data_path, vid_names
    """
    parameters = RunParameters(is_parallel=is_parallel)
    max_num_of_events = None
    if len(sys.argv) >= 2:  # argv[0] is script name
        data_path, fish_name, max_num_of_events, parameters.is_full_video, vid_type, \
            parameters.start_frame, parameters.end_frame, parameters.event_number, \
            parameters.is_parallel, parameters.scale_area, parameters.input_video_has_plate, parameters.is_fast_run,\
            parameters.visualize_movie, parameters.is_control_data, parameters.is_fish_only \
            = parse_input_from_command(vid_type, allowed_vid_types, start_frame=parameters.start_frame,
                                       end_frame=parameters.end_frame, event_number=parameters.event_number,
                                       scale_area=parameters.scale_area, max_num_of_events=max_num_of_events,
                                       is_parallel=is_parallel)
        print("(Given values) Working on fish ", fish_name, ' from ', data_path, ' vid ', vid_type, parameters)
    else:  # default - mount to shared data - windows path!
        data_path = "E:\\Lab-Shared\\Data\\FeedingAssay2020\\"
        fish_name = "20200720-f2"
        max_num_of_events = 2  # for tests only
        print("(HARD-CODED values) Working on fish ", fish_name, ' from ', data_path, ' vid ', vid_type, parameters)

    # folders matching 'cut' script
    if parameters.is_full_video:
        # todo this can be easily change to multiple fish
        input_folder = os.path.join(data_path, fish_name, "raw_whole_movie")
        mat_output_folder = os.path.join(data_path, fish_name, "processed_data_whole_movie_9.9", "parallel_parts")
        video_output_folder = os.path.join(data_path, fish_name, "debug_movies_whole_movie")
        video_output_folder_para = os.path.join(data_path, fish_name, "debug_paramecia_movies_full")
    else:
        #add_i, add_o = "", "_vel_seg_tail_fix"
        add_i, add_o = "", "_23.8"
        if parameters.is_control_data:
            add_i, add_o = "control_", "_control"
        if parameters.is_parallel:
            mat_output_folder = os.path.join(data_path, fish_name, "processed_data" + add_o, "parallel_parts")
        else:
            mat_output_folder = os.path.join(data_path, fish_name, "processed_data" + add_o)
        input_folder = os.path.join(data_path, fish_name, add_i + "events")  # todo this can be easily change to multiple fish
        video_output_folder = os.path.join(data_path, fish_name, "debug_movies" + add_o)
        video_output_folder_para = os.path.join(data_path, fish_name, "debug_paramecia_movies" + add_o)

    search_in_folder = input_folder
    names_only = True
    if vid_type not in allowed_vid_types:  # todo patch for using this function on non video inputs
        search_in_folder = video_output_folder
        names_only = False

    if is_vid_names:
        if fish_name == "*":
            vid_names = [f for f in glob.glob(os.path.join(search_in_folder, "*" + vid_type))]
        else:
            if names_only:
                vid_names = [f for f in os.listdir(search_in_folder) if f.lower().endswith(vid_type)]
            else:
                vid_names = [os.path.join(search_in_folder, f) for f in os.listdir(search_in_folder) if f.lower().endswith(vid_type)]

        if max_num_of_events is not None:
            vid_names = vid_names[0:max_num_of_events]
        parameters.vid_names = vid_names

    parameters.fish_name = fish_name
    return input_folder, mat_output_folder, video_output_folder, video_output_folder_para, data_path, parameters
