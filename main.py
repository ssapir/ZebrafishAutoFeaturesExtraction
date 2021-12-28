import numpy as np
import os
import numpy as np
import cv2
from tqdm import tqdm

# Home made
from classic_cv_trackers.abstract_and_common_trackers import ClassicCvAbstractTrackingAPI
from fish_preprocessed_data import FishPreprocessedData  # this is the output
from classic_cv_trackers import fish_tracking  # all used trackers

# utils
from utils import noise_cleaning, video_utils
from utils.main_utils import save_debug_avi, save_middle_mat_file, create_dirs_if_missing, get_info_from_event_name, \
    get_parameters, FishOutput, FishContoursAnnotationOutput, tracker_outputs_to_event, save_annotation_data, \
    RunParameters

# parameters, currently not changed between runs, therefore these are globals
ONE_SEC_N_FRAMES = 500  # 500 = 1s in 500fps.
MAX_NUM_OF_FRAMES_DEBUG_VID = 30 * ONE_SEC_N_FRAMES  # matching 100GB usage, and 50k movie cache
IS_CV2_VID_WRITE = True  # False - use ffmpeg
DISABLE_FRAMES_PROGRESS_BAR = False


"""
README PLEASE

This function is the main starting point of the regular analysis pipeline.
This code can run on any-size of fish raw/avi input file, and output the relevant mat files together with 'debug' videos

* The output is created by a class FishPreprocessedData 
* The trackers (=code analysing frames) inherits abstract class ClassicCvAbstractTrackingAPI (API + common impl).
* Input parameters are written in main_utils.get_parameters function

Please note.
1. While the main usage is for building a new feature, new analysis, etc, which usually runs on few fish and short video 
segments (locally0, it is also used for larger scale via scripts and parallel run (pipeline_scripts folder contains the 
extra needed code, except for cluster scripts outside this Repo). 
Hence, when you finish your work, please make sure the scripts there work as well.

2. Since we run on limited resources per frame, please try to make trackers as stateless as possible, and if not- make
sure the memory is limited by <5G per video (or as close to this as possible) to allow your code to run fast on all fish
(Fast means a coffee break, or if we must- a lunch break, for 1 event video (avg. 500 frames)).
"""


# ------------------------------------------- Loops logic ---------------------------------------------------

# --------- This is the main function/logic of this script (main point to change) - 1 video loop ------------
def analyse_one_video(dir_path, video, fish_tracker, para_tracker, fps, first_frame_n, n_frames, noise_frame,
                      video_output_folder, video_output_folder_para, mat_output_folder, filename, event_id, fish_name):
    """Receive an open video, noise frame and trackers, and returned list of analysed data (numpy types).
    This function also dump small annotated video files to output folder (for debug).

    :return: output per tracker.
    Please note that data used for debug is saved before returning the output
    """
    def dump_content_and_reset(frame_start):  # when the video is too large, output is reset as well
        save_debug_avi(video_output_folder, filename, fish_tracker.name, frame_start, frame_number,
                       video_frames, fps, IS_CV2_VID_WRITE=IS_CV2_VID_WRITE)
        video_frames.clear()  # reset memory - start accumulate frames for next video
        save_annotation_data(video_output_folder, filename, fish_tracker.name, frame_start, frame_number,
                             fish_contours_output, fish_output)
        frame_start = frame_number + 1  # next movie starts from here
        fish_contours_output.reset(frame_start=frame_start, n_frames=min(n_frames, frame_start + MAX_NUM_OF_FRAMES_DEBUG_VID))
        return frame_start

    def add_post_process_outputs():
        if post_process_outputs is not None:
            fish_output.is_bout_frame_list = post_process_outputs['is_bout_frame_list']
            fish_output.velocity_norms = post_process_outputs['velocity_norms']
            updated_tail_tip_status_list = \
                np.bitwise_and(fish_output.tail_tip_status_list,
                               post_process_outputs['is_tail_point_diff_norm_below_threshold'])
            fish_output.tail_tip_status_list = updated_tail_tip_status_list
            fish_output.fish_status_list = \
                np.bitwise_and(fish_output.fish_status_list,
                               post_process_outputs['is_head_origin_diff_norm_below_threshold'])
            fish_output.fish_status_list = \
                np.bitwise_and(fish_output.fish_status_list,
                               post_process_outputs['is_head_angle_diff_norm_below_threshold'])

    def append_outputs(relative_ind, ind):  # todo refactor to be dynamic
        if fish_analysis_output is not None and fish_analysis_output.is_ok:
            # lists are np.array of nX2 size for n points
            fish_output.origin_head_points_list[ind] = fish_analysis_output.fish_head_origin_point
            fish_output.destination_head_points_list[ind] = fish_analysis_output.fish_head_destination_point
            if fish_analysis_output.eyes_data is not None:
                fish_output.eyes_abs_angle_list[ind] = fish_analysis_output.eyes_data.abs_angle_deg
                fish_output.eyes_head_dir_diff_angle_list[ind] = fish_analysis_output.eyes_data.diff_from_fish_direction_deg
                fish_output.eyes_areas_pixels_list[ind] = fish_analysis_output.eyes_data.contour_areas
            fish_output.fish_status_list[ind] = True
            fish_output.is_head_prediction_list[ind] = fish_analysis_output.is_prediction
            if fish_analysis_output.tail_data is not None:
                fish_output.tail_tip_point_list[ind] = fish_analysis_output.tail_data.tail_tip_point
                fish_output.tail_path_list[ind] = fish_analysis_output.tail_data.tail_path
                fish_output.tail_tip_status_list[ind] = True
            elif ind > 0:
                fish_output.tail_tip_point_list[ind] = fish_output.tail_tip_point_list[ind - 1]  # copy prev
                fish_output.tail_path_list[ind] = fish_output.tail_path_list[ind - 1]  # copy prev
            # contours for merged-video creation - has relative index since these are dumped between runs
            fish_contours_output.fish_contour[relative_ind] = fish_analysis_output.fish_contour
            fish_contours_output.eyes_contour[relative_ind] = fish_analysis_output.eyes_contour
            if fish_analysis_output.eyes_data is not None:
                ellipses = fish_analysis_output.eyes_data.ellipses
                fish_contours_output.ellipse_centers[relative_ind] = [ellipse.ellipse_center for ellipse in ellipses]
                fish_contours_output.ellipse_angles[relative_ind] = [ellipse.ellipse_direction for ellipse in ellipses]
                fish_contours_output.ellipse_axes[relative_ind] = \
                    [(ellipse.ellipse_major, ellipse.ellipse_minor) for ellipse in ellipses]
            # todo: add skeleton?

        paramecium_output_list.append(para_output)

        if annotated_frame is not None:  # None in case of exception - shouldnt happen
            if IS_CV2_VID_WRITE:
                video_frames.append(annotated_frame)
            else:
                video_frames.append(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    frame_start = first_frame_n
    fish_output = FishOutput()  # this struct holds the information extracted from analysis (subset of all data)
    fish_output.reset(frame_start=frame_start, n_frames=n_frames)
    fish_contours_output = FishContoursAnnotationOutput()
    fish_contours_output.reset(frame_start=frame_start, n_frames=min(n_frames, frame_start + MAX_NUM_OF_FRAMES_DEBUG_VID))
    paramecium_output_list = []
    video_frames = []

    if para_tracker is not None:
        para_tracker.pre_process(dir_path, fish_name, event_id, noise_frame)  # todo should not be here if slow
    else:
        para_output = None

    for frame_number in tqdm(range(first_frame_n, n_frames + 1), disable=DISABLE_FRAMES_PROGRESS_BAR,
                             desc="current frame"):
        ok, frame = video.read()
        if not ok:
            print("Error- video stopped due to read (wrong n_frames)! (frame_num={0}, n_frames={1}, file={2}".format(
                frame_number, n_frames, filename
            ))
            break  # stop on last frame or on an error. todo how to know which one?

        annotated_frame, fish_analysis_output = fish_tracker.analyse(frame, noise_frame, fps, frame_number)
        if para_tracker is not None:
            _, para_output = para_tracker.analyse(frame, noise_frame, fps, frame_number, additional=[fish_analysis_output])

        append_outputs(frame_number - frame_start, frame_number - first_frame_n)

        if frame_number % MAX_NUM_OF_FRAMES_DEBUG_VID == 0:  # preserve memory- delete frames above this size
            frame_start = dump_content_and_reset(frame_start)

    post_process_outputs = fish_tracker.post_process(input_frames_list=None, analysis_data_outputs=fish_output)
    add_post_process_outputs()

    # remained unsaved frames (=for small input files), are saved here
    # (both debug video & annotations used to create future presentation videos)
    if len(video_frames) > 0:
        save_debug_avi(video_output_folder, filename, fish_tracker.name, frame_start, n_frames, video_frames, fps,
                       IS_CV2_VID_WRITE=IS_CV2_VID_WRITE)
    if len(fish_contours_output.fish_contour) > 0:
        save_annotation_data(video_output_folder, filename, fish_tracker.name, frame_start, n_frames,
                             fish_contours_output, fish_output)

    # todo this is mem issue for large movie
    if para_tracker is not None:
        para_annotated_frames_list, para_output = para_tracker.post_process(video_frames, paramecium_output_list)
        save_debug_avi(video_output_folder_para, filename, para_tracker.name, frame_start, n_frames,
                       para_annotated_frames_list, fps, IS_CV2_VID_WRITE=IS_CV2_VID_WRITE)

    # todo save para_annotated_frame
    # todo validate len?
    return fish_output, para_output


def main_loop(is_parallel, output_fps, start_frame, end_frame, input_folder, mat_output_folder, video_output_folder,
              video_output_folder_para, vidnames_dict, fish_tracker, para_tracker, visualize_movie=False):
    """Run over all given fish (based on vidnames_dict values), and creates all needed output.
    Note: this code can run either in parallel or serial manner (the parallel version combined the outputs later).

    :param is_parallel:
    :param output_fps: used when saving avi
    :param start_frame: for all movies
    :param end_frame: for all movies
    :param input_folder:
    :param mat_output_folder:
    :param video_output_folder:
    :param vidnames_dict: holds the data needed for this loop, video names as additional parameters
    :param fish_tracker: instance of tracker
    :param visualize_movie: True- will show the end result while calculating. Default should be false
    :return: None
    """
    for fish_name in tqdm(vidnames_dict.keys(), desc="current fish"):
        vidnames_list = vidnames_dict[fish_name]['videos']  # events
        noise_frame = vidnames_dict[fish_name]['noise']
        print("Fish ", fish_name, " videos ", vidnames_list)

        events = []
        for vid_data_dict in tqdm(vidnames_list, desc="current event (video)"):
            vidname = vid_data_dict['name']
            event_id = vid_data_dict['event_number']
            filename = vidname.split('.')[0]

            video, fps, ok, n_frames, first_frame_n = \
                video_utils.open(input_folder, vidname, start_frame=start_frame)

            if not ok:
                print("Not calculating bad video ", vidname)
                continue

            if end_frame is not None:
                n_frames = min(end_frame, n_frames)

            print("Start calculating ", vidname, " frames: {0}-{1}".format(first_frame_n, n_frames))
            fish_output, paramecium_output = \
                analyse_one_video(input_folder, video, fish_tracker, para_tracker, output_fps, first_frame_n, n_frames, noise_frame,
                                  video_output_folder, video_output_folder_para, mat_output_folder, filename, event_id,
                                  fish_name)

            video_utils.release(video, visualize_movie=visualize_movie)

            # Create event instance
            curr_event_data, ok = tracker_outputs_to_event(event_id, filename, fish_output, paramecium_output)
            events.append(curr_event_data)

            if is_parallel:  # create partial output with indication of frame numbers (to collect later)
                save_middle_mat_file(mat_output_folder, filename, first_frame_n, n_frames, event_id, fish_output,
                                     paramecium_output, fish_name)

            fish_tracker.print_time_statistics()
            print("Paramecium:")
            para_tracker.print_time_statistics()

        if not is_parallel:  # create and save combined fish (when parallel- external script does that)
            current_fish = FishPreprocessedData(fish_name, events)
            name = os.path.join(mat_output_folder, (fish_name + "_preprocessed.mat").lower())
            print("End. Saving fish...", name)
            current_fish.export_to_matlab(name)


# ------------------------------------------- Loops data -------------------------------------------


def build_events_data_dict(data_path, vid_names, requested_event_number=None, noise_folder="frames_for_noise"):
    """Build the data needed by main_loop function.
    Since main_loop saves mat file per fish, it calculate for each fish, its needed data

    data_path is path for input fish folders (FA2020).
    vid_names contains all requested files.

    :return: fish_videos_dict with fields needed by main loop
    """
    fish_videos_dict = {}

    # Get information from event names
    for name in vid_names:
        fish_name, event_number = get_info_from_event_name(name)
        if fish_name not in fish_videos_dict.keys():
            fish_videos_dict[fish_name] = {'videos': [], 'noise': None}  # These are the fields

        if requested_event_number is None or event_number == requested_event_number:
            fish_videos_dict[fish_name]['videos'].append({'name': name, 'event_number': int(event_number)})

    # sort based on event number
    for fish_name in fish_videos_dict.keys():
        fish_videos_dict[fish_name]['videos'].sort(key=lambda data_dict: data_dict['event_number'])

    # Add noise frame metadata to dict
    for fish_name in fish_videos_dict.keys():
        noise_frames_folder = os.path.join(data_path, fish_name, noise_folder)
        if os.path.exists(noise_frames_folder):
            fish_videos_dict[fish_name]['noise'] = noise_cleaning.static_noise_frame_from_full_event(
                noise_frames_folder)
        else:
            fish_videos_dict[fish_name]['noise'] = np.array([])

    return fish_videos_dict


def get_trackers(scale_area=None, input_video_has_plate=True, visualize_movie=False, is_fast_run=False,
                 is_fish_only=False, fish_tracker_class=fish_tracking.ContourBasedTracking):
    """Init all used trackers. This function should only pass meta parameters for the run

    :param is_fish_only:
    :param is_fast_run:
    :param input_video_has_plate: does input video contains plate (=that should be searched & removed)
    :param scale_area: parameter relates to the zoom level of current video (relative to lab's baseline)
    :param visualize_movie:
    :param fish_tracker_class:
    :return:
    """
    kwargs = {}
    if scale_area is not None:  # override only when needed
        kwargs['scale_area'] = scale_area
    return fish_tracker_class(visualize_movie=visualize_movie, input_video_has_plate=input_video_has_plate,
                              is_fast_run=is_fast_run, **kwargs), \
        ClassicCvAbstractTrackingAPI(visualize_movie=visualize_movie and not is_fish_only, is_fast_run=is_fast_run,
                          is_tracker_disabled=is_fish_only)


# ------------------------------------------- Main pipeline -------------------------------------------

# run me as: python main.py <data_path> <fish_folder_name>. Example: python main.py /ems/data/FA2020 20200720-f3
if __name__ == '__main__':
    parameters: RunParameters
    # Note. folders are for specific fish, but this code can easily run on the whole data-set
    input_folder, mat_output_folder, video_output_folder, video_output_folder_paramecia, data_path, parameters = \
        get_parameters(is_vid_names=True)

    create_dirs_if_missing([mat_output_folder, video_output_folder, video_output_folder_paramecia])

    output_fps = 30  # lab's default
    visualize_movie = parameters.visualize_movie
    fish_tracker, para_tracker = get_trackers(visualize_movie=visualize_movie, scale_area=parameters.scale_area,
                                              input_video_has_plate=parameters.input_video_has_plate,
                                              is_fast_run=parameters.is_fast_run, is_fish_only=parameters.is_fish_only)

    videos_dict = build_events_data_dict(data_path, parameters.vid_names, parameters.event_number)

    main_loop(parameters.is_parallel, output_fps, parameters.start_frame, parameters.end_frame, input_folder,
              mat_output_folder, video_output_folder, video_output_folder_paramecia, videos_dict,
              fish_tracker, para_tracker, visualize_movie=visualize_movie)
