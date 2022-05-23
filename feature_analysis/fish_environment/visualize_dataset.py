import math
import os
import traceback

from tqdm import tqdm
import numpy as np
import cv2

from classic_cv_trackers import fish_tracking
from classic_cv_trackers.abstract_and_common_trackers import ClassicCvAbstractTrackingAPI as cls, Colors
from feature_analysis.fish_environment.fish_processed_data import FishAndEnvDataset, SingleFishAndEnvData, \
    ParameciumRelativeToFish, ExpandedEvent, is_paramecia_in_fov, get_target_paramecia_index, pixels_mm_converters, \
    DISTANCE_LIST_IN_MM, ANGLES
from feature_analysis.fish_environment.env_utils import heatmap, resize, rotate_image, \
    PlotsCMDParameters
from scripts.python_scripts.main_annotate_presentation_movie import build_struct_for_fish_annotation, annotate_single_frame
from utils import video_utils
from utils.geometric_functions import fix_angle_range
from utils.main_utils import get_parameters, load_annotation_data, FishOutput, FishContoursAnnotationOutput, \
    create_dirs_if_missing, RunParameters, parse_video_name

IS_CV2_VID_WRITE = True
DISABLE_FRAMES_PROGRESS_BAR = False


def annotate_single_frame_exp(frame: np.ndarray, fish_output: FishOutput,
                              fish_contours_output: FishContoursAnnotationOutput, event_number: int, frame_number: int,
                              start_frame: int, fish_mat: SingleFishAndEnvData, event: ExpandedEvent, is_hunt_list,
                              target_para_ind: int = None, distances_to_fov=[1, 2, 3, 4, 5], DIST=DISTANCE_LIST_IN_MM,
                              names=['narrow', 'forward', 'front', 'ang_270', 'tail'],
                              name_colors=[Colors.WHITE, Colors.CYAN, Colors.GREEN, Colors.PINK, Colors.YELLOW],
                              visualize_video_output=False,
                              annotate_fish=True, annotate_metadata=True, annotate_paramecia=True, show_heatmap_nearby=False,
                              # change output
                              column_left_side=10, row_left_side=15, space_bet_text_rows=25, col_right_side=55,
                              fontsize=6, text_color=Colors.GREEN, text_font=cv2.FONT_HERSHEY_SIMPLEX, bold=2):
    parameters = PlotsCMDParameters()  # todo move outside

    an_frame = frame.copy()

    text_size = cv2.getTextSize(text="check", fontFace=text_font, fontScale=1, thickness=1)[1]
    font_size = lambda size: size / text_size

    one_mm_in_pixels, _ = pixels_mm_converters()

    frame_number -= start_frame
    if event.head.origin_points.x.shape[0] <= frame_number:
        print("Error. event length {0}, vs frame_number {1}".format(event.head.origin_points.x.shape[0], frame_number))
        return an_frame
    paramecium_exp: ParameciumRelativeToFish = event.paramecium
    origin = [event.head.origin_points.x[frame_number], event.head.origin_points.y[frame_number]]
    head_direction_angle = event.head.directions_in_deg[frame_number]

    for i, (name, (from_angle, to_angle)) in enumerate(ANGLES.items()):
        if name not in names:
            continue
        for a in [1, -1]:
            if from_angle is None or np.isnan(from_angle):
                angle = a * to_angle / 2 - head_direction_angle
            else:
                angle = a * (to_angle - from_angle) + a * from_angle / 2 - head_direction_angle
            for j, curr_distance in enumerate(distances_to_fov):
                endy = origin[1] + curr_distance * one_mm_in_pixels * math.sin(math.radians(angle))
                endx = origin[0] + curr_distance * one_mm_in_pixels * math.cos(math.radians(angle))
                cv2.putText(an_frame, "{0}".format(curr_distance),
                            cls.point_to_int([endx + 1, endy + 1]), text_font, font_size(4), Colors.RED, 1)
            length = (max(distances_to_fov) + 0.5) * one_mm_in_pixels
            endy = origin[1] + length * math.sin(math.radians(angle))
            endx = origin[0] + length * math.cos(math.radians(angle))
            to_p =[endx, endy]
            cv2.line(an_frame, cls.point_to_int(origin), cls.point_to_int(to_p), Colors.RED, 1)
            cv2.putText(an_frame, "{0}".format(name[:4]),
                        cls.point_to_int([to_p[0] + 2 * a, to_p[1] + 2 * a]), text_font, font_size(3),
                        name_colors[names.index(name)], 1)

    _, n_paramecia = paramecium_exp.status_points.shape
    for para_ind in range(n_paramecia):
        center = paramecium_exp.center_points[frame_number, para_ind, :]
        if np.isnan(center).all() or center is None:
            continue

        if target_para_ind is not None and para_ind == target_para_ind:
            size_ = 8
            cv2.rectangle(an_frame, cls.point_to_int([center[0] - size_, center[1] - size_]),
                          cls.point_to_int([center[0] + size_, center[1] + size_]), Colors.WHITE, 2)

        for i, (name, (from_angle, to_angle)) in enumerate(ANGLES.items()):
            if name not in names:
                continue
            curr_distance = max(distances_to_fov)
            if is_paramecia_in_fov(paramecium_exp, frame_number, para_ind, curr_distance, from_angle, to_angle):
                size_ = 12 + names.index(name)
                cv2.rectangle(an_frame, cls.point_to_int([center[0] - size_, center[1] - size_]),
                              cls.point_to_int([center[0] + size_, center[1] + size_]), name_colors[names.index(name)], 1)

        # if len(paramecium_exp.edge_points.shape) > 2:
        #     edges = paramecium_exp.edge_points[frame_number, para_ind, :, :]
        #     for i in [0, 2]:
        #         from_p, to_p = edges[i:i+2, :]
        #         if not np.isnan([from_p, to_p]).all():
        #             pass
                    # cv2.line(an_frame, cls.point_to_int(from_p), cls.point_to_int(center), color, 3)
                    # cv2.circle(an_frame, cls.point_to_int(from_p), color=color, radius=2, thickness=cv2.FILLED)
                # cv2.circle(an_frame, cls.point_to_int(to_p), color=color, radius=2, thickness=cv2.FILLED)

        # cv2.putText(an_frame, "  d {0:.2f}".format(paramecium_exp.diff_from_fish_angle_deg[frame_number, para_ind]),
        #             cls.point_to_int([center[0], center[1]]), text_font, font_size(fontsize), Colors.GREEN)
        # cv2.putText(an_frame, "  d {0:.2f}".format(paramecium_exp.diff_from_fish_angle_deg[frame_number, para_ind]),
        #             cls.point_to_int([center[0], center[1]]), text_font, font_size(fontsize), Colors.GREEN)
        # cv2.putText(an_frame, "  f {0:.2f}".format(paramecium_exp.field_angle[frame_number, para_ind]),
        #             cls.point_to_int([center[0], center[1] + 12]), text_font, font_size(fontsize), Colors.PURPLE)
        # if not np.isnan(origin).all() and origin is not None:
        #     cv2.line(an_frame, cls.point_to_int(origin), cls.point_to_int(center), color)
        #     where = np.mean([origin, center], axis=0)
        #     cv2.putText(an_frame, "{0:.1f}".format(paramecium_exp.distance_from_fish_in_mm[frame_number, para_ind]),
        #                 cls.point_to_int(where), text_font, font_size(fontsize), Colors.RED, 1)

    # add stuff for paramecia
    an_frame = annotate_single_frame(
        an_frame, fish_output, fish_contours_output, event_number, frame_number, start_frame, fish_mat, is_hunt_list,
        visualize_video_output=False, annotate_fish=annotate_fish, fontsize=fontsize, space_bet_text_rows=space_bet_text_rows,
        annotate_metadata=annotate_metadata, annotate_paramecia=annotate_paramecia, is_adding_eyes_text=False,
        text_color=text_color, column_left_side=column_left_side, row_left_side=row_left_side, col_right_side=col_right_side)  # change row left side to hide text (which is overridden here)
    if not annotate_metadata:
        cv2.rectangle(an_frame, (column_left_side, 0), (column_left_side + 120, row_left_side + 110), Colors.BLACK,
                      cv2.FILLED)

    ibis, indices = ExpandedEvent.inter_bout_interval_range(event)

    col_right_side_text = 680
    row_right_side_text = 20

    cv2.putText(an_frame, event.outcome_str, (col_right_side_text, row_right_side_text), text_font,
                font_size(5), Colors.WHITE, thickness=1)
    row_right_side_text += space_bet_text_rows
    for i, (name, (from_angle, to_angle)) in enumerate(ANGLES.items()):
        if name not in names:
            continue
        q = np.sum(event.paramecium.field_of_view_status[frame_number, :, i, np.where(np.array(DIST) == max(distances_to_fov))[0]])
        cv2.putText(an_frame, "N {0}: {1}".format(name, q), (col_right_side_text, row_right_side_text), text_font, font_size(4),
                    Colors.WHITE, thickness=1)
        row_right_side_text += space_bet_text_rows

    if frame_number in ibis:
        ibi_index = [True if a.start <= frame_number <= a.stop else False for a in indices].index(True)
        cv2.putText(an_frame, "Inter bout interval ({0} / {1})".format(ibi_index + 1, len(indices)),
                    (column_left_side, row_left_side + 6*space_bet_text_rows), text_font, font_size(fontsize),
                    Colors.PINK, thickness=bold)
        if target_para_ind is not None:
            vt, vo = None, None
            if len(paramecium_exp.velocity_towards_fish.shape) == 1 and ibi_index == 0:
                vt = paramecium_exp.velocity_towards_fish[target_para_ind]
                vo = paramecium_exp.velocity_orthogonal[target_para_ind]
            elif len(paramecium_exp.velocity_towards_fish.shape) == 2:
                vt = paramecium_exp.velocity_towards_fish[ibi_index, target_para_ind]
                vo = paramecium_exp.velocity_orthogonal[ibi_index, target_para_ind]
            if vt is not None and vo is not None:
                cv2.putText(an_frame, "Vt tar: {0:.2f}".format(vt), (col_right_side_text, row_right_side_text), text_font,
                            font_size(4), Colors.WHITE, thickness=bold)
                row_right_side_text += space_bet_text_rows
                cv2.putText(an_frame, "Vo tar: {0:.2f}".format(vo), (col_right_side_text, row_right_side_text), text_font,
                            font_size(4), Colors.WHITE, thickness=bold)
                row_right_side_text += space_bet_text_rows

    visualize_fish, _, _ = heatmap(event, frame_number, target_paramecia_center=None, parameters=parameters)
    if show_heatmap_nearby:
        result = np.hstack([resize(an_frame), resize(cv2.cvtColor(visualize_fish.astype(np.uint8), cv2.COLOR_GRAY2RGB))])
    else:
        result= an_frame
    if visualize_video_output:
        cv2.imshow('result', result)
        cv2.waitKey(60)

    return result


def add_annotation_to_raw_movies(video_data):
    for fish_name in tqdm(video_data.keys(), desc="current fish"):
        fish = video_data[fish_name]['fish']
        video_list = video_data[fish_name]['videos']
        for data_name, raw_movie, input_raw_folder, data_folder in tqdm(video_list, desc="current event"):
            print("Annotate contours on movie for fish ", data_name, input_raw_folder, raw_movie, data_folder)
            fish_name, event_number, frame_start, frame_end = parse_video_name(data_name)
            video, fps, ok, n_frames, first_frame_n = video_utils.open(input_raw_folder,
                                                                       raw_movie, start_frame=frame_start)

            if not ok:
                print("Error- video not opened (file={0})".format(raw_movie))
                break  # stop on last frame or on an error.

            video_frames = []
            fish_output, fish_contours_output = load_annotation_data(os.path.join(data_folder, data_name))
            # Smooth hunt detection
            outputs = [build_struct_for_fish_annotation(fish_contours_output, fish_output, frame_number, frame_start)
                       for frame_number in range(frame_start, frame_end + 1)]
            is_hunt = np.array([fish_tracking.ContourBasedTracking.is_hunting(25, output) for output in outputs])
            is_hunt = np.convolve(is_hunt, np.ones((51,))) >= 10

            events = [ev for ev in fish.events if ev.event_id == event_number]
            if len(events) != 1:
                print("Error. For fish {0} found {1} events for event-num {2}".format(fish.name, len(events),
                                                                                      event_number))
                continue
            event: ExpandedEvent = events[0]
            paramecium: ParameciumRelativeToFish = event.paramecium

            target_para_ind = paramecium.target_paramecia_index

            for frame_number in tqdm(range(frame_start, frame_end + 1), disable=DISABLE_FRAMES_PROGRESS_BAR,
                                     desc="current frame"):
                ok, frame = video.read()
                if not ok:
                    print("Error- video stopped due to read! (frame_num={0}, file={1}".format(frame_number, raw_movie))
                    break  # stop on last frame or on an error.
                try:
                    if fish_output is not None:
                        video_frames.append(annotate_single_frame_exp(frame, fish_output, fish_contours_output,
                                                                      event_number=event_number, start_frame=frame_start,
                                                                      frame_number=frame_number, fish_mat=fish,
                                                                      event=event, is_hunt_list=is_hunt,
                                                                      DIST=fish.distances_for_fov_in_mm,
                                                                      target_para_ind=target_para_ind))
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    video_frames.append(frame)
                # else:
                #     video_frames.append(frame)

            video.release()
            save_presentation_movie(video_output_folder, event_number, fish_name, 25, frame_end, frame_start, video_frames)
            video_frames.clear()


def save_presentation_movie(video_output_folder, event_number, fish_name, fps, frame_end, frame_start, video_frames):
    output_name = os.path.join(video_output_folder,
                               "{0}-{1}_presentation_frame_{2}_to_{3}.avi".format(
                                   fish_name, event_number, frame_start, frame_end).lower())
    print("End. Saving video output...", output_name)
    if IS_CV2_VID_WRITE:
        video_utils.create_video_with_cv2_exe(output_name, video_frames[1:], fps)
    else:
        video_utils.create_video_with_local_ffmpeg_exe(output_name, video_frames[1:], fps)


if __name__ == '__main__':
    vid_type = ".npz"
    parameters: RunParameters
    input_folder, _, _, _, data_path, parameters = get_parameters(vid_type=vid_type, is_vid_names=True)

    fullfile = "" #os.path.join(data_path, "data_set_features", "fish_env_dataset.mat")
    d2 = data_path.replace("Data\FeedingAssay2020", "Analysis\FeedingAssaySapir").replace("Data/FeedingAssay2020", "Analysis/FeedingAssaySapir")

    vid_names = parameters.vid_names
    if parameters.event_number is not None:
        vid_names = [f for f in vid_names if "-{0}_preproc".format(parameters.event_number) in f]
    print("Visualizing ", vid_names)

    if parameters.fish_name == '*':
        if not os.path.exists(fullfile):
            raise Exception("Missing file ", fullfile)
        dataset = FishAndEnvDataset.import_from_matlab(fullfile)
    else:
        #processed_path = os.path.join(os.path.join(d2, "dataset_features-checked_fish", "data_set_features"), "all_fish")
        processed_path = os.path.join(os.path.join(d2, "data_set_features", "test_target"), "all_fish")
        output_path = os.path.join(processed_path, parameters.fish_name + "_env_processed.mat")
        # 20200720 - f2_env_processed_small_hunt_abort
        dataset = FishAndEnvDataset([SingleFishAndEnvData.import_from_matlab(output_path)])
    print("Loaded ", vid_names)

    #video_output_folder = os.path.join(data_path, "dataset_debug_movies")
    video_output_folder = os.path.join(d2, "dataset_debug_movies_test_target")
    create_dirs_if_missing([video_output_folder])

    video_data = {}
    print(parameters.vid_names)
    for full_name in [f for f in vid_names if f.lower().endswith(vid_type)]:
        data_name = os.path.basename(full_name)
        data_folder = os.path.dirname(full_name)
        fish_name, event_number, frame_start, frame_end = parse_video_name(data_name)

        raw_movie = ("{0}-{1}.raw".format(fish_name, event_number)).lower()
        avi_movie = ("{0}-{1}.avi".format(fish_name, event_number)).lower()
        if os.path.exists(os.path.join(input_folder.replace("*", fish_name), raw_movie)):
            movie_file_name = raw_movie
        elif os.path.exists(os.path.join(input_folder.replace("*", fish_name), avi_movie)):
            movie_file_name = avi_movie
        else:
            print("Error. Missing raw/avi movie for fish ", avi_movie)
            continue

        if fish_name not in video_data.keys():
            fish: SingleFishAndEnvData = [curr for curr in dataset.fish_processed_data_set if curr.name == fish_name]
            if len(fish) == 1:
                video_data[fish_name] = {'fish': fish[0], 'videos': []}
            else:
                print("Error. lacking fish named ", fish_name)
        video_data[fish_name]['videos'].append((data_name, movie_file_name,
                                                input_folder.replace("*", fish_name), data_folder))
    add_annotation_to_raw_movies(video_data=video_data)
