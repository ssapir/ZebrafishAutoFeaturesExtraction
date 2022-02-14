import os
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import cv2

# hand made fun
from classic_cv_trackers import fish_tracking
from classic_cv_trackers import paramecium_tracking
from classic_cv_trackers.abstract_and_common_trackers import Colors
from classic_cv_trackers.fish_tracking import FrameAnalysisData
from feature_analysis.fish_environment.fish_processed_data import pixels_mm_converters
from fish_preprocessed_data import FishPreprocessedData
from utils import video_utils
from utils.main_utils import get_parameters, load_annotation_data, FishOutput, FishContoursAnnotationOutput, \
    parse_video_name, RunParameters

IS_CV2_VID_WRITE = True
DISABLE_FRAMES_PROGRESS_BAR = True


def metadata_annotate_single_frame(frame: np.ndarray, fish_data: FishPreprocessedData, event_number, frame_number,
                                   is_hunt_list=[], visualize_video_output=False,
                                   column_left_side=10, row_left_side=15, space_between_rows=25, fontsize=6,
                                   text_color=Colors.GREEN, text_font=cv2.FONT_HERSHEY_SIMPLEX, bold=2):
    event = [ev for ev in fish_data.events if ev.event_id == event_number]
    if len(event) != 1:
        print("Error. For fish {0} found {1} events for event-num {2}".format(fish_data.metadata.name,
                                                                              len(event), event_number))
        return frame
    event = event[0]

    # calculate size of current font to allow readable api for font scaling calculation
    text_size = cv2.getTextSize(text="check", fontFace=text_font, fontScale=1, thickness=bold)[1]

    font_size = lambda size: size / text_size

    # todo this is patch to work with old movies: override previous text and add all labels from scratch
    # cv2.rectangle(frame, (column_left_side, 0), (column_left_side + 120, row_left_side + 110), Colors.BLACK, cv2.FILLED)
    #
    # cv2.putText(frame, "Paramecia load: {0}".format(fish_data.metadata.num_of_paramecia_in_plate),
    #             (column_left_side, row_left_side), text_font, font_size(fontsize), Colors.WHITE, thickness=bold)
    #
    # row_left_side += space_between_rows
    #
    # complex_str = ""
    # if event.is_complex_hunt:
    #     complex_str = " (complex)"
    # cv2.putText(frame, "Outcome: {0}".format(event.outcome_str + complex_str), (column_left_side, row_left_side),
    #             text_font, font_size(fontsize), Colors.WHITE, thickness=bold)
    #
    # row_left_side += space_between_rows
    # cv2.putText(frame, "Age: {0}".format(fish_data.metadata.age_dpf), (column_left_side, row_left_side), text_font,
    #             font_size(fontsize), Colors.WHITE, thickness=bold)
    #
    # row_left_side += space_between_rows

    cv2.putText(frame, "" + str(int(1000 * frame_number / 500)) +"ms", (column_left_side, row_left_side), text_font, font_size(fontsize),
                Colors.WHITE, thickness=bold)

    # add hunt. Override fish_tracker hunt
    row_left_side += space_between_rows
    result = "Hunt"
    color = Colors.WHITE
    if len(is_hunt_list) > 0 and not is_hunt_list[frame_number - 1]:
        result = "No-hunt"
        color = Colors.GREEN
    cv2.putText(frame, result, (column_left_side, row_left_side), text_font, font_size(fontsize), color, thickness=bold)
    row_left_side += space_between_rows

    # Add scale bar
    # shape is 900x896. Plate's diameter is 20mm. The plate's distance from the edge is ~33 pixels
    one_mm_in_pixels, one_pixel_in_mm = pixels_mm_converters()
    n_mm = 2

    # start_point = (column_left_side, int(frame.shape[1] - 10))
    # end_point = (start_point[0] + one_mm_in_pixels * n_mm, start_point[1])
    # cv2.line(frame, start_point, end_point, Colors.WHITE, thickness=2)
    # cv2.putText(frame, "{0}mm".format(n_mm), (start_point[0], start_point[1] - 10), text_font,
    #             fontScale=font_size(16), color=Colors.WHITE)
    # import matplotlib.pyplot as plt
    # plt.add_artist(ScaleBar(one_pixel_in_mm, "mm", location='lower left', color='w', box_alpha=0,
    #                        font_properties={"size": 32}, fixed_value=n_mm))

    if visualize_video_output:
        cv2.imshow('result', resize(frame))
        cv2.waitKey(60)

    return frame


def annotate_single_frame(frame: np.ndarray, fish_output: FishOutput,
                          fish_contours_output: FishContoursAnnotationOutput, event_number: int, frame_number: int,
                          start_frame: int, fish_mat: FishPreprocessedData, is_hunt_list,
                          visualize_video_output=False,
                          annotate_fish=True, annotate_metadata=True, annotate_paramecia=True, is_adding_eyes_text=True,
                          # change output
                          column_left_side=10, row_left_side=90, space_bet_text_rows=25, fontsize=6, col_right_side=55,
                          text_color=Colors.GREEN, text_font=cv2.FONT_HERSHEY_SIMPLEX, bold=2):

    an_frame = frame.copy()

    # calculate size of current font to allow readable api for font scaling calculation
    text_size = cv2.getTextSize(text="check", fontFace=text_font, fontScale=1, thickness=1)[1]
    font_size = lambda size: size / text_size

    # cv2.putText(an_frame, "# " + str(int(frame_number)), (column_left_side, row_left_side), text_font, font_size(fontsize),
    #             Colors.GREEN, thickness=bold)
    # row_left_side += space_bet_text_rows

    # Draw similar to fish tracking - identical code
    if annotate_fish and fish_output.fish_status_list[frame_number - start_frame]:
        output = build_struct_for_fish_annotation(fish_contours_output, fish_output, frame_number, start_frame)
        fish_tracking.ContourBasedTracking.draw_output_on_annotated_frame(an_frame,
                                                                          fontsize=font_size(fontsize),
                                                                          text_color=text_color,
                                                                          output=output,
                                                                          is_bout=output.tail_data.is_bout,
                                                                          velocity_norms=output.tail_data.velocity_norms,
                                                                          row_left_side_text=row_left_side,
                                                                          row_right_side_text=row_left_side,
                                                                          col_left_side_text=column_left_side,
                                                                          col_right_side_text=col_right_side,
                                                                          space_between_text_rows=space_bet_text_rows,
                                                                          is_adding_eyes_text=is_adding_eyes_text)
    if annotate_paramecia:
        output = build_struct_for_paramecia_annotation(fish_mat, frame_number, start_frame, event_number)
        paramecium_tracking.ParameciumTracker.draw_output_on_annotated_frame(an_frame, output=output)
    if annotate_metadata:
        an_frame = metadata_annotate_single_frame(an_frame, fish_data=fish_mat,
                                                  is_hunt_list=is_hunt_list, fontsize=fontsize, text_color=text_color,
                                                  event_number=event_number, frame_number=frame_number,
                                                  column_left_side=column_left_side, row_left_side=row_left_side,
                                                  space_between_rows=space_bet_text_rows,
                                                  text_font=text_font, bold=bold)

    if visualize_video_output:
        cv2.imshow('result', resize(an_frame))
        cv2.waitKey(60)

    return an_frame


def build_struct_for_fish_annotation(fish_contours_output, fish_output, frame_number, start_frame):  # todo refactor
    # set parameters for annotation (identical to fish output to allow future reuse of draw function)
    output = FrameAnalysisData()
    # output = namedtuple('fish', ['fish_contour', 'eyes_contour', 'fish_head_origin_point',
    #                              'fish_head_destination_point', 'eyes_data', 'tail_data',])
    output.fish_contour = fish_contours_output.fish_contour[frame_number - start_frame]
    output.eyes_contour = fish_contours_output.eyes_contour[frame_number - start_frame]
    output.fish_head_origin_point = fish_output.origin_head_points_list[frame_number - start_frame]
    output.fish_head_destination_point = fish_output.destination_head_points_list[frame_number - start_frame]
    output.is_ok = fish_output.fish_status_list[frame_number - start_frame]
    output.is_prediction = fish_output.is_head_prediction_list[frame_number - start_frame]

    output.tail_data = namedtuple('tail', ['tail_tip_point', 'tail_path', 'is_bout', 'velocity_norms'])
    output.tail_data.tail_tip_point = fish_output.tail_tip_point_list[frame_number - start_frame]
    output.tail_data.tail_path = fish_output.tail_path_list[frame_number - start_frame]
    output.tail_data.is_bout = fish_output.is_bout_frame_list[frame_number - start_frame]
    output.tail_data.velocity_norms = fish_output.velocity_norms[frame_number - start_frame]

    if not np.isnan(fish_output.eyes_head_dir_diff_angle_list[frame_number - start_frame]).any():
        output.eyes_data = namedtuple('eyes', ['eyes_contour', 'diff_from_fish_direction_deg', 'abs_angle_deg'])
        output.eyes_data.diff_from_fish_direction_deg = fish_output.eyes_head_dir_diff_angle_list[frame_number - start_frame]
        output.eyes_data.abs_angle_deg = fish_output.eyes_abs_angle_list[frame_number - start_frame]
        output.eyes_data.contour_areas = fish_output.eyes_areas_pixels_list[frame_number - start_frame]

        ellipse = namedtuple('ellipse', ['ellipse_center', 'ellipse_major', 'ellipse_minor', 'ellipse_direction'])
        output.eyes_data.ellipses = []
        for i in range(len(fish_contours_output.ellipse_angles[frame_number - start_frame])):
            el = ellipse(ellipse_center=fish_contours_output.ellipse_centers[frame_number - start_frame][i],
                         ellipse_major=fish_contours_output.ellipse_axes[frame_number - start_frame][i][0],
                         ellipse_minor=fish_contours_output.ellipse_axes[frame_number - start_frame][i][1],
                         ellipse_direction=fish_contours_output.ellipse_angles[
                             frame_number - start_frame][i],
                         )
            output.eyes_data.ellipses.append(el)
    return output


def build_struct_for_paramecia_annotation(fish_data: FishPreprocessedData, frame_number: int, start_frame: int,
                                          event_number: int):
    output = paramecium_tracking.ParameciumOutput()
    event = [ev for ev in fish_data.events if ev.event_id == event_number]
    if len(event) != 1:
        print("Error. For fish {0} found {1} events for event-num {2}".format(fish_data.metadata.name, len(event), event_number))
        return output
    paramecium = event[0].paramecium

    output.is_ok = True
    output.color = paramecium.color_points
    output.area = paramecium.area_points[frame_number - start_frame, :]
    output.center = paramecium.center_points[frame_number - start_frame, :, :]
    output.status = paramecium.status_points[frame_number - start_frame, :]
    output.ellipse_dirs = paramecium.ellipse_dirs[frame_number - start_frame]
    output.ellipse_minors = paramecium.ellipse_minors[frame_number - start_frame]
    output.ellipse_majors = paramecium.ellipse_majors[frame_number - start_frame]
    output.bbox = paramecium.bounding_boxes[frame_number - start_frame]

    return output


def resize(f):
    return cv2.resize(f, (round(f.shape[0] / 1.5), round(f.shape[1] / 1.5)))


def add_annotation_to_raw_movies(input_folder, data_folder, video_data):
    for fish_name in tqdm(video_data.keys(), desc="current fish"):
        fish = video_data[fish_name]['fish']
        video_list = video_data[fish_name]['videos']
        for data_name, raw_movie in tqdm(video_list, desc="current event"):
            print("Annotate contours on movie for fish ", data_name)
            fish_name, event_number, frame_start, frame_end = parse_video_name(data_name)
            video, fps, ok, n_frames, first_frame_n = video_utils.open(input_folder,
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
            for frame_number in tqdm(range(frame_start, frame_end + 1), disable=DISABLE_FRAMES_PROGRESS_BAR,
                                     desc="current frame"):
                ok, frame = video.read()
                if not ok:
                    print("Error- video stopped due to read! (frame_num={0}, file={1}".format(frame_number, raw_movie))
                    break  # stop on last frame or on an error.
                if fish_output is not None:
                    video_frames.append(annotate_single_frame(frame, fish_output, fish_contours_output,
                                                              event_number=event_number, start_frame=frame_start,
                                                              frame_number=frame_number, fish_mat=fish,
                                                              is_hunt_list=is_hunt))
                else:
                    video_frames.append(frame)

            video.release()
            save_presentation_movie(event_number, fish_name, fps, frame_end, frame_start, video_frames)
            video_frames.clear()


def save_presentation_movie(event_number, fish_name, fps, frame_end, frame_start, video_frames):
    output_name = os.path.join(video_output_folder,
                               "{0}-{1}_presentation_frame_{2}_to_{3}.avi".format(
                                   fish_name, event_number, frame_start, frame_end).lower())
    print("End. Saving video output...", output_name)
    if IS_CV2_VID_WRITE:
        video_utils.create_video_with_cv2_exe(output_name, video_frames, fps)
    else:
        video_utils.create_video_with_local_ffmpeg_exe(output_name, video_frames, fps)


# run me as: python main_annotate_presentation_movie.py <data_path> <fish_folder_name>. Add --full for the original raw
# Example: python main_annotate_presentation_movie.py \\ems\Lab-Shared\Data\FeedingAssay2020 20200722-f1
# Example2: python main_annotate_presentation_movie.py \\ems\Lab-Shared\Data\FeedingAssay2020 20200722-f1 --full
if __name__ == '__main__':
    params: RunParameters
    input_folder, mat_inputs_folder, video_output_folder, _, data_path, params = get_parameters()
    # change locally folder name within get_parameters ti target different folder

    if not os.path.exists(video_output_folder):
        raise Exception("Error: ", video_output_folder, " not found (Error in usage!).")

    video_data = {}
    for data_name in [f for f in os.listdir(video_output_folder) if f.lower().endswith(".npz")]:
        fish_name, event_number, frame_start, frame_end = parse_video_name(data_name)
        if params.event_number is not None and params.event_number != event_number:
            continue
        raw_movie = ("{0}-{1}.raw".format(fish_name, event_number)).lower()
        avi_movie = ("{0}-{1}.avi".format(fish_name, event_number)).lower()
        mat_file = (fish_name + "_preprocessed.mat").lower()

        raw_file_exists = os.path.exists(os.path.join(input_folder, raw_movie))
        avi_file_exists = os.path.exists(os.path.join(input_folder, avi_movie))
        if raw_file_exists:
            movie_file_name = raw_movie
        elif avi_file_exists:
            movie_file_name = avi_movie
        else:
            print("Error. Missing raw/avi movie for fish ", avi_movie)
            continue
    
        if not os.path.exists(os.path.join(mat_inputs_folder, mat_file)):
            print("Error. Missing mat for fish ", mat_file)
        else:
            if fish_name not in video_data.keys():
                fish = FishPreprocessedData.import_from_matlab(os.path.join(mat_inputs_folder, mat_file))
                if fish.metadata.age_dpf == -1:
                    print("Note: ", mat_file, " has no metadata attached.")
                video_data[fish_name] = {'fish': fish, 'videos': []}
            video_data[fish_name]['videos'].append((data_name, movie_file_name))
    add_annotation_to_raw_movies(input_folder=input_folder, data_folder=video_output_folder,
                                 video_data=video_data)
