import os
import traceback

import numpy as np
from tqdm import tqdm

from fish_preprocessed_data import FishPreprocessedData, Event
from classic_cv_trackers.paramecium_tracking import ParameciumOutput
from utils.main_utils import tracker_outputs_to_event, FishOutput, get_parameters, get_mat_file_name_parts

def main(input_folder, output_folder, mat_files):
    files = {}
    print("Loading saved data")
    for name in tqdm(mat_files):
        event_number, fish_name, frame_start, frame_end = get_mat_file_name_parts(name)
        print(fish_name, ": ", frame_start, "-", frame_end)
        fish = FishPreprocessedData.import_from_matlab(os.path.join(input_folder, name))
        if len(fish.events) > 1:
            print("Error. Parallel data has multiple events within a single event file ", len(fish.events))
            continue
        if fish.events[0].event_id != event_number:
            print("Error. Parallel data has a different event-id than expected: ", fish.events[0].event_id, event_number)
            continue

        if fish_name not in files.keys():
            files[fish_name] = {}
        if event_number not in files[fish_name].keys():
            files[fish_name][event_number] = []
        files[fish_name][event_number].append({'filename': fish.events[0].event_name,
                                               'start': frame_start, 'end': frame_end,
                                               'fish': fish})

    print("Combining data")
    for fish_name in tqdm(files.keys(), desc="current fish"):
        events = []
        for event_id in tqdm(files[fish_name].keys(), desc="current event"):
            fish_output, paramecium_output = combine_output(files[fish_name][event_id])
            filename = files[fish_name][event_id][0]['filename']
            curr_event_data, ok = tracker_outputs_to_event(event_id, filename, fish_output, paramecium_output)
            events.append(curr_event_data)

        events.sort(key=lambda e: e.event_id)
        current_fish = FishPreprocessedData(fish_name, events)
        name = os.path.join(output_folder, (fish_name + "_preprocessed.mat").lower())
        print("End. Saving fish...", name)
        current_fish.export_to_matlab(name)


def expand_shape_if_needed(value, value_to_append):
    if len(value.shape) > len(value_to_append.shape) and \
            value.shape[-1] == 1:  # match dimension of a list (expand if needed)
        value_to_append = np.expand_dims(value_to_append, axis=-1)
    return value_to_append


def combine_output(parts):  # todo refactor this is horrible
    fish_output = FishOutput()
    paramecium_output = ParameciumOutput()
    parts = sorted(parts, key=lambda curr: curr['end'])
    # validate
    starting_frames, ending_frames = [], []
    for d in parts:
        starting_frames.append(d['start'])
        ending_frames.append(d['end'])
    if (np.array(starting_frames[1:]) - np.array(ending_frames[:-1]) != 1).any():
        print("Lacking frames (will put nans instead)")

    # To prevent errors: start with empty nan & false values, and fill only existing parts
    fish_output.reset(frame_start=1, n_frames=max(ending_frames))
    n_paramecia = parts[0]['fish'].events[0].paramecium.color_points.shape[0]
    paramecium_output.reset(n_paramecia=n_paramecia, n_frames=max(ending_frames))
    print(parts)
    for d in tqdm(parts, desc="current part"):
        fish, frame_start, frame_end = d['fish'], d['start'], d['end']
        curr: Event = fish.events[0]
        ok = True
        if len(curr.fish_tracking_status_list) > (frame_end - frame_start + 1):
            # print("slice " + str(len(curr.fish_tracking_status_list)) + " > " + str(frame_end - frame_start))
            st = curr.fish_tracking_status_list[(frame_start - 1):frame_end]
            tail_st = curr.tail_tip_status_list[(frame_start - 1):frame_end]
            head_pred_st = curr.is_head_prediction_list[(frame_start - 1):frame_end]
            tail_bout = curr.tail.is_bout_frame_list[(frame_start - 1):frame_end]
            fish_area_in_pixels = curr.fish_area_in_pixels[(frame_start - 1):frame_end]
            orig_x, orig_y = curr.head.origin_points.x[(frame_start - 1):frame_end], \
                             curr.head.origin_points.y[(frame_start - 1):frame_end]
            dest_x, dest_y = curr.head.destination_points.x[(frame_start - 1):frame_end], \
                             curr.head.destination_points.y[(frame_start - 1):frame_end]
            eyes = curr.head.eyes_abs_angle_deg[(frame_start - 1):frame_end, :]
            eyes_head_diff = curr.head.eyes_head_dir_diff_ang[(frame_start - 1):frame_end, :]
            eyes_areas_pixels = curr.head.eyes_areas_pixels[(frame_start - 1):frame_end, :]
            tail_tip_x, tail_tip_y = curr.tail.tail_tip_point_list.x[(frame_start - 1):frame_end], \
                                     curr.tail.tail_tip_point_list.y[(frame_start - 1):frame_end]
            tail_path_list = curr.tail.tail_path_list[(frame_start - 1):frame_end]
            interpolated_tail_path = curr.tail.interpolated_tail_path[(frame_start - 1):frame_end]
            tail_vel = curr.tail.velocity_norms[(frame_start - 1):frame_end, :]
            paramecium_center = curr.paramecium.center_points[(frame_start - 1):frame_end, :, :]
            paramecium_status = curr.paramecium.status_points[(frame_start - 1):frame_end, :]
            paramecium_area = curr.paramecium.area_points[(frame_start - 1):frame_end, :]
            paramecium_ell_maj = curr.paramecium.ellipse_majors[(frame_start - 1):frame_end, :]
            paramecium_ell_min = curr.paramecium.ellipse_minors[(frame_start - 1):frame_end, :]
            paramecium_ell_dir = curr.paramecium.ellipse_dirs[(frame_start - 1):frame_end, :]
            paramecium_bbox = curr.paramecium.bounding_boxes[(frame_start - 1):frame_end, :]
            paramecium_color = curr.paramecium.color_points
        elif len(curr.fish_tracking_status_list) < frame_end - frame_start:
            print("Error. " + str(len(curr.fish_tracking_status_list)) + "<" + str(frame_end - frame_start))
            ok = False
        else:
            st = curr.fish_tracking_status_list
            head_pred_st = curr.is_head_prediction_list
            tail_st = curr.tail_tip_status_list
            tail_bout = curr.tail.is_bout_frame_list
            fish_area_in_pixels = curr.fish_area_in_pixels
            orig_x, orig_y = curr.head.origin_points.x, curr.head.origin_points.y
            dest_x, dest_y = curr.head.destination_points.x, curr.head.destination_points.y
            eyes = curr.head.eyes_abs_angle_deg
            eyes_head_diff = curr.head.eyes_head_dir_diff_ang
            eyes_areas_pixels = curr.head.eyes_areas_pixels
            tail_tip_x, tail_tip_y = curr.tail.tail_tip_point_list.x, curr.tail.tail_tip_point_list.y
            tail_path_list = curr.tail.tail_path_list
            tail_vel = curr.tail.velocity_norms
            interpolated_tail_path = curr.tail.interpolated_tail_path
            paramecium_center = curr.paramecium.center_points
            paramecium_status = curr.paramecium.status_points
            paramecium_area = curr.paramecium.area_points
            paramecium_ell_maj = curr.paramecium.ellipse_majors
            paramecium_ell_min = curr.paramecium.ellipse_minors
            paramecium_ell_dir = curr.paramecium.ellipse_dirs
            paramecium_bbox = curr.paramecium.bounding_boxes
            paramecium_color = curr.paramecium.color_points
        if ok:
            print(frame_start)
            print(frame_end)
            # quickfix - todo refactor
            paramecium_status = expand_shape_if_needed(paramecium_output.status, paramecium_status)
            paramecium_area = expand_shape_if_needed(paramecium_output.area, paramecium_area)
            paramecium_ell_maj = expand_shape_if_needed(paramecium_output.ellipse_majors, paramecium_ell_maj)
            paramecium_ell_min = expand_shape_if_needed(paramecium_output.ellipse_minors, paramecium_ell_min)
            paramecium_ell_dir = expand_shape_if_needed(paramecium_output.ellipse_dirs, paramecium_ell_dir)
            paramecium_bbox = expand_shape_if_needed(paramecium_output.bbox, paramecium_bbox)

            print(fish_output.origin_head_points_list.shape)
            fish_output.fish_status_list[(frame_start - 1):frame_end] = np.array(st)
            fish_output.fish_area_in_pixels[(frame_start - 1):frame_end] = np.array(fish_area_in_pixels)
            fish_output.is_head_prediction_list[(frame_start - 1):frame_end] = np.array(head_pred_st)
            fish_output.origin_head_points_list[(frame_start - 1):frame_end, 0] = np.array(orig_x)
            fish_output.origin_head_points_list[(frame_start - 1):frame_end, 1] = np.array(orig_y)
            fish_output.destination_head_points_list[(frame_start - 1):frame_end, 0] = np.array(dest_x)
            fish_output.destination_head_points_list[(frame_start - 1):frame_end, 1] = np.array(dest_y)
            fish_output.eyes_abs_angle_list[(frame_start - 1):frame_end] = np.array(eyes)
            fish_output.eyes_head_dir_diff_angle_list[(frame_start - 1):frame_end] = np.array(eyes_head_diff)
            fish_output.eyes_areas_pixels_list[(frame_start - 1):frame_end] = np.array(eyes_areas_pixels)
            try:
                paramecium_output.center[(frame_start - 1):frame_end] = np.array(paramecium_center)
                paramecium_output.status[(frame_start - 1):frame_end] = np.array(paramecium_status)
                paramecium_output.area[(frame_start - 1):frame_end] = np.array(paramecium_area)
                paramecium_output.ellipse_majors[(frame_start - 1):frame_end] = np.array(paramecium_ell_maj)
                paramecium_output.ellipse_minors[(frame_start - 1):frame_end] = np.array(paramecium_ell_min)
                paramecium_output.ellipse_dirs[(frame_start - 1):frame_end] = np.array(paramecium_ell_dir)
                paramecium_output.bbox[(frame_start - 1):frame_end] = np.array(paramecium_bbox)
                paramecium_output.color = np.array(paramecium_color)
            except Exception as e:
                print("Failure: ", e)
                traceback.print_tb(e.__traceback__)
            try:
                fish_output.is_bout_frame_list[(frame_start - 1):frame_end] = np.array(tail_bout)
                fish_output.tail_tip_point_list[(frame_start - 1):frame_end, 0] = np.array(tail_tip_x)
                fish_output.tail_tip_point_list[(frame_start - 1):frame_end, 1] = np.array(tail_tip_y)
                fish_output.tail_path_list[(frame_start - 1):frame_end] = tail_path_list
                fish_output.velocity_norms[(frame_start - 1):frame_end] = np.array(tail_vel)
                fish_output.tail_tip_status_list[(frame_start - 1):frame_end] = np.array(tail_st)
                fish_output.interpolated_tail_path[(frame_start - 1):frame_end] = interpolated_tail_path
            except Exception as e:
                print("Failure: ", e)
                traceback.print_tb(e.__traceback__)
    return fish_output, paramecium_output


# Can work both on events parallel and full parallel
if __name__ == '__main__':
    _, mat_inputs_folder, _, _, _, _ = get_parameters(is_parallel=True)
    _, mat_outputs_folder, _, _, _, _ = get_parameters(is_parallel=False)

    mat_files = [f for f in os.listdir(mat_inputs_folder)
                 if "_frame_" in f and "_to_" in f and f.lower().endswith(".mat")]
    print("Files: ", mat_files)
    main(mat_inputs_folder, mat_outputs_folder, mat_files)
