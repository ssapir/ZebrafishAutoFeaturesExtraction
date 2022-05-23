import copy
import logging
import numpy as np
from tqdm import tqdm
from matplotlib import rcParams

from feature_analysis.fish_environment.env_utils import PlotsCMDParameters, CombineAgeGroups, HeatmapNParameciaType, \
    HeatmapType, outcome_to_map, OutcomeMapType, FeedingType
from feature_analysis.fish_environment.fish_processed_data import FishAndEnvDataset, SingleFishAndEnvData, \
    ParameciumRelativeToFish, ExpandedEvent, is_points_outside_plate, get_fov_angles
from feature_analysis.fish_environment.feature_utils import get_dataset, get_folder_and_file_names, recursive_fix_key_for_mat, \
    parse_input_from_command, frames_to_secs_converter, get_parameters, distance_name_to_value, is_fish_filtered
from utils.matlab_data_handle import save_mat_dict


def calc_paramecia_counts(dataset: FishAndEnvDataset, parameters: PlotsCMDParameters, age=None, age_list=[]):
    def max_sign(v, is_pos=True, is_mean=False):
        v = np.array(v)
        d = v[v > 0 & ~np.isnan(v)] if is_pos else v[v < 0 & ~np.isnan(v)]
        if len(d) == 0:
            return np.nan
        if is_mean:
            return np.nanmean(d)
        return np.nanmax(d) if is_pos else np.nanmin(d)
    def one_para(v):
        if len(v) == 1:
            return v[0]
        return np.nan
    def to_list(v):
        if isinstance(v, (list, np.ndarray)) and np.array(v).shape != ():  # patch for nan
            return copy.deepcopy(v)
        if np.isnan(v):
            return [np.nan]
        return [v]

    if len(dataset.fish_processed_data_set) == 0:
        logging.error("Zero length for {0}".format(dataset.fish_processed_data_set))
        return {}

    fish: SingleFishAndEnvData = dataset.fish_processed_data_set[0]
    distance_names = ["d_{0}_mm".format(dist).replace(".", "_dot_") for dist in fish.distances_for_fov_in_mm]
    angles_dict = fish.angles_for_fov

    m = {}
    small_distanes = False
    if small_distanes:  # 1/8
        add_d = [[0, 0.125], [0.25, 0.375], [0.5, 0.625], [0.75, 0.875], [1, 1.125], [1.25, 1.375], [1.5, 1.625],
                 [1.75, 1.875], [2, 2.125], [2.25, 2.375], [2.5, 2.625], [2.75, 2.875], [3, 3.125]]
    else:  # 1/4
        add_d = [[0, 0.375], [0.5, 0.875], [1, 1.375], [1.5, 1.875], [2, 2.375], [2.5, 2.875]]
    distance_pairs = [[0, 1.5], [1.5, 3], [0, 3], [1.5, 3.5], [1.5, 4],
                      [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]] + add_d
    for from_v, to_v in distance_pairs:
        m["{0}-{1}mm".format(from_v, to_v)] = ["d_{0}_mm".format(a).replace(".", "_dot_")
                                               for a in fish.distances_for_fov_in_mm if from_v <= a <= to_v]

    field_names = ["vel_to_fish_{0}_mm_sec", "orth_vel_{0}_mm_sec", "field_angle_{0}_deg",
                   "diff_angle_{0}_deg", "distance_{0}_mm", "dist_tar_{0}_mm"]
    orig_names = ["velocity_towards_fish_mm_sec", "orthogonal_velocity_mm_sec", "field_angle_deg",
                  "diff_from_fish_angle_deg", "distance_from_fish_in_mm", "distance_from_target_in_mm"]
    stat_names = ["mean", "sum", "min", "max", "strong", "max_pos", "max_neg", "mean_pos", "mean_neg", "one_para"]
    stat_funcs = [np.nanmean, np.nansum, np.nanmin, np.nanmax, lambda d: d[np.argmax(np.abs(d))],
                  lambda v: max_sign(v, is_pos=True), lambda v: max_sign(v, is_pos=False),
                  lambda v: max_sign(v, is_pos=True, is_mean=True), lambda v: max_sign(v, is_pos=False, is_mean=True),
                  lambda v: one_para(v)]

    velocitiy_values = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    ibi_str_name = lambda ibi: "ibi_{0}".format(ibi)

    empty_fields = {}
    for field in field_names:
        for what in stat_names:
            empty_fields[field.format(what)] = {}

    fov_counters = {}
    fov_fish_counters = {}
    event_ibi_fields = ['event_ibi_dur_sec_last', 'event_ibi_dur_sec_first', 'event_ibi_dur_sec_2nd',
                        'event_ibi_dur_sec_3rd', 'event_ibi_dur_sec_4th', 'event_ibi_dur_sec_5th',
                        'event_ibi_dur_sec_sum', 'event_ibi_dur_sec_mean', "event_end_frame_ind"]
    event_ibi_per_bout_fields = ["features_frame_from_0", "bout_start_frame_from_0", "bout_end_frame_from_0", "event_ibi_dur_sec"]
    for key in parameters.combine_outcomes.keys():
        fov_counters[key], fov_fish_counters[key] = {}, {}
        fov_counters[key]['event_data'] = {'fish_names': [], 'event_names': [], 'event_dur_sec': [],
                                           'event_is_valid': []}
        fov_fish_counters[key]['event_data'] = {'fish_names': [], 'n_valid_events': [], 'event_dur_sec': []}
        for ibi_k in event_ibi_fields:
            fov_counters[key]['event_data'][ibi_k] = []
            fov_fish_counters[key]['event_data'][ibi_k] = []
        for ibi_k in event_ibi_per_bout_fields:
            fov_counters[key]['event_data'][ibi_k] = {}
            for velocities_frame_number in velocitiy_values:
                fov_counters[key]['event_data'][ibi_k][ibi_str_name(velocities_frame_number)] = []

        for d_key in m.keys():
            fov_counters[key][d_key], fov_fish_counters[key][d_key] = {}, {}
            for angle_key in fish.angles_for_fov.keys():
                fov_counters[key][d_key][angle_key] = {'n_paramecia': {}}
                fov_counters[key][d_key][angle_key].update(copy.deepcopy(empty_fields))
                fov_fish_counters[key][d_key][angle_key] = {'n_paramecia': {}}
                fov_fish_counters[key][d_key][angle_key].update(copy.deepcopy(empty_fields))

                for feature in fov_counters[key][d_key][angle_key].keys():
                    for velocities_frame_number in velocitiy_values:
                        if ibi_str_name(velocities_frame_number) not in fov_counters[key][d_key][angle_key][feature].keys():
                            fov_counters[key][d_key][angle_key][feature][ibi_str_name(velocities_frame_number)] = []
                            fov_fish_counters[key][d_key][angle_key][feature][ibi_str_name(velocities_frame_number)] = []

    event: ExpandedEvent
    paramecium: ParameciumRelativeToFish
    for fish in tqdm(dataset.fish_processed_data_set,
                     desc="Para current fish age {0}".format(age if not parameters.is_combine_age else age_list)):
        if is_fish_filtered(parameters=parameters, fish=fish, age=age, age_list=age_list):
            continue

        if len(fish.events[0].paramecium.status_points) == 0:
            print("Error. fish {0} no paramecia".format(fish.name))

        paramecium: ParameciumRelativeToFish
        print("Existing {1} events for {0}".format(fish.name, len(fish.events)))
        for key in fov_counters.keys():
            a = [outcome_to_map(event.outcome_str, parameters) for event in fish.events]
            print(key, len([k for k, i in a if not i and key == k]))

        for event in tqdm(fish.events, desc="current event", disable=True):
            if event.is_inter_bout_interval_only:  # already IBI
                starting_bout_indices, ending_bout_indices = event.starting_bout_indices, event.ending_bout_indices
            else:
                starting_bout_indices, ending_bout_indices = ExpandedEvent.start_end_bout_indices(event)  # calc

            # Validate
            if len(starting_bout_indices) == 0 or len(event.paramecium.status_points) == 0:
                continue

            if (parameters.heatmap_type == HeatmapType.target_only or
                parameters.heatmap_type == HeatmapType.residuals) and \
                    np.isnan(event.paramecium.target_paramecia_index):
                logging.error("Paramecia in fov has nan target index for {0}".format(event.event_name))
                continue  # no data

            if len(event.paramecium.velocity_towards_fish.shape) == 2 and \
               event.paramecium.velocity_towards_fish.shape[0] != min(len(starting_bout_indices), len(ending_bout_indices)):
                logging.error("Event {0} vel shape {1} ibi {2}".format(event.event_name,
                    event.paramecium.velocity_towards_fish.shape,
                    min(len(starting_bout_indices), len(ending_bout_indices))))

            for velocities_frame_number in velocitiy_values:
                is_valid = True
                if min(len(starting_bout_indices), len(ending_bout_indices)) < abs(velocities_frame_number):
                    is_valid = False
                if len(event.paramecium.velocity_towards_fish.shape) == 2 \
                   and event.paramecium.velocity_towards_fish.shape[0] < abs(velocities_frame_number):
                    is_valid = False
                elif len(event.paramecium.velocity_towards_fish.shape) == 1 and abs(velocities_frame_number) > 1:
                    is_valid = False

                frame_number = None
                if is_valid:
                    if event.is_inter_bout_interval_only:  # already IBI
                        frame_number = np.where(event.frame_indices == starting_bout_indices[velocities_frame_number])[0][0]
                    else:
                        frame_number = starting_bout_indices[velocities_frame_number]  # frame number is relative to whole video

                    if not event.fish_tracking_status_list[frame_number]:
                        is_valid = False

                for key in fov_counters.keys():
                    event_key, ignored = outcome_to_map(event.outcome_str, parameters)
                    if key != event_key or ignored:
                        continue

                    # todo ignore predictions
                    # add general features
                    paramecium = event.paramecium
                    paramecia_indices = [_ for _ in range(0, paramecium.field_angle.shape[1])]
                    if not np.isnan(event.paramecium.target_paramecia_index):
                        if parameters.heatmap_type == HeatmapType.residuals:
                            paramecia_indices.remove(event.paramecium.target_paramecia_index)
                        elif parameters.heatmap_type == HeatmapType.target_only:
                            paramecia_indices = [event.paramecium.target_paramecia_index]

                    if is_valid:
                        result = get_fov_features(distance_names, angles_dict, event, fov_counters, frame_number, key, m,
                                                  paramecia_indices, paramecium, to_list, velocities_frame_number,
                                                  frame_name=starting_bout_indices[velocities_frame_number])

                    if velocities_frame_number == -1:  # for last one only
                        event_duration = frames_to_secs_converter(event.event_frame_ind)
                        cum_durations = [frames_to_secs_converter(e_ibi - s_ibi)
                                         for (e_ibi, s_ibi) in zip(event.starting_bout_indices[1:], event.ending_bout_indices)
                                         if e_ibi - s_ibi > 0]
                        get_cum = lambda ind, len_thr: cum_durations[ind] if len(cum_durations) > len_thr else np.nan
                        fov_counters[key]['event_data']['fish_names'].append(fish.name)
                        fov_counters[key]['event_data']['event_names'].append(event.event_name)
                        fov_counters[key]['event_data']['event_end_frame_ind'].append(event.event_frame_ind)
                        fov_counters[key]['event_data']['event_is_valid'].append(is_valid)
                        fov_counters[key]['event_data']['event_dur_sec'].append(event_duration)
                        fov_counters[key]['event_data']['event_ibi_dur_sec_sum'].append(np.nansum(cum_durations))
                        fov_counters[key]['event_data']['event_ibi_dur_sec_mean'].append(np.nanmean(cum_durations))
                        fov_counters[key]['event_data']['event_ibi_dur_sec_first'].append(get_cum(0, 0))
                        fov_counters[key]['event_data']['event_ibi_dur_sec_last'].append(get_cum(-1, 0))
                        fov_counters[key]['event_data']['event_ibi_dur_sec_2nd'].append(get_cum(1, 1))
                        fov_counters[key]['event_data']['event_ibi_dur_sec_3rd'].append(get_cum(2, 2))
                        fov_counters[key]['event_data']['event_ibi_dur_sec_4th'].append(get_cum(3, 3))
                        fov_counters[key]['event_data']['event_ibi_dur_sec_5th'].append(get_cum(4, 4))
                        try:  # todo this should replace the manual before
                            fov_counters[key]['event_data']["event_ibi_dur_sec"][
                                ibi_str_name(velocities_frame_number)].append(
                                get_cum(velocities_frame_number, velocities_frame_number))
                        except Exception as e:
                            print(e)

                    # for all velocities! Metadata for frames
                    fov_counters[key]['event_data']["features_frame_from_0"][
                        ibi_str_name(velocities_frame_number)].append(starting_bout_indices[velocities_frame_number])
                    fov_counters[key]['event_data']["bout_start_frame_from_0"][
                        ibi_str_name(velocities_frame_number)].append(starting_bout_indices[velocities_frame_number])
                    fov_counters[key]['event_data']["bout_end_frame_from_0"][
                        ibi_str_name(velocities_frame_number)].append(ending_bout_indices[velocities_frame_number])

                    to_l = lambda f, d: np.nan if len(d) == 0 or np.isnan(d).all() else f(d)
                    for _, d_key in enumerate([a for a in fov_counters[key].keys() if a not in ['event_data']]):
                        for angle_ind, angle_key in enumerate(fov_counters[key][d_key].keys()):
                            if is_valid:
                                fov_counters[key][d_key][angle_key]['n_paramecia'][ibi_str_name(velocities_frame_number)].append(
                                    result[d_key][angle_key]['n_paramecia'])
                            else:
                                fov_counters[key][d_key][angle_key]['n_paramecia'][ibi_str_name(velocities_frame_number)].append(np.nan)

                            for field, orig in zip(field_names, orig_names):
                                for what, f in zip(stat_names, stat_funcs):
                                    if is_valid:
                                        fov_counters[key][d_key][angle_key][field.format(what)][ibi_str_name(velocities_frame_number)].append(
                                            to_l(f, result[d_key][angle_key][orig]))
                                    else:
                                        fov_counters[key][d_key][angle_key][field.format(what)][
                                            ibi_str_name(velocities_frame_number)].append(to_l(f, [np.nan]))

    for key in fov_counters.keys():
        existing_fish = np.array(fov_counters[key]['event_data']['fish_names'])
        is_valid = np.array(fov_counters[key]['event_data']['event_is_valid'], dtype=bool)
        for curr_fish in tqdm(np.unique(existing_fish), desc="per_fish"):
            is_event_included = (existing_fish == curr_fish) & is_valid
            fov_fish_counters[key]['event_data']['fish_names'].append(curr_fish)
            fov_fish_counters[key]['event_data']['n_valid_events'].append(np.sum(is_event_included))
            for d_key in [a for a in fov_counters[key].keys() if a not in ['event_data']]:
                for angle_key in fov_counters[key][d_key].keys():
                    for feature in fov_counters[key][d_key][angle_key].keys():
                        for ibi_n in fov_counters[key][d_key][angle_key][feature].keys():
                            all_fish_data = np.array(fov_counters[key][d_key][angle_key][feature][ibi_n])
                            curr_fish_data = all_fish_data[is_event_included]
                            fov_fish_counters[key][d_key][angle_key][feature][ibi_n].append(
                                np.nanmean(curr_fish_data) if len(curr_fish_data) > 0 else np.nan)
            for ev_key in [a for a in fov_fish_counters[key]['event_data'].keys() if a not in ['fish_names', 'event_is_valid', 'n_valid_events']]:
                all_fish_data = np.array(fov_counters[key]['event_data'][ev_key])
                curr_fish_data = all_fish_data[is_event_included]
                fov_fish_counters[key]['event_data'][ev_key].append(
                    np.nanmean(curr_fish_data) if len(curr_fish_data) > 0 else np.nan)

    return fov_counters, fov_fish_counters


def get_fov_features(distances_fish, angles_dict, event, fov_counters, frame_number, key, m, paramecia_indices,
                     paramecium, to_list, velocities_frame_number, frame_name=""):
    def get_data(data, curr_frame_number, para_indices):
        if len(data.shape) == 1:  # patch (1d instead of 2d)
            return to_list(np.squeeze(data[para_indices]))
        return to_list(np.squeeze(data[curr_frame_number, para_indices]))

    empty_fields = {'velocity_towards_fish_mm_sec': [],
                    'orthogonal_velocity_mm_sec': [],
                    'field_angle_deg': [],
                    'diff_from_fish_angle_deg': [],
                    'distance_from_fish_in_mm': [],
                    'distance_from_target_in_mm': []}

    origin = [event.head.origin_points.x[frame_number], event.head.origin_points.y[frame_number]]
    head_angle = event.head.directions_in_deg[frame_number]
    if np.isnan(np.array(origin + [head_angle])).any():
        logging.error("get_fov_features nan origin/head direction for event {0}".format(event.event_name))
        return {}

    result = {}
    distances = [a for a in fov_counters[key].keys() if a not in ['event_data']]
    for dist in distances:
        temp = {}  # sum fields for d
        for angle_key in fov_counters[key][dist].keys():
            temp[angle_key] = {'n_paramecia': 0}
            temp[angle_key].update(copy.deepcopy(empty_fields))

        for angle_ind, angle_key in enumerate(fov_counters[key][dist].keys()):
            from_angle, to_angle = angles_dict[angle_key]
            check_angles = get_fov_angles(head_angle, from_angle=from_angle, to_angle=to_angle)
            added_indices = []
            distances_to_ignore = [_ for _ in np.setdiff1d(distances_fish, m[dist])
                                   if distances_fish.index(_) <= np.max([distances_fish.index(c) for c in m[dist]])]
            for d_key in distances_to_ignore:  # due to FOV cumulative values, we ignore everything of smaller distance
                dist_ind = distances_fish.index(d_key)
                paramecia_status = \
                    event.paramecium.field_of_view_status[frame_number, paramecia_indices, angle_ind, dist_ind]
                paramecia_indices2 = np.array(paramecia_indices)[paramecia_status == 1]
                added_indices.extend(paramecia_indices2)
            added_indices = list(np.unique(added_indices))
            for d_key in m[dist]:
                dist_ind = distances_fish.index(d_key)
                paramecia_status = \
                    event.paramecium.field_of_view_status[frame_number, paramecia_indices, angle_ind, dist_ind]

                # Find added paramecia indices - only within FOV and that were not already added
                paramecia_indices2 = np.array(paramecia_indices)[paramecia_status == 1]
                curr_indices = []
                if dist_ind == 0 or len(added_indices) == 0:
                    curr_indices = paramecia_indices2
                elif dist_ind > 0 and len(added_indices) > 0:
                    curr_indices = np.setdiff1d(paramecia_indices2, added_indices)
                added_indices.extend(curr_indices)

                if len(curr_indices) == 0:
                    distance_value = distance_name_to_value(d_key)
                    if distance_value.replace('.', '', 1).isdigit():
                        # set nan in n-paramecia, not zero, if fov is outside plate area
                        is_fov_points_outside_plate = is_points_outside_plate(
                            distance_mm=float(distance_value), angles_list=check_angles, origin=origin)
                        if is_fov_points_outside_plate.all():
                            temp[angle_key]['n_paramecia'] = np.nan
                            if float(distance_value) <= 5.0:  # log for non-trivial filtering
                                logging.debug("Event {0} frame {1} fish FOV {2} (angles {3}-{4} deg {5}) outside plate".format(
                                    event.event_name, frame_name, distance_value, check_angles[0], check_angles[-1], angle_key))
                        continue  # don't continue counting - no para in fov
                    else:
                        logging.error("distance-key {0} not a valid number {1}".format(d_key, distance_value))
                    continue  # don't continue counting - no para in fov

                if len(curr_indices) > 0:
                    temp[angle_key]['n_paramecia'] += len(curr_indices)
                    temp[angle_key]['velocity_towards_fish_mm_sec'].extend(
                        get_data(paramecium.velocity_towards_fish, velocities_frame_number, curr_indices))
                    temp[angle_key]['orthogonal_velocity_mm_sec'].extend(
                        get_data(paramecium.velocity_orthogonal, velocities_frame_number, curr_indices))
                    temp[angle_key]['field_angle_deg'].extend(
                        get_data(paramecium.field_angle, frame_number, curr_indices))
                    temp[angle_key]['diff_from_fish_angle_deg'].extend(
                        get_data(paramecium.diff_from_fish_angle_deg, frame_number, curr_indices))
                    temp[angle_key]['distance_from_fish_in_mm'].extend(
                        get_data(paramecium.distance_from_fish_in_mm, frame_number, curr_indices))
                    temp[angle_key]['distance_from_target_in_mm'].extend(
                        get_data(paramecium.distance_from_target_in_mm, frame_number, curr_indices))
        result[dist] = copy.deepcopy(temp)
    return result


def main(existing_ages, dataset, parameters: PlotsCMDParameters):
    paramecia_in_fov, fov_fish_counters = {}, {}
    age_list = parameters.combine_ages.keys() if parameters.is_combine_age else [None] + existing_ages
    for age in age_list:
        age_key = "all" if age is None else str(age)
        if parameters.is_combine_age:
            paramecia_in_fov[age_key], fov_fish_counters[age_key] = \
                calc_paramecia_counts(dataset, parameters=parameters,
                                      age_list=[int(_) for _ in parameters.combine_ages[age]])
        else:
            paramecia_in_fov[age_key], fov_fish_counters[age_key] = \
                calc_paramecia_counts(dataset, parameters=parameters, age=age)

    save_mat_dict(fullpath_output_prefix + "_all_fov_per_events.mat",
                  recursive_fix_key_for_mat(paramecia_in_fov))
    save_mat_dict(fullpath_output_prefix + "_all_fov_per_fish.mat",
                  recursive_fix_key_for_mat(fov_fish_counters))
    print(fullpath_output_prefix + "_all_fov_per_event.mat")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parameters, data_path, should_run_all_metadata_permutations = get_parameters()

    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = '12'

    dataset: FishAndEnvDataset = get_dataset(data_path, parameters=parameters, is_inter_bout_intervals=True)
    print("Path {0}, arguments: {1}. Mats {2}".format(data_path, parameters, parameters.mat_names))

    heatmap_folder, metadata_folder, heatmap_parent_folder, counters_json, heatmap_counters_json, heatmap_data_json, \
        fullpath_output_prefix, add_name, add_heatmap_name = \
        get_folder_and_file_names(data_path_=data_path, parameters=parameters, age_groups=parameters.age_groups)

    existing_ages = list(set([fish.age_dpf for fish in dataset.fish_processed_data_set]))  # unique only
    existing_ages.sort()
    if -1 in existing_ages:
        existing_ages.remove(-1)

    print("Ages: ", existing_ages)

    main(existing_ages, dataset, parameters)
