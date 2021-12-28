import copy
import glob
import logging
import os
import numpy as np
from tqdm import tqdm

from feature_analysis.fish_environment.env_utils import PlotsCMDParameters, CombineAgeGroups, HeatmapNParameciaType, \
    HeatmapType, outcome_to_map, OutcomeMapType
from feature_analysis.fish_environment.fish_processed_data import FishAndEnvDataset, SingleFishAndEnvData, \
    ParameciumRelativeToFish, ExpandedEvent
from feature_analysis.fish_environment.features_utils import get_dataset, get_folder_and_file_names, recursive_fix_key_for_mat, \
    parse_input_from_command
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

    m = {}
    for from_v, to_v in [[0, 1.5], [1.5, 3], [0, 3], [3, 9],
                         [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
                         [0, 0.125], [0.25, 0.375], [0.5, 0.5 + 0.125], [0.75, 0.75 + 0.125], [1, 1.125],
                         [1.25, 1.25 + 0.125], [1.5, 1.5 + 0.125], [1.75, 1.75 + 0.125], [2, 2.125],
                         [2.25, 2.25 + 0.125], [2.5, 2.5 + 0.125], [2.75, 2.75 + 0.125], [3, 3.125]]:
        m["{0}-{1}mm".format(from_v, to_v)] = ["d_{0}_mm".format(a).replace(".", "_dot_")
                                               for a in fish.distances_for_fov_in_mm if from_v <= a <= to_v]

    field_names = ["vel_to_fish_{0}_mm_sec", "orth_vel_{0}_mm_sec", "field_angle_{0}_deg",
                   "diff_angle_{0}_deg", "distance_{0}_mm"]
    orig_names = ["velocity_towards_fish_mm_sec", "orthogonal_velocity_mm_sec", "field_angle_deg",
                  "diff_from_fish_angle_deg", "distance_from_fish_in_mm"]
    stat_names = ["mean", "sum", "min", "max", "strong", "max_pos", "max_neg", "mean_pos", "mean_neg", "one_para"]
    stat_funcs = [np.nanmean, np.nansum, np.nanmin, np.nanmax, lambda d: d[np.argmax(np.abs(d))],
                  lambda v: max_sign(v, is_pos=True), lambda v: max_sign(v, is_pos=False),
                  lambda v: max_sign(v, is_pos=True, is_mean=True), lambda v: max_sign(v, is_pos=False, is_mean=True),
                  lambda v: one_para(v)]

    empty_fields = {}
    for field in field_names:
        for what in stat_names:
            empty_fields[field.format(what)] = []

    fov_counters = {}
    for key in parameters.combine_outcomes.keys():
        fov_counters[key] = {}
        for d_key in m.keys():
            fov_counters[key][d_key] = {}
            for angle_key in fish.angles_for_fov.keys():
                fov_counters[key][d_key][angle_key] = {'n_paramecia': []}
                fov_counters[key][d_key][angle_key].update(copy.deepcopy(empty_fields))

    event: ExpandedEvent
    paramecium: ParameciumRelativeToFish
    for fish in tqdm(dataset.fish_processed_data_set,
                     desc="Para current fish age {0}".format(age if not parameters.is_combine_age else age_list)):
        if not (not parameters.is_combine_age and (age is None or fish.age_dpf == age) or
                (parameters.is_combine_age and fish.age_dpf in age_list)):
            continue

        if parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.all and \
                fish.num_of_paramecia_in_plate not in parameters.valid_n_paramecia:
            print("(all) Ignoring fish {0} with n_paramecia={1}".format(fish.name, fish.num_of_paramecia_in_plate))
            continue

        if parameters.heatmap_n_paramecia_type != HeatmapNParameciaType.all:  # todo refactor me
            if (parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n30 and fish.num_of_paramecia_in_plate != 30) or \
               (parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n50 and fish.num_of_paramecia_in_plate != 50) or \
               (parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n70 and fish.num_of_paramecia_in_plate != 70):
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

            velocities_frame_number = -1
            if event.is_inter_bout_interval_only:  # already IBI
                frame_number = np.where(event.frame_indices == starting_bout_indices[velocities_frame_number])[0][0]
            else:
                frame_number = starting_bout_indices[velocities_frame_number]  # frame number is relative to whole video

            if not event.fish_tracking_status_list[frame_number]:
                continue  # invalid

            if (parameters.heatmap_type == HeatmapType.target_only or
                parameters.heatmap_type == HeatmapType.no_target) and \
                    np.isnan(event.paramecium.target_paramecia_index):
                logging.error("Paramecia in fov has nan target index for {0}".format(event.event_name))
                continue  # no data

            for key in fov_counters.keys():
                event_key, ignored = outcome_to_map(event.outcome_str, parameters)
                if key != event_key or ignored:
                    continue

                # todo ignore predictions
                # add general features
                paramecium = event.paramecium
                paramecia_indices = [_ for _ in range(0, paramecium.field_angle.shape[1])]
                if not np.isnan(event.paramecium.target_paramecia_index):
                    if parameters.heatmap_type == HeatmapType.no_target:
                        paramecia_indices.remove(event.paramecium.target_paramecia_index)
                    elif parameters.heatmap_type == HeatmapType.target_only:
                        paramecia_indices = [event.paramecium.target_paramecia_index]

                # add FOV features (4D matrix) per event, not statistics
                result = get_fov_features(distance_names, event, fov_counters, frame_number, key, m,
                                          paramecia_indices, paramecium, to_list, velocities_frame_number)
                # empty_fields = {'velocity_towards_fish_mm_sec': [], 'orthogonal_velocity_mm_sec': [],
                #                 'field_angle_deg': [],
                #                 'diff_from_fish_angle_deg': [], 'distance_from_fish_in_mm': []}
                to_l = lambda f, d: np.nan if len(d) == 0 or np.isnan(d).all() else f(d)
                for _, d_key in enumerate(fov_counters[key].keys()):
                    for angle_ind, angle_key in enumerate(fov_counters[key][d_key].keys()):
                        fov_counters[key][d_key][angle_key]['n_paramecia'].append(result[d_key][angle_key]['n_paramecia'])

                        for field, orig in zip(field_names, orig_names):
                            for what, f in zip(stat_names, stat_funcs):
                                fov_counters[key][d_key][angle_key][field.format(what)].append(
                                    to_l(f, result[d_key][angle_key][orig]))

    return fov_counters


def get_fov_features(distances_fish, event, fov_counters, frame_number, key, m, paramecia_indices, paramecium,
                     to_list, velocities_frame_number):
    def get_data(data, curr_frame_number, para_indices):
        if len(data.shape) == 1:  # patch (1d instead of 2d)
            return to_list(np.squeeze(data[para_indices]))
        return to_list(np.squeeze(data[curr_frame_number, para_indices]))

    empty_fields = {'velocity_towards_fish_mm_sec': [],
                    'orthogonal_velocity_mm_sec': [],
                    'field_angle_deg': [],
                    'diff_from_fish_angle_deg': [],
                    'distance_from_fish_in_mm': []}

    result = {}
    distances = fov_counters[key].keys()
    for dist in distances:
        temp = {}  # sum fields for d
        for angle_key in fov_counters[key][dist].keys():
            temp[angle_key] = {'n_paramecia': 0}
            temp[angle_key].update(copy.deepcopy(empty_fields))

        for angle_ind, angle_key in enumerate(fov_counters[key][dist].keys()):
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

                paramecia_indices2 = np.array(paramecia_indices)[paramecia_status == 1]
                if len(paramecia_indices2) <= 0:
                    continue

                curr_indices = []
                if dist_ind == 0 or len(added_indices) <= 0:
                    curr_indices = paramecia_indices2
                elif dist_ind > 0 and len(added_indices) > 0:
                    curr_indices = np.setdiff1d(paramecia_indices2, added_indices)
                added_indices.extend(curr_indices)

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

        result[dist] = copy.deepcopy(temp)
    return result


def main(existing_ages, dataset, parameters: PlotsCMDParameters):
    paramecia_in_fov = {}
    age_list = parameters.combine_ages.keys() if parameters.is_combine_age else [None] + existing_ages
    for age in age_list:
        age_key = "all" if age is None else str(age)
        if parameters.is_combine_age:
            paramecia_in_fov[age_key] = \
                calc_paramecia_counts(dataset, parameters=parameters,
                                      age_list=[int(_) for _ in parameters.combine_ages[age]])
        else:
            paramecia_in_fov[age_key] = calc_paramecia_counts(dataset, parameters=parameters, age=age)

    save_mat_dict(heatmap_data_json.replace(".mat", "_features_in_fov.mat"),
                  recursive_fix_key_for_mat(paramecia_in_fov))
    print(heatmap_data_json.replace(".mat", "_features_in_fov.mat"))


# Run with: \\ems.elsc.huji.ac.il\avitan-lab\Lab-Shared\Analysis\FeedingAssaySapir * --outcome_map_type hit_miss_abort --is_combine_age --heatmap_type=target_only
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data_path, args = parse_input_from_command()
    parameters = PlotsCMDParameters()
    should_run_all_metadata_permutations = args.is_all_metadata_permutations
    parameters.fish = args.fish_name
    parameters.mat_names = glob.glob(os.path.join(data_path, "dataset_features-checked_fish", "data_set_features", "inter_bout_interval", "*.mat"))
    parameters.gaussian = args.gaussian
    parameters.is_bounding_box = args.is_bounding_box
    parameters.is_combine_age = args.is_combine_age
    parameters.heatmap_n_paramecia_type = args.heatmap_n_paramecia_type
    parameters.heatmap_type = args.heatmap_type
    parameters.outcome_map_type = args.outcome_map_type
    parameters.is_save_per_fish_heatmap = args.is_save_per_fish
    parameters.is_heatmap = not args.no_heatmap
    parameters.is_metadata = not args.no_metadata
    if args.age_groups == CombineAgeGroups.v2:
        parameters.combine_ages = parameters.combine_ages_v2
    elif args.age_groups == CombineAgeGroups.v3:
        parameters.combine_ages = parameters.combine_ages_v3
    else:
        parameters.combine_ages = parameters.combine_ages_v1

    from matplotlib import rcParams

    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = '12'

    dataset: FishAndEnvDataset = get_dataset(data_path, parameters=parameters, is_inter_bout_intervals=True)
    if parameters.outcome_map_type == OutcomeMapType.hit_miss_abort:
        parameters.combine_outcomes = {'hit-spit': ['hit', 'spit'], 'miss': ['miss'], 'abort': ['abort']}
    elif parameters.outcome_map_type == OutcomeMapType.hit_miss_abort_es_abort_noes:
        parameters.combine_outcomes = {'hit-spit': ['hit', 'spit'], 'miss': ['miss'],
                                       'abort,escape': ['abort,escape'],
                                       'abort,no-escape': ['abort,no-escape']}
    elif parameters.outcome_map_type == OutcomeMapType.strike_abort:
        parameters.combine_outcomes = {'strike': ['hit', 'miss', 'spit'], 'abort': 'abort'}
    else:  # all- default
        parameters.combine_outcomes = {'hit': ['hit'], 'spit': ['spit'], 'miss': ['miss'], 'abort': ['abort']}
    print("Path {0}, arguments: {1}. Mats {2}".format(data_path, parameters, parameters.mat_names))

    heatmap_folder, metadata_folder, heatmap_parent_folder, counters_json, heatmap_counters_json, heatmap_data_json, \
    add_name, add_heatmap_name = \
        get_folder_and_file_names(data_path_=data_path, parameters=parameters, age_groups=args.age_groups)

    existing_ages = list(set([fish.age_dpf for fish in dataset.fish_processed_data_set]))  # unique only
    existing_ages.sort()
    if -1 in existing_ages:
        existing_ages.remove(-1)

    print("Ages: ", existing_ages)

    main(existing_ages, dataset, parameters)
