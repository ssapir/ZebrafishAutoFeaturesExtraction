import json
from collections import Counter

from scipy.ndimage import gaussian_filter

from feature_analysis.fish_environment.env_utils import OutcomeMapType
from feature_analysis.fish_environment.feature_utils import get_y, distance_name_to_value, get_parameters, \
    save_fig_fixname, get_folder_and_file_names
from feature_analysis.fish_environment.plot_utils import plot_scatter_bar, plot_densities, heatmap_plot
from utils.main_utils import create_dirs_if_missing
from utils.matlab_data_handle import load_mat_dict

from tqdm import tqdm
import os
import traceback
import copy
import logging
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIG_SIZE = (8, 6)  # 6 height


def to_list(v):
    if isinstance(v, (list, np.ndarray)):
        if np.array(v).shape == ():  # patch for array of int with empty shape
            return np.array([v])
        return copy.deepcopy(v)
    return [v]


# todo wrong since we should sum per event the correct distances
def paramecia_fov_dict_combined_distances(is_per_fish=False, is_fish_mean=False, key='all'):
    def quickfix(v):
        if "diff_from_fish_angle" in key:
            return copy.deepcopy(np.abs(v))
        if "velocity" in key:
            v = copy.deepcopy(v)
            return np.extract(np.array(np.abs(v)) <= 15, v)
        return v

    # m = {'0-2mm': ["d_1_0_mm", "d_2_0_mm"], '2-5mm': ["d_3_0_mm", "d_4_0_mm", "d_5_0_mm"]}
    # m = {'0-3mm': ["d_1_0_mm", "d_2_0_mm", "d_3_0_mm"], '3-6mm': ["d_4_0_mm", "d_5_0_mm", "d_6_0_mm"]}

    m = {'0-2mm': ['a_0_2mm'],
         '0-1.75mm': ['a_0_1_75mm'],
         '1.5-3mm': ['a_1_5_3mm'],
         '1.75-3mm': ['a_1_5_3mm'],
         '1.74-4mm': ['a_1_75_4mm'],
         '0-3mm': ['a_0_3mm'],
         }
    per_age_statistics = {}
    flipped_per_age_statistics = {}
    per_age_for_hist_statistics = {}
    flipped_per_age_for_hist_statistics = {}

    if is_per_fish:
        data = per_fish_paramecia_in_fov
    elif is_fish_mean:
        data = paramecia_mean_per_fish_in_fov
    else:
        data = paramecia_in_fov
    for age_ind, (age_name, values) in enumerate(data.items()):
        if age_name.startswith("__"):
            continue
        for outcome, inner_values in values.items():
            for (d_name, distances_list) in m.items():
                for distance_mm in distances_list:
                    inner_inner_values = inner_values[distance_mm]
                    new_values = {}
                    for k in inner_inner_values.keys():
                        new_values[k] = []
                    for j, (angle_type, inner_inner_inner_values) in enumerate(inner_inner_values.items()):
                        if angle_type not in per_age_statistics.keys():
                            per_age_statistics[angle_type] = {}
                            per_age_for_hist_statistics[angle_type] = {}
                            flipped_per_age_statistics[angle_type] = {}
                            flipped_per_age_for_hist_statistics[angle_type] = {}
                        if d_name not in per_age_statistics[angle_type].keys():
                            per_age_statistics[angle_type][d_name] = {}
                            per_age_for_hist_statistics[angle_type][d_name] = {}
                            flipped_per_age_statistics[angle_type][d_name] = {}
                            flipped_per_age_for_hist_statistics[angle_type][d_name] = {}
                        if outcome not in per_age_statistics[angle_type][d_name].keys():
                            per_age_statistics[angle_type][d_name][outcome] = {}
                            per_age_for_hist_statistics[angle_type][d_name][outcome] = {}
                        if age_name not in per_age_statistics[angle_type][d_name][outcome].keys():
                            per_age_statistics[angle_type][d_name][outcome][age_name] = {'x': [], 'y': []}
                            per_age_for_hist_statistics[angle_type][d_name][outcome][age_name] = {'x': age_ind, 'y': []}
                        if age_name not in flipped_per_age_statistics[angle_type][d_name].keys():
                            flipped_per_age_statistics[angle_type][d_name][age_name] = {}
                            flipped_per_age_for_hist_statistics[angle_type][d_name][age_name] = {}
                        if outcome not in flipped_per_age_statistics[angle_type][d_name][age_name].keys():
                            flipped_per_age_statistics[angle_type][d_name][age_name][outcome] = {'x': [], 'y': []}
                            flipped_per_age_for_hist_statistics[angle_type][d_name][age_name][outcome] = {'x': [],
                                                                                                          'y': []}

                        add = quickfix(to_list(inner_inner_inner_values[key]))
                        add = [_ for _ in add if _ <= 5 and key == "all_diff"]
                        x_add = [0] * len(add)
                        new_values[angle_type].extend(add)

                        per_age_statistics[angle_type][d_name][outcome][age_name]['x'].extend(x_add)
                        per_age_statistics[angle_type][d_name][outcome][age_name]['y'].extend(add)
                        flipped_per_age_statistics[angle_type][d_name][age_name][outcome]['x'].extend(x_add)
                        flipped_per_age_statistics[angle_type][d_name][age_name][outcome]['y'].extend(add)
    return per_age_statistics, flipped_per_age_statistics, per_age_for_hist_statistics, \
           flipped_per_age_for_hist_statistics


def paramecia_event_data_dict(is_per_fish=False, key='all', key_2nd=None, f=lambda x: x, quickfix=lambda x: x,
                              data=None, is_fish_mean=False):
    per_age_statistics = {}
    flipped_per_age_statistics = {}

    if data is None:
        if is_fish_mean:
            data = paramecia_mean_per_fish_in_fov
        else:
            data = paramecia_in_fov
    for age_ind, (age_name, values) in enumerate(data.items()):
        if age_name.startswith("__"):
            continue
        for outcome, inner_values in values.items():
            inner_inner_values = inner_values.get('event_data', {})
            if inner_inner_values == {}:
                logging.error("Missing event data in {0} {1}".format(age_name, outcome))
                continue
            if outcome not in per_age_statistics.keys():
                per_age_statistics[outcome] = {}
            if age_name not in per_age_statistics[outcome].keys():
                per_age_statistics[outcome][age_name] = {'x': [], 'y': []}
                if key_2nd is not None:
                    per_age_statistics[outcome][age_name]['y2'] = []
            if age_name not in flipped_per_age_statistics.keys():
                flipped_per_age_statistics[age_name] = {}
            if outcome not in flipped_per_age_statistics[age_name].keys():
                flipped_per_age_statistics[age_name][outcome] = {'x': [], 'y': []}
                if key_2nd is not None:
                    flipped_per_age_statistics[age_name][outcome]['y2'] = []
            add, add_hist, add_2nd = [], [], []
            pad = lambda v1, v2: copy.deepcopy(v1) if (max(len(v1), len(v2)) - len(v1)) == 0 \
                else np.pad(copy.deepcopy(v1), pad_width=(max(len(v1), len(v2)) - len(v1), 0),
                            mode='constant', constant_values=np.nan)
            add = quickfix(to_list(get_y(inner_inner_values, k=key)))
            if key_2nd is not None:
                add_2nd = quickfix(to_list(inner_inner_values[key_2nd]))
                add = pad(add, add_2nd)
                add_2nd = pad(add_2nd, add)
            x_add = [np.nan] * len(add)
            per_age_statistics[outcome][age_name]['x'].extend(x_add)
            per_age_statistics[outcome][age_name]['y'].extend(f(add))
            flipped_per_age_statistics[age_name][outcome]['x'].extend(x_add)
            flipped_per_age_statistics[age_name][outcome]['y'].extend(f(add))

            if key_2nd is not None:
                per_age_statistics[outcome][age_name]['y2'].extend(f(add_2nd))
                flipped_per_age_statistics[age_name][outcome]['y2'].extend(f(add_2nd))

    return per_age_statistics, flipped_per_age_statistics


def paramecia_distanecs_data_dict(distances_map, key='all', key_2nd=None, f=lambda x, d, a: x, quickfix=lambda x: x,
                                  data=None, is_fish_mean=False, distance_f=distance_name_to_value):
    def get_add_values(values_dict, distance_value):  # inner_inner_inner_values
        add, add_2nd = [], []
        pad = lambda v1, v2: copy.deepcopy(v1) if (max(len(v1), len(v2)) - len(v1)) == 0 \
            else np.pad(copy.deepcopy(v1), pad_width=(max(len(v1), len(v2)) - len(v1), 0),
                        mode='constant', constant_values=np.nan)

        if values_dict[key] == {}:
            add = quickfix(to_list([]))
        else:
            add = quickfix(to_list(get_y(values_dict, k=key)))
        if key_2nd is not None:
            add_2nd = quickfix(to_list(values_dict[key_2nd]))
            add = pad(add, add_2nd)
            add_2nd = pad(add_2nd, add)
        x_add = [np.nan] * len(add)
        if distance_value is not None:
            x_add = to_list(distance_value) * len(add)

        return x_add, add, add_2nd

    per_age_statistics = {}

    if data is None:
        if is_fish_mean:
            data = paramecia_mean_per_fish_in_fov
        else:
            data = paramecia_in_fov

    for age_ind, (age_name, values) in enumerate(data.items()):
        if age_name.startswith("__"):
            continue
        for outcome, inner_values in values.items():
            for distance_key, only_distances in distances_map.items():
                for i, (distance_mm, inner_inner_values) in enumerate(inner_values.items()):
                    if only_distances != [] and distance_mm.replace("_dot_", ".") not in only_distances:
                        continue
                    distance_value = distance_f(distance_mm)
                    distance_value = float(distance_value) if distance_value.replace('.', '', 1).isdigit() else None

                    for j, (angle_type, inner_inner_inner_values) in enumerate(inner_inner_values.items()):
                        if angle_type not in per_age_statistics.keys():
                            per_age_statistics[angle_type] = {}
                        if outcome not in per_age_statistics[angle_type].keys():
                            per_age_statistics[angle_type][outcome] = {}
                        if age_name not in per_age_statistics[angle_type][outcome].keys():
                            per_age_statistics[angle_type][outcome][age_name] = {}
                        if distance_key not in per_age_statistics[angle_type][outcome][age_name].keys():
                            per_age_statistics[angle_type][outcome][age_name][distance_key] = {'x': [], 'y': []}
                            if key_2nd is not None:
                                per_age_statistics[angle_type][outcome][age_name][distance_key]['y2'] = []

                        x_add, add, add_2nd = get_add_values(inner_inner_inner_values, distance_value)
                        per_age_statistics[angle_type][outcome][age_name][distance_key]['x'].extend(x_add)
                        per_age_statistics[angle_type][outcome][age_name][distance_key]['y'].extend(
                            f(add, distance_mm, angle_type))
                        if key_2nd is not None:
                            per_age_statistics[angle_type][outcome][age_name][distance_key]['y2'].extend(
                                f(add_2nd, distance_mm, angle_type))

    return per_age_statistics



def paramecia_fov_dict(is_per_fish=False, key='all', hist_key='n_paramecia', key_2nd=None, only_distances=[],
                       distance_f=distance_name_to_value, f=lambda x, dist, ang: x, data=None, is_fish_mean=False):
    def quickfix(v):
        if "diff_from_fish_angle" in key:
            return copy.deepcopy(np.abs(v))
        if "vel" in key:
            v = np.array(copy.deepcopy(v))
            # v[np.isnan(v)] = 0
            return np.extract(np.array(np.abs(v)) <= 10, v)
        # if "velocity" in key:
        #     v = copy.deepcopy(v)
        #     return np.extract(np.array(np.abs(v)) <= 8, v)
        return v

    per_age_statistics = {}
    flipped_per_age_statistics = {}
    per_age_for_hist_statistics = {}
    flipped_per_age_for_hist_statistics = {}
    levels = []

    if data is None:
        if is_per_fish:
            data = per_fish_paramecia_in_fov
        elif is_fish_mean:
            data = paramecia_mean_per_fish_in_fov
        else:
            data = paramecia_in_fov
    for age_ind, (age_name, values) in enumerate(data.items()):
        if age_name.startswith("__"):
            continue
        for outcome, inner_values in values.items():
            for i, (distance_mm, inner_inner_values) in enumerate(inner_values.items()):
                # d = distance_mm.replace("d_", "").replace("a_", "").replace("_", " ")
                distance_value = distance_f(distance_mm)
                if distance_value.replace('.', '', 1).isdigit():
                    distance_value = float(distance_value)
                else:
                    distance_value = None
                if only_distances != [] and distance_mm.replace("_dot_", ".") not in only_distances:
                    continue
                # levels.append(d if distance_value is None else distance_value)
                for j, (angle_type, inner_inner_inner_values) in enumerate(inner_inner_values.items()):
                    if angle_type not in per_age_statistics.keys():
                        per_age_statistics[angle_type] = {}
                        per_age_for_hist_statistics[angle_type] = {}
                        flipped_per_age_statistics[angle_type] = {}
                        flipped_per_age_for_hist_statistics[angle_type] = {}
                    if outcome not in per_age_statistics[angle_type].keys():
                        per_age_statistics[angle_type][outcome] = {}
                        per_age_for_hist_statistics[angle_type][outcome] = {}
                    if age_name not in per_age_statistics[angle_type][outcome].keys():
                        per_age_statistics[angle_type][outcome][age_name] = {'x': [], 'y': []}
                        per_age_for_hist_statistics[angle_type][outcome][age_name] = {'x': age_ind, 'y': []}
                        if key_2nd is not None:
                            per_age_statistics[angle_type][outcome][age_name]['y2'] = []
                            per_age_for_hist_statistics[angle_type][outcome][age_name]['y2'] = []
                    if age_name not in flipped_per_age_statistics[angle_type].keys():
                        flipped_per_age_statistics[angle_type][age_name] = {}
                        flipped_per_age_for_hist_statistics[angle_type][age_name] = {}
                    if outcome not in flipped_per_age_statistics[angle_type][age_name].keys():
                        flipped_per_age_statistics[angle_type][age_name][outcome] = {'x': [], 'y': []}
                        flipped_per_age_for_hist_statistics[angle_type][age_name][outcome] = {'x': [], 'y': []}
                        if key_2nd is not None:
                            flipped_per_age_statistics[angle_type][age_name][outcome]['y2'] = []
                            flipped_per_age_for_hist_statistics[angle_type][age_name][outcome]['y2'] = []
                    add, add_hist, add_2nd = [], [], []
                    if is_per_fish:
                        for fish_name, data_dict in inner_inner_inner_values.items():
                            add.extend(quickfix(to_list(data_dict[key][0]['ibi__1'])))
                            if key_2nd is not None:
                                add_2nd.extend(quickfix(to_list(data_dict[key_2nd])))
                            x_add = [np.nan] * len(add)
                            if distance_value is not None:
                                x_add = to_list(distance_value) * len(add)
                                if hist_key is not None:
                                    add_hist.extend(to_list(distance_value) * int(data_dict[hist_key]))
                    else:
                        pad = lambda v1, v2: copy.deepcopy(v1) if (max(len(v1), len(v2)) - len(v1)) == 0 \
                            else np.pad(copy.deepcopy(v1), pad_width=(max(len(v1), len(v2)) - len(v1), 0),
                                        mode='constant', constant_values=np.nan)

                        # print(key, inner_inner_inner_values[key])
                        if inner_inner_inner_values[key] == {}:
                            add = quickfix(to_list([]))
                        else:
                            add = quickfix(to_list(get_y(inner_inner_inner_values, k=key)))
                        if key_2nd is not None:
                            add_2nd = quickfix(to_list(inner_inner_inner_values[key_2nd]))
                            add = pad(add, add_2nd)
                            add_2nd = pad(add_2nd, add)

                        x_add = [np.nan] * len(add)
                        if distance_value is not None:
                            x_add = to_list(distance_value) * len(add)
                            if hist_key is not None:
                                add_hist = to_list(distance_value) * int(inner_inner_inner_values[hist_key])
                    # print(angle_type, outcome, age_name, distance_value, add, x_add)
                    per_age_statistics[angle_type][outcome][age_name]['x'].extend(x_add)
                    per_age_statistics[angle_type][outcome][age_name]['y'].extend(f(add, distance_mm, angle_type))
                    flipped_per_age_statistics[angle_type][age_name][outcome]['x'].extend(x_add)
                    flipped_per_age_statistics[angle_type][age_name][outcome]['y'].extend(f(add, distance_mm, angle_type))

                    if hist_key is not None:
                        per_age_for_hist_statistics[angle_type][outcome][age_name]['y'].extend(add_hist)
                        flipped_per_age_for_hist_statistics[angle_type][age_name][outcome]['y'].extend(add_hist)
                    if key_2nd is not None:
                        per_age_statistics[angle_type][outcome][age_name]['y2'].extend(f(add_2nd, distance_mm, angle_type))
                        flipped_per_age_statistics[angle_type][age_name][outcome]['y2'].extend(f(add_2nd, distance_mm, angle_type))
                        # if len(add) != len(add_2nd):
                        #     print(len(add), len(add_2nd), angle_type, age_name, outcome)

    return per_age_statistics, flipped_per_age_statistics, per_age_for_hist_statistics, \
           flipped_per_age_for_hist_statistics, levels


def load_mat_dict_no_headers(full_path):
    v = load_mat_dict(full_path)
    v.pop("__header__")
    v.pop("__version__")
    v.pop("__globals__")
    return v


def fix_name(key):
    field2title = {'event_dur_sec': 'Event duration (s)',
                   'event_ibi_dur_sec_sum': 'Sum inter-bout-intervals (s)',
                   'event_ibi_dur_sec_mean': 'Mean inter-bout-intervals (s)',
                   'event_ibi_dur_sec_first': 'First inter-bout-interval duration (s)',
                   'event_ibi_dur_sec_last': 'Last inter-bout-interval duration (s)'}
    if key in field2title.keys():
        return field2title[key]
    return key.replace("_", " ").replace("mm sec", "(mm/sec)") \
        .replace("vel to fish max", "Max velocity").replace("vel to fish min", "Min velocity") \
        .replace("vel to fish mean", "Mean velocity").replace("distance", "Distance").replace("in mm", "(mm)") \
        .replace("diff from fish angle deg", "Relative angle (deg)")


def plot_all_paramecia_properties(plot_dir, outcomes_list=[], outcome_map={}, age_map={}, age_list=[],
                                  filter_keys=lambda k: True, is_event_data=False, is_fish_mean=False, split_also=False):
    plots = []
    rec = lambda d: list([v for k, v in d.items() if k != 'event_data'])[0]
    all_keys = rec(rec(rec(rec(paramecia_in_fov)))).keys()
    if is_event_data:
        all_keys = rec(rec(paramecia_in_fov))['event_data'].keys()
    print(all_keys)
    all_keys = [k for k in all_keys if filter_keys(k)]
    for key in all_keys:
        n = key + ("_per_fish" if is_fish_mean else "")
        print("sapir2", key, outcomes_list)
        #     todo is label pretty enough?
        if is_event_data:
            plots.extend(plot_event_property(general_filename=n, plot_dir=plot_dir, y_label="Paramecia",
                                             outcomes_list=outcomes_list, outcome_map=outcome_map, age_map=age_map,
                                             is_fish_mean=is_fish_mean,
                                             age_list=age_list, x_label=fix_name(key), key=key))
        else:
            plots.extend(plot_paramecia_property(general_filename=n, plot_dir=plot_dir, y_label="Paramecia",
                                                 outcomes_list=outcomes_list, outcome_map=outcome_map, age_map=age_map,
                                                 is_fish_mean=is_fish_mean, split_also=split_also,
                                                 age_list=age_list, x_label=fix_name(key), key=key))
    vel_keys = all_keys  # [k for k in all_keys if "velocity" in k and ("min" in k or "max" in k) and "mean" not in k and "to_fish" in k]
    print(vel_keys)
    if len(vel_keys) == 2 and not is_event_data:  # todo fixme
        logging.info("plot_all_paramecia_properties - rel plot between velocities")
        plots.extend(plot_paramecia_property(general_filename="angle_dist"+ ("_per_fish" if is_fish_mean else ""), plot_dir=plot_dir,
                                             outcomes_list=outcomes_list, outcome_map=outcome_map, age_map=age_map,
                                             age_list=age_list, is_fish_mean=is_fish_mean,
                                             x_label=fix_name(vel_keys[0]), y_label=fix_name(vel_keys[1]),
                                             key=vel_keys[0], key_2nd=vel_keys[1]))

    logging.info("plot_all_paramecia_properties - done")
    logging.info("Running over {0}".format(all_keys))

    # total_df, per_fov_df = paramecia_properties_df(all_keys, paramecia_data=paramecia_data, paramecia_in_fov=paramecia_in_fov)
    #
    # common_sns_args = {'aspect': .7, 'height': 6}
    # hist_args = {'common_norm': False, 'kde': True, 'common_norm': False, 'kind': "hist"}  # 'discrete': True,
    # violin_args = {'kind': "violin", 'scale_hue': False, 'scale': "count", 'cut': 0}

    # for key in all_keys:
    #     general_filename = key
    #     y_label = key.replace("_", " ").replace("mm sec", "(mm/sec)").replace('in mm', '(mm)')
    #     for angle_type in per_fov_df['Angle (type)'].unique():
    #         curr_title = angle_type + " angle"
    #         d = per_fov_df[per_fov_df['Angle (type)'] == angle_type]
    #         d_total = per_fov_df[per_fov_df['Distance (mm)'] == per_fov_df['Distance (mm)'].max()]
    #         d_total = d_total[d_total['Angle (type)'] == angle_type]
    #         d_total[key] = d_total[key].astype(float)
    #         try:
    #             if key == 'distance_from_fish_in_mm':
    #                 hist_args['binwidth'] = 0.4
    #
    #             for stat in ["probability"]:
    #                 g = sns.displot(hue='Outcome', x=key, col="Age (dpf)",
    #                                 data=d_total, stat=stat, **hist_args, **common_sns_args)
    #                 g.tight_layout()
    #                 g.fig.subplots_adjust(top=0.9)  # adjust the Figure
    #                 g.fig.suptitle(curr_title)
    #                 image_fish_scatter_path = os.path.join(plot_dir,
    #                                                        save_fig_fixname(add_name + general_filename + "_" + angle_type +
    #                                                            "_total_hist_" + stat + ".png"))
    #                 g.savefig(image_fish_scatter_path)
    #                 plots.extend([image_fish_scatter_path])
    #                 plt.close()
    #
    #                 g = sns.displot(hue="Age (dpf)", x=key, col='Outcome',
    #                                 data=d_total, stat=stat, **hist_args)
    #                 g.tight_layout()
    #                 g.fig.subplots_adjust(top=0.9)  # adjust the Figure
    #                 g.fig.suptitle(curr_title)
    #                 image_fish_scatter_path = os.path.join(plot_dir,
    #                                                        save_fig_fixname(add_name + general_filename + "_" + angle_type +
    #                                                        "_total_flip_hist_" + stat + ".png"))
    #                 g.savefig(image_fish_scatter_path)
    #                 plots.extend([image_fish_scatter_path])
    #                 plt.close()
    #
    #         except Exception as e:
    #             print(e)
    #             traceback.print_tb(e.__traceback__)

    return plots


def plot_event_property(general_filename, plot_dir, x_label, y_label, key, key_2nd=None, title_add="", is_fish_mean=False,
                        is_per_angle=True, outcomes_list=[], outcome_map={}, age_map={}, age_list=[]):
    density_args = {'y_label': y_label, 'x_label': x_label, 'dpi': dpi, 'is_legend_outside': True,
                    'all_together': False,
                    'fig_size': FIG_SIZE, 'colormap': get_color,
                    'is_outer_keys_subplots': False,
                    'palette': default_colors}
    plots = []
    per_age_statistics, flipped_per_age_statistics = paramecia_event_data_dict(key=key, key_2nd=None,
                                                                               is_fish_mean=is_fish_mean)
    curr_name = general_filename
    for is_flip in [True, False] if len(age_list) > 1 else [False]:  # flip = [age][outcome]
        n = curr_name + "_flip" if is_flip else curr_name
        plots.extend(plot_scatter_bar(add_name, age_names=age_list, outcome_names=outcomes_list,
                                      counters=flipped_per_age_statistics,
                                      with_lines=not is_flip,
                                      is_cut_bottom=key in ["diff_from_fish_angle_deg",
                                                            "distance_from_fish_in_mm", "field_angle_deg"],
                                      is_significant_only=not parameters.is_combine_age,
                                      with_p_values=parameters.is_combine_age,
                                      key="y", dpi=dpi,
                                      is_combine_age=parameters.is_combine_age,
                                      curr_title=fix_name(key), with_title=True,
                                      plot_dir=plot_dir,
                                      filename=n + "_with_scatter.pdf",
                                      y_label=fix_name(key), split_per_outcome=False,
                                      is_subplot=True, with_sup_title=False, p_val_color="k",
                                      title_rename_map=age_map if not is_flip else outcome_map,
                                      xlabel_rename_map=age_map if is_flip else outcome_map,
                                      colormap=get_color))

    return plots


def plot_paramecia_property(general_filename, plot_dir, x_label, y_label, key, key_2nd=None, title_add="", is_fish_mean=False,
                            is_per_angle=True, outcomes_list=[], outcome_map={}, age_map={}, age_list=[], split_also=False):
    density_args = {'y_label': y_label, 'x_label': x_label, 'dpi': dpi, 'is_legend_outside': True,
                    'all_together': False,
                    'fig_size': FIG_SIZE, 'colormap': get_color,
                    'is_outer_keys_subplots': False,
                    'palette': default_colors}
    scatter_args = {'is_significant_only': not parameters.is_combine_age, 'with_p_values': parameters.is_combine_age,
                    'dpi': dpi, 'key': "y", 'p_val_color': 'k',
                    'is_combine_age': parameters.is_combine_age,
                    'with_title': True, 'with_sup_title': False, 'curr_title': "",
                    'y_label': x_label, 'plot_dir': plot_dir}

    plots = []

    m = {'0-1.5mm': ['a_0_1.5mm'],
         '1.5-3mm': ['a_1.5_3mm'],
         '0-3mm': ['a_0_3mm'],
         '0-3.5mm': ['a_0_3.5mm'],
         '0-4mm': ['a_0_4mm'],
         '1.5-3.5mm': ['a_1.5_3.5mm'],
         '1.5-4mm': ['a_1.5_4mm']}

    # add figures with inner comparison of distances
    per_age_distance_statistics = paramecia_distanecs_data_dict(m, key=key, key_2nd=None, is_fish_mean=is_fish_mean)
    if key_2nd is not None:
        per_age_distance_statistics = paramecia_distanecs_data_dict(m, key=key, key_2nd=key_2nd,
                                                                    is_fish_mean=is_fish_mean)
    get_color_adapted = lambda dist, age: get_color(age, outcome)
    density_adapted = density_args.copy()
    density_adapted['colormap'] = get_color_adapted
    for angle_type in per_age_distance_statistics.keys():
        for outcome in per_age_distance_statistics[angle_type].keys():
            curr_name = general_filename + "_" + angle_type.replace("-", "_") + "_" + outcome.replace("-", "_")
            outer_keys_map = age_map
            inner_keys_map = {}
            outer_keys_list = age_list
            inner_keys_list = list(m.keys())
            what = "age"
            plots.extend(plot_densities(curr_name, plot_dir, per_age_distance_statistics[angle_type][outcome],
                                        is_kde=True, is_hist=False, is_box=False,
                                        outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                                        outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map,
                                        what=what, title_add=title_add, add_name=add_name, **density_adapted))
            plots.extend(plot_scatter_bar(add_name, age_names=outer_keys_list, outcome_names=inner_keys_list,
                                          filename=curr_name + "_with_scatter.pdf",
                                          counters=per_age_distance_statistics[angle_type][outcome],
                                          with_lines=True,  # true is age outside
                                          split_per_outcome=True, is_subplot=False,
                                          title_rename_map=outer_keys_map,
                                          xlabel_rename_map=inner_keys_map,
                                          colormap=get_color_adapted,
                                          **scatter_args))

    for (name, distance_values) in m.items():
        per_age_statistics, flipped_per_age_statistics, _, _, _ = \
            paramecia_fov_dict(key=key, hist_key=None, only_distances=distance_values, is_fish_mean=is_fish_mean)
        if key_2nd is not None:
            per_age_statistics, flipped_per_age_statistics, _, _, _ = \
                paramecia_fov_dict(key=key, hist_key=None, key_2nd=key_2nd, is_fish_mean=is_fish_mean)

        curr_name = general_filename + "_" + name.replace("-", "_")
        for angle_type in per_age_statistics.keys():
            title_add = " (angle {0})".format(angle_type)
            for is_flip in [True, False] if len(age_list) > 1 else [False]:  # flip = [age][outcome]
                n = curr_name + "_flip_" + angle_type if is_flip else curr_name + "_" + angle_type
                what = "age" if is_flip else "outcome"
                if is_per_angle:
                    d_hist = flipped_per_age_statistics[angle_type] if is_flip else per_age_statistics[angle_type]
                else:
                    d_hist = flipped_per_age_statistics if is_flip else per_age_statistics

                outer_keys_map = age_map if is_flip else outcome_map
                inner_keys_map = age_map if not is_flip else outcome_map
                outer_keys_list = outcomes_list if not is_flip else age_list
                inner_keys_list = outcomes_list if is_flip else age_list
                if key_2nd is not None:
                    density_args['is_legend_outside'] = False
                    density_args['is_outer_keys_subplots'] = True
                    plots.extend(plot_densities(n, plot_dir, d_hist, is_kde=False, is_hist=False, is_box=False,
                                                is_rel=True, is_joint=False,
                                                outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                                                outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map,
                                                what="age" if not is_flip else "outcome", f=lambda x: np.abs(x),
                                                title_add=title_add, add_name=add_name, **density_args))
                    density_args['is_outer_keys_subplots'] = False
                    plots.extend(plot_densities(n, plot_dir, d_hist, is_kde=False, is_hist=False, is_box=False,
                                                is_rel=True, is_joint=True,
                                                outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                                                outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map,
                                                what="age" if not is_flip else "outcome", f=lambda x: np.abs(x),
                                                title_add=title_add, add_name=add_name, **density_args))
                    density_args['is_legend_outside'] = True
                    density_args['is_outer_keys_subplots'] = True
                else:
                    print("sapir3", outer_keys_list, inner_keys_list)
                    # plots.extend(
                    #     plot_densities(n, plot_dir, d_hist, is_kde=False, is_hist=False, is_box=False, is_cdf=True,
                    #                    outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                    #                    outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map,
                    #                    what=what, title_add=title_add, add_name=add_name, **density_args))
                    plots.extend(plot_densities(n, plot_dir, d_hist, is_kde=True, is_hist=False, is_box=False,
                                                outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                                                outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map,
                                                what=what, title_add=title_add, add_name=add_name, **density_args))
                    # plots.extend(plot_densities(n, plot_dir, d_hist, is_kde=False, is_hist=True, is_box=False,
                    #                             outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                    #                             outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map,
                    #                             what=what, title_add=title_add, add_name=add_name, **density_args))
                    # plots.extend(plot_densities(n, plot_dir, d_hist, is_kde=False, is_hist=False, is_box=False,
                    #                             outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                    #                             outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map,
                    #                             what="age" if not is_flip else "outcome",
                    #                             title_add=title_add, add_name=add_name, **density_args))
                    plots.extend(plot_scatter_bar(add_name, age_names=age_list, outcome_names=outcomes_list,
                                                  counters=flipped_per_age_statistics[angle_type],
                                                  with_lines=not is_flip,
                                                  is_cut_bottom=key in ["diff_from_fish_angle_deg",
                                                                        "distance_from_fish_in_mm", "field_angle_deg"],
                                                  filename=n + "_with_scatter.pdf",
                                                  split_per_outcome=False, is_subplot=True,
                                                  title_rename_map=age_map if not is_flip else outcome_map,
                                                  xlabel_rename_map=age_map if is_flip else outcome_map,
                                                  colormap=get_color,
                                                  **scatter_args))

                    if split_also:
                        plots.extend(plot_scatter_bar(add_name, age_names=age_list, outcome_names=outcomes_list,
                                                      counters=flipped_per_age_statistics[angle_type],
                                                      with_lines=not is_flip,
                                                      is_cut_bottom=key in ["diff_from_fish_angle_deg",
                                                                            "distance_from_fish_in_mm", "field_angle_deg"],
                                                      filename=n + "_with_scatter.pdf",
                                                      split_per_outcome=True, is_subplot=False,
                                                      title_rename_map=age_map if not is_flip else outcome_map,
                                                      xlabel_rename_map=age_map if is_flip else outcome_map,
                                                      colormap=get_color,
                                                      **scatter_args))

    return plots


def get_color(name, name_2):
    default_colors = sns.color_palette('colorblind')
    color_map = {'hit:5-7': default_colors[0], 'hit:14-15': default_colors[2],
                 'abort:5-7': default_colors[1], 'abort:14-15': default_colors[8],
                 'miss:5-7': default_colors[4], 'miss:14-15': default_colors[5]}  # todo fix miss
    if 'abort' in name.lower() or "hit" in name.lower() or "miss" in name.lower():
        key = name.lower().replace("-","_").replace("hit_spit", "hit") + ":" + name_2.lower().replace("a_", "").replace("_", "-")
    elif 'abort' in name_2.lower() or "hit" in name_2.lower() or "miss" in name_2.lower():
        key = name_2.lower().replace("-","_").replace("hit_spit", "hit") + ":" + name.lower().replace("a_", "").replace("_", "-")
    else:
        return default_colors[0 if '5_7' in name_2 else 2]  # todo
    return color_map[key]


def plot_n_paramecia_distributions(plot_dir, data, general_filename="n_paramecia_fov", x_label="Distance in mm",
                                   y_label="Paramecia", dpi=600, is_fish_mean=False,
                                   outcomes_list=[], outcome_map={}, age_map={}, age_list=[]):
    def normalize(y_v, distance_m, angle):
        print(distance_m, angle)
        return y_v
    plots = []
    density_args = {'y_label': y_label, 'x_label': x_label, 'dpi': dpi, 'is_legend_outside': False,
                    'all_together': False,
                    'fig_size': FIG_SIZE, 'colormap': get_color,  # default_colors[ind*2 + int(i)],
                    'is_outer_keys_subplots': True}
    #
    # m = {'0-3mm': ['a_0_3mm'],
    #      '3-6mm': ['a_3_6mm'],
    #      '0-6mm': ['a_0_6mm']}
    #
    # m = {
    #     # '0-3mm': ['a_0_1mm', 'a_1_2mm', 'a_2_3mm'],
    #     #  '3-6mm': ['a_3_4mm', 'a_4_5mm', 'a_5_6mm'],
    #      '0-6mm': ['a_0_1mm', 'a_1_2mm', 'a_2_3mm', 'a_3_4mm', 'a_4_5mm', 'a_5_6mm']}

    # m = {'0-2mm': ['a_0_2mm'],
    #      '0-1.75mm': ['a_0_1_75mm'],
    #      '1.5-3mm': ['a_1_5_3mm'],
    #      '1.74-4mm': ['a_1_75_4mm'],
    #      '0-3mm': ['a_0_3mm'],
    #      }

    #
    # for from_v, to_v in [[0, 1.5], [1.5, 3], [0, 3], [3, 9],
    #                      [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
    #                      [0, 0.125], [0.25, 0.375], [0.5, 0.5 + 0.125], [0.75, 0.75 + 0.125], [1, 1.125],
    #                      [1.25, 1.25 + 0.125], [1.5, 1.5 + 0.125], [1.75, 1.75 + 0.125], [2, 2.125],
    #                      [2.25, 2.25 + 0.125], [2.5, 2.5 + 0.125], [2.75, 2.75 + 0.125], [3, 3.125]]:
    rec = lambda d: list([v for k, v in d.items() if k != 'event_data'])[0]

    distances_in_data = [k.replace("_dot_", ".") for k in rec(rec(data)).keys() if k != 'event_data']
    for ignore in ['a_0_1mm', 'a_1_2mm', 'a_2_3mm', 'a_0_1.5mm', 'a_1.5_3mm', 'a_0_3mm', 'a_3_9mm']:
        if ignore in distances_in_data:
            distances_in_data.remove(ignore)
    distances_in_data = sorted(distances_in_data, key=distance_name_to_value)
    if not (sum(np.diff([float(distance_name_to_value(k)) for k in distances_in_data]) == 0.5) >= 5 or
            sum(np.diff([float(distance_name_to_value(k)) for k in distances_in_data]) == 0.25) >=5):
        logging.error("Bad distance calculation in n paramecia {0}".format(distances_in_data))
        return

    m = {'0-9mm': distances_in_data}
    logging.info(m)

    k, h_k = 'n_paramecia', 'n_paramecia'
    for (distance_name, distance_values) in m.items():
        per_age_statistics, flipped_per_age_statistics, _, _, _ = \
            paramecia_fov_dict(key=k, hist_key=None, only_distances=distance_values, f=normalize, data=data,
                               is_fish_mean=is_fish_mean)
        if fake_n_para is not None:
            fake_stats, flipped_fake_stats, _, _, _ = \
                paramecia_fov_dict(key=k, hist_key=None, only_distances=distance_values, f=normalize, data=fake_n_para['fake_all']['n_para_30'])

        # for k, h_k in [('all_diff', 'n_paramecia_diff')]:
        # per_age_statistics, flipped_per_age_statistics, per_age_for_hist_statistics, \
        #     flipped_per_age_for_hist_statistics, levels = \
        #     paramecia_fov_dict(key=k, hist_key=h_k)
        # per_age_statistics_comb, flipped_per_age_statistics_comb, per_age_for_hist_statistics_comb, \
        #     flipped_per_age_for_hist_statistics_comb, _ = \
        #     paramecia_fov_dict(key=k, hist_key=h_k, only_distances=distance_values)#paramecia_fov_dict_combined_distances(key=k)
        curr_name = general_filename + distance_name.replace("-", "_")
        for angle_type in per_age_statistics.keys():
            try:
                title_add = " (angle {0})".format(angle_type)
                for is_flip in [True]:  # flip = [age][outcome]
                    n = curr_name + "_flip_" + angle_type if is_flip else curr_name + "_" + angle_type
                    what = "age" if is_flip else "outcome"
                    d = flipped_per_age_statistics[angle_type] if is_flip else per_age_statistics[angle_type]
                    d_fake=None
                    if fake_n_para is not None and angle_type in flipped_fake_stats.keys():
                        d_fake = flipped_fake_stats[angle_type] if is_flip else fake_stats[angle_type]
                    outer_keys_map = age_map if is_flip else outcome_map
                    inner_keys_map = age_map if not is_flip else outcome_map
                    outer_keys_list = outcomes_list if not is_flip else age_list
                    inner_keys_list = outcomes_list if is_flip else age_list
                    density_args_no_hist = density_args.copy()
                    density_args_no_hist["y_label"] = "number of paramecia"
                    plots.extend(plot_densities(n, plot_dir, d, is_kde=False, is_hist=False, title_add=title_add,
                                                outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                                                outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map, d_fake=d_fake,
                                                what=what, add_name=add_name + add_heatmap_name, is_cum_line=False,
                                                **density_args_no_hist))
                    plots.extend(plot_densities(n, plot_dir, d, is_kde=False, is_hist=False, title_add=title_add,
                                                outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                                                outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map, d_fake=d_fake,
                                                what=what, add_name=add_name + add_heatmap_name, is_cum_line=True,
                                                **density_args_no_hist))
                    # plots.extend(plot_densities(n, plot_dir, d_hist, is_kde=True, is_hist=False, title_add=title_add,
                    #                             outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                    #                             outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map,
                    #                             what=what, add_name=add_name + add_heatmap_name, **density_args))
                    # plots.extend(plot_densities(n, plot_dir, d_hist, is_kde=False, is_hist=True, title_add=title_add,
                    #                             outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                    #                             outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map,
                    #                             what=what, add_name=add_name + add_heatmap_name, **density_args))
                    # plots.extend(plot_densities(n, plot_dir, d_hist, is_kde=True, is_hist=False, is_cdf=False, title_add=title_add,
                    #                             outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                    #                             outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map,
                    #                             what=what, add_name=add_name + add_heatmap_name, **density_args))
                    # plots.extend(plot_densities(n, plot_dir, d_hist, is_kde=False, is_hist=False, is_cdf=False, title_add=title_add,
                    #                             outer_keys_list=outer_keys_list, inner_keys_list=inner_keys_list,
                    #                             outer_keys_map=outer_keys_map, inner_keys_map=inner_keys_map,
                    #                             what=what, add_name=add_name + add_heatmap_name, **density_args))
                    # # for distance_name in flipped_per_age_statistics_comb[angle_type].keys():
                    # plots.extend(plot_scatter_bar(add_name, age_names=age_list, outcome_names=outcomes_list,
                    #                               counters=flipped_per_age_statistics_comb[angle_type], #[distance_name],
                    #                               with_lines=not is_flip,
                    #                               is_cut_bottom=True,
                    #                               is_significant_only=not parameters.is_combine_age,
                    #                               with_p_values=parameters.is_combine_age,
                    #                               key="y", dpi=dpi,
                    #                               is_combine_age=parameters.is_combine_age,
                    #                               curr_title="",
                    #                               plot_dir=plot_dir,
                    #                               filename=n + "_with_scatter_{0}.pdf".format(distance_name.replace("-","_")),
                    #                               y_label="Number of paramecia", split_per_outcome=False,
                    #                               is_subplot=True, with_sup_title=False, p_val_color="k",
                    #                               title_rename_map=age_map if not is_flip else outcome_map,
                    #                               xlabel_rename_map=age_map if is_flip else outcome_map,
                    #                               colormap=lambda ind: default_colors[int(ind)]))
            except Exception as e:
                print(e)
                traceback.print_tb(e.__traceback__)
    # print(plots)


def plot_metadata(parameters, dpi=300, outcome_names_sub=['hit-spit', "miss", 'abort'], key="per_fish"):
    fraction_images_paths = []
    if parameters.is_combine_age:
        age_names = counters.keys()
        age_names = sorted(age_names, key=lambda k: int(k.split("-")[0]))
    else:
        age_names = sorted(counters.keys(), key=lambda k: int(k) if k != 'all' else 10000)
    outcome_names = counters[sorted([k for k in counters.keys() if counters[k] != {}])[0]].keys()
    color_map = {'hit-spit': default_colors[0], 'abort': default_colors[1], "miss": default_colors[2]}
    for i in [1]:
        with_lines = True if i == 2 else False
        add = "lines_" if with_lines else ""
        is_significant_only = not parameters.is_combine_age and not with_lines
        image_stats2_path = plot_scatter_bar(add_name, age_names, outcome_names_sub, counters, with_lines=False,
                                             is_significant_only=is_significant_only, max_value=1.1,
                                             with_p_values=parameters.is_combine_age,  # , with_p_values=not with_lines
                                             key="per_fish", dpi=dpi, is_combine_age=parameters.is_combine_age,
                                             curr_title="Outcome fraction across age", plot_dir=metadata,
                                             filename=add + "_sep_event_outcome_fraction_with_scatter.pdf",
                                             y_label="Fraction", split_per_outcome=True,
                                             is_subplot=False, with_sup_title=False, p_val_color="k",
                                             title_rename_map=outcome_map, colormap=get_color)
        fraction_images_paths.extend(image_stats2_path)
        image_stats2_path = plot_scatter_bar(add_name, age_names, outcome_names_sub, counters, with_lines=True,
                                             is_significant_only=is_significant_only, max_value=1.1,
                                             with_p_values=parameters.is_combine_age,  # , with_p_values=not with_lines
                                             key="per_fish", dpi=dpi, is_combine_age=parameters.is_combine_age,
                                             curr_title="Outcome fraction across age", plot_dir=metadata,
                                             filename=add + "_sep_event_outcome_fraction_with_scatter.pdf",
                                             y_label="Fraction", split_per_outcome=False, xlabel_rename_map=outcome_map,
                                             is_subplot=True, with_sup_title=False, p_val_color="k",
                                             title_rename_map=age_map, colormap=get_color)

        for age_c in counters.keys():
            fig1, ax1 = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=dpi)
            sizes, labels, colorsarray = [], [], []
            for outcome in sorted(counters[age_c].keys()):
                y = counters[age_c][outcome][key]
                labels.append(outcome_map.get(outcome))
                sizes.append(np.nanmean(y))
                colorsarray.append(get_color("a_5_7", outcome))
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colorsarray,counterclock=False)
            image_fish_scatter_path = os.path.join(metadata,
                                                   save_fig_fixname(age_c + "_same_color_outcome_fraction_pie.jpg"))
            plt.savefig(image_fish_scatter_path)
            print(image_fish_scatter_path)

    print(fraction_images_paths)



def calc_outcomes_df_fractions(remove_zero=False, angles=["forward", "narrow", "front", "front_sides"]):
    outcome_name = lambda outcome: outcome_map.get(outcome, outcome)
    age_name = lambda age: age_map.get(age, age)
    df_total = None
    for age in age_list:
        for angle in angles:
            paired_outcomes = [(p1, p2, outcomes_list[p1], outcomes_list[p2])
                               for p1 in range(len(outcomes_list)) for p2 in range(p1 + 1, len(outcomes_list))]
            for (p1, p2, outcome1, outcome2) in paired_outcomes:
                m = {'0-2mm': ["d_1_0_mm", "d_2_0_mm"], '2-5mm': ["d_3_0_mm", "d_4_0_mm", "d_5_0_mm"]}
                for (name, distances_list) in m.items():
                    outcome1_freq = np.zeros(20)
                    outcome2_freq = np.zeros(20)
                    tot_arr = np.zeros(20)
                    for distance in distances_list:
                        values_outcome1 = Counter(paramecia_in_fov[age][outcome1][distance][angle]["all_diff"])
                        values_outcome2 = Counter(paramecia_in_fov[age][outcome2][distance][angle]["all_diff"])
                        unique_keys = np.unique(list(values_outcome1.keys()) + list(values_outcome2.keys()))
                        for key in unique_keys:
                            outcome1_freq[int(key)] += values_outcome1.get(key, 0)
                            outcome2_freq[int(key)] += values_outcome2.get(key, 0)
                            tot_arr[int(key)] += (values_outcome1.get(key, 0) + values_outcome2.get(key, 0))
                    tot_arr[tot_arr == 0] = np.nan
                    df1 = pd.DataFrame(outcome1_freq, columns=["Sum".format(outcome_name(outcome1))])
                    df1.insert(0, "Frequency", outcome1_freq / tot_arr)
                    df1.insert(0, "Total", tot_arr)
                    df1.insert(0, "Outcome", outcome_name(outcome1))
                    df1.insert(0, "Number of paramecia", np.arange(0, len(outcome1_freq)))
                    outcome1_freq /= tot_arr
                    df1.insert(0, "SEM", np.std(outcome1_freq) / np.sqrt(len(tot_arr)))
                    df2 = pd.DataFrame(outcome2_freq, columns=["Sum".format(outcome_name(outcome2))])
                    df2.insert(0, "Frequency", outcome2_freq / tot_arr)
                    df2.insert(0, "Total", tot_arr)
                    df2.insert(0, "Outcome", outcome_name(outcome2))
                    df2.insert(0, "Number of paramecia", np.arange(0, len(outcome2_freq)))
                    outcome2_freq /= tot_arr
                    # should ignore zero?
                    df2.insert(0, "SEM", np.std(outcome2_freq) / np.sqrt(len(tot_arr)))
                    df = pd.concat([df1, df2])
                    df.insert(0, "Outcome pair", "{0}:{1}".format(outcome_name(outcome1), outcome_name(outcome2)))
                    df.insert(0, "Distance (mm)", name)
                    df.insert(0, "Angle", angle)
                    df.insert(0, "Age (dpf)", age_name(age))
                    if df_total is None:
                        df_total = df
                    else:
                        df_total = pd.concat([df_total, df])
    print(df_total)
    if remove_zero:
        return df_total.loc[df_total.Frequency > 0, :]
    return df_total


def plot_all_heatmaps(outcome_name, age_name, outs=[".pdf", ".jpg"]):
    FIG_SIZE_H = (FIG_SIZE[1] * 1.3, FIG_SIZE[1])
    for _, age in tqdm(enumerate(age_list)):
        add = ""  # if age is None else " {0} age".format(age)
        age_key = "all" if age is None else str(age).replace("a_", "age_")
        age_key_c = "all" if age is None else str(age).replace("a_", "").replace("_", "-")
        total_maps = copy.deepcopy(total_all_age[age_key])
        max_v = 0
        for i, key in tqdm(enumerate(outcomes_list)):
            print(key)
            counter = total_heatmap_counters[age_key_c][key.replace("_", "-")]
            name = outcome_name(key)
            if total_heatmap_counters[age_key_c][key.replace("_", "-")]["n_events"] > 0:
                np_map = np.array(total_maps[key]).astype(float).copy() / counter["n_events"]
            else:
                np_map = np.array(total_maps[key]).astype(float).copy()  # zero
            if True:  # parameters.gaussian:
                np_map = gaussian_filter(np_map, 5)
            max_v = max(max_v, np.max(np_map))

        # f, axes = plt.subplots(1, len(outcomes_list), figsize=FIG_SIZE_H, dpi=dpi, sharex=True, sharey=True)
        # if not isinstance(axes, list):
        #     axes = [axes]
        for i, key in tqdm(enumerate(outcomes_list)):
            f, axes = plt.subplots(1, 1, figsize=FIG_SIZE_H, dpi=dpi, sharex=True, sharey=True)
            if not isinstance(axes, list):
                axes = [axes]
            counter = total_heatmap_counters[age_key_c][key.replace("_", "-")]
            name = outcome_name(key)
            if total_heatmap_counters[age_key_c][key.replace("_", "-")]["n_events"] > 0:
                np_map = np.array(total_maps[key]).astype(float).copy() / counter["n_events"]
            else:
                np_map = np.array(total_maps[key]).astype(float).copy()  # zero
            if True:  # parameters.gaussian:
                np_map = gaussian_filter(np_map, 5)
            np_map /= max_v
            heatmap_plot(np_map=np_map, name="v2_" + age, title=name, plot_dir=heatmap_folder, ax=axes[0], max_val=1,
                         with_cbar=True, f=f)  # i!=0)
            if True:  # i == 0:
                axes[0].set_ylabel("Distance from fish (mm)")
            axes[0].set_xlabel("Distance from fish (mm)")

            for o in outs:
                image_path = os.path.join(heatmap_folder, save_fig_fixname(
                    age.replace(" ", "_") + key.replace(" ", "_") + "_norm_target" + o))
                print(image_path)
                try:
                    plt.savefig(image_path)
                except Exception as e:
                    print(e)
                    traceback.print_tb(e.__traceback__)
            plt.close()

        # f, axes = plt.subplots(1, len(outcomes_list), figsize=FIG_SIZE_H, dpi=dpi, sharex=True, sharey=True)
        # if not isinstance(axes, list):
        #     axes = [axes]
        for i, key in tqdm(enumerate(outcomes_list)):
            f, axes = plt.subplots(1, 1, figsize=FIG_SIZE_H, dpi=dpi, sharex=True, sharey=True)
            if not isinstance(axes, list):
                axes = [axes]
            counter = total_heatmap_counters[age_key_c][key.replace("_", "-")]
            name = outcome_name(key)
            if total_heatmap_counters[age_key_c][key.replace("_", "-")]["n_events"] > 0:
                np_map = np.array(total_maps[key]).astype(float).copy() / counter["n_events"]
            else:
                np_map = np.array(total_maps[key]).astype(float).copy()  # zero
            # np_map /= np.max(np_map)
            if True:  # parameters.gaussian:
                np_map = gaussian_filter(np_map, 5)
            heatmap_plot(np_map=np_map, name="v2_" + age, title=name, plot_dir=heatmap_folder, ax=axes[0],
                         with_cbar=True, f=f)  # i!=0)
            if True:  # i == 0:
                axes[0].set_ylabel("Distance from fish (mm)")
            axes[0].set_xlabel("Distance from fish (mm)")

            for o in outs:
                image_path = os.path.join(heatmap_folder, save_fig_fixname(
                    age.replace(" ", "_") + key.replace(" ", "_") + "_usual_target" + o))
                try:
                    plt.savefig(image_path)
                    print(image_path)
                except Exception as e:
                    print(e)
                    traceback.print_tb(e.__traceback__)
            plt.close()


def get_maps_for_plots(parameters):
    default_colors = sns.color_palette('colorblind')
    # [sns.color_palette('colorblind')[1], sns.color_palette('colorblind')[8]]
    if parameters.outcome_map_type == OutcomeMapType.hit_miss_abort_es_abort_noes:
        outcomes_list = ["hit_spit", "abort,escape"]  # todo in counters it's hit-spit
        outcome_map = {"hit_spit": "hit", "hit-spit":  "hit", "abort,escape": "abort"}
        # outcomes_list = ["abort,no_escape", "abort,escape"]  # todo in counters it's hit-spit
        # outcome_map = {"hit_spit": "hit"}
    else:
        outcomes_list = ["hit_spit", "miss", "abort"]  # todo in counters it's hit-spit
        # outcomes_list = ["hit_spit", "abort"]  # todo in counters it's hit-spit
        # outcomes_list = ["hit_spit"]  # todo in counters it's hit-spit
        # outcomes_list = ["abort"]  # todo in counters it's hit-spit
        outcome_map = {"hit_spit": "Hit", "hit-spit": "Hit", "miss": "Miss", "abort,escape": "Abort", "abort": "Abort"}
    age_list = ['a_5_7', 'a_14_15']
    age_map = {'a_5_7': '5-7', 'a_14_15': '14-15'}
    all_together = True

    if outcomes_list == ["abort"]:
        # default_colors = [sns.color_palette('colorblind')[1], sns.color_palette('colorblind')[3]]
        default_colors = [sns.color_palette('colorblind')[1], sns.color_palette('colorblind')[8],
                          sns.color_palette('colorblind')[9]]
    elif outcomes_list == ["hit_spit"]:
        # default_colors = [sns.color_palette('colorblind')[0], sns.color_palette('colorblind')[9]]
        default_colors = [sns.color_palette('colorblind')[0], sns.color_palette('colorblind')[2],
                          sns.color_palette('colorblind')[3]]
    else:
        default_colors = sns.color_palette('colorblind')
    if all_together:
        default_colors = [sns.color_palette('colorblind')[0], sns.color_palette('colorblind')[2],
                          sns.color_palette('colorblind')[1], sns.color_palette('colorblind')[8]]

    return outcomes_list, outcome_map, age_list, age_map, default_colors


if __name__ == '__main__':
    dpi = 300
    logging.basicConfig(level=logging.INFO)
    parameters, data_path, should_run_all_metadata_permutations = get_parameters(with_mat=False)

    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = '22'

    heatmap_folder, metadata_folder, heatmap_parent_folder, counters_json, heatmap_counters_json, heatmap_data_json, \
    fullpath_output_prefix, add_name, add_heatmap_name = \
        get_folder_and_file_names(data_path_=data_path, parameters=parameters, age_groups=parameters.age_groups)

    paramecia_in_fov = load_mat_dict_no_headers(fullpath_output_prefix + "_all_fov_per_events.mat")
    paramecia_mean_per_fish_in_fov = load_mat_dict_no_headers(fullpath_output_prefix + "_all_fov_per_fish.mat")
    fake_n_para = load_mat_dict_no_headers(os.path.join(data_path, "random_n_paramecia_densities_20_to_70.mat"))

    counters = {}
    with open(counters_json, 'r') as outfile:
        counters = json.load(outfile)
    if parameters.is_save_per_fish_heatmap:
        per_fish_paramecia_in_fov = \
            load_mat_dict_no_headers(fullpath_output_prefix + "_per_fish_paramecia_in_fov.mat")

    output_folder = os.path.join(data_path, "dataset_plots_presentation", "new_bout_target",
                                 str(parameters.feeding_type), parameters.heatmap_type.name)
    n_paramecia = os.path.join(output_folder, "n_para")
    metadata = os.path.join(output_folder, "Metadata")
    heatmap_folder = os.path.join(output_folder, "Heat")
    ibi_folder = os.path.join(output_folder, "ibi_dur")
    visual_folder = os.path.join(output_folder, "VisualLoad")
    velocities_folder = os.path.join(output_folder, "Velocities")
    create_dirs_if_missing([n_paramecia, heatmap_folder, metadata, ibi_folder])
    create_dirs_if_missing([visual_folder, velocities_folder])

    outcomes_list, outcome_map, age_list, age_map, default_colors = get_maps_for_plots(parameters=parameters)

    plot_metadata(parameters)

    for is_fish_mean in [False, True]:
        plot_all_paramecia_properties(visual_folder, outcomes_list=outcomes_list, outcome_map=outcome_map,
                                      age_list=age_list, age_map=age_map, is_fish_mean=is_fish_mean,
                                      filter_keys=lambda k: (("field" in k  and "neg" not in k) or ("dist" in k and "neg" not in k)))
        plot_all_paramecia_properties(n_paramecia, outcomes_list=outcomes_list, outcome_map=outcome_map,
                                      age_list=age_list, age_map=age_map, is_fish_mean=is_fish_mean, split_also=True,
                                      filter_keys=lambda k: ("n_para" in k))
        plot_all_paramecia_properties(ibi_folder, outcomes_list=outcomes_list, outcome_map=outcome_map,
                                      age_list=age_list, age_map=age_map, is_event_data=True, is_fish_mean=is_fish_mean,
                                      filter_keys=lambda k: ("event_dur_sec" in k or "event_ibi_dur_sec_last" in k))
        plot_all_paramecia_properties(ibi_folder, outcomes_list=outcomes_list, outcome_map=outcome_map,
                                      age_list=age_list, age_map=age_map, is_event_data=True, is_fish_mean=is_fish_mean,
                                      filter_keys=lambda k: ("event_ibi_dur_sec_first" in k or "event_ibi_dur_sec_2nd" in k))
        plot_all_paramecia_properties(velocities_folder, outcomes_list=outcomes_list, outcome_map=outcome_map,
                                      age_list=age_list, age_map=age_map, is_fish_mean=is_fish_mean,
                                      filter_keys=lambda k: ("vel_to" in k and ("neg" in k or "pos" in k or "mean_mm" in k)))

    plot_n_paramecia_distributions(n_paramecia, paramecia_in_fov, general_filename="n_paramecia_fov_part", x_label="Distance from fish (mm)",
                                   y_label="Paramecia", dpi=dpi, outcomes_list=outcomes_list, outcome_map=outcome_map,
                                   age_list=age_list, age_map=age_map, is_fish_mean=False)
    #
    # plot_n_paramecia_distributions(n_paramecia, paramecia_mean_per_fish_in_fov, general_filename="n_paramecia_per_fish_fov", x_label="Distance from fish (mm)",
    #                                y_label="Paramecia", dpi=dpi, outcomes_list=outcomes_list, outcome_map=outcome_map,
    #                                age_list=age_list, age_map=age_map, is_fish_mean=True)

    total_all_age = load_mat_dict(heatmap_data_json)
    with open(heatmap_counters_json, 'r') as outfile:
        total_heatmap_counters = json.load(outfile)

    outcome_name = lambda outcome: outcome_map.get(outcome, outcome)
    age_name = lambda age: age_map.get(age, age)
    print(outcomes_list)
    plot_all_heatmaps(outcome_name, age_name)


    #
    # df_total = calc_outcomes_df_fractions(remove_zero=True)
    # plot_dir = n_paramecia  # shahar: limit by 5 the total amount, diff outcomes, diff strike zones . How err?
    # # print(df_total)
    # for pair in df_total["Outcome pair"].unique():
    #     fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=dpi)  # FIG_SIZE
    #     df = df_total.loc[(df_total["Outcome pair"] == pair) & (df_total["Outcome"] == "abort")
    #                       & (df_total["Distance (mm)"] == "2-5mm")]
    #     g = sns.relplot(data=df, x="Number of paramecia", y="Frequency", hue="Angle", style="Angle",
    #                     col="Age (dpf)", kind="line", markers=True, dashes=False)
    #     axes = g.axes.flatten()
    #     for ax in axes:
    #         ax.axhline(0.5, ls='--', linewidth=1, color='gray')
    #     g.set_axis_labels("Number of paramecia", "Abort frequency").tight_layout(w_pad=0)
    #     image_path = os.path.join(plot_dir, save_fig_fixname(add_name + "abort_freq_all_angles.jpg"))
    #     try:
    #         plt.savefig(image_path)
    #     except Exception as e:
    #         print(e)
    #         traceback.print_tb(e.__traceback__)
    #     plt.close()
    #     fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=dpi)  # FIG_SIZE
    #     df = df_total.loc[(df_total["Outcome pair"] == pair) & (df_total["Outcome"] == "abort")
    #                       & (df_total["Distance (mm)"] == "2-5mm")
    #                       & ((df_total["Angle"] == "narrow") | (df_total["Angle"] == "forward"))]
    #     g = sns.relplot(data=df, x="Number of paramecia", y="Frequency", hue="Angle", style="Angle",
    #                     col="Age (dpf)", kind="line", markers=True, dashes=False)
    #     axes = g.axes.flatten()
    #     for ax in axes:
    #         ax.axhline(0.5, ls='--', linewidth=1, color='gray')
    #     g.set_axis_labels("Number of paramecia", "Abort frequency").tight_layout(w_pad=0)
    #     image_path = os.path.join(plot_dir, save_fig_fixname(add_name + "abort_freq_nar_fwd_angles.jpg"))
    #     try:
    #         plt.savefig(image_path)
    #     except Exception as e:
    #         print(e)
    #         traceback.print_tb(e.__traceback__)
    #     plt.close()
    # # df_total.to_csv(os.path.join(plot_dir, save_fig_fixname(add_name + "abort_freq.csv")))
    #
    # df.insert(0, "Outcome", [outcome_name(outcome1), outcome_name(outcome2)])
    # df.insert(0, "Outcome pair", "{0}:{1}".format(outcome_name(outcome1), outcome_name(outcome2)))
    # df.insert(0, "Distance (mm)", distance)
    # df.insert(0, "Age (dpf)", age_name(age))
    # print(df_total)
    # max_num = np.where(tot_arr < 5)[0][0]
    # max_num = 5 # mm
    # plt.errorbar(np.arange(max_num), abort_freq[:max_num], yerr=error, capsize=5)
