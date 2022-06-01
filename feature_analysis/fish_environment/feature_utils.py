import argparse
import copy
import glob
import json
import logging
import os
import re
import sys
import traceback
import warnings

import cv2
import skimage.measure as measure
from fpdf import FPDF
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance

from tqdm import tqdm
import numpy as np
import pandas as pd

from feature_analysis.fish_environment.env_utils import heatmap, \
    heatmap_per_event_type, HeatmapType, outcome_to_map, PlotsCMDParameters, OutcomeMapType, HeatmapNParameciaType, \
    CombineAgeGroups, FeedingType
from feature_analysis.fish_environment.fish_processed_data import FishAndEnvDataset, SingleFishAndEnvData, \
    ExpandedEvent, ParameciumRelativeToFish, pixels_mm_converters
from scripts.python_scripts.main_metadata import read_file
from utils.main_utils import create_dirs_if_missing
from utils.matlab_data_handle import save_mat_dict, load_mat_dict
from utils.video_utils import VideoFromRaw


class MyPDF(FPDF):
    def __init__(self, orientation='P', unit='mm', format='A4', footer_text=""):
        super().__init__(orientation=orientation, unit=unit, format=format)
        self.footer_text = footer_text

    def footer(self):
        self.set_y(-10)  # Go to 1 cm from bottom
        self.set_font('Arial', 'I', 6)
        self.set_text_color(128)  # Text color in gray
        self.cell(0, 5, self.footer_text, 0, 1, 'C')
        self.cell(0, 5, 'Page %s' % self.page_no(), 0, 1, 'C')


FIG_SIZE = (8, 6)  # 6 height


def save_fig_fixname(image_path):
    return image_path.replace("-", "_").replace(",", "_").replace(" ", "_")


def count_abort_fraction(dataset: FishAndEnvDataset, heatmap_n_paramecia_type: HeatmapNParameciaType,
                         valid_n_paramecia: int, age=None):
    heatmap_keys = parameters.combine_outcomes.keys()
    counters = {}
    for key in heatmap_keys:
        counters[key] = {"per_fish": [], "n_events": 0, 'n_events_per_fish': []}

    fish: SingleFishAndEnvData
    event: ExpandedEvent
    for fish in tqdm(dataset.fish_processed_data_set, desc="current fish", disable=True):
        if is_fish_filtered(parameters=parameters, fish=fish, age=age, age_list=[]):
            continue

        # outcomes_map = [outcome_to_map(_.outcome_str, parameters) for _ in fish.events]
        # event_outcomes = set([v for v, ignored in outcomes_map if not ignored])
        for key in heatmap_keys:
            if True:  # any([key in _ for _ in event_outcomes]):
                counters[key]["per_fish"].append(0)
                counters[key]["n_events_per_fish"].append(0)

        for key in heatmap_keys:
            for event in tqdm(fish.events, desc="current event", disable=True):
                event_key, ignored = outcome_to_map(event.outcome_str, parameters)
                if key == event_key and not ignored:
                    counters[key]["n_events"] += 1
                    counters[key]["per_fish"][-1] += 1
                    counters[key]["n_events_per_fish"][-1] += 1

            if len(counters[key]["per_fish"]) > 0:
                counters[key]["per_fish"][-1] /= len(fish.events)
            else:
                print("Fish {0} have no {1} events".format(fish.metadata.name, key))
    return counters


def get_y(data_dict, f=lambda x: x, k='y'):
    if k != '':
        if isinstance(data_dict[k], list) and len(data_dict[k]) > 0 and isinstance(data_dict[k][0], dict):
            return np.array(f(data_dict[k][0]['ibi__1']))
        elif isinstance(data_dict[k], dict):
            return np.array(f(data_dict[k]['ibi__1']))
        else:
            return np.array(f(data_dict[k]))
    return np.array(f(data_dict[k]))


def calc_paramecia_counts(dataset: FishAndEnvDataset, parameters: PlotsCMDParameters, age=None, age_list=[],
                          calc_per_fish=False):
    def to_list(v):
        if isinstance(v, (list, np.ndarray)) and np.array(v).shape != ():  # patch for nan
            return copy.deepcopy(v)
        if np.isnan(v):
            return [np.nan]
        return [v]

    if len(dataset.fish_processed_data_set) == 0:
        logging.error("Zero length for {0}".format(dataset.fish_processed_data_set))
        return {}, {}, {}, {}

    fish: SingleFishAndEnvData = dataset.fish_processed_data_set[0]

    empty_fields = {'velocity_towards_fish_mm_sec': [], 'velocity_to_fish_mean_mm_sec': [], 'velocity_to_fish_max_mm_sec': [], 'velocity_to_fish_min_mm_sec': [],
                    'orthogonal_velocity_mm_sec': [], 'orth_velocity_mean_mm_sec': [],'orth_velocity_max_mm_sec': [],'orth_velocity_min_mm_sec': [],
                    'field_angle_deg': [], 'field_angle_sum_deg': [], 'field_angle_mean_deg': [], 'field_angle_max_deg': [], 'field_angle_min_deg': [], 
                    'diff_from_fish_angle_deg': [], 'distance_from_fish_in_mm': []}

    fov_counters, fov_per_fish_counters = {}, {}
    counters, per_fish_counters = {}, {}  # todo is per fish needed? (counters is age group)
    for key in parameters.combine_outcomes.keys():
        fov_counters[key], fov_per_fish_counters[key] = {}, {}
        counters[key], per_fish_counters[key] = copy.deepcopy(empty_fields), {}
        for d in fish.distances_for_fov_in_mm:
            d_key = "d_{0}_mm".format(d).replace(".", "_dot_")
            fov_counters[key][d_key], fov_per_fish_counters[key][d_key] = {}, {}
            for angle_key in fish.angles_for_fov.keys():
                fov_counters[key][d_key][angle_key], fov_per_fish_counters[key][d_key][angle_key] = \
                    {'n_paramecia': 0, 'f_paramecia': 0, 'n_events': 0, 'all': [], 'all_diff': [],
                     'n_paramecia_diff': 0}, {}
                fov_counters[key][d_key][angle_key].update(copy.deepcopy(empty_fields))

    event: ExpandedEvent
    paramecium: ParameciumRelativeToFish
    for fish in tqdm(dataset.fish_processed_data_set,
                     desc="Para current fish age {0}".format(age if not parameters.is_combine_age else age_list)):
        if is_fish_filtered(parameters=parameters, fish=fish, age=age, age_list=age_list):
            continue

        if len(fish.events[0].paramecium.status_points) == 0:
            print("Error. fish {0} no paramecia".format(fish.name))

        paramecium: ParameciumRelativeToFish
        for event in tqdm(fish.events, desc="current event", disable=True):
            if event.is_inter_bout_interval_only:  # already IBI
                starting_bout_indices, ending_bout_indices = event.starting_bout_indices, event.ending_bout_indices
            else:
                starting_bout_indices, ending_bout_indices, err_code = ExpandedEvent.start_end_bout_indices(event)  # calc

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
                parameters.heatmap_type == HeatmapType.residuals) and \
                    np.isnan(event.paramecium.target_paramecia_index):
                logging.error("Paramecia in fov has nan target index for {0}".format(event.event_name))
                continue  # no data

            for key in fov_counters.keys():
                event_key, ignored = outcome_to_map(event.outcome_str, parameters)
                if key != event_key or ignored:
                    continue

                if fish.name not in per_fish_counters[key].keys():
                    per_fish_counters[key][fish.name] = copy.deepcopy(empty_fields)

                # todo ignore predictions

                # add general features
                paramecium = event.paramecium
                paramecia_indices = [_ for _ in range(0, paramecium.field_angle.shape[1])]
                if not np.isnan(event.paramecium.target_paramecia_index):
                    if parameters.heatmap_type == HeatmapType.residuals:
                        paramecia_indices.remove(event.paramecium.target_paramecia_index)
                    elif parameters.heatmap_type == HeatmapType.target_only:
                        paramecia_indices = [event.paramecium.target_paramecia_index]

                if len(paramecium.velocity_towards_fish.shape) == 1:  # patch (1d instead of 2d)
                    vel_towards = np.squeeze(paramecium.velocity_towards_fish[paramecia_indices])
                    vel_orthog = np.squeeze(paramecium.velocity_orthogonal[paramecia_indices])
                else:
                    vel_towards = np.squeeze(
                        paramecium.velocity_towards_fish[velocities_frame_number, paramecia_indices])
                    vel_orthog = np.squeeze(paramecium.velocity_orthogonal[velocities_frame_number, paramecia_indices])

                per_fish_counters[key][fish.name]['velocity_towards_fish_mm_sec'].extend(to_list(vel_towards))
                per_fish_counters[key][fish.name]['orthogonal_velocity_mm_sec'].extend(to_list(vel_orthog))
                per_fish_counters[key][fish.name]['field_angle_deg'].extend(to_list(
                    np.squeeze(paramecium.field_angle[frame_number, paramecia_indices])))
                per_fish_counters[key][fish.name]['diff_from_fish_angle_deg'].extend(to_list(
                    np.squeeze(paramecium.diff_from_fish_angle_deg[frame_number, paramecia_indices])))
                per_fish_counters[key][fish.name]['distance_from_fish_in_mm'].extend(to_list(
                    np.squeeze(paramecium.distance_from_fish_in_mm[frame_number, paramecia_indices])))

                counters[key]['velocity_towards_fish_mm_sec'].extend(to_list(vel_towards))
                counters[key]['orthogonal_velocity_mm_sec'].extend(to_list(vel_orthog))
                counters[key]['field_angle_deg'].extend(to_list(
                    np.squeeze(paramecium.field_angle[frame_number, paramecia_indices])))
                counters[key]['diff_from_fish_angle_deg'].extend(to_list(
                    np.squeeze(paramecium.diff_from_fish_angle_deg[frame_number, paramecia_indices])))
                counters[key]['distance_from_fish_in_mm'].extend(to_list(
                    np.squeeze(paramecium.distance_from_fish_in_mm[frame_number, paramecia_indices])))

                # add FOV features (4D matrix)
                prev_paramecia_indices = []
                for dist_ind, d_key in enumerate(fov_counters[key].keys()):
                    for angle_ind, angle_key in enumerate(fov_counters[key][d_key].keys()):
                        paramecia_status = \
                            event.paramecium.field_of_view_status[frame_number, paramecia_indices, angle_ind, dist_ind]
                        fov_counters[key][d_key][angle_key]['n_paramecia'] += np.sum(paramecia_status)
                        fov_counters[key][d_key][angle_key]['n_events'] += 1
                        fov_counters[key][d_key][angle_key]['all'].append(np.sum(paramecia_status))

                        if dist_ind > 0:
                            prev_paramecia_status = event.paramecium.field_of_view_status[frame_number,
                                                                                          paramecia_indices, angle_ind,
                                                                                          dist_ind - 1]
                            d = np.sum(paramecia_status) - np.sum(prev_paramecia_status)
                            fov_counters[key][d_key][angle_key]['all_diff'].append(d)
                            fov_counters[key][d_key][angle_key]['n_paramecia_diff'] += d

                        elif dist_ind == 0:
                            d = np.sum(paramecia_status)
                            fov_counters[key][d_key][angle_key]['all_diff'].append(d)
                            fov_counters[key][d_key][angle_key]['n_paramecia_diff'] += d

                        paramecia_indices2 = np.array(paramecia_indices)[paramecia_status == 1]
                        vel_towards_prev, vel_orthog_prev = [], []
                        if len(paramecium.velocity_towards_fish.shape) == 1:  # patch (1d instead of 2d)
                            vel_towards = np.squeeze(paramecium.velocity_towards_fish[paramecia_indices2])
                            vel_orthog = np.squeeze(paramecium.velocity_orthogonal[paramecia_indices2])
                            if dist_ind > 0 and len(prev_paramecia_indices) > 0:
                                vel_towards_prev = np.squeeze(paramecium.velocity_towards_fish[prev_paramecia_indices])
                                vel_orthog_prev = np.squeeze(paramecium.velocity_orthogonal[prev_paramecia_indices])
                        else:
                            vel_towards = np.squeeze(
                                paramecium.velocity_towards_fish[velocities_frame_number, paramecia_indices2])
                            vel_orthog = np.squeeze(
                                paramecium.velocity_orthogonal[velocities_frame_number, paramecia_indices2])
                            if dist_ind > 0 and len(prev_paramecia_indices) > 0:
                                vel_towards_prev = np.squeeze(
                                    paramecium.velocity_towards_fish[velocities_frame_number, prev_paramecia_indices])
                                vel_orthog_prev = np.squeeze(
                                    paramecium.velocity_orthogonal[velocities_frame_number, prev_paramecia_indices])

                        if len(paramecia_indices2) > 0:
                            if dist_ind == 0:
                                fov_counters[key][d_key][angle_key]['velocity_towards_fish_mm_sec'].extend(
                                    to_list(vel_towards))
                                fov_counters[key][d_key][angle_key]['orthogonal_velocity_mm_sec'].extend(
                                    to_list(vel_orthog))
                                fov_counters[key][d_key][angle_key]['field_angle_deg'].extend(to_list(
                                    np.squeeze(paramecium.field_angle[frame_number, paramecia_indices2])))
                                fov_counters[key][d_key][angle_key]['diff_from_fish_angle_deg'].extend(to_list(
                                    np.squeeze(paramecium.diff_from_fish_angle_deg[frame_number, paramecia_indices2])))
                                fov_counters[key][d_key][angle_key]['distance_from_fish_in_mm'].extend(to_list(
                                    np.squeeze(paramecium.distance_from_fish_in_mm[frame_number, paramecia_indices2])))
                            if dist_ind > 0 and len(prev_paramecia_indices) > 0:
                                vel_tow = to_list(np.setdiff1d(to_list(vel_towards), to_list(vel_towards_prev)))
                                vel_ort = to_list(np.setdiff1d(to_list(vel_orthog), to_list(vel_orthog_prev)))
                                field_ang = to_list(np.setdiff1d(to_list(np.squeeze(paramecium.field_angle[frame_number, paramecia_indices2])),
                                                                 to_list(np.squeeze(paramecium.field_angle[frame_number, prev_paramecia_indices]))))
                                diff_ang = to_list(np.setdiff1d(to_list(np.squeeze(paramecium.diff_from_fish_angle_deg[frame_number, paramecia_indices2])),
                                                                to_list(np.squeeze(paramecium.diff_from_fish_angle_deg[frame_number, prev_paramecia_indices]))))
                                dist_from = to_list(np.setdiff1d(to_list(np.squeeze(paramecium.distance_from_fish_in_mm[frame_number, paramecia_indices2])),
                                                                 to_list(np.squeeze(paramecium.distance_from_fish_in_mm[frame_number, prev_paramecia_indices]))))
                                fov_counters[key][d_key][angle_key]['velocity_towards_fish_mm_sec'].extend(vel_tow)
                                if len(vel_tow) > 0: # and vel_tow != [np.nan]:
                                    fov_counters[key][d_key][angle_key]['velocity_to_fish_mean_mm_sec'].append(np.nanmean(vel_tow))
                                    fov_counters[key][d_key][angle_key]['velocity_to_fish_max_mm_sec'].append(np.nanmax(vel_tow))
                                    fov_counters[key][d_key][angle_key]['velocity_to_fish_min_mm_sec'].append(np.nanmin(vel_tow))
                                # else:
                                    # for n in ['velocity_to_fish_mean_mm_sec', 'velocity_to_fish_max_mm_sec', 'velocity_to_fish_min_mm_sec']:
                                    #     fov_counters[key][d_key][angle_key][n].append(np.nan)
                                print(key, d_key, angle_key)
                                print(len(fov_counters[key][d_key][angle_key]['velocity_towards_fish_mm_sec']),
                                      fov_counters[key][d_key][angle_key]['velocity_towards_fish_mm_sec'])
                                print(len(fov_counters[key][d_key][angle_key]['velocity_to_fish_mean_mm_sec']),
                                      fov_counters[key][d_key][angle_key]['velocity_to_fish_mean_mm_sec'])
                                fov_counters[key][d_key][angle_key]['orthogonal_velocity_mm_sec'].extend(vel_ort)
                                if len(vel_ort) > 0: # and vel_ort != [np.nan]:
                                    fov_counters[key][d_key][angle_key]['orth_velocity_mean_mm_sec'].append(np.nanmean(vel_ort))
                                    fov_counters[key][d_key][angle_key]['orth_velocity_max_mm_sec'].append(np.nanmax(vel_ort))
                                    fov_counters[key][d_key][angle_key]['orth_velocity_min_mm_sec'].append(np.nanmin(vel_ort))
                                else:
                                    for n in ['orth_velocity_mean_mm_sec', 'orth_velocity_max_mm_sec', 'orth_velocity_min_mm_sec']:
                                        fov_counters[key][d_key][angle_key][n].append(np.nan)
                                fov_counters[key][d_key][angle_key]['field_angle_deg'].extend(field_ang)
                                if len(field_ang) > 0: # and field_ang != [np.nan]:
                                    fov_counters[key][d_key][angle_key]['field_angle_sum_deg'].append(np.nansum(field_ang))
                                    fov_counters[key][d_key][angle_key]['field_angle_mean_deg'].append(np.nanmean(field_ang))
                                    fov_counters[key][d_key][angle_key]['field_angle_max_deg'].append(np.nanmax(field_ang))
                                    fov_counters[key][d_key][angle_key]['field_angle_min_deg'].append(np.nanmin(field_ang))
                                else:
                                    for n in ['field_angle_sum_deg', 'field_angle_mean_deg', 'field_angle_max_deg', 'field_angle_min_deg']:
                                        fov_counters[key][d_key][angle_key][n].append(np.nan)
                                fov_counters[key][d_key][angle_key]['diff_from_fish_angle_deg'].extend(diff_ang)
                                fov_counters[key][d_key][angle_key]['distance_from_fish_in_mm'].extend(dist_from)

                        if calc_per_fish:
                            if fish.name not in fov_per_fish_counters[key][d_key][angle_key].keys():
                                fov_per_fish_counters[key][d_key][angle_key][fish.name] = \
                                    {'n_paramecia': 0, 'f_paramecia': 0, 'n_events': 0, 'all': [], 'all_diff': [],
                                     'n_paramecia_diff': 0}
                            fov_per_fish_counters[key][d_key][angle_key][fish.name]['n_paramecia'] += np.sum(
                                paramecia_status)
                            fov_per_fish_counters[key][d_key][angle_key][fish.name]['n_events'] += 1
                            fov_per_fish_counters[key][d_key][angle_key][fish.name]['all'].append(
                                np.sum(paramecia_status))

                    prev_paramecia_indices = paramecia_indices2

    # complete fractions (to have normalized values)
    for key in fov_counters.keys():
        for d_key in fov_counters[key].keys():
            for angle_key in fov_counters[key][d_key].keys():
                n = fov_counters[key][d_key][angle_key]['n_events']
                n = n if n > 0 else 1
                fov_counters[key][d_key][angle_key]['f_paramecia'] = \
                    fov_counters[key][d_key][angle_key]['n_paramecia'] / n
                for fish_name in fov_per_fish_counters[key][d_key][angle_key].keys():
                    n_2 = fov_per_fish_counters[key][d_key][angle_key][fish_name]['n_events']
                    n_2 = n_2 if n_2 > 0 else 1
                    fov_per_fish_counters[key][d_key][angle_key][fish_name]['f_paramecia'] = \
                        fov_per_fish_counters[key][d_key][angle_key][fish_name]['n_paramecia'] / n_2

    print(fov_counters)
    return fov_counters, {'per_fish': fov_per_fish_counters}, counters, {'per_fish': per_fish_counters}


def calc_heat_maps(dataset: FishAndEnvDataset, parameters: PlotsCMDParameters, age=None, age_list=[],
                   calc_per_fish=False):
    heatmap_keys = parameters.combine_outcomes.keys()
    is_combine_age = parameters.is_combine_age

    total_maps = heatmap_per_event_type(heatmap_keys=heatmap_keys)
    counters = {}
    for key in total_maps.keys():
        counters[key] = {"n_fish": 0, "n_events": 0}

    per_fish_maps = {}
    fish: SingleFishAndEnvData
    event: ExpandedEvent
    for fish in tqdm(dataset.fish_processed_data_set,
                     desc="current fish age {0}".format(age if not is_combine_age else age_list)):
        if is_fish_filtered(parameters=parameters, fish=fish, age=age, age_list=age_list):
            continue

        if len(fish.events[0].paramecium.status_points) == 0:
            print("Error. fish {0} no paramecia".format(fish.name))

        fish_maps = heatmap_per_event_type(heatmap_keys=heatmap_keys)
        fish_counters = {}
        for key in fish_maps.keys():
            fish_counters[key] = {"n_fish": 1, "n_events": 0}
        outcomes_map = [outcome_to_map(_.outcome_str, parameters) for _ in fish.events]
        event_outcomes = set([v for v, ignored in outcomes_map if not ignored])
        for key in total_maps.keys():
            if any([key in _ for _ in event_outcomes]):
                counters[key]["n_fish"] += 1

        for event in tqdm(fish.events, desc="current event", disable=True):
            if event.is_inter_bout_interval_only:  # already IBI
                starting_bout_indices, ending_bout_indices = event.starting_bout_indices, event.ending_bout_indices
            else:
                starting_bout_indices, ending_bout_indices, err_code = ExpandedEvent.start_end_bout_indices(event)  # calc

            # Validate
            if len(starting_bout_indices) == 0:
                print("Error. fish {0} event {1} no bout indices".format(fish.name, event.event_id))
                continue
            if len(event.paramecium.status_points) == 0:
                continue

            if event.is_inter_bout_interval_only:  # already IBI
                frame_indices = event.frame_indices  # as saved in IBI
                frame_number = np.where(frame_indices == starting_bout_indices[-1])[0][0]
            else:
                frame_number = starting_bout_indices[-1]  # frame number is relative to whole video\

            _, result_maps, rotated_head_dest = heatmap(event, frame_number=frame_number, parameters=parameters)
            for key in result_maps.keys():
                event_key, ignored = outcome_to_map(event.outcome_str, parameters)
                if key == event_key and not ignored:
                    counters[key]["n_events"] += 1
                    total_maps[key] += result_maps[key]
                    if calc_per_fish:
                        fish_maps[key] += result_maps[key]
                        fish_counters[key]["n_events"] += 1
                elif ignored:
                    print(event.outcome_str, event_key, " ignored")

        if calc_per_fish:
            across_fish_max = 0
            normalized_fish_maps = {}
            contour_properties_fish = {}
            for key in fish_maps.keys():
                np_map = copy.deepcopy(fish_maps[key].astype(float))
                if fish_counters[key]["n_events"] > 0:
                    np_map /= fish_counters[key]["n_events"]
                if np.max(np_map) > across_fish_max:
                    across_fish_max = np.max(np_map)
                normalized_fish_maps[key] = np_map
            for key in fish_maps.keys():
                np_map = copy.deepcopy(fish_maps[key].astype(float))
                if fish_counters[key]["n_events"] > 0:
                    np_map /= fish_counters[key]["n_events"]
                # cant save mask since output mat file is too large to be saved
                contour_properties_fish[key] = calc_contour_and_properties_from_heatmap(np_map=np_map, with_mask=False,
                                                                                        label="per_fish {0}".format(
                                                                                            key),
                                                                                        across_all_max=across_fish_max)

            per_fish_maps[fish.name] = {"heatmaps": copy.deepcopy(fish_maps), 'counters': copy.deepcopy(fish_counters),
                                        'normalized_heatmaps': copy.deepcopy(normalized_fish_maps),
                                        'across_fish_max': across_fish_max,
                                        'age_dpf': fish.age_dpf,
                                        'contour_properties': contour_properties_fish}

    return total_maps, counters, {'per_fish': per_fish_maps}  # total maps is not normalized


def calc_contour_and_properties_from_heatmap(np_map, across_all_max, dpi=300, with_mask=True, figsize=(10, 10),
                                             label=""):
    """

    :param np_map: numpy array of heatmap
    :param across_all_max: max value of heatmap (should be the max value between comparable heatmaps)
    :param dpi: fig resolution. default in plt is 100, but here it is higher
    :param with_mask: should dict value contain mask as well (per fish- impossible since result is too large mat)
    :param figsize: heatmap figure size. This is saved in result dictionary to allow area calculation
    :return: dict in form: {l0: {areas: [...], diameter: [...]}, l1: {...}, l2: {...}}
    """
    one_mm_in_pixels, one_pixel_in_mm = pixels_mm_converters()

    # calculate contourf of current map (all levels, with values between 0 and across_all_max)
    f2, ax2 = plt.subplots(figsize=figsize, dpi=dpi)
    xxi, yyi = np.meshgrid(np.arange(1, VideoFromRaw.FRAME_ROWS + 1, 1),
                           np.arange(1, VideoFromRaw.FRAME_COLS + 1, 1))
    cs = ax2.contourf(xxi, yyi, np.flipud(np_map), vmin=0, vmax=np.max(np_map), levels=10)
    # if len(cs.levels) != 10:
    print(label, len(cs.levels), cs.levels)

    result_per_level = {}
    for i, level in enumerate(cs.levels):
        key = "l{0}".format(i)  # to allow mat saving/loading of this output, the key should be valid name
        level_perc = (i + 1) * 100 / len(cs.levels)
        # create heatmap img of specific level using contour function and fig canvas
        fig = plt.figure(2, figsize=figsize)
        plt.contour(cs, levels=[level])
        plt.axis('off')  # no axes
        plt.tight_layout(pad=0)  # no margins
        fig.canvas.draw()  # get figure's canvas as NumPy array
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        (h, w) = img.shape[:2]  # todo np_map? all properties should be relative to img
        (cX, cY) = (w // 2, h // 2)
        total_area = float(img.shape[0] * img.shape[1])
        total_map_area = float(np_map.shape[0] * np_map.shape[1])
        scale_to_mm = one_pixel_in_mm * total_map_area / total_area  # todo make no sense

        # find contours and its properties using mask and skimage functions ()
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
        label_image = measure.label(mask)
        regions = measure.regionprops(label_image)

        # save region properties as dictionary (/mat struct) for current level
        d = {'threshold_value': level, 'img_shape': img.shape, 'image_center': (cX, cY), 'level_percentage': level_perc,
             'areas': [], 'filled_areas': [], 'normed_filled_areas': [], 'diameter': [],
             'com_distance_from_origin': [], 'normed_com_distance_from_origin': [],
             'major_axis_lengths': [], 'minor_axis_lengths': [], 'eccentricity': [], 'center_of_mass': []}
        if with_mask:
            d['mask'] = copy.deepcopy(mask)
        for r in regions:
            d['areas'].append(r.area)
            d['filled_areas'].append(r.filled_area)
            d['normed_filled_areas'].append(float(r.filled_area) * scale_to_mm)
            d['diameter'].append(r.equivalent_diameter)
            d['major_axis_lengths'].append(r.major_axis_length)
            d['minor_axis_lengths'].append(r.minor_axis_length)
            d['eccentricity'].append(r.eccentricity)
            d['center_of_mass'].append((r.centroid[1], r.centroid[0]))
            dist = distance.euclidean((r.centroid[1], r.centroid[0]), (cX, cY))
            d['com_distance_from_origin'].append(dist)
            d['normed_com_distance_from_origin'].append(float(dist) * scale_to_mm)
        result_per_level[key] = copy.deepcopy(d)
        plt.close()

    # todo needed? cs.allsegs is list of lists... in shape: [level0segs, level1segs, ...],
    #  level0segs = [polygon0, polygon1, ...] & polygon0 = [[x0, y0], [x1, y1], ...]
    # manual_areas = []
    # for seg in cs.allsegs:  # todo this is generic and doesnt find diff contours for each level
    #     x = seg[0][:, 0] if True else seg[0][0][:, 0]
    #     y = seg[0][:, 1] if True else seg[0][0][:, 1]
    #     area = np.abs(0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y)))
    #     manual_areas.append(area)
    # print(areas, manual_areas, len(cs.allsegs), [r.area for r in regions])

    plt.close()
    return result_per_level


def get_dataset(data_path_, parameters: PlotsCMDParameters, age=None, is_inter_bout_intervals=True):
    def get_info_from_event_name(name):
        """Search for pattern: <date-digits>-f<digits>-<digits><smt>.avi/mp4 (- or _ as separators)
        to extract fish_name, event_number
        If not found, split to find name

        :param name:
        :return:
        """
        pattern = re.match(r'^(?=(\d+(?:-|_)f(\d+))).*\.mat$', name)
        # Match: <date-digits>-f<digits>-<digits><smt>.avi/mp4
        event_number = -1
        if pattern is None:  # hard coded if the above didnt work
            pass
        else:
            fish_name = pattern.group(1)
        return fish_name, int(event_number)

    def fish_mat_path(curr_fish, mat_path):
        if is_inter_bout_intervals:
            if mat_path.endswith("inter_bout_interval"):
                processed_path = mat_path
            else:
                processed_path = os.path.join(os.path.join(mat_path[:mat_path.find("data_set_features")], "data_set_features"), "inter_bout_interval")
            return os.path.join(processed_path, curr_fish + "_ibi_processed.mat")
        else:
            if mat_path.endswith("inter_bout_interval"):
                processed_path = mat_path.replace("inter_bout_interval", "all_fish")
            else:
                processed_path = os.path.join(os.path.join(mat_path[:mat_path.find("data_set_features")], "data_set_features"), "all_fish")
            return os.path.join(processed_path, curr_fish + "_env_processed.mat")

    def get_fish_age(excel_full_path):
        curr_age = None
        try:
            if os.path.exists(excel_full_path):
                metadata_df = read_file(excel_full_path)
                metadata_df = metadata_df[metadata_df.fishName == curr_fish]  # only current fish (make sure)
                metadata_df = metadata_df.iloc[0]
                curr_age = int(metadata_df.age)
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
        return curr_age

    if parameters.fish == '*':  # todo refactor me
        # dataset = FishAndEnvDataset.import_from_matlab(fullfile)  # mat too large
        fish_names = []

        vid_names = parameters.mat_names
        if parameters.event_number is not None:
            vid_names = [f for f in vid_names if "-{0}_preproc".format(parameters.event_number) in f]

        videos_path = np.unique([os.path.dirname(v) for v in vid_names])
        for vid in vid_names:
            curr_fish, _ = get_info_from_event_name(os.path.basename(vid))
            if age is not None:  # read metadata to choose fish based on age, faster
                curr_age = get_fish_age(os.path.join(data_path_, curr_fish, "{0}-frames.csv".format(curr_fish)))
                if curr_fish not in fish_names and (curr_age is not None and curr_age == age):
                    fish_names.append(curr_fish)
            elif curr_fish not in fish_names:
                fish_names.append(curr_fish)

        all_fish = []
        for curr_fish in tqdm(fish_names, desc="curr fish"):
            curr = fish_mat_path(curr_fish, videos_path[0])
            if os.path.exists(curr):
                try:
                    all_fish.append(SingleFishAndEnvData.import_from_matlab(curr))
                except Exception as e:
                    logging.error("Error: ", curr, e)
                    traceback.print_tb(e.__traceback__)
        print("# fish in dataset: ", len([f for f in all_fish if len(f.events) > 0]))
        dataset = FishAndEnvDataset([f for f in all_fish if len(f.events) > 0])
    else:
        dataset = FishAndEnvDataset([SingleFishAndEnvData.import_from_matlab(fish_mat_path(parameters.fish))])

    return dataset


def create_data_for_plots(counters_json_, heatmap_counters_json_, parameters: PlotsCMDParameters, dataset=None):
    counters = {}
    if parameters.is_reading_json and os.path.exists(counters_json_):
        with open(counters_json_, 'r') as outfile:
            counters = json.load(outfile)
            print("loaded json. Plotting...")
            existing_ages = [int(k) for k in counters.keys() if k != "all"]

    if counters == {}:
        existing_ages = [fish.age_dpf for fish in dataset.fish_processed_data_set]
        print("loaded dataset. saving json...")
        for age in [None] + existing_ages:
            age_key = "all" if age is None else str(age)
            counters[age_key] = count_abort_fraction(dataset, age=age,
                                                     heatmap_n_paramecia_type=parameters.heatmap_n_paramecia_type,
                                                     valid_n_paramecia=parameters.valid_n_paramecia)

    existing_ages = list(set(existing_ages))  # unique only
    existing_ages.sort()
    if -1 in existing_ages:
        existing_ages.remove(-1)
    if '-1' in counters.keys():
        counters.pop('-1')

    print("Ages: ", existing_ages)
    combine_age_counters = {}
    if parameters.is_combine_age:
        for key, value_list in parameters.combine_ages.items():
            combine_age_counters[key] = {}
            for age in value_list & counters.keys():
                for outcome in counters[age].keys():
                    if outcome not in combine_age_counters[key].keys():  # new - copy
                        combine_age_counters[key][outcome] = copy.deepcopy(counters[age][outcome])
                    else:  # extend
                        for inner_key in combine_age_counters[key][outcome].keys():
                            if isinstance(counters[age][outcome][inner_key], list):
                                combine_age_counters[key][outcome][inner_key].extend(
                                    counters[age][outcome][inner_key].copy())
                            else:
                                combine_age_counters[key][outcome][inner_key] += counters[age][outcome][inner_key]

    return counters, combine_age_counters, existing_ages, \
           calc_heatmaps_main(existing_ages, dataset, parameters=parameters)


def frames_to_secs_converter(n_curr_frames):
    return n_curr_frames / float(VideoFromRaw.FPS)


def is_fish_filtered(parameters, fish, age=None, age_list=[]):
    if len(age_list) == 0:
        if not (age is None or fish.age_dpf == age):
            return True
    else:
        if not (not parameters.is_combine_age and (age is None or fish.age_dpf == age) or
                (parameters.is_combine_age and fish.age_dpf in age_list)):
            return True

    if parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.all and \
       fish.num_of_paramecia_in_plate not in parameters.valid_n_paramecia:
        print("(all) Ignoring fish {0} with n_paramecia={1}".format(fish.name, fish.num_of_paramecia_in_plate))
        return True
    if parameters.heatmap_n_paramecia_type != HeatmapNParameciaType.all:  # todo refactor me
        if (parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n30 and fish.num_of_paramecia_in_plate != 30) or \
                (parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n50 and fish.num_of_paramecia_in_plate != 50) or \
                (parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n70 and fish.num_of_paramecia_in_plate != 70):
            return True
    if parameters.feeding_type != FeedingType.all_feeding:
        if FeedingType.map_feeding_str(fish.feeding_str) != parameters.feeding_type:
            return True

    return False


def calc_bout_durations(dataset, parameters: PlotsCMDParameters, age=None, age_list=[], calc_per_fish=False):
    heatmap_keys = parameters.combine_outcomes.keys()
    r = {"event_durations_per_fish": {}}
    for what in ["sum", "mean", "sem", "first", "last", "n"]:
        r["event_ibi_dur_per_fish_" + what] = {}

    event: ExpandedEvent
    paramecium: ParameciumRelativeToFish
    for fish in tqdm(dataset.fish_processed_data_set,
                     desc="Duration current fish age {0}".format(age if not parameters.is_combine_age else age_list)):
        if is_fish_filtered(parameters=parameters, fish=fish, age=age, age_list=age_list):
            continue

        for what in r.keys():
            r[what][fish.name] = {'all': []}
            for k in heatmap_keys:
                r[what][fish.name][k] =[]
        for event in fish.events:
            event_duration = frames_to_secs_converter(event.event_frame_ind)
            cum_durations = [frames_to_secs_converter(e_ibi - s_ibi)
                             for (e_ibi, s_ibi) in zip(event.starting_bout_indices[1:], event.ending_bout_indices)
                             if e_ibi - s_ibi > 0]
            r["event_ibi_dur_per_fish_sum"][fish.name]['all'].append(np.nansum(cum_durations))
            r["event_ibi_dur_per_fish_mean"][fish.name]['all'].append(np.nanmean(cum_durations))
            r["event_ibi_dur_per_fish_sem"][fish.name]['all'].append(np.nanstd(cum_durations) / len(cum_durations))
            r["event_ibi_dur_per_fish_first"][fish.name]['all'].append(cum_durations[0] if len(cum_durations) > 0 else np.nan)
            r["event_ibi_dur_per_fish_last"][fish.name]['all'].append(cum_durations[-1] if len(cum_durations) > 0 else np.nan)
            r["event_ibi_dur_per_fish_n"][fish.name]['all'].append(len(cum_durations))
            r["event_durations_per_fish"][fish.name]['all'].append(event_duration)
            event_key, ignored = outcome_to_map(event.outcome_str, parameters)
            for k in heatmap_keys:
                if k == event_key and not ignored:
                    r["event_ibi_dur_per_fish_sum"][fish.name][k].append(np.nansum(cum_durations))
                    r["event_ibi_dur_per_fish_mean"][fish.name][k].append(np.nanmean(cum_durations))
                    r["event_ibi_dur_per_fish_sem"][fish.name][k].append(np.nanstd(cum_durations) / len(cum_durations))
                    r["event_ibi_dur_per_fish_first"][fish.name][k].append(cum_durations[0] if len(cum_durations) > 0 else np.nan)
                    r["event_ibi_dur_per_fish_last"][fish.name][k].append(cum_durations[-1] if len(cum_durations) > 0 else np.nan)
                    r["event_ibi_dur_per_fish_n"][fish.name][k].append(len(cum_durations))
                    r["event_durations_per_fish"][fish.name][k].append(event_duration)
    return r


def calc_durations_main(existing_ages, dataset, parameters: PlotsCMDParameters):
    event_durations = {}
    age_list = parameters.combine_ages.keys() if parameters.is_combine_age else [None] + existing_ages
    for age in age_list:
        age_key = "all" if age is None else str(age)
        if parameters.is_combine_age:
            event_durations[age_key] = calc_bout_durations(dataset, parameters=parameters,
                                                           age_list=[int(_) for _ in parameters.combine_ages[age]],
                                                           calc_per_fish=False)
        else:
            event_durations[age_key] = \
                calc_heat_maps(dataset, parameters=parameters, age=age, calc_per_fish=False)

    return event_durations


def calc_heatmaps_main(existing_ages, dataset, parameters: PlotsCMDParameters):
    heatmap_keys = parameters.combine_outcomes.keys()

    total_all_age = {}
    total_heatmap_counters = {}
    contour_properties = {}
    per_fish_maps = {}
    paramecia_in_fov = {}
    per_fish_paramecia_in_fov = {}
    paramecia_data = {}
    per_fish_paramecia_data = {}

    if not parameters.is_heatmap:
        return total_all_age, 0, [], total_heatmap_counters, per_fish_maps, contour_properties, \
               paramecia_in_fov, per_fish_paramecia_in_fov, paramecia_data, per_fish_paramecia_data

    if parameters.is_save_per_fish_heatmap:
        parameters_per_fish = copy.deepcopy(parameters)
        parameters_per_fish.is_combine_age = False
        _, _, per_fish_maps = calc_heat_maps(dataset, parameters=parameters_per_fish, age=None, calc_per_fish=True)
        _, per_fish_paramecia_in_fov, _, per_fish_paramecia_data = \
            calc_paramecia_counts(dataset, parameters=parameters_per_fish, age=None, calc_per_fish=True)

    across_all_max = 0
    age_list = parameters.combine_ages.keys() if parameters.is_combine_age else [None] + existing_ages
    for age in age_list:
        age_key = "all" if age is None else str(age)
        if parameters.is_combine_age:
            total_all_age[age_key], total_heatmap_counters[age_key], _ = \
                calc_heat_maps(dataset, parameters=parameters, age_list=[int(_) for _ in parameters.combine_ages[age]],
                               calc_per_fish=False)
            paramecia_in_fov[age_key], _, paramecia_data[age_key], _ = \
                calc_paramecia_counts(dataset, parameters=parameters,
                                      age_list=[int(_) for _ in parameters.combine_ages[age]], calc_per_fish=False)
        else:
            total_all_age[age_key], total_heatmap_counters[age_key], _ = \
                calc_heat_maps(dataset, parameters=parameters, age=age, calc_per_fish=False)
            paramecia_in_fov[age_key], _, paramecia_data[age_key], _ = \
                calc_paramecia_counts(dataset, parameters=parameters, age=age, calc_per_fish=False)

        total_maps = copy.deepcopy(total_all_age[age_key])
        if age is not None:  # dont account all ages for across_all_max since it's above others
            for key in total_maps.keys():
                if total_heatmap_counters[age_key][key]["n_events"] > 0:
                    np_map = total_maps[key].astype(float).copy() / total_heatmap_counters[age_key][key]["n_events"]
                else:
                    np_map = total_maps[key].astype(float).copy()  # zero
                if parameters.gaussian:
                    np_map = gaussian_filter(np_map, 5)
                if np.max(np_map) > across_all_max:
                    across_all_max = np.max(np_map)
                print(age, key, np.max(np_map))

        contour_properties[age_key] = {}
        for key in total_maps.keys():
            if total_heatmap_counters[age_key][key]["n_events"] > 0:
                np_map = total_maps[key].astype(float).copy() / total_heatmap_counters[age_key][key]["n_events"]
            else:
                np_map = total_maps[key].astype(float).copy()  # zero
            contour_properties[age_key][key] = calc_contour_and_properties_from_heatmap(np_map=np_map, label=key,
                                                                                        across_all_max=across_all_max)

    return total_all_age, across_all_max, age_list, total_heatmap_counters, per_fish_maps, contour_properties, \
           paramecia_in_fov, per_fish_paramecia_in_fov, paramecia_data, per_fish_paramecia_data


def get_folder_and_file_names(data_path_, parameters: PlotsCMDParameters, age_groups, create_folders=False):
    output_folder = os.path.join(data_path_, "features", "new_bout_target")
    heatmap_parent_folder = output_folder # os.path.join(output_folder, "Heat_maps")
    metadata_folder = output_folder # os.path.join(output_folder, "Metadata")

    # global prefix - should indicate what changes global counts etc
    # heatmap params are within heatmap path
    add_name = "age_{0}_".format(age_groups) if parameters.is_combine_age else ""
    add_name += "{0}_".format(parameters.outcome_map_type)
    add_name += "{0}_pd_".format(parameters.heatmap_n_paramecia_type)
    add_name += str(parameters.feeding_type) + "_"

    add_heatmap_name = str(parameters.heatmap_type)
    heatmap_folder = os.path.join(heatmap_parent_folder, add_heatmap_name)

    counters_json = os.path.join(metadata_folder, add_name + 'count_metadata.txt')
    heatmap_counters_json = os.path.join(heatmap_parent_folder,
                                         add_name + add_heatmap_name + '_heatmap_count_metadata.txt')
    heatmap_data_json = os.path.join(heatmap_parent_folder,
                                     add_name + add_heatmap_name + '_heatmap_data.mat')

    if create_folders:
        create_dirs_if_missing([heatmap_folder, os.path.join(heatmap_folder, "all_fish"), metadata_folder,
                                os.path.join(metadata_folder, "figures"),
                                os.path.join(heatmap_folder, "contour_properties")])


    fullpath_output_prefix = os.path.join(heatmap_parent_folder, add_name + add_heatmap_name)
    return heatmap_folder, metadata_folder, heatmap_parent_folder, counters_json, heatmap_counters_json, \
           heatmap_data_json, fullpath_output_prefix, add_name, add_heatmap_name


def recursive_fix_key_for_mat(curr_d: dict):
    result = {}
    for k, v in curr_d.items():
        if isinstance(v, dict):
            v = recursive_fix_key_for_mat(v)
        if isinstance(k, (int, float)) or k[0].isdigit():
            k = "a_{0}".format(k)
        result[k.replace("-", "_").replace(".", "_dot_").replace(" ", "_")] = copy.deepcopy(v)
    return result


def save_plot_outputs(counters_json, heatmap_counters_json, counters, dataset, total_heatmap_counters, total_all_age,
                      per_fish_maps, contour_properties, paramecia_in_fov, per_fish_paramecia_in_fov, paramecia_data,
                      per_fish_paramecia_data):
    with open(counters_json, 'w') as outfile:
        json.dump(counters, outfile, sort_keys=True, indent=4)

    if dataset is not None and parameters.is_heatmap:
        total_all_age_for_mat, total_all_age_for_mat_normalized = {}, {}
        for key in total_all_age.keys():
            d, d_normalized = {}, {}
            for inner_key in total_all_age[key].keys():
                data = copy.deepcopy(total_all_age[key][inner_key])
                d[inner_key.replace("-", "_")] = copy.deepcopy(data)
                if total_heatmap_counters[key][inner_key]["n_events"] > 0:
                    data = data.astype(float).copy() / total_heatmap_counters[key][inner_key]["n_events"]
                d_normalized[inner_key.replace("-", "_")] = data

            total_all_age_for_mat["age_" + key.replace("-", "_")] = d
            total_all_age_for_mat_normalized["age_" + key.replace("-", "_")] = d_normalized
        save_mat_dict(heatmap_data_json, total_all_age_for_mat)  # heatmaps
        #save_mat_dict(fullpath_output_prefix + "_n_events_normalized.mat", total_all_age_for_mat_normalized)
        #save_mat_dict(fullpath_output_prefix + "_contour_properties.mat", recursive_fix_key_for_mat(contour_properties))
        #save_mat_dict(fullpath_output_prefix + "_paramecia_in_fov.mat", recursive_fix_key_for_mat(paramecia_in_fov))
        #save_mat_dict(fullpath_output_prefix + "_paramecia_data.mat", recursive_fix_key_for_mat(paramecia_data))
        if parameters.is_save_per_fish_heatmap:
            save_mat_dict(fullpath_output_prefix + "_per_fish.mat", recursive_fix_key_for_mat(per_fish_maps))
            save_mat_dict(fullpath_output_prefix + "_per_fish_paramecia_in_fov.mat",
                          recursive_fix_key_for_mat(per_fish_paramecia_in_fov))
            save_mat_dict(fullpath_output_prefix + "_per_fish_paramecia_data.mat",
                          recursive_fix_key_for_mat(per_fish_paramecia_data))

        with open(heatmap_counters_json, 'w') as outfile:
            json.dump(total_heatmap_counters, outfile, sort_keys=True, indent=4)


def paramecia_properties_df(all_keys=None, paramecia_data=None, paramecia_in_fov=None):
    df_list, df_per_fov_list = [], []
    for age_name, fish_data in paramecia_data.items():
        if age_name.startswith("__"):
            continue
        for outcome, inner_values in fish_data.items():
            df = pd.DataFrame.from_dict(inner_values)
            df.insert(0, "Age (dpf)", age_name.replace("a_", "").replace("_", "-"))
            df.insert(1, "Outcome", outcome)
            df.insert(2, "Angle (type)", '360 deg')
            df.insert(3, "Distance (mm)", 'all')
            df_list.append(df)
    for age_name, fish_data in paramecia_in_fov.items():
        if age_name.startswith("__"):
            continue
        for outcome, inner_values in fish_data.items():
            for i, (distance_mm, inner_inner_values) in enumerate(inner_values.items()):
                distance_value = distance_mm.replace("d_", "").replace("_mm", "").replace("_", ".")
                if distance_value.replace('.', '', 1).isdigit():
                    distance_value = float(distance_value)
                else:
                    distance_value = distance_mm.replace("d_", "").replace("_", " ")
                for j, (angle_type, inner_inner_inner_values) in enumerate(inner_inner_values.items()):
                    if all_keys is not None:
                        # quick patch- remove non property fields
                        inner_inner_inner_values = \
                            dict([(k, np.array(v, dtype='float')) for k, v in inner_inner_inner_values.items() if k in all_keys])
                    if np.array([v.size == 1 for v in inner_inner_inner_values.values()]).all():
                        df = pd.DataFrame.from_records([inner_inner_inner_values])
                    else:
                        df = pd.DataFrame.from_dict(inner_inner_inner_values)
                    df.insert(0, "Age (dpf)", age_name.replace("a_", "").replace("_", "-"))
                    df.insert(1, "Outcome", outcome)
                    df.insert(2, "Angle (type)", angle_type)
                    df.insert(3, "Distance (mm)", distance_value)
                    df["Distance (mm)"] = df["Distance (mm)"].astype('category')
                    df_per_fov_list.append(df)
    return pd.concat(df_list), pd.concat(df_per_fov_list)


def parse_input_from_command():
    parser = argparse.ArgumentParser(description='Plot fish data.')
    parser.add_argument('data_path', type=str, help='Full path to data folder (events)')
    parser.add_argument('fish_name', default="*",
                        type=str, help='Fish folder name (inside events). If empty- run all')
    parser.add_argument('--gaussian', default=False, action='store_true',
                        help='Heatmaps with gaussian filter')
    parser.add_argument('--is_bounding_box', default=False, action='store_true',
                        help='Heatmaps with bounding box (default: false, center only)')
    parser.add_argument('--is_combine_age', default=False, action='store_true',
                        help='Plots should combine ages (to range) (default: false, discrete values)')
    parser.add_argument('--heatmap_type', default=HeatmapType.all_para, type=lambda v: HeatmapType[v],
                        choices=list(HeatmapType),
                        help='Heatmaps with background only, target only or both (default: heatmap.all means both)')
    parser.add_argument('--outcome_map_type', default=OutcomeMapType.all_outcome, type=lambda v: OutcomeMapType[v],
                        choices=list(OutcomeMapType), help='Ways to combine outcomes')
    parser.add_argument('--heatmap_n_paramecia_type', default=HeatmapNParameciaType.all,
                        type=lambda v: HeatmapNParameciaType[v], choices=list(HeatmapNParameciaType),
                        help='Ways to combine outcomes')
    parser.add_argument('--feeding_type', default=FeedingType.all_feeding,
                        type=lambda v: FeedingType[v], choices=list(FeedingType),
                        help='')
    parser.add_argument('--age_groups', default=CombineAgeGroups.v2,
                        type=lambda v: CombineAgeGroups[v], choices=list(CombineAgeGroups),
                        help='Ways to combine ages (exact range in code)')
    parser.add_argument('--is_save_per_fish', default=False, action='store_true',
                        help='Heatmaps created for each fish (default: false)')
    parser.add_argument('--no_heatmap', default=False, action='store_true',
                        help='Skip heatmap creation, only make metadata (default: false)')
    parser.add_argument('--no_metadata', default=False, action='store_true',
                        help='Skip metadata creation, only make heatmap (default: false)')
    parser.add_argument('--is_all_metadata_permutations', default=False, action='store_true',
                        help='Create metadata for all values (default: false)')

    # print(parser.print_help())
    args = parser.parse_args(sys.argv[1:])
    return args.data_path, args


def quicktry(key='all', is_per_fish=False):
    def to_list(v):
        if isinstance(v, list):
            return copy.deepcopy(v)
        return [v]

    def paramecia_fov_dict():
        per_age_statistics = {}
        flipped_per_age_statistics = {}
        per_age_for_hist_statistics = {}
        levels = []
        key = 'all'
        hist_key = 'n_paramecia'
        if is_per_fish:
            data = per_fish_paramecia_in_fov
        else:
            data = paramecia_in_fov
        for age_ind, (age_name, values) in enumerate(data.items()):
            if age_name.startswith("__"):
                continue
            for outcome, inner_values in values.items():
                for i, (distance_mm, inner_inner_values) in enumerate(inner_values.items()):
                    d = distance_mm.replace("d_", "").replace("_", " ")
                    distance_value = distance_mm.replace("d_", "").replace("_mm", "").replace("_", ".")
                    if distance_value.replace('.', '', 1).isdigit():
                        distance_value = float(distance_value)
                    else:
                        distance_value = None
                    levels.append(d if distance_value is None else distance_value)
                    for j, (angle_type, inner_inner_inner_values) in enumerate(inner_inner_values.items()):
                        if angle_type not in per_age_statistics.keys():
                            per_age_statistics[angle_type] = {}
                            per_age_for_hist_statistics[angle_type] = {}
                            flipped_per_age_statistics[angle_type] = {}
                        if outcome not in per_age_statistics[angle_type].keys():
                            per_age_statistics[angle_type][outcome] = {}
                            per_age_for_hist_statistics[angle_type][outcome] = {}
                        if age_name not in per_age_statistics[angle_type][outcome].keys():
                            per_age_statistics[angle_type][outcome][age_name] = {'x': [], 'y': []}
                            per_age_for_hist_statistics[angle_type][outcome][age_name] = {'x': age_ind, 'y': []}
                        if age_name not in flipped_per_age_statistics[angle_type].keys():
                            flipped_per_age_statistics[angle_type][age_name] = {}
                        if outcome not in flipped_per_age_statistics[angle_type][age_name].keys():
                            flipped_per_age_statistics[angle_type][age_name][outcome] = {'x': [], 'y': []}
                        add, add_hist = [], []
                        if is_per_fish:
                            for fish_name, data_dict in inner_inner_inner_values.items():
                                add.extend(to_list(data_dict[key]))
                                x_add = [np.nan] * len(add)
                                if distance_value is not None:
                                    x_add = to_list(distance_value) * len(add)
                                    add_hist.extend(to_list(distance_value) * int(data_dict[hist_key]))
                        else:
                            add = to_list(inner_inner_inner_values[key])
                            x_add = [np.nan] * len(add)
                            if distance_value is not None:
                                x_add = to_list(distance_value) * len(add)
                                add_hist = to_list(distance_value) * int(inner_inner_inner_values[hist_key])
                        # print(angle_type, outcome, age_name, distance_value, add, x_add)
                        per_age_statistics[angle_type][outcome][age_name]['x'].extend(x_add)
                        per_age_statistics[angle_type][outcome][age_name]['y'].extend(add)
                        flipped_per_age_statistics[angle_type][age_name][outcome]['x'].extend(x_add)
                        flipped_per_age_statistics[angle_type][age_name][outcome]['y'].extend(add)
                        per_age_for_hist_statistics[angle_type][outcome][age_name]['y'].extend(add_hist)
        return per_age_statistics, flipped_per_age_statistics, per_age_for_hist_statistics, levels

    def fov_dict_to_pd(curr_dict, is_flipped=False):
        df_list = []
        for angle_type in curr_dict.keys():
            for inner_key in curr_dict[angle_type].keys():
                for inner_inner_key in curr_dict[angle_type][inner_key].keys():
                    df = pd.DataFrame.from_dict(curr_dict[angle_type][inner_key][inner_inner_key])
                    df['x'] = df['x'].astype("category")
                    df.rename(columns={'x': 'Distance (mm)', 'y': 'Number of paramecia'}, inplace=True)
                    if is_flipped:  # inner_key=age, inner_inner_key=outcome
                        df.insert(0, "Age (dpf)", inner_key.replace("a_", "").replace("_", "-"))
                        df.insert(1, "Outcome", inner_inner_key)
                    else:  # inner_key=outcome, inner_inner_key=age
                        df.insert(0, "Age (dpf)", inner_inner_key.replace("a_", "").replace("_", "-"))
                        df.insert(1, "Outcome", inner_key)
                    df.insert(2, "Angle (type)", angle_type)
                    df_list.append(df)
        return pd.concat(df_list)

    def paramecia_properties_df():
        df_list, df_per_fov_list = [], []
        for age_name, fish_data in paramecia_data.items():
            if age_name.startswith("__"):
                continue
            for outcome, inner_values in fish_data.items():
                df = pd.DataFrame.from_dict(inner_values)
                df.insert(0, "Age (dpf)", age_name.replace("a_", "").replace("_", "-"))
                df.insert(1, "Outcome", outcome)
                df.insert(2, "Angle (type)", '360 deg')
                df.insert(3, "Distance (mm)", 'all')
                df_list.append(df)
        for age_name, fish_data in paramecia_in_fov.items():
            if age_name.startswith("__"):
                continue
            for outcome, inner_values in fish_data.items():
                for i, (distance_mm, inner_inner_values) in enumerate(inner_values.items()):
                    distance_value = distance_mm.replace("d_", "").replace("_mm", "").replace("_", ".")
                    if distance_value.replace('.', '', 1).isdigit():
                        distance_value = float(distance_value)
                    else:
                        distance_value = distance_mm.replace("d_", "").replace("_", " ")
                    for j, (angle_type, inner_inner_inner_values) in enumerate(inner_inner_values.items()):
                        # quick patch- remove non property fields
                        n_paramecia = inner_inner_inner_values.pop('n_paramecia')
                        inner_inner_inner_values.pop('f_paramecia')
                        n_events = inner_inner_inner_values.pop('n_events')
                        inner_inner_inner_values.pop('all')
                        df = pd.DataFrame.from_dict(inner_inner_inner_values)
                        df.insert(0, "Age (dpf)", age_name.replace("a_", "").replace("_", "-"))
                        df.insert(1, "Outcome", outcome)
                        df.insert(2, "Angle (type)", angle_type)
                        df.insert(3, "Distance (mm)", distance_value)
                        df_per_fov_list.append(df)
        return pd.concat(df_list), pd.concat(df_per_fov_list)

    # load
    paramecia_in_fov = load_mat_dict(fullpath_output_prefix + "_paramecia_in_fov.mat")
    paramecia_data = load_mat_dict(fullpath_output_prefix + "_paramecia_data.mat")

    counters, combine_age_counters, existing_ages, \
    (total_all_age, across_all_max, age_list, total_heatmap_counters, per_fish_maps,
     contour_properties, paramecia_in_fov, per_fish_paramecia_in_fov, paramecia_data, per_fish_paramecia_data) = \
        create_data_for_plots(counters_json_=counters_json, dataset=dataset,
                              heatmap_counters_json_=heatmap_counters_json, parameters=parameters)
    # per_age_statistics, flipped_per_age_statistics = paramecia_properties_dict('distance_from_fish_in_mm')

    total_df, per_fov_df = paramecia_properties_df()
    print(per_fov_df)
    print("hi")


def get_parameters(with_mat=True):
    data_path, args = parse_input_from_command()
    parameters = PlotsCMDParameters()
    should_run_all_metadata_permutations = args.is_all_metadata_permutations
    parameters.fish = args.fish_name
    parameters.mat_names = []
    if with_mat:
        parameters.mat_names = glob.glob(
            os.path.join(data_path, "data_set_features", "new_bout_target", "inter_bout_interval", "*.mat"))
    logging.info("Mat names: {0}".format(parameters.mat_names))
    parameters.gaussian = args.gaussian
    parameters.fish = args.fish_name
    parameters.is_bounding_box = args.is_bounding_box
    parameters.is_combine_age = args.is_combine_age
    parameters.heatmap_n_paramecia_type = args.heatmap_n_paramecia_type
    parameters.feeding_type = args.feeding_type
    parameters.heatmap_type = args.heatmap_type
    parameters.outcome_map_type = args.outcome_map_type
    parameters.is_save_per_fish_heatmap = args.is_save_per_fish
    parameters.is_heatmap = not args.no_heatmap
    parameters.is_metadata = not args.no_metadata
    parameters.age_groups = CombineAgeGroups.v2
    if args.age_groups == CombineAgeGroups.v2:
        parameters.combine_ages = parameters.combine_ages_v2
    elif args.age_groups == CombineAgeGroups.v3:
        parameters.combine_ages = parameters.combine_ages_v3
    else:
        parameters.combine_ages = parameters.combine_ages_v1

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

    return parameters, data_path, should_run_all_metadata_permutations


def distance_name_to_value (distance_str):
    """a_1_1_5mm => 1-1.5mm should return 1

    Old: lambda v: v.replace("d_", "").replace("a_", "").replace("mm", "").split("_")[1].replace("_", ".")
    :param distance_str:
    :return:
    """
    if distance_str.startswith("a_"):
        v = distance_str.replace("a_", "").replace("mm", "").replace("_dot_", ".")
        if len(v.split("_")) == 2:
            return v.split("_")[1]
        elif len(v.split("_")) == 3 and "_5" in v:
            v = v.replace("_5", ".5")
            if len(v.split("_")) == 2:
                return v.split("_")[1]
        logging.error("Unknown distance {0}".format(distance_str))
        return np.nan
    if distance_str.startswith("d_") and distance_str.count("_dot_") == 1:
        return distance_str.replace("d_", "").replace("_mm", "").replace("mm", "").replace("_dot_", ".")
    return distance_str.replace("d_", "").replace("a_", "").replace("mm", "").split("_")[1].replace("_dot_", ".")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parameters: PlotsCMDParameters
    parameters, data_path, should_run_all_metadata_permutations = get_parameters()

    from matplotlib import rcParams, pyplot as plt

    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = '12'

    # read this first/outside to allow loop over parameters
    dataset = None
    if parameters.is_heatmap or not parameters.is_reading_json:
        dataset: FishAndEnvDataset = get_dataset(data_path, parameters=parameters, is_inter_bout_intervals=True)

    if should_run_all_metadata_permutations:
        all_para = [HeatmapNParameciaType.all, HeatmapNParameciaType.n30, HeatmapNParameciaType.n50,
                    HeatmapNParameciaType.n70]
        all_outcome = [OutcomeMapType.strike_abort, OutcomeMapType.hit_miss_abort,
                       OutcomeMapType.hit_miss_abort_es_abort_noes, OutcomeMapType.all_outcome]
        all_feeding = [FeedingType.all_feeding, FeedingType.before_feeding, FeedingType.after_feeding]
        for heatmap_n_paramecia_type in tqdm(all_para, desc="heatmap_n_paramecia_type"):
            for outcome_map_type in tqdm(all_outcome, desc="outcome_map_type"):
                for feeding_type in tqdm(all_feeding, desc="feeding_map_type"):
                    for is_combine_age in [True, False]:
                        parameters.is_combine_age = is_combine_age
                        parameters.heatmap_n_paramecia_type = heatmap_n_paramecia_type
                        parameters.outcome_map_type = outcome_map_type
                        parameters.feeding_type = feeding_type
                        if parameters.outcome_map_type == OutcomeMapType.hit_miss_abort:
                            parameters.combine_outcomes = {'hit-spit': ['hit', 'spit'], 'miss': ['miss'],
                                                           'abort': ['abort']}
                        elif parameters.outcome_map_type == OutcomeMapType.strike_abort:
                            parameters.combine_outcomes = {'strike': ['hit', 'miss', 'spit'], 'abort': 'abort'}
                        elif parameters.outcome_map_type == OutcomeMapType.hit_miss_abort_es_abort_noes:
                            parameters.combine_outcomes = {'hit-spit': ['hit', 'spit'], 'miss': ['miss'],
                                                           'abort,escape': ['abort,escape'],
                                                           'abort,no-escape': ['abort,no-escape']}
                        else:  # all- default
                            parameters.combine_outcomes = {'hit': ['hit'], 'spit': ['spit'], 'miss': ['miss'],
                                                           'abort': ['abort']}

                        if is_combine_age:
                            ages = [(CombineAgeGroups.v1, parameters.combine_ages_v1),
                                    (CombineAgeGroups.v2, parameters.combine_ages_v2),
                                    (CombineAgeGroups.v3, parameters.combine_ages_v3)]
                        else:
                            ages = [(CombineAgeGroups.v1, parameters.combine_ages_v1)]  # not affecting anything

                        for group_version, ages_group in ages:
                            parameters.combine_ages = ages_group
                            print("Path {0}, arguments: {1}".format(data_path, parameters))

                            heatmap_folder, metadata_folder, heatmap_parent_folder, counters_json, heatmap_counters_json, \
                            heatmap_data_json, fullpath_output_prefix, add_name, add_heatmap_name = \
                                get_folder_and_file_names(data_path_=data_path, parameters=parameters,
                                                          age_groups=group_version)

                            counters, combine_age_counters, existing_ages, \
                            (total_all_age, across_all_max, age_list, total_heatmap_counters, per_fish_maps,
                             contour_properties, paramecia_in_fov, per_fish_paramecia_in_fov,
                             paramecia_data, per_fish_paramecia_data) = \
                                create_data_for_plots(counters_json_=counters_json, dataset=dataset,
                                                      heatmap_counters_json_=heatmap_counters_json, parameters=parameters)

                            #  ***************** Save plots and outputs *****************
                            save_plot_outputs(counters_json, heatmap_counters_json,
                                              combine_age_counters if parameters.is_combine_age else counters,
                                              dataset, total_heatmap_counters, total_all_age, per_fish_maps,
                                              contour_properties,
                                              paramecia_in_fov, per_fish_paramecia_in_fov, paramecia_data,
                                              per_fish_paramecia_data)
    else:
        print("Path {0}, arguments: {1}".format(data_path, parameters))

        heatmap_folder, metadata_folder, heatmap_parent_folder, counters_json, heatmap_counters_json, heatmap_data_json, \
        fullpath_output_prefix, add_name, add_heatmap_name = \
            get_folder_and_file_names(data_path_=data_path, parameters=parameters, age_groups=parameters.age_groups)

        #quicktry()

        existing_ages = list(set([fish.age_dpf for fish in dataset.fish_processed_data_set]))  # unique only
        print([fish.age_dpf for fish in dataset.fish_processed_data_set])
        existing_ages.sort()
        if -1 in existing_ages:
            existing_ages.remove(-1)

        print("Ages: ", existing_ages)
        print([fish.name for fish in dataset.fish_processed_data_set])
        # durations_data = calc_durations_main(existing_ages,dataset, parameters)
        # print("save ", fullpath_output_prefix + "_duration_properties.mat")
        # save_mat_dict(fullpath_output_prefix + "_duration_properties.mat",
        #               recursive_fix_key_for_mat(durations_data))
        #sys.exit()

        # quicktry()
        counters, combine_age_counters, existing_ages, \
        (total_all_age, across_all_max, age_list, total_heatmap_counters, per_fish_maps,
         contour_properties, paramecia_in_fov, per_fish_paramecia_in_fov, paramecia_data, per_fish_paramecia_data) = \
            create_data_for_plots(counters_json_=counters_json, dataset=dataset,
                                  heatmap_counters_json_=heatmap_counters_json, parameters=parameters)

        #  ***************** Save plots and outputs *****************
        print("save ", heatmap_counters_json)
        save_plot_outputs(counters_json, heatmap_counters_json,
                           combine_age_counters if parameters.is_combine_age else counters,
                           dataset, total_heatmap_counters, total_all_age, per_fish_maps, contour_properties,
                           paramecia_in_fov, per_fish_paramecia_in_fov, paramecia_data, per_fish_paramecia_data)
