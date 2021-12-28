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
import scipy
import skimage
import skimage.measure as measure
from fpdf import FPDF
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from scipy import stats
from scipy.spatial import distance

from tqdm import tqdm
import numpy as np
from numpy import format_float_scientific
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
import matplotlib.cm as cm
from pandas.plotting import table
import pandas as pd
import seaborn as sns

from classic_cv_trackers.abstract_and_common_trackers import Colors
from feature_analysis.fish_environment.env_utils import heatmap, \
    heatmap_per_event_type, HeatmapType, outcome_to_map, PlotsCMDParameters, OutcomeMapType, HeatmapNParameciaType, \
    CombineAgeGroups
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
        if not (age is None or fish.age_dpf == age):
            continue

        if heatmap_n_paramecia_type == HeatmapNParameciaType.all and \
                fish.num_of_paramecia_in_plate not in valid_n_paramecia:
            print("(all2) Ignoring fish {0} with n_paramecia={1}".format(fish.name, fish.num_of_paramecia_in_plate))
            continue

        if heatmap_n_paramecia_type != HeatmapNParameciaType.all:  # todo refactor me
            if (heatmap_n_paramecia_type == HeatmapNParameciaType.n30 and fish.num_of_paramecia_in_plate != 30) or \
                    (heatmap_n_paramecia_type == HeatmapNParameciaType.n50 and fish.num_of_paramecia_in_plate != 50) or \
                    (heatmap_n_paramecia_type == HeatmapNParameciaType.n70 and fish.num_of_paramecia_in_plate != 70):
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
        if not (not parameters.is_combine_age and (age is None or fish.age_dpf == age) or
                (parameters.is_combine_age and fish.age_dpf in age_list)):
            continue

        if parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.all and \
                fish.num_of_paramecia_in_plate not in parameters.valid_n_paramecia:
            print("(all) Ignoring fish {0} with n_paramecia={1}".format(fish.name, fish.num_of_paramecia_in_plate))
            continue

        if parameters.heatmap_n_paramecia_type != HeatmapNParameciaType.all:  # todo refactor me
            if (
                    parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n30 and fish.num_of_paramecia_in_plate != 30) or \
                    (
                            parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n50 and fish.num_of_paramecia_in_plate != 50) or \
                    (
                            parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n70 and fish.num_of_paramecia_in_plate != 70):
                continue

        if len(fish.events[0].paramecium.status_points) == 0:
            print("Error. fish {0} no paramecia".format(fish.name))

        paramecium: ParameciumRelativeToFish
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

                if fish.name not in per_fish_counters[key].keys():
                    per_fish_counters[key][fish.name] = copy.deepcopy(empty_fields)

                # todo ignore predictions

                # add general features
                paramecium = event.paramecium
                paramecia_indices = [_ for _ in range(0, paramecium.field_angle.shape[1])]
                if not np.isnan(event.paramecium.target_paramecia_index):
                    if parameters.heatmap_type == HeatmapType.no_target:
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
                        print("sapir, ", paramecia_indices, " => ", paramecia_indices2)
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
        if not (not is_combine_age and (age is None or fish.age_dpf == age) or
                (is_combine_age and fish.age_dpf in age_list)):
            continue

        if parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.all and \
                fish.num_of_paramecia_in_plate not in parameters.valid_n_paramecia:
            print("(all) Ignoring fish {0} with n_paramecia={1}".format(fish.name, fish.num_of_paramecia_in_plate))
            continue

        if parameters.heatmap_n_paramecia_type != HeatmapNParameciaType.all:  # todo refactor me
            if (
                    parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n30 and fish.num_of_paramecia_in_plate != 30) or \
                    (
                            parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n50 and fish.num_of_paramecia_in_plate != 50) or \
                    (
                            parameters.heatmap_n_paramecia_type == HeatmapNParameciaType.n70 and fish.num_of_paramecia_in_plate != 70):
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
                starting_bout_indices, ending_bout_indices = ExpandedEvent.start_end_bout_indices(event)  # calc

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


def generate_heatmap(np_map, name, title, max_val, across_all_max, plot_dir, n_events, xlim=None, ylim=None,
                     curr_pdf=None, dpi=300, fig_size=FIG_SIZE, cutoff=0.5, max_length=10000):
    """
    The function creates and save all heatmap plots (including contours), both as figure and within curr_pdf
    :param curr_pdf: if not None, all figures will be added to it
    :param title: title of figure
    :param max_val: max value for color bar, relative to specific condition
    :param across_all_max: max value for color bar, comparable to all other heatmaps.
    :param name: name of the file to save
    :param np_map: np array of heat map
    :param plot_dir: directory to save into
    :return: None
    """

    def density_calc(np_map):
        one_mm_in_pixels, _ = pixels_mm_converters()
        # create locations
        if np_map.max() < 1:  # normalized -> this code assumes int values
            d = (np_map * n_events).astype(int)
        else:
            d = np_map.astype(int)
        # for i in range(1, d.max() + 1):
        #     for j in range(i):
        y1, y2 = np.array([]), np.array([])
        for i in [_ for _ in np.unique(d) if _ >= d.max() * cutoff and d.max() > 0]:
            locs = np.where(d >= i)
            y1 = np.append(y1, locs[0])
            y2 = np.append(y2, locs[1])
        if len(y1) > max_length or len(y2) > max_length:  # protect against map with too many pixels returned
            y1, y2 = np.array([]), np.array([])

        # y1, y2 = d.sum(axis=1), d.sum(axis=0)  # todo this is less good since will sum low pixel values
        y1 = np_map.shape[0] / 2 - y1
        y1 *= one_pixel_in_mm
        y2 = np_map.shape[1] / 2 - y2
        y2 *= one_pixel_in_mm
        return y2, y1

    def polar_histogram(np_map, ntheta, nr):
        r = np.sqrt(np_map[:, 0] ** 2 + np_map[:, 1] ** 2)
        theta = np.arctan2(np_map[:, 1], np_map[:, 0])
        r_edges = np.linspace(0, r.max(), nr)
        theta_edges = np.linspace(0, 2 * np.pi, ntheta + 1)
        H, _, _ = np.histogram2d(r, theta, [r_edges, theta_edges])
        Theta, R = np.meshgrid(theta_edges, r_edges)
        return Theta, R, H, r_edges

    _, one_pixel_in_mm = pixels_mm_converters()
    n_mm = 2

    nr = 20
    ntheta = 36

    palette = colors.LinearSegmentedColormap.from_list('rg', ["w", "r"], N=256)  # from white to red

    ax: plt.Axes
    f, ax = plt.subplots(figsize=fig_size, constrained_layout=True, dpi=dpi)
    c = ax.pcolormesh(np_map, cmap=palette, vmax=max_val, vmin=0)  # 0.000001
    # c = ax.pcolormesh(np_map, cmap='seismic', norm=colors.CenteredNorm())  # center around 0 (for negative heatmaps)
    f.colorbar(c, ax=ax.invert_yaxis())
    plt.title(title, fontsize=15)
    ax.add_artist(ScaleBar(one_pixel_in_mm, "mm", location='lower left', color='k', box_alpha=0,
                           font_properties={"size": 6}, fixed_value=n_mm))
    ax.annotate(".", (0.5, 0.5), xycoords='axes fraction', ha='center', color='tab:cyan', size=12)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    image_path = os.path.join(plot_dir, save_fig_fixname(name.replace(" ", "_") + ".jpg"))
    try:
        plt.savefig(image_path)
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
    plt.close()

    f, ax = plt.subplots(figsize=fig_size, constrained_layout=True, dpi=dpi)
    c = ax.pcolormesh(np_map, cmap=palette, vmax=across_all_max, vmin=0)  # 0.000001
    # c = ax.pcolormesh(np_map, cmap='seismic', norm=colors.CenteredNorm())  # center around 0 (for negative heatmaps)
    f.colorbar(c, ax=ax.invert_yaxis())
    plt.title(title, fontsize=15)
    ax.add_artist(ScaleBar(one_pixel_in_mm, "mm", location='lower left', color='k', box_alpha=0,
                           font_properties={"size": 6}, fixed_value=n_mm))
    ax.annotate(".", (0.5, 0.5), xycoords='axes fraction', ha='center', color='tab:cyan', size=12)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    image_norm_path = os.path.join(plot_dir, save_fig_fixname(name.replace(" ", "_") + "_normed" + ".jpg"))
    try:
        plt.savefig(image_norm_path)
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
    plt.close()

    delta = 1
    x = np.arange(1, VideoFromRaw.FRAME_ROWS + 1, delta)
    y = np.arange(1, VideoFromRaw.FRAME_COLS + 1, delta)
    xxi, yyi = np.meshgrid(x, y)

    image_cont_path = os.path.join(plot_dir, save_fig_fixname(name.replace(" ", "_") + "_contour" + ".jpg"))
    f2, ax2 = plt.subplots(figsize=fig_size, constrained_layout=True, dpi=dpi)
    try:
        cs = ax2.contourf(xxi, yyi, np.flipud(np_map), cmap=palette, vmin=0, vmax=np.max(np_map), levels=10)
        CS2 = ax2.contour(cs, levels=cs.levels[len(cs.levels) // 5::])

        print("plot: ", len(cs.levels), cs.levels, title)
        cb = f2.colorbar(cs)
        cb.add_lines(CS2)
        cb.set_label('Levels')
        plt.title(title)
        ax2.add_artist(ScaleBar(one_pixel_in_mm, "mm", location='lower left', color='k', box_alpha=0,
                                font_properties={"size": 6}, fixed_value=n_mm))
        ax2.annotate(".", (0.5, 0.5), xycoords='axes fraction', ha='center', color='tab:cyan', size=12)
        plt.savefig(image_cont_path)
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
    plt.close()

    image_kde_path = os.path.join(plot_dir, save_fig_fixname(name.replace(" ", "_") + "_kde" + ".jpg"))
    f2 = plt.figure(figsize=fig_size, dpi=dpi)
    try:
        x, y = density_calc(np.array(np_map))
        curr = sns.jointplot(x=x, y=y, kind="kde")
        curr.set_axis_labels(xlabel="Distance from fish com (mm)", ylabel="Distance from fish com (mm)")
        curr.fig.suptitle(title)
        # curr.ax_joint.collections[0].set_alpha(0)
        curr.fig.tight_layout()
        curr.fig.subplots_adjust(top=0.95)  # Reduce plot to make room
        curr.savefig(image_kde_path)
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
    plt.close()

    if xlim is not None and ylim is not None:
        image_kde_path2 = os.path.join(plot_dir, save_fig_fixname(name.replace(" ", "_") + "_kde_norm" + ".jpg"))
        f2 = plt.figure(figsize=fig_size, dpi=dpi)
        try:
            x, y = density_calc(np.array(np_map))
            curr = sns.jointplot(x=x, y=y, kind="kde", xlim=xlim, ylim=ylim)
            curr.set_axis_labels(xlabel="Distance from fish com (mm)", ylabel="Distance from fish com (mm)")
            curr.fig.suptitle(title)
            # curr.ax_joint.collections[0].set_alpha(0)
            curr.fig.tight_layout()
            curr.fig.subplots_adjust(top=0.95)  # Reduce plot to make room
            curr.savefig(image_kde_path2)
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
        plt.close()

    # f2 = plt.figure(figsize=FIG_SIZE)
    # ax_polar = plt.subplot(projection="polar")
    # Theta, R, H, r_edges = polar_histogram(np_map, ntheta=ntheta, nr=nr)
    # c = ax_polar.pcolormesh(Theta, R, H, cmap=palette, vmin=0)
    # f2.colorbar(c, ax=ax_polar.invert_yaxis())
    # ax_polar.set_theta_zero_location("N")  # zero at north
    # ax_polar.set_theta_direction(-1)  # CCW
    # ax_polar.set_thetagrids(range(0, 360, int(360 / ntheta)))  # set the gridlines
    # ax_polar.set_rgrids(r_edges)  # set the gridlines
    # ax_polar.invert_yaxis()  # fix r for being 0 in the middle
    # ax_polar.set_rorigin(-(r_edges.max() / 20))
    # # ax_polar.set_thetamin(40)
    # # ax_polar.set_thetamax(360 - 40)
    # ax_polar.set_xticklabels([])  # no r ticks
    # ax_polar.set_yticklabels([])  # no theta ticks
    # ax_polar.grid(True)
    # plt.title(title, fontsize=20)
    # ax_polar.annotate("C", (0.5, 0.5), xycoords='axes fraction', ha='center', color='tab:cyan', size=6)
    #
    # image_coord_path = os.path.join(plot_dir, "polar_" + name.replace(" ", "_") + ".jpg")
    # try:
    #     plt.savefig(image_coord_path)
    # except Exception as e:
    #     print(e)
    # plt.close()

    if curr_pdf is None:
        return
    try:
        curr_pdf.add_page()
        curr_pdf.image(image_path, 0, 0, 200, 150)
        curr_pdf.image(image_norm_path, 0, 160, 200, 150)
        curr_pdf.add_page()
        curr_pdf.image(image_cont_path, 0, 0, 200, 150)
        curr_pdf.image(image_kde_path, 0, 160, 200, 150)
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)


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
            processed_path = os.path.join(os.path.join(mat_path[:mat_path.find("data_set_features")], "data_set_features"), "inter_bout_interval")
            return os.path.join(processed_path, curr_fish + "_ibi_processed.mat")
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
                    print("Error: ", curr, e)
                    traceback.print_tb(e.__traceback__)
        dataset = FishAndEnvDataset(all_fish)
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


def calc_bout_durations(dataset, parameters: PlotsCMDParameters, age=None, age_list=[], calc_per_fish=False):
    heatmap_keys = parameters.combine_outcomes.keys()
    r = {"event_durations_per_fish": {}}
    for what in ["sum", "mean", "sem", "first", "last", "n"]:
        r["event_ibi_dur_per_fish_" + what] = {}

    event: ExpandedEvent
    paramecium: ParameciumRelativeToFish
    for fish in tqdm(dataset.fish_processed_data_set,
                     desc="Duration current fish age {0}".format(age if not parameters.is_combine_age else age_list)):
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


def get_folder_and_file_names(data_path_, parameters: PlotsCMDParameters, age_groups, create_folders=True):
    output_folder = os.path.join(data_path_, "dataset_extracted_features")
    heatmap_parent_folder = os.path.join(output_folder, "Heat_maps")
    metadata_folder = os.path.join(output_folder, "Metadata")

    # global prefix - should indicate what changes global counts etc
    # heatmap params are within heatmap path
    add_name = "combined_age_{0}_".format(age_groups) if parameters.is_combine_age else ""
    add_name += "{0}_outcome_".format(parameters.outcome_map_type)
    add_name += str(parameters.heatmap_n_paramecia_type) + "_paramecia_"

    add_heatmap_name = "bbox_" if parameters.is_bounding_box else ""
    add_heatmap_name += str(parameters.heatmap_type) + "_heatmap"
    heatmap_folder = os.path.join(heatmap_parent_folder, add_heatmap_name)

    counters_json = os.path.join(metadata_folder, add_name + 'count_metadata.txt')
    heatmap_counters_json = os.path.join(heatmap_parent_folder,
                                         add_name + add_heatmap_name + '_count_metadata.txt')
    heatmap_data_json = os.path.join(heatmap_parent_folder,
                                     add_name + add_heatmap_name + '_data.mat')

    if create_folders:
        create_dirs_if_missing([heatmap_folder, os.path.join(heatmap_folder, "all_fish"), metadata_folder,
                                os.path.join(metadata_folder, "figures"),
                                os.path.join(heatmap_folder, "contour_properties")])
    return heatmap_folder, metadata_folder, heatmap_parent_folder, counters_json, heatmap_counters_json, \
           heatmap_data_json, add_name, add_heatmap_name


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
        save_mat_dict(heatmap_data_json, total_all_age_for_mat)
        save_mat_dict(heatmap_data_json.replace(".mat", "_n_events_normalized.mat"), total_all_age_for_mat_normalized)
        save_mat_dict(heatmap_data_json.replace(".mat", "_contour_properties.mat"),
                      recursive_fix_key_for_mat(contour_properties))
        save_mat_dict(heatmap_data_json.replace(".mat", "_paramecia_in_fov.mat"),
                      recursive_fix_key_for_mat(paramecia_in_fov))
        save_mat_dict(heatmap_data_json.replace(".mat", "_paramecia_data.mat"),
                      recursive_fix_key_for_mat(paramecia_data))
        if parameters.is_save_per_fish_heatmap:
            save_mat_dict(heatmap_data_json.replace(".mat", "_per_fish.mat"), recursive_fix_key_for_mat(per_fish_maps))
            save_mat_dict(heatmap_data_json.replace(".mat", "_per_fish_paramecia_in_fov.mat"),
                          recursive_fix_key_for_mat(per_fish_paramecia_in_fov))
            save_mat_dict(heatmap_data_json.replace(".mat", "_per_fish_paramecia_data.mat"),
                          recursive_fix_key_for_mat(per_fish_paramecia_data))

        with open(heatmap_counters_json, 'w') as outfile:
            json.dump(total_heatmap_counters, outfile, sort_keys=True, indent=4)


def plot_counters(counters: dict, plot_dir, curr_pdf=None, is_combine_age=False, add_name="", dpi=100):
    if is_combine_age:
        age_names = counters.keys()
    else:
        age_names = sorted(counters.keys(), key=lambda k: int(k) if k != 'all' else 10000)
    outcome_names = counters[sorted([k for k in counters.keys() if counters[k] != {}])[0]].keys()
    n_events = np.zeros(shape=(len(age_names), len(outcome_names)))
    n_fish = np.zeros(shape=(len(age_names), len(outcome_names)))
    stat_mean_events = np.zeros(shape=(len(age_names), len(outcome_names)))
    stat_std_events = np.zeros(shape=(len(age_names), len(outcome_names)))
    for i, age in enumerate(age_names):
        outcomes_dict = counters[age]
        for j, outcome in enumerate(outcomes_dict.keys()):
            values_dict = outcomes_dict[outcome]
            n_events[i, j] = values_dict["n_events"]
            n_fish[i, j] = len(values_dict["per_fish"])
            stat_mean_events[i, j] = np.mean(values_dict["per_fish"])
            stat_std_events[i, j] = np.std(values_dict["per_fish"])

    df_fish = pd.DataFrame(n_fish, columns=outcome_names)
    df_fish.insert(0, "age_dpf", age_names)
    df = pd.DataFrame(n_events, columns=outcome_names)
    df.insert(0, "age_dpf", age_names)

    # show 'all' as title if exists
    add_n_fish_title, add_n_event_title = "", ""
    if 'all' in age_names:
        add_n_fish_title = " (all: {0})".format(int(np.max(n_fish[age_names.index('all'), :])))
        df_fish.drop(df_fish[df_fish["age_dpf"] == 'all'].index, inplace=True)
        add_n_event_title = " (all: {0})".format(int(np.max(n_events[age_names.index('all'), :])))
        df.drop(df[df["age_dpf"] == 'all'].index, inplace=True)

    stats_df = pd.concat([pd.DataFrame(stat_mean_events, columns=[_ + "_mean" for _ in outcome_names]),
                          pd.DataFrame(stat_std_events, columns=[_ + "_std" for _ in outcome_names])], axis=1)
    stats_df.insert(0, "age_dpf", age_names)

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=dpi)
    axes = df.plot(x="age_dpf", y=outcome_names, kind="bar", ax=ax)
    for bars in axes.containers:
        for bar in bars:
            plt.text(bar.get_x(), bar.get_height() * 1.01, int(bar.get_height()))
    plt.title("# of events per outcome & age" + add_n_event_title)
    plt.xticks(rotation=0)
    plt.ylabel("Number of events")
    plt.xlabel("Age (dpf)")
    image_path = os.path.join(plot_dir, save_fig_fixname(add_name + "event_outcome_number.jpg"))
    try:
        plt.savefig(image_path)
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=dpi)
    axes = df_fish.plot(x="age_dpf", y=outcome_names, kind="bar", ax=ax)
    for bars in axes.containers:
        for bar in bars:
            plt.text(bar.get_x(), bar.get_height() * 1.01, int(bar.get_height()))
    plt.title("# of fish per outcome & age" + add_n_fish_title)
    plt.ylabel("Number of fish")
    plt.xlabel("Age (dpf)")
    plt.xticks(rotation=0)
    image_fish_path = os.path.join(plot_dir, save_fig_fixname(add_name + "event_fish_number.jpg"))
    try:
        plt.savefig(image_fish_path)
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
    # plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=dpi)
    axes = stats_df[['age_dpf'] + [_ + "_mean" for _ in outcome_names]].plot(kind='bar', x="age_dpf",
                                                                             yerr=stats_df[[_ + "_std" for _ in
                                                                                            outcome_names]].values.T,
                                                                             alpha=0.2, capsize=2, ax=ax)
    ax.set_xlabel("Age (dpf)")
    ax.set_ylabel("Outcome fraction")
    ax.set_title("Outcome fraction across age")
    plt.xticks(rotation=0)

    image_stats_path = os.path.join(plot_dir, save_fig_fixname(add_name + "event_outcome_fraction.jpg"))
    try:
        plt.savefig(image_stats_path)
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
    plt.close()

    fig, ax = plt.subplots()
    table(ax, np.round(stats_df, 2), loc="center")
    ax.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    fig.tight_layout()

    image_stats_table_path = os.path.join(plot_dir, save_fig_fixname(add_name + "event_outcome_fraction_table.jpg"))
    try:
        plt.savefig(image_stats_table_path)
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
    plt.close()

    fig, ax = plt.subplots()
    table(ax, np.round(df_fish, 2), loc="center")
    ax.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    fig.tight_layout()

    image_table_path = os.path.join(plot_dir, save_fig_fixname(add_name + "event_status_table.jpg"))
    try:
        plt.savefig(image_table_path)
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
    plt.close()

    curr_pdf.add_page()
    curr_pdf.image(image_path, 0, 0, 200, 150)
    curr_pdf.image(image_fish_path, 0, 145, 200, 150)
    curr_pdf.add_page()
    curr_pdf.image(image_stats_path, 0, 0, 200, 150)
    curr_pdf.image(image_stats_table_path, 0, 145, 200, 150)


def plot_all():
    """
    Lack additional plots. Generate heatmaps and basic metadata
    :return:
    """
    if parameters.is_metadata:
        curr_pdf = MyPDF(footer_text=add_name.replace("_", " "))
        plot_counters(combine_age_counters if parameters.is_combine_age else counters,
                      plot_dir=os.path.join(metadata_folder, "figures"), add_name=add_name,
                      curr_pdf=curr_pdf, is_combine_age=parameters.is_combine_age)
        curr_pdf.output(os.path.join(metadata_folder, add_name + "summary.pdf"), "F")
        print("Done ", os.path.join(metadata_folder, add_name + "summary.pdf"))
        plt.close('all')
        logging.info("Plot metadata - done")

    if dataset is None or not parameters.is_heatmap:
        print("Skipping heatmaps")
        return

    curr_pdf = MyPDF(footer_text=(add_name + add_heatmap_name).replace("_", " "))
    for age in age_list:
        add = "" if age is None else " {0} age".format(age)
        age_key = "all" if age is None else str(age)
        total_maps = copy.deepcopy(total_all_age[age_key])
        for key in total_maps.keys():
            add_title = "(#fish = {0}, #events = {1})".format(total_heatmap_counters[age_key][key]["n_fish"],
                                                              total_heatmap_counters[age_key][key]["n_events"])
            name = "Paramecia " + str(key) + " heatmap" + add
            if total_heatmap_counters[age_key][key]["n_events"] > 0:
                np_map = total_maps[key].astype(float).copy() / total_heatmap_counters[age_key][key]["n_events"]
            else:
                np_map = total_maps[key].astype(float).copy()  # zero
            if parameters.gaussian:
                name += " gaus 5"
                np_map = gaussian_filter(np_map, 5)
            xlim, ylim = None, None
            if parameters.heatmap_type == HeatmapType.target_only:  # patch todo remove
                # if key.lower() in ["hit-spit", "hit", "spit"]:
                #     xlim, ylim = (-0.5, 0.5), (0.8, 2.2)
                # if key.lower() in ["miss"]:
                #     xlim, ylim = (-1, 1), (0.5, 3.5)
                # if key.lower() in ["abort"]:
                #     xlim, ylim = (-4, 4), (0, 8)
                xlim, ylim = (-4, 4), (-4, 4)
            elif parameters.heatmap_type == HeatmapType.no_target:  # patch todo remove
                xlim, ylim = (-30, 30), (-30, 30)
            generate_heatmap(np_map=np_map, name=add_name + name, title=name + add_title, max_val=np.max(np_map),
                             n_events=total_heatmap_counters[age_key][key]["n_events"], xlim=xlim, ylim=ylim,
                             plot_dir=heatmap_folder, curr_pdf=curr_pdf, across_all_max=across_all_max)
    name = "_plots.pdf"
    pref = "" if not parameters.gaussian else "gaussian_"
    curr_pdf.output(os.path.join(heatmap_parent_folder, pref + add_name + add_heatmap_name + name), "F")
    plt.close('all')
    logging.info("Plot total heatmaps - done")

    if parameters.is_save_per_fish_heatmap:
        fish_pdf = MyPDF(footer_text=(add_name + add_heatmap_name).replace("_", " "))
        try:
            for fish_name, fish_data_dict in tqdm(list(per_fish_maps.values())[0].items(), desc="Per-fish-heatmap"):
                for outcome in fish_data_dict["heatmaps"].keys():
                    age_dpf = fish_data_dict['age_dpf']
                    add_title = " (#events = {0}, age={1})".format(fish_data_dict["counters"][outcome]["n_events"],
                                                                   age_dpf)
                    name = fish_name.replace("-", "_") + " " + str(outcome)
                    np_map = fish_data_dict['normalized_heatmaps'][outcome]
                    if parameters.gaussian:
                        name += " g=5"
                        np_map = gaussian_filter(np_map, 5)
                    generate_heatmap(np_map=np_map, name=add_name + name, title=name + add_title,
                                     max_val=np.max(np_map), n_events=fish_data_dict["counters"][outcome]["n_events"],
                                     plot_dir=os.path.join(heatmap_folder, "all_fish"),
                                     curr_pdf=fish_pdf, across_all_max=fish_data_dict["across_fish_max"])
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
        # fish_pdf.output(os.path.join(heatmap_parent_folder,
        #                              pref + add_name + add_heatmap_name + "_per_fish_heatmap_plots.pdf"), "F")
        plt.close('all')
        logging.info("Plot fish heatmaps - done")

    logging.info("Plot contour properties - done")
    # todo per fish


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
    parser.add_argument('--heatmap_type', default=HeatmapType.all, type=lambda v: HeatmapType[v],
                        choices=list(HeatmapType),
                        help='Heatmaps with background only, target only or both (default: heatmap.all means both)')
    parser.add_argument('--outcome_map_type', default=OutcomeMapType.all, type=lambda v: OutcomeMapType[v],
                        choices=list(OutcomeMapType), help='Ways to combine outcomes')
    parser.add_argument('--heatmap_n_paramecia_type', default=HeatmapNParameciaType.all,
                        type=lambda v: HeatmapNParameciaType[v], choices=list(HeatmapNParameciaType),
                        help='Ways to combine outcomes')
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


# Run with: \\ems.elsc.huji.ac.il\avitan-lab\Lab-Shared\Analysis\FeedingAssaySapir * --is_bounding_box --outcome_map_type hit_miss_abort --is_combine_age --heatmap_type=target_only
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
    print( parameters.mat_names)

    # read this first/outside to allow loop over parameters
    dataset = None
    if parameters.is_heatmap or not parameters.is_reading_json:
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
    print("Path {0}, arguments: {1}".format(data_path, parameters))

    heatmap_folder, metadata_folder, heatmap_parent_folder, counters_json, heatmap_counters_json, heatmap_data_json, \
    add_name, add_heatmap_name = \
        get_folder_and_file_names(data_path_=data_path, parameters=parameters, age_groups=args.age_groups)

    existing_ages = list(set([fish.age_dpf for fish in dataset.fish_processed_data_set]))  # unique only
    existing_ages.sort()
    if -1 in existing_ages:
        existing_ages.remove(-1)

    print("Ages: ", existing_ages)
    print([fish.name for fish in dataset.fish_processed_data_set])
    durations_data = calc_durations_main(existing_ages,dataset, parameters)
    print("save ", heatmap_data_json.replace(".mat", "_duration_properties.mat"))
    save_mat_dict(heatmap_data_json.replace(".mat", "_duration_properties.mat"),
                  recursive_fix_key_for_mat(durations_data))

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

    plot_all()
