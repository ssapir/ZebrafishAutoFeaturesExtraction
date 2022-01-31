import enum
import logging

import numpy as np
import cv2

from classic_cv_trackers.abstract_and_common_trackers import Colors, ClassicCvAbstractTrackingAPI as cls
from feature_analysis.fish_environment.fish_processed_data import ExpandedEvent, rotate_data, ParameciaStatus
from utils.video_utils import VideoFromRaw


def resize(f):
    return cv2.resize(f, (round(f.shape[0] / 1.5), round(f.shape[1] / 1.5)))


def rotate_image(img, head_point, direction, to_center=True):
    """
    The method rotate the image such that the fish head center will be in the image center, and his direction
     will be upward
    :param to_center: True = move to center, False = move from center
    :param head_point: head center of the fish
    :param direction: fish direction in degrees relative to horizontal
    :return: it update the image to be the rotated image.
    """
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    if to_center:
        rotation_mat = cv2.getRotationMatrix2D((cX, cY), -direction + 90, 1)  # fix direction relative to vertical line
        shift_mat = np.float32([[1, 0, cX - head_point[0]], [0, 1, cY - head_point[1]]])
    else:  # from center
        rotation_mat = cv2.getRotationMatrix2D((head_point[0], head_point[1]), direction - 90, 1)
        shift_mat = np.float32([[1, 0, head_point[0] - cX], [0, 1, head_point[1] - cY]])
    return cv2.warpAffine(cv2.warpAffine(img, shift_mat, (w, h)), rotation_mat, (w, h))


def zero_mask():
    return np.zeros((VideoFromRaw.FRAME_COLS, VideoFromRaw.FRAME_ROWS))


def heatmap_per_event_type(heatmap_keys=["abort", "hit", "miss", "spit"]):
    result_maps = {}
    for key in heatmap_keys:
        result_maps[key] = zero_mask()
    return result_maps


class HeatmapNParameciaType(enum.Enum):
    all = 1
    n50 = 2
    n30 = 3
    n70 = 4

    def __str__(self):
        return self.name


class CombineAgeGroups(enum.Enum):
    v1 = 1
    v2 = 2
    v3 = 3

    def __str__(self):
        return self.name


class HeatmapType(enum.Enum):
    all_para = 1
    target_only = 2
    residuals = 3

    def __str__(self):
        return self.name


class OutcomeMapType(enum.Enum):
    all_outcome = 1
    strike_abort = 2
    hit_miss_abort = 3
    hit_miss_abort_es_abort_noes = 4

    def __str__(self):
        return self.name


class FeedingType(enum.Enum):
    all_feeding = 1
    before_feeding = 2
    after_feeding = 3

    def __str__(self):
        return self.name

    @staticmethod
    def map_feeding_str(feeding_str):
        if 'before morning' in feeding_str:
            return FeedingType.before_feeding
        elif 'after' in feeding_str:
            return FeedingType.after_feeding
        return FeedingType.all_feeding


class PlotsCMDParameters:
    """Hold default args and cmd input overrides.
    This is passed to functions (combined as 1 struct since there are many)
    """
    heatmap_type = HeatmapType.all_para
    feeding_type = FeedingType.all_feeding
    heatmap_n_paramecia_type = HeatmapNParameciaType.all
    outcome_map_type = OutcomeMapType.all_outcome
    age_groups = CombineAgeGroups.v2
    valid_n_paramecia = [30, 50]
    is_reading_json = False
    is_save_per_fish_heatmap = False
    is_heatmap = True
    is_metadata = True
    is_bounding_box = True
    gaussian = False
    is_combine_age = True
    is_saving_fish = True
    combine_ages_v1 = {'5-6': ['5', '6'], '7-9': ['7', '8', '9'], '12-15': ['12', '13', '14', '15']}
    combine_ages_v2 = {'5-7': ['5', '6', '7'], '14-15': ['14', '15']}
    combine_ages_v3 = {'5': ['5'], '15': ['15']}
    combine_ages = {}
    combine_outcomes = {'hit': ['hit'], 'spit': ['spit'], 'miss': ['miss'], 'abort': ['abort']}
    ignore_outcomes = 'no_target'
    fish = "*"
    event_number = None
    mat_names = []

    def __str__(self):
        return ("is_bounding_box? {0}, gaussian? {1}, is_combine_age? {2}, heatmap_type? {3}, outcomes map: {4}, " + \
                "is_saving_per_fish? {5}, n_paramecia_type? {6}, feeding? {7}, age_groups? {8}").format(
            self.is_bounding_box, self.gaussian, self.is_combine_age, self.heatmap_type, self.combine_outcomes,
            self.is_save_per_fish_heatmap, self.heatmap_n_paramecia_type, self.feeding_type, self.combine_ages)


def outcome_to_map(outcome_str, parameters: PlotsCMDParameters):
    """

    :param outcome_str: hit/miss/abort-.../spit-...
    :param heatmap_keys: combine_outcomes keys (can be strike for example)
    :return:
    """
    if parameters.ignore_outcomes in outcome_str:
        return parameters.ignore_outcomes, True
    for key, values_list in parameters.combine_outcomes.items():
        if any([_ for _ in values_list if _ in outcome_str]):
            return key, False
    return "Error", True


def heatmap(event: ExpandedEvent, frame_number, parameters: PlotsCMDParameters,
            target_paramecia_center=None,
            valid_paramecia_statuses=[ParameciaStatus.FROM_IMG.value, ParameciaStatus.REPEAT_LAST.value,
                                      ParameciaStatus.DOUBLE_PARA.value]):
    heatmap_keys = parameters.combine_outcomes.keys()
    is_bounding_box = parameters.is_bounding_box

    result_maps = heatmap_per_event_type(heatmap_keys=heatmap_keys)

    vis = zero_mask()  # visualize paraecia vs fish
    curr_para_dist = zero_mask()

    if not event.fish_tracking_status_list[frame_number]:  # can be false (nan's)
        return vis, result_maps, []

    if (parameters.heatmap_type == HeatmapType.target_only or
        parameters.heatmap_type == HeatmapType.residuals) and \
            np.isnan(event.paramecium.target_paramecia_index):  # todo no target as well?
        logging.error("Skip heatmap. Paramecia in fov has nan target index for {0}".format(event.event_name))
        return vis, result_maps, []

    head_point = [event.head.origin_points.x[frame_number], event.head.origin_points.y[frame_number]]
    dest_point = [event.head.destination_points.x[frame_number], event.head.destination_points.y[frame_number]]
    head_dir = event.head.directions_in_deg[frame_number]

    rotated_head_dest = rotate_data(data=np.array(dest_point), head_point=cls.point_to_int(head_point),
                                    direction=head_dir)

    if target_paramecia_center is not None:  # visualize target if requested
        cv2.rectangle(vis, cls.point_to_int([target_paramecia_center[0] - 8, target_paramecia_center[1] - 8]),
                      cls.point_to_int([target_paramecia_center[0] + 8, target_paramecia_center[1] + 8]), Colors.WHITE)

    # add all paramecia of current event to plot
    _, n_paramecia = event.paramecium.status_points.shape
    for para_ind in range(n_paramecia):
        center = event.paramecium.center_points[frame_number, para_ind, :]
        status = event.paramecium.status_points[frame_number, para_ind]
        box = event.paramecium.bounding_boxes[frame_number, para_ind]

        # filter heatmap requested type
        if parameters.heatmap_type == HeatmapType.residuals and para_ind == event.paramecium.target_paramecia_index:
            logging.debug("Ignoring target {0} in event {1} for no_target heatmap".format(para_ind, event.event_name))
            continue
        elif parameters.heatmap_type == HeatmapType.target_only and para_ind != event.paramecium.target_paramecia_index:
            logging.debug(
                "Ignoring no-target {0} in event {1} for target_only heatmap".format(para_ind, event.event_name))
            continue

        if status not in valid_paramecia_statuses:
            continue

        if is_bounding_box:
            if not np.isnan(box).all():
                cv2.drawContours(curr_para_dist, [box.astype(int)], -1, Colors.WHITE, thickness=cv2.FILLED)
        else:
            if not np.isnan(center).all() and center is not None:
                cv2.circle(vis, cls.point_to_int(center), color=Colors.WHITE, radius=1, thickness=cv2.FILLED)
                cv2.circle(curr_para_dist, cls.point_to_int(center), color=Colors.WHITE, radius=1, thickness=cv2.FILLED)

    # Change to binary + rotate the relevant map (event outcome is only one value)
    for key in result_maps.keys():
        event_key, ignored = outcome_to_map(event.outcome_str, parameters)
        if key == event_key and not ignored:
            result_maps[key] = rotate_image(curr_para_dist, cls.point_to_int(head_point), head_dir)
            _, result_maps[key] = cv2.threshold(result_maps[key], 50, Colors.WHITE[0], cv2.THRESH_BINARY)
            result_maps[key] = (result_maps[key] > 0).astype(np.int)

    return rotate_image(vis, cls.point_to_int(head_point), head_dir), result_maps, rotated_head_dest
