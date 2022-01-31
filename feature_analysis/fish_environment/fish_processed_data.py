import enum
import logging
import math
from typing import List
import warnings

import cv2
from scipy.spatial import distance
from tqdm import tqdm
import numpy as np
import pandas as pd

from classic_cv_trackers.abstract_and_common_trackers import ClassicCvAbstractTrackingAPI as cls
from utils.geometric_functions import get_angle_to_horizontal, fix_angle_range

from fish_preprocessed_data import FishPreprocessedData, Paramecium, Head, Event, Metadata, \
    get_validated_list, Tail
from utils.matlab_data_handle import save_mat_dict, load_mat_dict
from utils.video_utils import VideoFromRaw

partial = False
ANGLES = {'narrow': (np.nan, 45), 'forward': (np.nan, 90), 'front_sides': (90, 180),
          'front': (np.nan, 180), #'ang_270': (np.nan, 270), 'wide_360': (np.nan, 360),
          'tail': (90, 225)}
#DISTANCE_LIST_IN_MM = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#DISTANCE_LIST_IN_MM = np.array(range(1, 41, 1)) / 4
#DISTANCE_LIST_IN_MM = np.concatenate([np.array(range(1, 41, 1)) / 8, np.array(range(21, 41, 1)) / 4])
#DISTANCE_LIST_IN_MM = np.concatenate([np.array(range(1, 21, 1)) / 4, np.array(range(11, 21, 1)) / 2])
#DISTANCE_LIST_IN_MM = np.array(range(1, 21, 1)) / 4
DISTANCE_LIST_IN_MM = np.concatenate([np.array(range(1, 31, 1)) / 8, np.array(range(4, 11, 1))])


class ParameciaStatus(enum.Enum):
    FROM_IMG = 0
    REPEAT_LAST = 1
    PREDICT = 2
    PREDICT_AND_IMG = 3
    DOUBLE_PARA = 4

    def __str__(self):
        return self.name


def pixels_mm_converters():
    """The plate is 70 pixels from edge, and 1cm radius
    one_mm_in_pixels should be 42 pixels per mm
    """
    img = np.zeros((VideoFromRaw.FRAME_COLS, VideoFromRaw.FRAME_ROWS))
    one_mm_in_pixels = np.ceil((img.shape[1] - 70) / 20)  # magic numbers are plate's edges
    one_pixel_in_mm = 1 / one_mm_in_pixels
    return one_mm_in_pixels, one_pixel_in_mm


def get_fig_center(h=VideoFromRaw.FRAME_COLS, w=VideoFromRaw.FRAME_ROWS):
    (cX, cY) = (w // 2, h // 2)
    return cX, cY


def is_points_outside_plate(origin, angle=None, angles_list=None, distance_mm=1, plate_radius_mm=10):
    def get_fov_end(angles_list_):
        length = distance_mm * one_mm_in_pixels
        endy = origin[1] + length * np.sin(np.radians(angles_list_))
        endx = origin[0] + length * np.cos(np.radians(angles_list_))
        return endx, endy

    if angle is None and angles_list is None:
        return np.array([False])
    if angles_list is None and angle is not None:
        angles_list = [angle]
    one_mm_in_pixels, _ = pixels_mm_converters()
    plate_radius_pxls = plate_radius_mm * one_mm_in_pixels
    (cX, cY) = get_fig_center()
    [endx, endy] = get_fov_end(angles_list)
    return abs(endx - cX)**2 + abs(endy - cY)**2 > plate_radius_pxls**2


def get_fov_angles(head_angle_deg, from_angle=None, to_angle=None, step_deg=20):
    if from_angle is None or np.isnan(from_angle):
        result = np.arange(head_angle_deg - to_angle / 2, head_angle_deg + to_angle / 2, step=step_deg)
    else:
        # both sides
        result = np.arange(head_angle_deg + from_angle / 2, head_angle_deg + (to_angle - from_angle) + from_angle / 2,
                           step=step_deg * 2)
        result = np.concatenate([result, np.arange(head_angle_deg - ((to_angle - from_angle) + from_angle / 2),
                                                   head_angle_deg - from_angle / 2, step=step_deg * 2)])
    return np.unique(np.array([fix_angle_range(angle) for angle in result] + [fix_angle_range(head_angle_deg)]))


def is_paramecia_in_fov(paramecium, frame_number, para_ind, distance_in_mm=None, from_angle=None, to_angle=None):
    """

    :param paramecium:
    :param frame_number:
    :param para_ind:
    :param distance:
    :param from_angle: can be None (if not upper limit)
    :param to_angle:
    :return:
    """
    is_in_fov = True
    if to_angle is not None and distance_in_mm is not None:
        para_dist = paramecium.distance_from_fish_in_mm[frame_number, para_ind]
        para_rel_ang = abs(paramecium.diff_from_fish_angle_deg[frame_number, para_ind])
        if abs(para_rel_ang) > 180:
            para_rel_ang = 360 - para_rel_ang
        if from_angle is None or np.isnan(from_angle):
            if not (para_dist < distance_in_mm and para_rel_ang < to_angle / 2):
                is_in_fov = False
        else:
            if not (para_dist < distance_in_mm and from_angle / 2 < para_rel_ang < (to_angle - from_angle) + from_angle / 2):
                is_in_fov = False
    return is_in_fov


def rotate_data(data, head_point, direction, h=VideoFromRaw.FRAME_COLS, w=VideoFromRaw.FRAME_ROWS, to_center=True):
    """
    The method rotate the data such that the fish head center will be in the image center, and his direction
     will be upward
    :param data: data to rotate and shift
    :param w: image width (center calc)
    :param h: image height (center calc)
    :param to_center: True = move to center, False = move from center
    :param head_point: head center of the fish
    :param direction: fish direction in degrees relative to horizontal
    :return: return result
    """
    (cX, cY) = (w // 2, h // 2)
    if to_center:
        rotation_mat = cv2.getRotationMatrix2D((cX, cY), -direction + 90, 1)  # fix direction relative to vertical line
        shift_mat = np.float32([[1, 0, cX - head_point[0]], [0, 1, cY - head_point[1]], [0, 0, 1]])
    else:  # from center
        rotation_mat = cv2.getRotationMatrix2D((head_point[0], head_point[1]), direction - 90, 1)
        shift_mat = np.float32([[1, 0, head_point[0] - cX], [0, 1, head_point[1] - cY], [0, 0, 1]])
    return np.dot(rotation_mat, np.dot(shift_mat, data.tolist() + [1]))  # todo out of limits?


def get_function_arg_names(f):
    import inspect
    result = list(inspect.signature(f).parameters.keys())
    if "self" in result:
        result.remove("self")
    if "cls" in result:
        result.remove("cls")
    return result


def get_class_members(class_instance):
    # result = [_ for _ in dir(class_instance) if not (_.startswith('__') and _.endswith('__'))
    #           and not callable(getattr(class_instance, _)) and not isinstance(getattr(type(class_instance), _), property)]
    # result = [_ for _ in result if _.startswith('_') or "_" + _ not in result]
    return [k for (k, v) in vars(class_instance).items() if not isinstance(v, property)]


def get_array_class_members(class_instance):
    result = get_class_members(class_instance)
    result = [_ for _ in result if isinstance(getattr(class_instance, _), (list, np.ndarray))]
    return result


def get_custom_class_members(class_instance):
    result = get_class_members(class_instance)
    result = [_ for _ in result if hasattr(getattr(class_instance, _), '__dict__')]
    return result


class ParameciumRelativeToFish(Paramecium):
    _distance_from_fish_in_mm = []
    _angle_deg_from_fish = []
    _diff_from_fish_angle_deg = []
    _edge_points = []
    _field_angle = []
    _field_of_view_status = []
    _ibi_length_in_secs = []
    _target_paramecia_index = None

    @property
    def distance_from_fish_in_mm(self):
        return self._distance_from_fish_in_mm

    @property
    def angle_deg_from_fish(self):
        return self._angle_deg_from_fish

    @property
    def diff_from_fish_angle_deg(self):
        return self._diff_from_fish_angle_deg

    @property
    def edge_points(self):
        return self._edge_points

    @property
    def field_angle(self):
        return self._field_angle

    @property
    def field_of_view_status(self):
        return self._field_of_view_status

    @property
    def ibi_length_in_secs(self):
        return self._ibi_length_in_secs

    @property
    def velocity_norm(self):
        return self._velocity_norm

    @property
    def velocity_direction(self):
        return self._velocity_direction

    @property
    def velocity_towards_fish(self):
        return self._velocity_towards_fish

    @property
    def velocity_orthogonal(self):
        return self._velocity_orthogonal

    @property
    def target_paramecia_index(self):
        return self._target_paramecia_index

    def __init__(self, paramecium_to_copy: Paramecium, distance_from_fish_in_mm=[], angle_deg_from_fish=[],
                 diff_from_fish_angle_deg=[], edge_points=[], field_angle=[], field_of_view_status=[],
                 ibi_length_in_secs=[], target_paramecia_ind=np.nan,
                 velocity_norm=[], velocity_direction=[], velocity_towards_fish=[], velocity_orthogonal=[]):
        super().__init__(center=paramecium_to_copy.center_points, area=paramecium_to_copy.area_points,
                         status=paramecium_to_copy.status_points, color=paramecium_to_copy.color_points,
                         ellipse_majors=paramecium_to_copy.ellipse_majors,
                         ellipse_minors=paramecium_to_copy.ellipse_minors,
                         ellipse_dirs=paramecium_to_copy.ellipse_dirs,
                         bounding_boxes=paramecium_to_copy.bounding_boxes)
        self._distance_from_fish_in_mm = get_validated_list(distance_from_fish_in_mm, float)
        self._angle_deg_from_fish = get_validated_list(angle_deg_from_fish, float)
        self._diff_from_fish_angle_deg = get_validated_list(diff_from_fish_angle_deg, float)
        self._edge_points = get_validated_list(edge_points, float)
        self._field_angle = get_validated_list(field_angle, float)
        self._velocity_norm = get_validated_list(velocity_norm, float)
        self._velocity_direction = get_validated_list(velocity_direction, float)
        self._velocity_towards_fish = get_validated_list(velocity_towards_fish, float)
        self._velocity_orthogonal = get_validated_list(velocity_orthogonal, float)
        self._ibi_length_in_secs = get_validated_list(ibi_length_in_secs, float)
        self._field_of_view_status = get_validated_list(field_of_view_status, float)
        self._target_paramecia_index = target_paramecia_ind

    def export_to_struct(self):  # this is an example of saving points only (centers) of one trajectory
        result = super().export_to_struct()
        result['distance_from_fish_in_mm'] = np.array(self._distance_from_fish_in_mm, dtype=np.float)
        result['angle_deg_from_fish'] = np.array(self._angle_deg_from_fish, dtype=np.float)
        result['diff_from_fish_angle_deg'] = np.array(self._diff_from_fish_angle_deg, dtype=np.float)
        result['edge_points'] = np.array(self._edge_points, dtype=np.float)
        result['field_angles_deg'] = np.array(self._field_angle, dtype=np.float)
        result['velocity_norm_mm_sec'] = np.array(self._velocity_norm, dtype=np.float)
        result['velocity_direction_mm_sec'] = np.array(self._velocity_direction, dtype=np.float)
        result['velocity_towards_fish_mm_sec'] = np.array(self._velocity_towards_fish, dtype=np.float)
        result['velocity_orthogonal_mm_sec'] = np.array(self._velocity_orthogonal, dtype=np.float)
        result['field_of_view_status'] = np.array(self._field_of_view_status, dtype=np.float)  # can be nan
        result['ibi_length_in_secs'] = np.array(self._ibi_length_in_secs, dtype=np.float)  # can be nan
        result['target_paramecia_index_from_0'] = self._target_paramecia_index
        return result

    @classmethod
    def import_from_struct(cls, data):  # match ctor
        return cls(Paramecium.import_from_struct(data),
                   distance_from_fish_in_mm=data['distance_from_fish_in_mm'],
                   angle_deg_from_fish=data['angle_deg_from_fish'],
                   diff_from_fish_angle_deg=data['diff_from_fish_angle_deg'],
                   edge_points=data['edge_points'],
                   field_angle=data['field_angles_deg'],
                   field_of_view_status=data['field_of_view_status'],
                   ibi_length_in_secs=data['ibi_length_in_secs'],
                   target_paramecia_ind=data.get('target_paramecia_index_from_0', np.nan),
                   velocity_norm=data['velocity_norm_mm_sec'],
                   velocity_direction=data['velocity_direction_mm_sec'],
                   velocity_towards_fish=data['velocity_towards_fish_mm_sec'],
                   velocity_orthogonal=data['velocity_orthogonal_mm_sec'])

    @staticmethod
    def velocity2angle(velocity_towards_fish, orthogonal_velocity):
        """
        :param velocity_towards_fish: paramecia velocity on axis towards fish
        :param orthogonal_velocity: paramecia velocity on axis orthogonal to towards fish
        :return:
        """
        if orthogonal_velocity == 0:# todo fix 1
            return 0 if velocity_towards_fish > 0 else math.degrees(math.pi)
        return math.degrees(abs(math.atan(velocity_towards_fish / orthogonal_velocity) - math.pi / 2))

    @staticmethod
    def frames_to_secs_converter(n_curr_frames):
        return n_curr_frames / float(VideoFromRaw.FPS)

    def calc_ibi_velocity(self, head: Head, event_name: str, starting_bout_indices, ending_bout_indices):
        _, n_paramecia = self.center_points.shape[0:2]
        n_frames = len([(s - e) for (s, e) in zip(starting_bout_indices[1:], ending_bout_indices)])
        self._velocity_norm = np.full((n_frames, n_paramecia), fill_value=np.nan)
        self._velocity_direction = np.full((n_frames, n_paramecia), fill_value=np.nan)
        self._velocity_towards_fish = np.full((n_frames, n_paramecia), fill_value=np.nan)
        self._velocity_orthogonal = np.full((n_frames, n_paramecia), fill_value=np.nan)
        self._ibi_length_in_secs = np.full((n_frames, n_paramecia), fill_value=np.nan)

        head_origin_x: np.ndarray = head.origin_points.x
        head_origin_y: np.ndarray = head.origin_points.y
        head_direction_angle_list: np.ndarray = head.directions_in_deg
        if len(head_origin_x) < n_frames or len(head_origin_x) != len(head_origin_y):
            logging.error("Error. Fish event {0} has wrong number of paramecia centers relative to fish".format(
                event_name))
            return False
        if not np.array([(s - e) > 0 for (s, e) in zip(starting_bout_indices[1:], ending_bout_indices)]).all():
            logging.error("Error. Fish event {0} has wrong IBIs (skip paramecia): start {1} end {2}".format(
                event_name, starting_bout_indices, ending_bout_indices))
            return False

        if n_frames < 1:
            logging.error("Fish event {0} has wrong #IBIs (skip paramecia): start {1} end {2}".format(
                event_name, starting_bout_indices, ending_bout_indices))
            return False

        one_mm_in_pixels, _ = pixels_mm_converters()

        # IBI: start-next-bout minus end-curr-bout
        for i, (frame_ind, prev_frame_ind) in tqdm(enumerate(zip(starting_bout_indices[1:], ending_bout_indices)),
                                                   desc="frame number", disable=True):
            ibi_frames = frame_ind - prev_frame_ind
            if ibi_frames < 0:
                logging.error("Fish event {0} has negative ibi frames for start frame {1} & end frame {2}. Skip".format(
                    event_name, frame_ind, prev_frame_ind))
                continue
            elif ibi_frames < 10:
                logging.warning(
                    "Fish event {0} has < 10 ibi length for frames {2}-{1}".format(
                        event_name, frame_ind, prev_frame_ind))
            head_origin_point = [head_origin_x[frame_ind], head_origin_y[frame_ind]]
            head_direction_angle = head_direction_angle_list[frame_ind]
            to_mm_per_seconds = 1.0 / (one_mm_in_pixels * self.frames_to_secs_converter(ibi_frames))
            per_seconds = 1.0 / (self.frames_to_secs_converter(ibi_frames))
            if np.isnan(head_origin_point).any() or np.isnan(head_direction_angle).any():
                logging.error(
                    "Fish event {0} has nan values in either of frames {1} & {2} for head origin/angle. Skip".format(
                     event_name, frame_ind, prev_frame_ind))
                continue

            if not (np.isnan([head_origin_x[prev_frame_ind], head_origin_y[prev_frame_ind]]).any()):
                prev_fish_head = [head_origin_x[prev_frame_ind], head_origin_y[prev_frame_ind]]
                prev_direction = head_direction_angle_list[prev_frame_ind]
                logging.info("{0}: d head (pixels): {1}".format(event_name, distance.euclidean(prev_fish_head, head_origin_point)))
                logging.info("{0}: d dir (deg): {1}".format(event_name, np.diff([prev_direction, head_direction_angle])))

            for para_ind in tqdm(range(n_paramecia), desc="paramecia number", disable=True):
                point = self.center_points[frame_ind, para_ind, :]
                prev_point = self.center_points[prev_frame_ind, para_ind, :]
                if np.isnan(prev_point).any() or np.isnan(point).any():
                    # print("Error. Fish event ", event_name, " has nan values in either of frames ", frame_ind,
                    #       prev_frame_ind, " paramecia ", para_ind)
                    pass
                else:
                    center_shifted = rotate_data(data=point, head_point=cls.point_to_int(head_origin_point),
                                                 direction=head_direction_angle)
                    prev_center = rotate_data(data=prev_point, direction=head_direction_angle,
                                              head_point=cls.point_to_int(head_origin_point))

                    self._velocity_norm[i, para_ind] = distance.euclidean(prev_center, center_shifted)
                    self._velocity_towards_fish[i, para_ind] = prev_center[1] - center_shifted[1]  # y axis after shift
                    self._velocity_norm[i, para_ind] *= to_mm_per_seconds
                    self._velocity_towards_fish[i, para_ind] *= to_mm_per_seconds
                    self._velocity_orthogonal[i, para_ind] = \
                        math.sqrt(abs(self._velocity_norm[i, para_ind] ** 2 -
                                      self._velocity_towards_fish[i, para_ind] ** 2))
                    self._velocity_direction[i, para_ind] = \
                        self.velocity2angle(self._velocity_towards_fish[i, para_ind],
                                            self._velocity_orthogonal[i, para_ind])
                    self._ibi_length_in_secs[i, para_ind] = self.frames_to_secs_converter(ibi_frames)
        return True

    def calc_paramecia_env_features(self, head: Head, event_name: str):
        """For each fish,

        self.center_points is #frames x #paramecia x 2
        self.status_points is #frames x #paramecia

        :return:
        """
        if len(self.center_points.shape) < 2:
            logging.error("Fish event {0} has wrong bad paramecia centers (shape {1}). Skip".format(
                event_name, self.center_points.shape))
            return False

        n_frames, n_paramecia = self.center_points.shape[0:2]
        if n_frames == 0 or n_paramecia == 0:
            logging.error("Fish event {0} has bad #paramecia {1} or #frames {2}. Skip".format(
                event_name, n_paramecia, n_frames))
            return False

        self._distance_from_fish_in_mm = np.full((n_frames, n_paramecia), fill_value=np.nan)
        self._angle_deg_from_fish = np.full((n_frames, n_paramecia), fill_value=np.nan)
        self._diff_from_fish_angle_deg = np.full((n_frames, n_paramecia), fill_value=np.nan)
        self._edge_points = np.full((n_frames, n_paramecia, 4, 2), fill_value=np.nan)
        self._field_angle = np.full((n_frames, n_paramecia), fill_value=np.nan)
        self._field_of_view_status = np.full((n_frames, n_paramecia, len(ANGLES.keys()), len(DISTANCE_LIST_IN_MM)),
                                             fill_value=False)

        head_origin_x: np.ndarray = head.origin_points.x
        head_origin_y: np.ndarray = head.origin_points.y
        head_direction_angle_list: np.ndarray = head.directions_in_deg
        if len(head_origin_x) != n_frames or len(head_origin_x) != len(head_origin_y):
            logging.error("Error. Fish event {0} has wrong number of paramecia centers relative to fish. Skip".format(
                event_name))
            return False

        one_mm_in_pixels, _ = pixels_mm_converters()

        for frame_ind in tqdm(range(n_frames), desc="frame number", disable=True):
            for para_ind in tqdm(range(n_paramecia), desc="paramecia number", disable=True):
                point = self.center_points[frame_ind, para_ind, :]
                head_origin_point = [head_origin_x[frame_ind], head_origin_y[frame_ind]]
                head_direction_angle = head_direction_angle_list[frame_ind]
                # major, minor, direction = self.ellipse_majors[frame_ind, para_ind], \
                #     self.ellipse_minors[frame_ind, para_ind], self.ellipse_dirs[frame_ind, para_ind]

                if not np.isnan([point, head_origin_point]).any():
                    self._distance_from_fish_in_mm[frame_ind, para_ind] = \
                        distance.euclidean(point, head_origin_point) / one_mm_in_pixels
                    direction_angle = get_angle_to_horizontal(head_origin_point, point)  # fish point of view
                    abs_angle_deg = fix_angle_range(direction_angle)
                    diff_angle = fix_angle_range(abs(direction_angle - head_direction_angle))
                    if diff_angle > 180:
                        diff_angle = abs(360 - diff_angle)
                    diff_from_fish_direction_deg = diff_angle * np.sign(direction_angle - head_direction_angle)
                    self._angle_deg_from_fish[frame_ind, para_ind] = abs_angle_deg
                    self._diff_from_fish_angle_deg[frame_ind, para_ind] = diff_from_fish_direction_deg

                    for i, (name, (from_angle, to_angle)) in enumerate(ANGLES.items()):
                        for j, curr_distance in enumerate(DISTANCE_LIST_IN_MM):
                            is_in_fov = is_paramecia_in_fov(self, frame_ind, para_ind, curr_distance, from_angle,
                                                            to_angle)
                            if is_in_fov:
                                self._field_of_view_status[frame_ind, para_ind, i, j] = True

                    # xc, yc = point
                    # edge_points = []
                    # for curr in [major, minor]:
                    #     xtop, ytop = xc + math.cos(direction) * curr / 2, yc + math.sin(direction) * curr / 2
                    #     xbot, ybot = xc - math.cos(direction) * curr / 2, yc - math.sin(direction) * curr / 2
                    #     edge_points += [[xtop, ytop], [xbot, ybot]]
                    # self._edge_points[frame_ind, para_ind, :, :] = edge_points

                    diff_angles = []
                    if len(self.bounding_boxes) > 0:
                        for para_point in self.bounding_boxes[frame_ind, para_ind, :, :]:  # edge_points
                            direction_angle = get_angle_to_horizontal(head_origin_point,
                                                                      para_point)  # fish point of view
                            diff_from_fish_direction_deg = fix_angle_range(
                                abs(direction_angle - head_direction_angle)) * np.sign(
                                direction_angle - head_direction_angle)
                            diff_angles.append(diff_from_fish_direction_deg)
                        high_angle, low_angle = max(diff_angles), min(diff_angles)

                        field_angle = fix_angle_range(abs(high_angle - low_angle))
                        if field_angle > 180:
                            field_angle = abs(360 - field_angle)
                        self._field_angle[frame_ind, para_ind] = field_angle

        if partial:
            del self._ellipse_minors
            self._ellipse_minors = []
            del self._ellipse_majors
            self._ellipse_majors = []
            del self._ellipse_dirs
            self._ellipse_dirs = []
            del self._edge_points
            self._edge_points = []

        return True


class ExpandedEvent(Event):
    def __init__(self, event_to_copy: Event, paramecium: ParameciumRelativeToFish,
                 starting_bout_indices=[], ending_bout_indices=[], frame_indices=[], is_inter_bout_interval_only=False):
        expected_keys = get_function_arg_names(Event.__init__)
        expected_keys = [(key, [k for k in event_to_copy.__dict__.keys() if k.endswith(key)][0])
                         for key in expected_keys if "paramecium" not in key]
        super().__init__(**dict([(k, event_to_copy.__dict__[inner_k]) for (k, inner_k) in expected_keys]),
                         paramecium=paramecium)
        # dynamically copy metadata expected keys
        expected_keys = get_function_arg_names(Event.set_metadata)
        self.set_metadata(**dict([(k, event_to_copy.__dict__[k]) for k in expected_keys]))

        add = " (event {0})".format(event_to_copy.event_name)
        self.starting_bout_indices = get_validated_list(starting_bout_indices, inner_type=(int, np.int32, np.int64), add=add)
        self.ending_bout_indices = get_validated_list(ending_bout_indices, inner_type=(int, np.int32, np.int64), add=add)
        self.frame_indices = get_validated_list(frame_indices, inner_type=(int, np.int32, np.int64), add=add)
        self.is_inter_bout_interval_only = is_inter_bout_interval_only

    @staticmethod
    def calc_velocity_norm(x, y):
        tail_points = np.array([x, y]).transpose()
        tail_point_differences = np.diff((tail_points[1:], tail_points[0:-1]), axis=0)[0]
        zeros_for_padding = np.zeros(shape=(1, 2),)
        tail_point_differences_padded = np.concatenate([zeros_for_padding, tail_point_differences])
        velocity_norms = np.linalg.norm(x=tail_point_differences_padded, axis=1)
        return velocity_norms

    @staticmethod
    def fix_is_bout_detection(tail: Tail, event_name, bout_threshold=3.1, min_len=6, min_len_ibi=5, allowed_holes=3):
        """Patch to fix issues in previous analysis

        :param event:
        :return:
        """
        def calc_start_end(is_bouts_):
            to_list = lambda x: x if isinstance(x, (list, np.ndarray)) else np.array([x])
            diffs = np.diff(np.asarray(np.append(is_bouts_, is_bouts_[-1]), dtype=int))  # -1 is end. 1 is start
            starting_indices = to_list(np.where(diffs == 1)[0])
            ending_indices = to_list(np.where(diffs == -1)[0])
            return starting_indices, ending_indices
        if np.isnan(tail.velocity_norms).all():
            logging.error("Event {0} has all nan velocity norms. Calculate it myself".format(event_name))
            tail.velocity_norms = ExpandedEvent.calc_velocity_norm(x=tail.tail_tip_point_list.x, y=tail.tail_tip_point_list.y)
        is_bouts = tail.velocity_norms >= bout_threshold
        # fill small holes- should be minimal since it pushes the start-end of segments by 1-3 frames
        is_bouts = np.array(pd.Series(is_bouts).rolling(window=1 * 2 + 1, min_periods=1).median()).astype(bool)

        # fix via loops and not rolling median/sum since it trims the edged
        start, end = calc_start_end(is_bouts)
        if len(start) == len(end):
            for s, e in zip(start[1:], end[:-2]):
                if s - e < min_len_ibi:
                    is_bouts[e:(s + 1)] = True
            start, end = calc_start_end(is_bouts)
            for s, e in zip(start, end):
                if e - s < min_len:
                    is_bouts[s:(e + 1)] = False
        else:
            logging.info("Event {0} has unequal start-end. Using rolling fix".format(event_name))
            is_bouts = np.array(pd.Series(is_bouts).rolling(window=allowed_holes * 2 + 1,
                                                            min_periods=1).median()).astype(bool)
            is_bouts_temp = np.array(pd.Series(is_bouts)
                                     .rolling(window=int(min_len * 1.5), min_periods=1).sum() >= min_len).astype(bool)
            is_bouts = np.bitwise_or(is_bouts, is_bouts_temp)

        return is_bouts

    @staticmethod
    def start_end_bout_indices(event: Event):
        """ todo bugs
        Example: 20200720-f2-3 will return [  5,  84, 157, 198], [ 43, 127, 180] (start & end respectively)
        :param event:
        :return:
        """
        to_list = lambda x: x if isinstance(x, (list, np.ndarray)) else np.array([x])
        is_bouts = event.tail.is_bout_frame_list[:event.event_frame_ind]
        diffs = np.diff(np.asarray(np.append(is_bouts, is_bouts[-1]), dtype=int))  # -1 is end. 1 is start
        starting_indices = to_list(np.where(diffs == 1)[0])
        ending_indices = to_list(np.where(diffs == -1)[0])
        if len(ending_indices) == 0:  # bout end is event frame ind
            ending_indices = to_list(event.event_frame_ind)

        # if len(starting_indices) + 1 == len(ending_indices) and ((starting_indices[1:]-ending_indices)>0).all():
        #     logging.debug("Event {0} all good".format(event.event_name))

        if len(ending_indices) == len(starting_indices):
            if np.array([(s - e) > 0 for (s, e) in zip(starting_indices[1:], ending_indices[:-1])]).all():
                ending_indices = ending_indices[:-1]
        elif len(ending_indices) == len(starting_indices) + 1:
            if np.array([(s - e) > 0 for (s, e) in zip(starting_indices, ending_indices[:-1])]).all():
                starting_indices = np.array([1] + starting_indices)

        # make sure IBIs are identified as expected (without length, but all is correct for IBI count)
        if not (len(ending_indices) + 1 == len(starting_indices) and ((starting_indices[1:]-ending_indices) > 0).all()) \
           or not (len(ending_indices) == len(starting_indices) and ((starting_indices[1:]-ending_indices[-1]) > 0).all()):
            logging.error("start_end_bout_indices- Fish event {0} has wrong IBIs: start {1} end {2}".format(
                event.event_name, starting_indices, ending_indices))

        return starting_indices, ending_indices

    @staticmethod
    def inter_bout_interval_range(event: Event):
        to_list = lambda x: x if isinstance(x, (list, np.ndarray)) else np.array([x])

        starting_bout_indices, ending_bout_indices = \
            to_list(event.starting_bout_indices), to_list(event.ending_bout_indices)
        if len(starting_bout_indices) == 0 or len(ending_bout_indices) == 0:
            starting_bout_indices, ending_bout_indices = ExpandedEvent.start_end_bout_indices(event)
        if len(starting_bout_indices) == 0 or len(ending_bout_indices) == 0:  # this is an error in start_end_bout_indices
            return []

        if len(starting_bout_indices) == len(ending_bout_indices):
            pass
        elif len(starting_bout_indices) == len(ending_bout_indices) + 1:
            ending_bout_indices = np.concatenate([ending_bout_indices, [event.event_frame_ind]])
        else:
            return []
        indices = [range(e + 1, s) for (s, e) in zip(starting_bout_indices[1:], ending_bout_indices[:-1])]
        return np.concatenate(indices), indices

    @classmethod
    def from_preprocessed(cls, event_to_copy: Event):
        """Copy common neeed data while expanding paramecium

        :param event_to_copy:
        """
        # filter errors in annotation
        if "no-target" in event_to_copy.outcome_str or event_to_copy.outcome_str == "":
            logging.info("Event {0} is filtered due to outcome {1}".format(event_to_copy.event_name,
                                                                           event_to_copy.outcome_str))
            return None, event_to_copy.event_name, "outcome-{0}".format(event_to_copy.outcome_str)

        event_to_copy.tail.is_bout_frame_list = ExpandedEvent.fix_is_bout_detection(event_to_copy.tail,
                                                                                    event_to_copy.event_name)

        paramecium = ParameciumRelativeToFish(event_to_copy.paramecium)
        is_data_good = paramecium.calc_paramecia_env_features(event_to_copy.head, event_to_copy.event_name)
        if not is_data_good:
            logging.info("{1} event {0} is filtered due to bad para-features results".format(event_to_copy.event_name,
                                                                                             event_to_copy.outcome_str))
            return None, event_to_copy.event_name, "bad-features"

        starting_bout_indices, ending_bout_indices = cls.start_end_bout_indices(event_to_copy)
        if len(starting_bout_indices) == 0 or len(ending_bout_indices) == 0:
            logging.info("{1} event {0} is filtered due to empty start/end (skip paramecia): start {2} end {3}".format(
                event_to_copy.event_name, event_to_copy.outcome_str, starting_bout_indices, ending_bout_indices))
            return None, event_to_copy.event_name, "zero-start-or-end"

        frame_indices = cls.get_frame_indices(starting_bout_indices, ending_bout_indices)
        is_data_good = paramecium.calc_ibi_velocity(event_to_copy.head, event_to_copy.event_name,
                                                    starting_bout_indices, ending_bout_indices)
        if not is_data_good:
            logging.info("{1} event {0} is filtered due to bad velocity results".format(event_to_copy.event_name,
                                                                                        event_to_copy.outcome_str))
            return None, event_to_copy.event_name, "bad-velocity"

        paramecium._target_paramecia_index = get_target_paramecia_index_expanded(starting_bout_indices,
                                                                                 ending_bout_indices,
                                                                                 event_to_copy.event_frame_ind,
                                                                                 paramecium,
                                                                                 event_to_copy.outcome_str,
                                                                                 event_name=event_to_copy.event_name)

        if np.isnan(paramecium.target_paramecia_index):
            logging.info("{1} event {0} is filtered due to nan target".format(event_to_copy.event_name,
                                                                              event_to_copy.outcome_str))
            return None, event_to_copy.event_name, "nan-target"

        result = cls(event_to_copy, paramecium, is_inter_bout_interval_only=False, frame_indices=frame_indices,
                     starting_bout_indices=starting_bout_indices, ending_bout_indices=ending_bout_indices)

        expected_keys = get_function_arg_names(result.set_metadata)
        result.set_metadata(**dict([(k, event_to_copy.__dict__[k]) for k in expected_keys]))
        return result, event_to_copy.event_name, "great"

    def export_to_struct(self):
        result = super().export_to_struct()
        for attrib in ["starting_bout_indices", "ending_bout_indices", "is_inter_bout_interval_only", "frame_indices"]:
            if hasattr(self, attrib):
                result[attrib] = getattr(self, attrib)
        return result

    @classmethod
    def import_from_struct(cls, data):
        event = Event.import_from_struct(data)
        paramecium = ParameciumRelativeToFish.import_from_struct(data['paramecium'])
        if all(_ in data.keys() for _ in
               ["starting_bout_indices", "ending_bout_indices", "is_inter_bout_interval_only"]):
            result = cls(event, paramecium,
                         starting_bout_indices=data["starting_bout_indices"],
                         ending_bout_indices=data["ending_bout_indices"],
                         frame_indices=data.get("frame_indices", cls.get_frame_indices(data["starting_bout_indices"],
                                                                                       data["ending_bout_indices"])),
                         is_inter_bout_interval_only=data["is_inter_bout_interval_only"])
        else:
            result = cls(event, paramecium, is_inter_bout_interval_only=False)
        return result

    def change_to_ibi_only_data(self, only_start_end=True):
        """Recursive search inner nd-arrrays. Assuming n_frames is 0th axis (besides that, no assumptions).

        :param only_start_end:
        :return:
        """

        def change_recursive(class_instance):
            change_arrays(class_instance)
            for attrib in get_custom_class_members(class_instance):
                change_recursive(getattr(class_instance, attrib))

        def change_arrays(class_instance):
            for attrib in get_array_class_members(class_instance):
                value = np.asarray(getattr(class_instance, attrib))
                if value.shape[0] == n_frames and np.max(frame_indices) < n_frames:
                    value = value[frame_indices]
                    setattr(class_instance, attrib, value)

        starting_bout_indices, ending_bout_indices = self.start_end_bout_indices(self)  # pass self to static method
        if len(starting_bout_indices) == 0 or len(ending_bout_indices) == 0:
            return

        n_frames = self.fish_tracking_status_list.shape[0]
        frame_indices = self.get_frame_indices(starting_bout_indices=starting_bout_indices,
                                               ending_bout_indices=ending_bout_indices, only_start_end=only_start_end)
        change_recursive(self)
        self.is_inter_bout_interval_only = True
        self.starting_bout_indices = starting_bout_indices
        self.ending_bout_indices = ending_bout_indices
        self.frame_indices = frame_indices

    @staticmethod
    def get_frame_indices(starting_bout_indices, ending_bout_indices, only_start_end=True):
        to_list = lambda x: x if isinstance(x, (list, np.ndarray)) else np.array([x])
        if only_start_end:
            starting_bout_indices, ending_bout_indices = to_list(starting_bout_indices), to_list(ending_bout_indices)
            if len(starting_bout_indices) == 0 or len(ending_bout_indices) == 0:
                return []
            frame_indices = np.concatenate((starting_bout_indices, ending_bout_indices))  # todo pairs?
        else:
            frame_indices, _ = ExpandedEvent.inter_bout_interval_range()
        return frame_indices


def get_target_paramecia_index(event: ExpandedEvent):
    """
    outcomes {0: 'abort,escape', 1: 'miss', 2: 'spit', 3: 'hit', 4: 'abort,no-escape', 5: 'abort,no-target'}

    Example: 20200720-f2-3 will return 54
    :param event:
    :return:
    """
    starting, ending = ExpandedEvent.start_end_bout_indices(event)
    return get_target_paramecia_index_expanded(starting, ending, event.event_frame_ind, event.paramecium,
                                               event.outcome_str, event.event_name)


def get_target_paramecia_index_expanded(starting, ending, event_frame_ind, para: ParameciumRelativeToFish,
                                        outcome_str, event_name, max_hit_distance_in_mm=5, max_hit_angle_in_deg=22.5,
                                        max_abort_distance_in_mm=15, max_abort_angle_in_deg=22.5):
    """
    outcomes {0: 'abort,escape', 1: 'miss', 2: 'spit', 3: 'hit', 4: 'abort,no-escape', 5: 'abort,no-target'}

    Example: 20200720-f2-3 will return 54
    :return:
    """
    def validate_max_distance_angle(paramecia_indices, max_distance_in_mm, max_angle_in_deg):
        # search back x frames for close enough (distance < 40, max angle < 10) paramecia
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            near_paramecia = np.where(np.nanmin(np.abs(from_fish_distances), axis=0) < max_distance_in_mm)[0]
            disappeared_paramecia = np.intersect1d(paramecia_indices, near_paramecia)  # make sure it's near
            in_front_paramecia = np.where(np.nanmin(np.abs(from_fish_diff_angles), axis=0) < max_angle_in_deg)[0]
            disappeared_paramecia = np.intersect1d(disappeared_paramecia, in_front_paramecia)  # make sure it's near
        return disappeared_paramecia

    def check_hit(recent_frame, older_frame):
        recent_para_indices = np.where(np.isnan(from_fish_distances[recent_frame, :]))[0]
        older_para_indices = np.where(np.isnan(from_fish_distances[older_frame, :]))[0]
        disappeared_paramecia = np.setdiff1d(recent_para_indices, older_para_indices)
        disappeared_paramecia = validate_max_distance_angle(disappeared_paramecia,
                                                            max_distance_in_mm=max_hit_distance_in_mm,
                                                            max_angle_in_deg=max_hit_angle_in_deg)
        if len(disappeared_paramecia) == 1:
            return disappeared_paramecia[0]
        return None

    to_frame_ind = event_frame_ind  # always end paramecia should be the one removed when event ended
    for n_ibis_back in [1, 2, 3]:
        if len(ending) < n_ibis_back:
            logging.error("Target paramecia return nan for {2} event {0} ({1} ibis!)".format(event_name, len(ending), outcome_str))
            return np.nan

        from_frame_ind = ending[-n_ibis_back]
        # start_frame_ind = starting[-n_ibis_back]

        from_fish_distances = para.distance_from_fish_in_mm[from_frame_ind:(to_frame_ind + 1), :]
        from_fish_diff_angles = para.diff_from_fish_angle_deg[from_frame_ind:(to_frame_ind + 1), :]
        from_fish_statuses = para.status_points[from_frame_ind:(to_frame_ind + 1), :]

        for curr_frame in range(from_fish_distances.shape[0]):  # ignore predictions in this function
            for ignore_status in [ParameciaStatus.PREDICT_AND_IMG, ParameciaStatus.PREDICT, ParameciaStatus.REPEAT_LAST]:
                from_fish_distances[curr_frame, from_fish_statuses[curr_frame, :] == ignore_status.value] = np.nan
                from_fish_diff_angles[curr_frame, from_fish_statuses[curr_frame, :] == ignore_status.value] = np.nan

        if from_fish_distances.shape[0] < 10:
            logging.info("Target paramecia is nan for {3} event {0} (#distances={1} for IBI {2})".format(
                event_name, from_fish_distances.shape, n_ibis_back, outcome_str))
            continue
        if np.isnan(from_fish_distances).all() or np.isnan(from_fish_diff_angles).all():
            logging.info("Target paramecia is nan for {4} event {0} (frames {1}-{2} has all nan) for IBI {3}".format(
                event_name, from_frame_ind, to_frame_ind, n_ibis_back, outcome_str))
            continue

        if "hit" in outcome_str or "spit" in outcome_str:  # search closest missing paramecia for hit event
            new_paramecia = check_hit(recent_frame=-1, older_frame=-10)
            if new_paramecia is not None:
                logging.info("Found paramecia for -1 & -10 para_ind={0} ({5} event {1} IBI={4} f: {2}-{3})".format(
                    new_paramecia + 1, event_name, from_frame_ind, to_frame_ind, n_ibis_back, outcome_str))
                return new_paramecia
            #new_paramecia = check_hit(recent_frame=-1, older_frame=start_frame_ind-from_frame_ind)
            #if new_paramecia is not None:
            #    logging.info("Found paramecia for -1 & {4} para_ind={0} ({1} IBI={3} f: {1}-{2})".format(
            #        new_paramecia + 1, event_name, from_frame_ind, to_frame_ind, n_ibis_back,
            #        start_frame_ind-from_frame_ind))
            #    return new_paramecia
            new_paramecia = check_hit(recent_frame=-1, older_frame=0)
            if new_paramecia is not None:
                logging.info("Found paramecia for -1 & 0 para_ind={0} ({5} event {1} IBI={4} f: {2}-{3})".format(
                    new_paramecia + 1, event_name, from_frame_ind, to_frame_ind, n_ibis_back, outcome_str))
                return new_paramecia

            # compare event start to end (might find several paramecia
            m_start = np.where(np.isnan(from_fish_distances[0, :]))[0]
            m_end = np.where(np.isnan(from_fish_distances[-1, :]))[0]
            new_paramecia = np.setdiff1d(m_end, m_start)

            new_paramecia = validate_max_distance_angle(new_paramecia,
                                                        max_distance_in_mm=max_hit_distance_in_mm,
                                                        max_angle_in_deg=max_hit_angle_in_deg)

            largest_angle_per_paramecia = np.nanmax(from_fish_diff_angles[:, new_paramecia], axis=0)
            smallest_dist_per_paramecia = np.nanmin(from_fish_distances[:, new_paramecia], axis=0)
            if largest_angle_per_paramecia.size == 0 or largest_angle_per_paramecia.size == 0:
                logging.info("Target paramecia is nan for {0} event {1} for IBI {2} (start-end search)".format(
                    outcome_str, event_name, n_ibis_back))
                continue
            if np.argmin(largest_angle_per_paramecia) == np.argmin(smallest_dist_per_paramecia):
                logging.info("Found paramecia for start-end, para_ind={0} ({5} event {1} IBI={4} f: {2}-{3})".format(
                    new_paramecia[np.argmin(smallest_dist_per_paramecia)] + 1,
                    event_name, from_frame_ind, to_frame_ind, n_ibis_back, outcome_str))
                return new_paramecia[np.argmin(smallest_dist_per_paramecia)]

        if "abort" in outcome_str or "miss" in outcome_str:
            if "abort" in outcome_str:
                if n_ibis_back == 1 and len(ending) > 1:  # for abort it's better to skip 1 bout (both escape and no)
                    continue
                elif n_ibis_back >= 1 and len(ending) == 1 and "no-escape" not in outcome_str:
                    if n_ibis_back > 1:
                        continue
                if "no-escape" not in outcome_str:
                    logging.info("{0} event {1} override frames IBI={2}: {3}-{4}".format(outcome_str, event_name,
                                                                                         n_ibis_back,
                                                                                         ending[-n_ibis_back],
                                                                                         starting[-1] + 1))
                    from_fish_distances = para.distance_from_fish_in_mm[ending[-n_ibis_back]:(starting[-1] + 1), :]
                    from_fish_diff_angles = para.diff_from_fish_angle_deg[ending[-n_ibis_back]:(starting[-1] + 1), :]
                    from_fish_statuses = para.status_points[ending[-n_ibis_back]:(starting[-1] + 1), :]
                    for curr_frame in range(from_fish_distances.shape[0]):  # ignore predictions in this function
                        for ignore_status in [ParameciaStatus.PREDICT_AND_IMG, ParameciaStatus.PREDICT,
                                              ParameciaStatus.REPEAT_LAST]:
                            from_fish_distances[curr_frame, from_fish_statuses[curr_frame, :] == ignore_status.value] = np.nan
                            from_fish_diff_angles[curr_frame, from_fish_statuses[curr_frame, :] == ignore_status.value] = np.nan

            # search closest distance from paramecia in front of fish
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                in_front_paramecia = np.where(np.nanmin(np.abs(from_fish_diff_angles), axis=0) < max_abort_angle_in_deg)[0]
                in_front_paramecia = validate_max_distance_angle(in_front_paramecia,
                                                                 max_distance_in_mm=max_abort_distance_in_mm,
                                                                 max_angle_in_deg=max_abort_angle_in_deg)
                smallest_dist_per_paramecia = np.nanmin(from_fish_distances[-10:, in_front_paramecia], axis=0)
                if smallest_dist_per_paramecia.size == 0 or np.isnan(smallest_dist_per_paramecia).all():
                    logging.info("No nearby paramecia in last 10 frames for {0} event {1}. Retry".format(outcome_str,
                                                                                                         event_name))
                    smallest_dist_per_paramecia = np.nanmin(from_fish_distances[:, in_front_paramecia], axis=0)
                    if smallest_dist_per_paramecia.size == 0 or np.isnan(smallest_dist_per_paramecia).all():
                        logging.info("Target paramecia is nan for {0} event {1} for IBI {2}".format(
                            outcome_str, event_name, n_ibis_back))
                        continue
                logging.info("Found paramecia for in-front distance, para_ind={0} ({5} event {1} IBI={4} f: {2}-{3})".format(
                    in_front_paramecia[np.nanargmin(smallest_dist_per_paramecia)] + 1,
                    event_name, from_frame_ind, to_frame_ind, n_ibis_back, outcome_str))
                return in_front_paramecia[np.nanargmin(smallest_dist_per_paramecia)]

    if "hit" in outcome_str or "spit" in outcome_str or ("abort" in outcome_str and "no-escape" not in outcome_str):
        # didnt find missing paramecia- try search by distance - patch
        to_frame_ind = event_frame_ind  # always end paramecia should be the one removed when event ended
        for n_ibis_back in [1, 2, 3]:
            if len(ending) < n_ibis_back:
                logging.error("Target paramecia return nan for {2} event {0} ({1} ibis!)".format(event_name, len(ending), outcome_str))
                return np.nan

            if "abort" in outcome_str and "no-escape" not in outcome_str:
                if len(ending) < n_ibis_back + 1:
                    logging.error(
                        "Target paramecia return nan for {2} event {0} ({1} ibis!)".format(event_name, len(ending), outcome_str))
                    return np.nan
                from_frame_ind = ending[-n_ibis_back-1]
            else:
                from_frame_ind = ending[-n_ibis_back]

            from_fish_distances = para.distance_from_fish_in_mm[from_frame_ind:(to_frame_ind + 1), :]
            from_fish_diff_angles = para.diff_from_fish_angle_deg[from_frame_ind:(to_frame_ind + 1), :]
            from_fish_statuses = para.status_points[from_frame_ind:(to_frame_ind + 1), :]

            for curr_frame in range(from_fish_distances.shape[0]):  # ignore predictions in this function
                for ignore_status in [ParameciaStatus.PREDICT_AND_IMG, ParameciaStatus.PREDICT,
                                      ParameciaStatus.REPEAT_LAST]:
                    from_fish_distances[curr_frame, from_fish_statuses[curr_frame, :] == ignore_status.value] = np.nan
                    from_fish_diff_angles[curr_frame, from_fish_statuses[curr_frame, :] == ignore_status.value] = np.nan

            if np.isnan(from_fish_distances).all() or np.isnan(from_fish_diff_angles).all():
                logging.info("Target paramecia is nan for {3} event {0} (frames {1}-{2} has all nan) for IBI {3}".format(
                    event_name, from_frame_ind, to_frame_ind, n_ibis_back, outcome_str))
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                in_front_paramecia = np.where(np.nanmin(np.abs(from_fish_diff_angles), axis=0) < max_hit_angle_in_deg)[0]
                in_front_paramecia = validate_max_distance_angle(in_front_paramecia,
                                                                 max_distance_in_mm=max_hit_distance_in_mm,
                                                                 max_angle_in_deg=max_hit_angle_in_deg)

                smallest_dist_per_paramecia = np.nanmin(from_fish_distances[-10:, in_front_paramecia], axis=0)
                if smallest_dist_per_paramecia.size == 0 or np.isnan(smallest_dist_per_paramecia).all():
                    logging.info("No nearby paramecia in last 10 frames for {0} event {1}. Retry".format(outcome_str,
                                                                                                         event_name))
                    smallest_dist_per_paramecia = np.nanmin(from_fish_distances[:, in_front_paramecia], axis=0)
                    if smallest_dist_per_paramecia.size == 0 or np.isnan(smallest_dist_per_paramecia).all():
                        logging.info("Target paramecia is nan for {0} event {1} for IBI {2}".format(
                            outcome_str, event_name, n_ibis_back))
                        continue
                logging.info("Found paramecia for in-front distance, para_ind={0} ({5} event {1} IBI={4} f: {2}-{3})".format(
                    in_front_paramecia[np.nanargmin(smallest_dist_per_paramecia)] + 1,
                    event_name, from_frame_ind, to_frame_ind, n_ibis_back, outcome_str))
                return in_front_paramecia[np.nanargmin(smallest_dist_per_paramecia)]

    logging.error("Target paramecia return nan for {0} event {1} (end of function)".format(outcome_str, event_name))
    return np.nan


class SingleFishAndEnvData:
    """

    """
    _events: List[ExpandedEvent]
    _metadata: Metadata
    _angles_for_fov = None
    _distances_for_fov_in_mm = None

    def __init__(self, metadata: Metadata, events: List[ExpandedEvent], angles_for_fov: dict,
                 distances_for_fov_in_mm: List[float]):
        self._events = events
        self._metadata = metadata
        self._angles_for_fov = angles_for_fov
        self._distances_for_fov_in_mm = distances_for_fov_in_mm

    @classmethod
    def from_preprocessed(cls, preprocessed_fish: FishPreprocessedData, ids=[]):
        events_all = [ExpandedEvent.from_preprocessed(event_to_copy) for event_to_copy in preprocessed_fish.events
                      if event_to_copy.event_id in ids or len(ids) == 0]
        events = [curr for (curr, event_name, reason) in events_all if curr is not None]
        thrown_events = [(event_name, reason) for (curr, event_name, reason) in events_all if curr is None]
        return cls(preprocessed_fish.metadata, events,
                   angles_for_fov=ANGLES, distances_for_fov_in_mm=DISTANCE_LIST_IN_MM), thrown_events

    def export_to_matlab(self, full_filename):
        """Save to a matlab struct data file
        :str full_filename: full path file name to file
        """
        save_mat_dict(full_filename, {'fish_data': self.export_to_struct()})

    def export_to_struct(self):
        events = [event.export_to_struct() for event in self._events]
        return {**self._metadata.export_to_struct(),
                'events': np.array(events, dtype=np.object),
                'distances_for_fov_in_mm': np.array(self._distances_for_fov_in_mm, dtype=np.float),
                'angles': self._angles_for_fov}

    @classmethod
    def import_from_struct(cls, data):
        if not isinstance(data['events'], (list, np.ndarray)):  # one list array is saved as object not list
            data['events'] = [data['events']]
        metadata = Metadata.import_from_struct(data)
        return cls(metadata, [ExpandedEvent.import_from_struct(event_d) for event_d in data['events']],
                   angles_for_fov=data['angles'], distances_for_fov_in_mm=data['distances_for_fov_in_mm'])

    @classmethod
    def import_from_matlab(cls, full_filename):
        """Initialize from a matlab struct data file
        :str full_filename: full path file name to file
        """
        # todo check keys? support versioning?
        data: dict = load_mat_dict(full_filename)['fish_data']
        if 'fish_list' in data.keys():  # todo quickfix
            data = data['fish_list'][0]
        return SingleFishAndEnvData.import_from_struct(data)

    # todo refactor (change in preproc before changing here)
    # add metadata values as properties to not change the class usage (current reading of these)
    @property
    def metadata(self):
        return self._metadata

    @property
    def name(self):
        return self._metadata.name

    @property
    def acclimation_time_min(self):
        return self._metadata.acclimation_time_min

    @property
    def num_of_paramecia_in_plate(self):
        return self._metadata.num_of_paramecia_in_plate

    @property
    def age_dpf(self):
        return self._metadata.age_dpf

    @property
    def feeding_str(self):
        return self._metadata.feeding_str

    @property
    def feeding(self):
        return self._metadata.feeding

    @property
    def events(self):
        return self._events

    @property
    def distances_for_fov_in_mm(self):
        return self._distances_for_fov_in_mm

    @property
    def angles_for_fov(self):
        return self._angles_for_fov


class FishAndEnvDataset:
    """

    """
    fish_processed_data_set: List[SingleFishAndEnvData]

    def __init__(self, fish_processed_data_set: List[SingleFishAndEnvData]):
        self.fish_processed_data_set = fish_processed_data_set

    @classmethod
    def from_preprocessed(cls, fish_preprocessed_data_set: List[FishPreprocessedData]):
        """
        :param fish_preprocessed_data_set: list of data (mat files) of several fish, which are dataset
        """
        fish_preprocessed_data_set = get_validated_list(fish_preprocessed_data_set, inner_type=FishPreprocessedData)
        fish_processed_data_set = [SingleFishAndEnvData.from_preprocessed(fish)
                                   for fish in tqdm(fish_preprocessed_data_set, desc="fish")]
        fish_processed_data_set = [fish for (fish, thrown_events) in fish_processed_data_set]
        return cls(fish_processed_data_set)

    @classmethod
    def import_from_matlab(cls, full_filename):
        """Initialize from a matlab struct data file
        :str full_filename: full path file name to file
        """
        # todo check keys? support versioning?
        data = load_mat_dict(full_filename)['fish_data']
        if not isinstance(data['fish_list'], (list, np.ndarray)):  # one list array is saved as object not list
            data['fish_list'] = [data['fish_list']]
        return cls([SingleFishAndEnvData.import_from_struct(curr) for curr in data['fish_list']])

    def export_to_matlab(self, full_filename):
        """Save to a matlab struct data file
        :str full_filename: full path file name to file
        """
        fish_list = [fish.export_to_struct() for fish in self.fish_processed_data_set]
        save_mat_dict(full_filename, {'fish_data': {'fish_list': np.array(fish_list, dtype=np.object), }
                                      })
