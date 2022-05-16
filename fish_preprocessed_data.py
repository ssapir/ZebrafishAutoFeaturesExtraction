from typing import List
import numpy as np
from utils.matlab_data_handle import load_mat_dict, save_mat_dict
from utils.geometric_functions import get_angle_to_horizontal

""" 
README PLEASE

This file contains the main output of the analysis, a class FishPreprocessedData which wraps a mat file (receives the 
analysis output & read/write the data with the needed fixes).
This class wraps one fish, which can have several events within. 

This output is called pre-process, since these are the very basic features needed. 
It holds the needed metadata (the fish name, age and additional human-annotated data about the experiment), 
as well as inner classes, separated by single definition (s.a. fish head vs tail, or fish's environment).

The classes can easily be replaced by simple inheritance & override (the classes can be decomposed), 
or be combined as data-set of all fish in more advanced stages (see examples).

The class FishPreprocessedData contains explanation on the functions, structure and use cases.

Please note that due to gaps between matlab-python, both this file and matlab_data_handle functions contains validations
and fixes. Those that appear here cannot be automatic (unlike those in matlab_data_handle), hence requires your 
specific declaration (for example, strings and lists).

Please note 2: the usage of properties is not here to annoy everyone, but rather to avoid incorrect usage of this class.
It wraps a mat file with class methods for loading/dumping data, therefore it should only have getters (not setters) 
"""


def to_valid_str(data):
    """savemat and loadmat doesn't recognize well empty char string (loaded as float array). Therefore empty str is " ".

    :param data:
    :return: valid string
    """
    if len(data) == 0:
        return np.str(" ")
    return np.str(data)


class Points:
    """For clarity, when we have simple point-per-frame list, it is saved as x & y arrays.
    todo should these be 2d matrix?
    """
    # constants + inner attributes
    INPUT_SIZE_PER_POINT = 2  # how many input params we expect- used for validation
    _x: np.ndarray
    _y: np.ndarray

    def __init__(self, x: List[float], y: List[float]):
        self._x = get_validated_list(x, inner_type=float)
        self._y = get_validated_list(y, inner_type=float)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    # functions - export and import are mirror of one another
    def export_to_struct(self):
        return {'x': self.x, 'y': self.y}

    @classmethod
    def import_from_struct(cls, data):
        if len(data) == 0:  # empty
            return cls([np.nan], [np.nan])
        return cls(data['x'], data['y'])

    @staticmethod
    def is_valid_points(points_list):
        return len(points_list) > 0 and len(np.array(points_list).shape) > 1 \
               and np.array(points_list).shape[1] == Points.INPUT_SIZE_PER_POINT

    @classmethod
    def from_array(cls, array_2d: np.ndarray, cast_to_float=False):
        """Create Points instance from 2d nd-array if valid (empty values otherwise)

        :param array_2d:
        :return:
        """

        if cls.is_valid_points(array_2d):
            if cast_to_float:
                return cls(array_2d[:, 0].astype(float), array_2d[:, 1].astype(float))
            return cls(array_2d[:, 0], array_2d[:, 1])
        return cls([], [])


class Head:
    # protected inner attributes
    _origin_points: Points
    _destination_points: Points
    _eyes_abs_angle_deg: np.ndarray
    _eyes_head_dir_diff_ang: np.ndarray
    _eyes_areas_pixels: np.ndarray

    @property
    def origin_points(self):
        """ Origin of the direction, example: point between eyes
        :return: list of 2D points (x,y). Format type: Points class (to allow points.x, points.y...)
        """
        return self._origin_points

    @property
    def destination_points(self):
        """ Destination of the direction, example: mouth
        :return: list of 2D points (x,y). Format type: Points class (to allow points.x, points.y...)
        """
        return self._destination_points

    @property
    def directions_in_deg(self):
        """ Angles in degrees, matching vector from origin to destinations, relative to x axis counter clockwise
        :return: list of float
        """
        return self._directions

    @property
    def eyes_abs_angle_deg(self):
        """ Angles in degrees, relative to x axis counter clockwise
        :return: list of (float, float) meaning nx2 shape
        """
        return self._eyes_abs_angle_deg

    @property
    def eyes_head_dir_diff_ang(self):
        """ Angles in degrees of diff between head direction and eyes, relative to x axis counter clockwise
        :return: list of (float, float) meaning nx2 shape
        """
        return self._eyes_head_dir_diff_ang

    @property
    def eyes_areas_pixels(self):
        return self._eyes_areas_pixels

    def __init__(self, origin_points: Points, destination_points: Points, eyes_abs_angle_list: np.ndarray,
                 eyes_head_dir_diff_ang_list: np.array, eyes_areas_pixels_list: np.array):
        """Creates instance, with validations for each data type (raise errors as early as possible)

        :param origin_points:
        :param destination_points:
        :param eyes_abs_angle_list:
        :param eyes_head_dir_diff_ang_list:
        """

        # Raise exception if type is incorrect
        validate_type(origin_points, Points)
        validate_type(destination_points, Points)

        self._origin_points = origin_points
        self._destination_points = destination_points

        # validate inner type is an expected + make sure list is a valid one (fixing 1 item use case)
        self._eyes_abs_angle_deg = np.array(get_validated_list(eyes_abs_angle_list, inner_type=float))
        self._eyes_head_dir_diff_ang = np.array(get_validated_list(eyes_head_dir_diff_ang_list, inner_type=float))
        self._eyes_areas_pixels = np.array(get_validated_list(eyes_areas_pixels_list, inner_type=float))

        # Eyes should be float list between 0-360
        self.validate_eyes_data(self._eyes_abs_angle_deg, self._eyes_head_dir_diff_ang)

        self._directions = []
        for origin_x, origin_y, destination_x, destination_y in \
                zip(self._origin_points.x, self._origin_points.y, self._destination_points.x,
                    self._destination_points.y):
            self._directions.append(get_angle_to_horizontal([origin_x, origin_y],
                                                            [destination_x, destination_y]))

    @staticmethod
    def validate_eyes_data(eyes_abs_angle_list, eyes_head_dir_diff_ang_list):
        """Raise exception if shape is incorrect or values of angles are out of range

        :param eyes_abs_angle_list:
        :param eyes_head_dir_diff_ang_list:
        :return: nothing
        """
        for curr_eyes_list, name_ in zip([eyes_abs_angle_list, eyes_head_dir_diff_ang_list],
                                         ["Eyes angles", "Diff of eyes-Head angle"]):
            if (len(curr_eyes_list.shape) != 2 or curr_eyes_list.shape[1] != 2) and curr_eyes_list.shape != (2,):  # validate nx2 size
                raise TypeError("{0} shape is {1} instead of nx2".format(name_, curr_eyes_list.shape))
            if np.min(np.abs(curr_eyes_list)) < 0 or np.max(np.abs(curr_eyes_list)) > 360:  # validate abs range
                raise TypeError("{0} values are not is {1}-{2} instead of 0-360".format(name_,
                                                                                        np.min(curr_eyes_list),
                                                                                        np.max(curr_eyes_list)))

    def export_to_struct(self):
        """
        :return: dictionary matching matlab struct format, to be saved.
        """
        return {'origin_points': self.origin_points.export_to_struct(),
                'destination_points': self.destination_points.export_to_struct(),
                'directions_in_deg': np.array(self.directions_in_deg),
                'eyes_abs_angles_in_deg': np.array(self.eyes_abs_angle_deg),
                'eyes_head_dir_diff_ang_in_deg': np.array(self.eyes_head_dir_diff_ang),
                'eyes_areas_pixels': np.array(self.eyes_areas_pixels)}

    @classmethod  # gets class and not instance of an object
    def import_from_struct(cls, data):
        """Load matlab data as Head class instance.
        Note: directions_in_deg is calculated from points, and not loaded.

        :param data: input data - dictionary matching export_to_struct output (mirror of this function)
        :return: Head class, filled with data.
        """
        return cls(Points.import_from_struct(data['origin_points']),
                   Points.import_from_struct(data['destination_points']),
                   data['eyes_abs_angles_in_deg'],
                   data['eyes_head_dir_diff_ang_in_deg'],
                   data.get('eyes_areas_pixels', []))


class Tail:
    def __init__(self, tail_tip_points: Points, is_bout_frame_list: List[bool], bout_start_frames: List,
                 bout_end_frames: List, tail_path_list: List[Points], interpolated_tail_path: List,
                 swimbladder_points_list: Points, tip_to_swimbladder_distance: List, velocity_norms: List[float]):
        """todo explain each parameter

        :param tail_tip_points: list of points referring to tips of the tail at different frames
        :param tail_path_list: list of arrays containing all of the midline (tail points) from tail tip to swimbladder, each entry refers to a different frame
        :param swimbladder_points_list: list of points referring to swimbladder points at different frames
        :param tip_to_swimbladder_distance: list of distances between tip of the tail to the swimbladder point at different frames
        :param is_bout_frame_list: List of boolean values indicating if the frame is during a bout or not
        :param bout_start_frames: List of bout start frames
        :param bout_end_frames: List of bout end frames
        :param velocity_norms:
        """
        validate_type(tail_tip_points, Points)
        validate_type(swimbladder_points_list, Points)

        # todo this is an error. The usage of properties protects the class attributes
        self.tail_tip_point_list = tail_tip_points
        self.swimbladder_points_list = swimbladder_points_list
        self.tip_to_swimbladder_distance = tip_to_swimbladder_distance
        self.tail_path_list = get_validated_list(tail_path_list, inner_type=Points)
        self.interpolated_tail_path = get_validated_list(interpolated_tail_path,inner_type=float)
        self.is_bout_frame_list = np.array(get_validated_list(is_bout_frame_list, inner_type=bool))
        self.bout_start_frames = np.array(get_validated_list(bout_start_frames, inner_type=(int, np.int64)))
        self.bout_end_frames = np.array(get_validated_list(bout_end_frames, inner_type=(int, np.int64)))
        self.velocity_norms = np.array(get_validated_list(velocity_norms, inner_type=float))

    def export_to_struct(self):
        tail_path_list_exported = [tail_path.export_to_struct() for tail_path in self.tail_path_list]

        return {'tail_tip_point_list': self.tail_tip_point_list.export_to_struct(),
                'tail_path_list': tail_path_list_exported,
                'interpolated_tail_path': self.interpolated_tail_path,
                'is_bout_frame_list': self.is_bout_frame_list,
                'bout_start_frames': self.bout_start_frames,
                'bout_end_frames': self.bout_end_frames,
                'swimbladder_points_list': self.swimbladder_points_list.export_to_struct(),
                'tip_to_swimbladder_distance': self.tip_to_swimbladder_distance,
                'velocity_norms': self.velocity_norms}

    @classmethod
    def import_from_struct(cls, data):
        if 'tail_path_list' not in data.keys() or len(data['tail_path_list']) == 0:  # todo patch due to bug above. should be removed
            tail_path_list_as_points = []
        else:
            if isinstance(data['tail_path_list'][0], dict):  # todo patch due to bug above. should be removed
                tail_path_list_as_points = [Points.import_from_struct(tail_path) for tail_path in data['tail_path_list']]
            else:
                tail_path_list_as_points = [Points.from_array(np.array(tail_path), cast_to_float=True) for tail_path in
                                            data['tail_path_list']]
        return cls(tail_tip_points=Points.import_from_struct(data['tail_tip_point_list']),
                   is_bout_frame_list=data.get('is_bout_frame_list', []),
                   bout_start_frames=data.get('bout_start_frames', []),
                   bout_end_frames=data.get('bout_end_frames', []),
                   tail_path_list=tail_path_list_as_points,
                   interpolated_tail_path=data.get('interpolated_tail_path', []),
                   swimbladder_points_list=Points.import_from_struct(data.get('swimbladder_points_list', [])),
                   tip_to_swimbladder_distance=data.get('tip_to_swimbladder_distance', np.nan),
                   velocity_norms=data.get('velocity_norms', []))


class Paramecium:
    @property
    def center_points(self):
        return self._center_points

    @property
    def area_points(self):
        return self._area_points

    @property
    def status_points(self):
        return self._status_points

    @property
    def color_points(self):
        return self._color_points

    @property
    def ellipse_majors(self):
        return self._ellipse_majors

    @property
    def ellipse_minors(self):
        return self._ellipse_minors

    @property
    def ellipse_dirs(self):
        return self._ellipse_dirs

    @property
    def bounding_boxes(self):
        return self._bounding_boxes

    def __init__(self, center: np.ndarray, area: np.ndarray, status: np.ndarray, color: np.ndarray,
                 ellipse_majors: np.ndarray, ellipse_minors: np.ndarray, ellipse_dirs: np.ndarray,
                 bounding_boxes: np.ndarray):
        """Format:
            n = rows represents number of frames in this event.
            m = columns represents number of paramecium in this event.

            :param center: nXmX2 where 2 = the centroid of each paramecia at each frames is (x,y).
            if the paramecia does not exists in this frame, this is none.
            :param area: nXm = number of pixels in paramecia contour.
            :param status: nXm contains the type of data.
            FROM_IMG = 0
            REPEAT_LAST = 1
            PREDICT = 2
            PREDICT_AND_IMG = 3
            DOUBLE_PARA = 4
            :param color: mX3 contains the type of data.
            :param ellipse_majors: nXm contains the major axis of the ellipse.
            :param ellipse_minors: nXm, minor axis.
            :param ellipse_dirs: nXm, ellipse direction (degrees- todo fix?).
            :param ellipse_dirs: nXmx4x2, bounding box points.
        """
        # int + nan = float type
        self._center_points = get_validated_list(center, inner_type=float)
        self._area_points = get_validated_list(area, inner_type=float)
        self._status_points = get_validated_list(status, inner_type=float)
        self._color_points = get_validated_list(color, inner_type=(int, np.int64))
        self._ellipse_majors = get_validated_list(ellipse_majors, inner_type=float)
        self._ellipse_minors = get_validated_list(ellipse_minors, inner_type=float)
        self._ellipse_dirs = get_validated_list(ellipse_dirs, inner_type=float)
        self._bounding_boxes = get_validated_list(bounding_boxes, inner_type=float)

    @classmethod
    def from_tracker_output(cls, output):
        return cls(center=output.center, area=output.area, status=output.status, color=output.color,
                   ellipse_majors=output.ellipse_majors, ellipse_minors=output.ellipse_minors,
                   ellipse_dirs=output.ellipse_dirs, bounding_boxes=output.bbox)

    def export_to_struct(self):  # this is an example of saving points only (centers) of one trajectory
        return {'center': self.center_points, 'area': self.area_points, 'status': self.status_points,
                'color': self.color_points, 'bounding_boxes': self.bounding_boxes,
                'ellipse_majors': self.ellipse_majors, 'ellipse_minors': self.ellipse_minors,
                'ellipse_dirs_in_deg': self.ellipse_dirs}

    @classmethod
    def import_from_struct(cls, data):  # match ctor
        return cls(center=data['center'], area=data.get('area', []), status=data['status'],
                   color=data.get('color',[]),
                   ellipse_majors=data.get('ellipse_majors', []),
                   ellipse_minors=data.get('ellipse_minors',[]),
                   ellipse_dirs=data.get('ellipse_dirs_in_deg',[]),
                   bounding_boxes=data.get('bounding_boxes',[]))


class Event:
    # protected inner attributes (should not be accessed from outside)
    _head: Head
    _tail: Tail
    _paramecium: Paramecium
    _fish_tracking_status_list: np.ndarray
    _tail_tip_status_list: np.ndarray
    _swimbladder_points_status_list: np.ndarray
    _fish_area_in_pixels: np.ndarray
    _event_name: str
    _event_id: int

    # metadata - default empty values
    is_complex_hunt = False
    outcome = -1
    outcome_str = to_valid_str("")
    comments = to_valid_str("")
    event_frame_ind = -1

    @property
    def head(self):
        return self._head

    @property
    def tail(self):
        return self._tail

    @property
    def paramecium(self):
        return self._paramecium

    @property
    def fish_area_in_pixels(self):
        return self._fish_area_in_pixels

    @property
    def fish_tracking_status_list(self):
        return self._fish_tracking_status_list

    @property
    def is_head_prediction_list(self):
        return self._is_head_prediction_list

    @property
    def tail_tip_status_list(self):
        return self._tail_tip_status_list

    @property
    def swimbladder_points_status_list(self):
        return self._swimbladder_points_status_list

    @property
    def event_id(self):
        return self._event_id

    @property
    def event_name(self):
        return self._event_name

    @property
    def fish_tracking_found_percentage(self):
        if len(self.fish_tracking_status_list) > 0:
            return sum(self.fish_tracking_status_list) / len(self.fish_tracking_status_list)
        return 0

    @property
    def fish_prediction_percentage(self):
        if len(self.is_head_prediction_list) > 0:
            return sum(self.is_head_prediction_list) / len(self.is_head_prediction_list)
        return 0

    @property
    def tail_tip_found_percentage(self):
        if len(self.tail_tip_status_list) > 0:
            return sum(self.tail_tip_status_list) / len(self.tail_tip_status_list)
        return 0

    @property
    def swimbladder_found_percentage(self):
        if len(self.swimbladder_points_status_list) > 0:
            return sum(self.swimbladder_points_status_list) / len(self.swimbladder_points_status_list)
        return 0

    def __init__(self, event_name: str, event_id: int, head: Head, tail: Tail, fish_tracking_status_list: List[bool],
                 is_head_prediction_list: List[bool], tail_tip_status_list: List[bool],  swimbladder_points_status_list: List[bool], paramecium: Paramecium,
                 fish_area_in_pixels: List[float]):

        validate_type(paramecium, Paramecium)
        validate_type(head, Head)
        validate_type(tail, Tail)
        validate_type(event_id, int)
        validate_type(event_name, str)

        self._head = head
        self._tail = tail
        self._paramecium = paramecium
        self._fish_tracking_status_list = get_validated_list(fish_tracking_status_list, inner_type=(bool, np.bool_))
        self._is_head_prediction_list = get_validated_list(is_head_prediction_list, inner_type=(bool, np.bool_))
        self._tail_tip_status_list = get_validated_list(tail_tip_status_list, inner_type=(bool, np.bool_))
        self._swimbladder_points_status_list = get_validated_list(swimbladder_points_status_list, inner_type=(bool, np.bool_))
        self._event_id = event_id
        self._event_name = to_valid_str(event_name)
        self._fish_area_in_pixels = get_validated_list(fish_area_in_pixels, inner_type=float)

    def set_metadata(self, is_complex_hunt: bool, outcome: int, outcome_str: str, comments: str, event_frame_ind: int):
        """Set metadata parameters from an external script (look at pipeline scripts)

        :param is_complex_hunt:
        :param outcome:
        :param outcome_str:
        :param comments:
        :param event_frame_ind:
        :return:
        """
        validate_type(is_complex_hunt, (bool, np.bool_))
        validate_type(outcome, (int, np.int64))
        validate_type(outcome_str, str)
        validate_type(comments, str)
        validate_type(event_frame_ind, (int, np.int64))

        self.is_complex_hunt = np.bool(is_complex_hunt)
        self.outcome = outcome
        self.outcome_str = to_valid_str(outcome_str)
        self.comments = to_valid_str(comments)
        self.event_frame_ind = event_frame_ind

    def export_to_struct(self):
        return {'event_name': self.event_name,
                'event_id': self.event_id,
                'head': self.head.export_to_struct(),
                'tail': self.tail.export_to_struct(),
                'fish_tracking_status_list': self.fish_tracking_status_list,
                'fish_tracking_found_percentage': self.fish_tracking_found_percentage,
                'is_head_prediction_list': self.is_head_prediction_list,
                'fish_area_in_pixels': self.fish_area_in_pixels,
                'fish_prediction_percentage': self.fish_prediction_percentage,
                'tail_tip_status_list': self.tail_tip_status_list,
                'tail_tip_found_percentage': self.tail_tip_found_percentage,
                'swimbladder_points_status_list': self.swimbladder_points_status_list,
                'swimbladder_found_percentage': self.swimbladder_found_percentage,
                "paramecium": self.paramecium.export_to_struct(),
                'is_complex_hunt': self.is_complex_hunt,
                'outcome': self.outcome,
                'outcome_str': self.outcome_str,
                'comments': self.comments,
                'event_frame_ind': self.event_frame_ind}

    @classmethod
    def import_from_struct(cls, data):
        result = cls(event_name=data['event_name'],
                     event_id=data['event_id'],
                     head=Head.import_from_struct(data['head']),
                     tail=Tail.import_from_struct(data['tail']),
                     fish_tracking_status_list=data['fish_tracking_status_list'],
                     is_head_prediction_list=data.get('is_head_prediction_list', []),
                     fish_area_in_pixels=data.get('fish_area_in_pixels', []),
                     tail_tip_status_list=data.get('tail_tip_status_list', []),
                     swimbladder_points_status_list=data.get('swimbladder_points_status_list', []),
                     paramecium=Paramecium.import_from_struct(data['paramecium']))
        if 'is_complex_hunt' in data.keys():
            result.set_metadata(data['is_complex_hunt'], data['outcome'], data['outcome_str'], data['comments'],
                                data.get('event_frame_ind', -1))
        return result

    @classmethod
    def from_tracker_output(cls,
                            event_name: str,
                            event_id: int,
                            origin_head_points_list: np.array,
                            destination_head_points_list: np.array,
                            fish_tracking_status_list: List[bool],
                            fish_area_in_pixels: List[float],
                            is_head_prediction_list: List[bool],
                            eyes_abs_angle_list: np.array,
                            eyes_head_dir_diff_ang_list: np.array,
                            eyes_areas_pixels_list: np.array,
                            tail_tip_point_list: np.array,
                            swimbladder_points_list: np.array,
                            tip_to_swimbladder_distance: List[float],
                            tail_tip_status_list: List[bool],
                            swimbladder_points_status_list: List[bool],
                            tail_path_list: np.array,
                            interpolated_tail_path: List[np.array],
                            is_bout_frame_list: List[bool],
                            bout_start_frames: List[int],
                            bout_end_frames: List[int],
                            velocity_norms: np.array,
                            paramecium_tracker_output):
        """ Initialize class instance from tracker output

        :param eyes_areas_pixels_list:
        :param event_name: current event full name
        :param event_id: current event_id (number only)
        :param fish_tracking_status_list: list indicating if current frame data is good
        :param fish_area_in_pixels: per frame area size (in pixels)
        :param is_head_prediction_list: list indicating if current frame head data is a prediction
        :param tail_tip_status_list: list indicating if the tail tip was found for a given frame number
        :param swimbladder_points_status_list: list indicating if the swimbladder point was found for a given frame number
        :param destination_head_points_list: np array as returned from tracker
        :param origin_head_points_list: np array as returned from tracker
        :param eyes_abs_angle_list: nx2 float angles list (pair of eyes, per frame)
        :param eyes_head_dir_diff_ang_list: nx2 float angles list (pair of eyes, per frame)
        :param tail_tip_point_list: nx2 integer pixel position of tail tip
        :param swimbladder_points_list: nx2 integer pixel position of swimbladder
        :param tip_to_swimbladder_distance: n float of distance from tip of tail to swimbladder (per frame)
        :param tail_path_list: list of lists of ordered points across the fish's midline
        :param interpolated_tail_path: np.array of shape (Frames,TailSegments,2) with points along the tail in equal distance. Tail Segments is always constant.
        :param is_bout_frame_list: list of boolean values indicating whether or not this frame is part of a bout
        :param bout_start_frames: list containing bout start frames
        :param bout_end_frames: list containing bout end frames
        :param velocity_norms: list of float values containing a convolution of the velocity norm, used to determine bouts
        :param paramecium_tracker_output: output data object (as defined by paramerium tracker)
        :return: class instance of one event, is instance ok (otherwise not saved)
        """
        # todo what else is mandatory for a valid mat file?
        if not Points.is_valid_points(origin_head_points_list) or \
                not Points.is_valid_points(destination_head_points_list):
            return None, False

        if not isinstance(paramecium_tracker_output, Paramecium):  # convert if not converted outside
            paramecium = Paramecium.from_tracker_output(paramecium_tracker_output)
        else:
            paramecium = paramecium_tracker_output

        validate_type(event_name, str)
        validate_type(event_id, int)

        tail_path_list_as_points = []
        for tail_path in tail_path_list:
            if type(tail_path) is not Points:
                tail_path = Points.from_array(tail_path, cast_to_float=True)

            tail_path_list_as_points.append(tail_path)
        return cls(event_name=event_name,
                   event_id=event_id,
                   head=Head(Points.from_array(np.array(origin_head_points_list)),
                             Points.from_array(
                                 np.array(destination_head_points_list)),
                             eyes_abs_angle_list,
                             eyes_head_dir_diff_ang_list,
                             eyes_areas_pixels_list),
                   tail=Tail(Points.from_array(np.array(tail_tip_point_list)),
                             is_bout_frame_list,
                             bout_start_frames,
                             bout_end_frames,
                             tail_path_list_as_points,
                             interpolated_tail_path,
                             Points.from_array(np.array(swimbladder_points_list)),
                             tip_to_swimbladder_distance,
                             velocity_norms),
                   fish_tracking_status_list=fish_tracking_status_list,
                   is_head_prediction_list=is_head_prediction_list,
                   tail_tip_status_list=tail_tip_status_list,
                   fish_area_in_pixels=fish_area_in_pixels,
                   swimbladder_points_status_list=swimbladder_points_status_list,
                   paramecium=paramecium), True
class Metadata:
    """ Separate class to allow reuse.
    This data is added by the script main_metadata.py, and is read from the relevant excel file (after main.py run)
    """
    _name: str

    # metadata values (initialized via script after fish is created)
    age_dpf = -1
    num_of_paramecia_in_plate = -1
    acclimation_time_min = -1
    feeding = -1
    feeding_str = to_valid_str("")

    @property
    def name(self):
        return self._name

    def __init__(self, name: str):
        validate_type(name, str)
        self._name = to_valid_str(name)

    def set_metadata(self, age_dpf: int, num_of_paramecia_in_plate: int, acclimation_time_min: int,
                     feeding: int, feeding_str: str):
        """Set and save additional metadata from excel file.

        :param age_dpf:
        :param num_of_paramecia_in_plate:
        :param acclimation_time_min:
        :param feeding: number indicating feeding state
        :param feeding_str: textual value of the number
        :return: nothing
        """
        validate_type(age_dpf, (int, np.int64))
        validate_type(num_of_paramecia_in_plate, (int, np.int64))
        validate_type(acclimation_time_min, (int, np.int64))
        validate_type(feeding, (int, np.int64))
        validate_type(feeding_str, str)

        self.age_dpf = age_dpf
        self.num_of_paramecia_in_plate = num_of_paramecia_in_plate
        self.acclimation_time_min = acclimation_time_min
        self.feeding = feeding
        self.feeding_str = to_valid_str(feeding_str)

    @classmethod
    def import_from_struct(cls, data):
        result = cls(data['name'])
        if 'age_dpf' in data.keys():
            result.set_metadata(data['age_dpf'], data['num_of_paramecia_in_plate'], data['acclimation_time_min'],
                                data['feeding'], data['feeding_str'])
        return result

    def export_to_struct(self):
        return {'name': self.name,
                'age_dpf': self.age_dpf,
                'num_of_paramecia_in_plate': self.num_of_paramecia_in_plate,
                'acclimation_time_min': self.acclimation_time_min,
                'feeding': self.feeding,
                'feeding_str': self.feeding_str}


class FishPreprocessedData:
    """ Contains: Name, and events containing head direction, tail points and paramecia trajectories data (any data
    used by further processing).
    Note: This class contained inner/nested classes, since these exists within the fish's context

    Functionality:
    - Preprocessed from videos: initialized directly from tracker output
    - Loaded/saved from/to mat files: to be further processed in both languages.

    Main flow:
    - analyse by trackers (result as numpy array or classes)
    - call Event.from_tracker_output to create Event instance (FishPreprocessedData receives list of events)
    - call export_to_matlab to save the output
    -- call import_from_matlab to load the output, to be re-processed, fixed, and saved again

    Main functions:
    - The functions are recursively called, s.t. each class can load and save itself.
      There are 2 levels:
      1. FishPreprocessedData.export_to_matlab and FishPreprocessedData.import_from_matlab works against mat files
      2. Inner classes (Event and everything inside) works with dictionaries.
         Its main functions:
         1. import_from_struct and export_to_struct, which are mirror of one another.
            Meaning, the dictionary you define by export is the one received by import.
            These functions are called by FishPreprocessedData.
         2. from_tracker_output works against relevant tracker output class.
            Event.from_tracker_output is called by main loop, inner classes are called recursively.

    Note:
        - export functions are 'normal' functions.
        - import functions and from_tracker_output are classmethods, meaning they creates a class instance by calling
          its ctor.
    """
    _events: np.ndarray
    _metadata: Metadata

    # add metadata values as properties to not change the class usage (current reading of these)
    @property
    def metadata(self):
        return self._metadata

    @property
    def events(self):
        return self._events

    def __init__(self, name: str, events: List[Event]):
        self._events = get_validated_list(events, inner_type=Event)
        self._metadata = Metadata(name)

    def set_metadata(self, age_dpf: int, num_of_paramecia_in_plate: int, acclimation_time_min: int,
                     feeding: int, feeding_str: str):
        """Set and save additional metadata from excel file.

        :param age_dpf:
        :param num_of_paramecia_in_plate:
        :param acclimation_time_min:
        :param feeding: number indicating feeding state
        :param feeding_str: textual value of the number
        :return: nothing
        """
        self._metadata.set_metadata(age_dpf=age_dpf, num_of_paramecia_in_plate=num_of_paramecia_in_plate,
                                    acclimation_time_min=acclimation_time_min, feeding=feeding, feeding_str=feeding_str)

    @classmethod
    def import_from_struct(cls, data):
        if not isinstance(data['events'], (list, np.ndarray)):  # one list array is saved as object not list
            data['events'] = [data['events']]

        result = cls(data['name'],
                     [Event.import_from_struct(event_d) for event_d in data['events']])
        metadata = Metadata.import_from_struct(data)
        result.set_metadata(metadata.age_dpf, metadata.num_of_paramecia_in_plate, metadata.acclimation_time_min,
                            metadata.feeding, metadata.feeding_str)
        return result

    def export_to_struct(self):
        events = [event.export_to_struct() for event in self.events]
        return {'fish_data': {**self._metadata.export_to_struct(),
                              'events': np.array(events, dtype=np.object)}}

    @classmethod
    def import_from_matlab(cls, full_filename):
        """Initialize from a matlab struct data file
        :str full_filename: full path file name to file
        """
        return cls.import_from_struct(load_mat_dict(full_filename)['fish_data'])

    def export_to_matlab(self, full_filename):
        """Save to a matlab struct data file
        :str full_filename: full path file name to file
        """
        save_mat_dict(full_filename, self.export_to_struct())


# Help functions for validations
def validate_type(obj, wanted_type=(list, np.ndarray)):
    """Protect code from usage mistakes (to prevent further loading/saving errors)
    :param obj: input data
    :param wanted_type: data to validate (raise excpetion if invalid)
    :return: nothing
    """
    if not isinstance(obj, wanted_type):
        raise TypeError("Instance of type {0} instead of {1}".format(type(obj), wanted_type))


def get_validated_list(obj, inner_type, add=""):
    """savemat and loadmat treat 1 object list as object, not list. This code make sure and fix type if needed.

    :param obj: input data
    :param inner_type: type of the list objects
    :return: list of valid object (fix if needed)
    """
    if isinstance(obj, inner_type) and inner_type is not object:  # saving list of 1 point
        obj = [obj]
    validate_type(obj, wanted_type=(list, np.ndarray))
    if len(obj) > 0:
        if (isinstance(inner_type, (list, tuple)) and np.array(obj).dtype not in inner_type) or \
                (not isinstance(inner_type, (list, tuple)) and np.array(obj).dtype != inner_type):
            raise TypeError("Instance of type {0} instead of {1} in array.{2}".format(np.array(obj).dtype, inner_type, add))
    return np.array(obj)
