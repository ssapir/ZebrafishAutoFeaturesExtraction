import logging

from tqdm import tqdm
import numpy as np
import cv2

from classic_cv_trackers.abstract_and_common_trackers import ClassicCvAbstractTrackingAPI, Colors
from classic_cv_trackers.common_tracking_features import EllipseData

# this file contains only my code, which was added for plotting (lacking actual tracking)


class ParameciumOutput:
    """Holds both paramecia output of 1 frame (analyse result) and of 1 event (post-process)
    """
    def __init__(self):
        self.center = []
        self.area = []
        self.status = []
        self.color = []
        self.is_ok = True
        self.ellipse_majors = []
        self.ellipse_minors = []
        self.ellipse_dirs = []
        self.bbox = []

    def reset(self, n_frames, n_paramecia):
        """Make sure values are of correct size (from n_frames and n_paramecia) & filled with nan values

        :param n_frames:
        :param n_paramecia:
        :return: nothing
        """
        for val in self.__dict__.values():  # clear all previous values if exist
            if isinstance(val, list):
                val.clear()

        self.center = np.full([n_frames, n_paramecia, 2], fill_value=np.nan)
        for key in ['area', 'ellipse_majors', 'ellipse_minors', 'ellipse_dirs']:
            self.__setattr__(key, np.full([n_frames, n_paramecia, 1], fill_value=np.nan))
        self.status = np.full(shape=[n_frames, n_paramecia, 1], fill_value=np.nan)
        self.color = np.zeros([n_paramecia, 3], dtype=int)
        self.bbox = np.full([n_frames, n_paramecia, 4, 2], fill_value=np.nan)

    @classmethod
    def output_from_current_frame(cls, frame_ind, output):
        """Return ParameciumOutput subset with data of specific frame (from ParameciumOutput of a movie)

        :param frame_ind:
        :param output:
        :return:
        """
        result = ParameciumOutput()
        for key in [key for key in output.__dict__.keys() if key not in ['color', 'is_ok']]:
            val: np.ndarray = output.__dict__[key][frame_ind, :, :]
            if len(val.shape) > 1 and val.shape[1] == 1:  # list
                val = val.flatten()
            result.__setattr__(key, val)
        result.color = output.color
        return result


class ParameciumTracker(ClassicCvAbstractTrackingAPI):
    def __init__(self, visualize_movie=False, is_fast_run=False, is_tracker_disabled=True):
        super().__init__(visualize_movie=visualize_movie)
        self.name = 'paramecium tracker'
        self.is_fast_run = is_fast_run
        self.is_tracker_disabled = is_tracker_disabled
        self.reset_data()
        logging.basicConfig(level=logging.INFO)
        if self.is_tracker_disabled:
            logging.info(self.name + " is disabled")

    def _pre_process(self, dir_path, fish_name, event_id, noise_frame):
        pass

    def _analyse(self, input_frame: np.array, noise_frame: np.array, fps: int, frame_number: int, additional=None) \
            -> (np.array, bool, np.array):
        """Returns annotated frame (for debug-video) as well as data (class matching tracker's output).
        The annotated frame is optional (return empty np.array for no video creation at this stage).
        per frame- clean and binary
        :param input_frame: current input from video. Either original or clean (depends on the boolean)
        :param noise_frame: single frame that represents the noise from the whole video of the fish (not only the event)
        :param fps: frames-per-second of the video
        :param additional: additional data structs, from previous tracker processing
        :return: annotated_frame, output class (depends on the specific tracker API)
        """
        # self.draw_output_on_annotated_frame(an_frame, output, add_text=True)
        return input_frame, ParameciumOutput()

    @classmethod
    def draw_output_on_annotated_frame(cls, an_frame, output: ParameciumOutput, add_text=False,
                                       text_font=cv2.FONT_HERSHEY_SIMPLEX):  # todo abstract function?
        """Draw ParameciumOutput on current frame. Used both for presentation movies & debug movies.

        :param an_frame:
        :param output:
        :param add_text: should add paramecia debug text (True) or not (False, default)
        :return:
        """
        if add_text:
            col_right_side_text = 680
            row_right_side_text = 20
            text_font = cv2.FONT_HERSHEY_SIMPLEX
            bold = 2
            cv2.putText(an_frame, "# para {0}".format(len(output.center)),
                        (col_right_side_text, row_right_side_text), text_font, 0.6, Colors.GREEN, bold)

        for i, center, color in zip(range(len(output.center)), output.center, output.color):
            if not np.isnan(center).all() and center is not None:
                if isinstance(color, np.ndarray):
                    color = color.tolist()
                if len(output.ellipse_dirs) > i and not np.isnan(output.ellipse_majors[i]):
                    maj, minor, orientation = output.ellipse_majors[i], output.ellipse_minors[i], output.ellipse_dirs[i]
                    if not np.isnan([maj, minor, orientation]).all():
                        # ellipse = (cls.point_to_int(center), cls.point_to_int((maj/2, minor/2)), np.rad2deg(orientation))  # similar to fitEllipse result
                        # x0, y0 = cls.point_to_int(center)
                        # x1 = x0 + math.cos(orientation) * 0.5 * maj
                        # y1 = y0 + math.sin(orientation) * 0.5 * maj
                        # x2 = x0 + math.sin(orientation) * 0.5 * minor
                        # y2 = y0 - math.cos(orientation) * 0.5 * minor
                        # cv2.ellipse(an_frame, ellipse, color, thickness=1)
                        # cv2.line(an_frame, cls.point_to_int([x0, y0]), cls.point_to_int([x1, y1]), color, 2)
                        # cv2.line(an_frame, cls.point_to_int([x0, y0]), cls.point_to_int([x2, y2]), color, 2)
                        pass
                    box = output.bbox[i]
                    if not np.isnan(box).all():
                        cv2.drawContours(an_frame, [box.astype(int)], -1, color, 2)
                # cv2.putText(an_frame, "{0}".format(i + 1),
                #             cls.point_to_int([center[0] + 7, center[1] + 7]), text_font, 0.4, Colors.GREEN)
                status = output.status[i]
                # if status != Para.FROM_IMG:
                #     map_colors = {Para.REPEAT_LAST: Colors.YELLOW, Para.PREDICT: Colors.PURPLE,
                #                   Para.PREDICT_AND_IMG: Colors.PINK, Para.DOUBLE_PARA: Colors.RED}
                #     cv2.circle(an_frame, cls.point_to_int(center), 10, map_colors[status])

    def _post_process(self, input_frames_list: np.array, analysis_data_outputs: ParameciumOutput = None):
        """
        create output of the tracker
         - clean data of paramecium that collided with fish and didn't appear again.
         - filter the paramecium by length of data- meaning in how many frames they appeared.
         - creates annotated frame, (two options, with or without the fish annotations)
         - creates output for matlab struct
        :param input_frames_list: list of input frames of full video
        :param analysis_data_outputs: struct holding analysis outputs
        :return: annotated_frames_list, outputs list (class depends on the specific tracker API)
        annotated_frames_list are saved as video for debug etc
        """
        pass

    @classmethod
    def add_params_to_orig_frames(cls, annotated_frames, output):
        """
        annotates the given frames with paramecium, assumes the frames are in color scale
        (for example with fish annotations)
        :param annotated_frames:
        :return: todo add shape information
        """
        event_marked_param = []
        for frame_ind, frame in enumerate(annotated_frames):
            if len(frame.shape) != 3:  # not color
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            output_curr_frame = ParameciumOutput.output_from_current_frame(frame_ind, output)
            cls.draw_output_on_annotated_frame(frame, output_curr_frame, add_text=True)
            cv2.putText(frame, "# " + str(int(frame_ind + 1)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.GREEN, 2)
            event_marked_param.append(frame)

        return event_marked_param

    @staticmethod
    def location_from_region_props(region_props):
        if region_props is None or region_props.centroid is None:
            return np.nan, np.nan
        return region_props.centroid[1], region_props.centroid[0]  # flip

    @staticmethod
    def ellipse_from_region_props(region_props):
        """Return EllipseData of paramecia ellipse center, major, minor and directions (np.nan if not existing)

        :param region_props:
        :return:
        """
        if region_props is None:
            return EllipseData(ellipse_center=(np.nan, np.nan), ellipse_major=np.nan, ellipse_minor=np.nan,
                               ellipse_direction=np.nan)
        center = ParameciumTracker.location_from_region_props(region_props)
        return EllipseData(ellipse_center=center, ellipse_major=region_props.major_axis_length,
                           ellipse_minor=region_props.minor_axis_length,
                           ellipse_direction=ParameciumTracker.direction_from_region_props(region_props))

    @staticmethod
    def direction_from_region_props(region_props):
        if region_props is None:
            return np.nan
        return 90 - np.rad2deg(region_props.orientation)

    @staticmethod
    def area_from_region_props(region_props):
        """Return float of paramecia area (np.nan if not existing)

        :param region_props:
        :return:
        """
        if region_props is None:
            return np.nan
        return region_props.area

    @staticmethod
    def bbox_from_region_props(region_props):
        """Return bounding box, rotated to the orientation of the ellipse, as a contour of 4 points
        :param region_props:
        :return:
        """
        if region_props is None:
            return np.full(shape=(4, 2), fill_value=np.nan)
        fixed_orientation = ParameciumTracker.direction_from_region_props(region_props)
        center = [np.mean([region_props.bbox[3], region_props.bbox[1]]),
                  np.mean([region_props.bbox[2], region_props.bbox[0]])]
        h, w = abs(region_props.bbox[3] - region_props.bbox[1]), abs(region_props.bbox[2] - region_props.bbox[0])
        return np.int0(cv2.boxPoints((center, (h, w), fixed_orientation)))
