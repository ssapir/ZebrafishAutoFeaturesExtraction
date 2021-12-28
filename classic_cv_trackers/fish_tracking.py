import networkx
import logging
from scipy.interpolate import interp1d

import cv2
import numpy as np
import math
from scipy.spatial import distance
from classic_cv_trackers.abstract_and_common_trackers import ClassicCvAbstractTrackingAPI, Colors
from classic_cv_trackers.common_tracking_features import EllipseData, absolute_angles_and_differences_in_deg
from utils.geometric_functions import get_angle_to_horizontal
from utils.main_utils import FishOutput
from utils.noise_cleaning import clean_plate


class FrameAnalysisData:  # todo make common fish output? after tail merge
    """
    Hold tracking output. todo - simplify output to list? (remove inner classes?)
    """
    class TailData:
        tail_tip_point = None

        def __init__(self, fish_tail_tip_point, tail_path):
            self.tail_tip_point = fish_tail_tip_point
            self.tail_path = tail_path

    class EyesData:
        ellipses = None
        abs_angle_deg = None  # relative to X axis, 0-360 deg
        diff_from_fish_direction_deg = None
        contour_areas = None

        def __init__(self, eyes_dict, p_from, p_to):
            self.ellipses = [EllipseData(eye['center'], eye['major'], eye['minor'], eye['angle'])
                             for eye in eyes_dict]

            head_direction_angle = get_angle_to_horizontal(p_from, p_to)  # 0-360 relative to y axis
            eye_angles = [ContourBasedTracking.fix_ellipse_angle(eye['angle'], head_direction_angle)
                          for eye in eyes_dict]
            self.abs_angle_deg, self.diff_from_fish_direction_deg = \
                absolute_angles_and_differences_in_deg(eye_angles, head_direction_angle)
            self.contour_areas = [eye['area'] for eye in eyes_dict]

    is_ok = False
    is_prediction = False
    fish_contour = None
    fish_segment = None  # full fish pixels (for masking) - todo use contour instead (should change usage as well)
    eyes_contour = None
    eyes_data = None
    fish_head_origin_point = (np.nan, np.nan)
    fish_head_destination_point = (np.nan, np.nan)
    tail_data = None


class ContourBasedTracking(ClassicCvAbstractTrackingAPI):
    """ Use contours with surrounding ellipse around the fish for head direction and eyes position.
    This tracker functionality can be used by other trackers, to get additional metadata s.a. eyes and fish contours

    Usage:
    tracker = ContourBasedTracking()
    annotated_frame, output = tracker.analyse(frame, noise_frame, fps)  # lists are np.array of nX2 size for n points
    if output.is_ok: # false when can't find fish
        output.fish_contour will conwtain the fish external contour only
        output.fish_segment will contain the fish coordinates, in cv2 coordinate system (as returned from findNonZero)

    """
    minimum_midline_contour_area = 2
    clean_bnw_for_tail_threshold = 55

    def __init__(self, visualize_movie=False, input_video_has_plate=True, is_fast_run=False,
                 logger_filename='', is_debug=True, scale_area=1, is_twin_view=False, save_n_back=6,
                 max_invalid_n_back=2, is_predicting_missing_eyes=True):
        ClassicCvAbstractTrackingAPI.__init__(self, visualize_movie=visualize_movie)
        if logger_filename != '':
            if is_debug:
                logging.basicConfig(filename=logger_filename, level=logging.DEBUG)
            else:
                logging.basicConfig(filename=logger_filename, level=logging.INFO)
        elif is_debug:
            logging.basicConfig(level=logging.DEBUG)

        self.name = "fish_contour"
        self.should_remove_plate = input_video_has_plate
        self.is_twin_view = is_twin_view
        self.is_fast_run = is_fast_run
        self.is_predicting_missing_eyes = is_predicting_missing_eyes
        self.save_n_back = save_n_back
        self.min_valid_n_back = min(save_n_back, abs(self.save_n_back - max_invalid_n_back))
        self.__reset_history()

        # thresholds - todo parameters?
        self.hunt_threshold = 25
        self.tail_point_deviation_threshold = 24  # Determined using a histogram from 35000 data points
        self.head_origin_deviation_threshold = 30
        self.bnw_threshold = 10
        self.bout_movement_threshold = 3.1
        self.min_bout_frame_length = 10

        if scale_area is not None and isinstance(scale_area, (float, int)):
            self.scale = scale_area  # scale parameters relating to sizes
            logging.debug("Fish tracker initiated with scale=" + str(self.scale))

    def __reset_history(self):
        if self.save_n_back is not None:
            self.count_n_back = 0
            self.history_analysis_data_outputs = FishOutput()
            self.history_analysis_data_outputs.reset(frame_start=1, n_frames=self.save_n_back + 1)

    def _pre_process(self, dir_path, fish_name, event_id, noise_frame) -> None:
        pass  # nothing for now

    def _post_process(self, input_frames_list: np.array, analysis_data_outputs: FishOutput=None) -> dict:
        """todo add documentation for tail

        :param input_frames_list:
        :param analysis_data_outputs:
        :return:
        """
        is_head_angle_diff_norm_below_threshold, is_head_origin_diff_norm_below_threshold = \
            self.__post_processing_find_head_errors(analysis_data_outputs)

        if len(np.where(is_head_origin_diff_norm_below_threshold == False)[0]) > 0:
            print("Frames of head above threshold: Origin ",
                  np.where(is_head_origin_diff_norm_below_threshold == False)[0] + 1)
        if len(np.where(is_head_angle_diff_norm_below_threshold == False)[0]) > 0:
            print("Frames of head above threshold: Angle ",
                  np.where(is_head_angle_diff_norm_below_threshold == False)[0] + 1)

        self.__reset_history()
        return {'is_bout_frame_list': np.array([]),
                'velocity_norms': np.array([]),
                'is_tail_point_diff_norm_below_threshold': np.array([]),
                'is_head_origin_diff_norm_below_threshold': is_head_origin_diff_norm_below_threshold,
                'is_head_angle_diff_norm_below_threshold': is_head_angle_diff_norm_below_threshold}

    @staticmethod
    def calc_points_diff(points):
        point_differences = np.diff(points, axis=0)
        point_differences_norm = np.linalg.norm(np.concatenate([point_differences[0:1, :], point_differences]), axis=1)
        return point_differences_norm

    @staticmethod
    def fill_nan_2d_interpolate(data_2d, kind='previous'):
        if np.isnan(data_2d).any():
            data_2d[:, 0] = ContourBasedTracking.fill_nan_1d_interpolate(data_2d[:, 0], kind=kind)
            data_2d[:, 1] = ContourBasedTracking.fill_nan_1d_interpolate(data_2d[:, 1], kind=kind)
        return data_2d

    @staticmethod
    def fill_nan_1d_interpolate(data_1d, kind='previous'):
        nans, x = np.isnan(data_1d), lambda z: z.nonzero()[0]
        if sum(~nans) >= 1:  # minimum needed to use interp1d (more accurate interpolation)
            data_1d[nans] = np.interp(x(nans), x(~nans), data_1d[~nans])
        if sum(~nans) >= 2:  # minimum needed to use interp1d (more accurate interpolation)
            f = interp1d(x(~nans), data_1d[~nans], kind=kind, fill_value="extrapolate")
            data_1d[nans] = f(x(nans))
        return data_1d

    def _analyse(self, input_frame: np.array, noise_frame: np.array, fps: float, frame_number: int, additional=None):
        """

        :param input_frame:
        :param noise_frame:
        :param fps:
        :param frame_number:
        :param additional: not used here. List with inputs from other trackers etc
        :return: annotated frame (for debug) & output struct
        """
        output = FrameAnalysisData()

        if self.should_remove_plate:
            cleaned = self.clean_plate_noise(input_frame, frame_number, scale=self.scale)
            if cleaned is None:  # error
                if self.save_n_back is not None and self.save_n_back > 0:
                    self.save_history(output, frame_number=frame_number)  # append nan that will be filled in post-proc. todo- use history to find?
                return input_frame, output  # default is_ok = False
        else:
            cleaned = input_frame.copy()

        an_frame = cleaned.copy()  # output frame

        # Step 1- fish contour - return with error if incorrect
        fish_contour = self.get_fish_contour(cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.uint8),
                                             scale=self.scale)
        if fish_contour is None:
            logging.debug("Frame " + str(frame_number) + " didn't find fish")
            if self.save_n_back is not None and self.save_n_back > 0:
                self.save_history(output, frame_number=frame_number)  # append nan that will be filled in post-proc. todo- use history to find?
            return an_frame, output  # default is_ok = False
        output.fish_contour = fish_contour
        cv2.drawContours(an_frame, [output.fish_contour], -1, Colors.RED, 1)  # drawn here for debugging

        # Create fish mask from contour (both fish and background (non fish)
        cleaned_fish, cleaned_non_fish, segment, mask, expanded_mask = \
            self.get_fish_segment_and_masks(an_frame, cleaned, fish_contour)
        output.fish_segment = segment.transpose(0, 2, 1).reshape(-1, 2)  # output shape (flatten): (#p, 2) - todo can remove?

        # Step 2- eyes - return with error if incorrect
        eyes_data, no_eyes_data = self.get_eyes_no_eyes_within_fish_segment(an_frame, cleaned_fish, frame_number,
                                                                            # scale = higher zoom with camera.
                                                                            # This increases fish area & also details
                                                                            # look better so we reduce eye color limit
                                                                            scale=self.scale,
                                                                            # todo this max_color is not good
                                                                            max_color=80-abs(self.scale-1) * 6)
        output.eyes_contour = [c['contour'] for c in eyes_data]

        if len(eyes_data) == 0 or len(eyes_data) == 1:
            logging.debug("Frame " + str(frame_number) + " didn't find eyes ({0} found)".format(len(eyes_data)))
            if not self.is_predicting_missing_eyes:  # error
                if self.save_n_back is not None and self.save_n_back > 0:
                    self.save_history(output, frame_number=frame_number)
                return an_frame, output  # default is_ok = False
            output.fish_head_origin_point = np.array([np.nan, np.nan])
            output.fish_head_destination_point = np.array([np.nan, np.nan])
        else:
            eyes_data, p_from, p_to, cleaned_fish_head, mask_head = \
                self.calc_fish_direction_from_eyes(cleaned_fish, eyes_data, frame_number, output, scale=self.scale)

            output.fish_head_origin_point = np.array(p_from)
            output.fish_head_destination_point = np.array(p_to)

            # Eyes difference and hunting
            output.eyes_data = FrameAnalysisData.EyesData(eyes_data, output.fish_head_origin_point,
                                                          output.fish_head_destination_point)
            # Eyes found - marked as ok for later
            output.is_ok = True

        # If saving history- use to fix errors of jumping eyes
        if self.save_n_back is not None:
            origin_points = self.history_analysis_data_outputs.origin_head_points_list.copy()
            origin_points[self.history_analysis_data_outputs.is_head_prediction_list, :] = np.nan
            if 0 < self.min_valid_n_back <= sum(~np.isnan(origin_points)[:, 0]):
                self.validate_and_fix_head_data(self.history_analysis_data_outputs, output, frame_number)  # todo recalc eye contours
                output.is_ok = True

        if self.save_n_back is not None and self.save_n_back > 0:  # todo prediction?
            self.save_history(output, frame_number=frame_number)

        if not output.is_ok:
            return an_frame, output  # default is_ok = False

        self.draw_output_on_annotated_frame(an_frame, output,
                                            hunt_threshold=self.hunt_threshold, redraw_fish_contours=False)

        if self.is_twin_view:  # show original and an side-by-side
            return np.hstack([input_frame, an_frame]), output
        else:
            return an_frame, output

    def save_history(self, output, frame_number):
        is_nans = np.isnan(self.history_analysis_data_outputs.origin_head_points_list[:self.save_n_back, 0])
        if self.count_n_back >= self.save_n_back:  # cyclic rotation (constant length list)
            self.history_analysis_data_outputs.origin_head_points_list = \
                np.roll(self.history_analysis_data_outputs.origin_head_points_list, -1, axis=0)
            self.history_analysis_data_outputs.destination_head_points_list = \
                np.roll(self.history_analysis_data_outputs.destination_head_points_list, -1, axis=0)
            self.history_analysis_data_outputs.is_head_prediction_list = \
                np.roll(self.history_analysis_data_outputs.is_head_prediction_list, -1, axis=0)

            self.history_analysis_data_outputs.origin_head_points_list[-1, :] = np.nan
            self.history_analysis_data_outputs.destination_head_points_list[-1, :] = np.nan
            self.history_analysis_data_outputs.is_head_prediction_list[-1] = False
            ind = -2  # last one is saved as nan
        else:
            self.count_n_back += 1
            ind = np.where(is_nans)[0][0]
        self.history_analysis_data_outputs.origin_head_points_list[ind, :] = output.fish_head_origin_point
        self.history_analysis_data_outputs.destination_head_points_list[ind, :] = output.fish_head_destination_point
        self.history_analysis_data_outputs.is_head_prediction_list[ind] = output.is_prediction

    @classmethod
    def draw_output_on_annotated_frame(cls, an_frame, output: FrameAnalysisData,
                                       redraw_fish_contours=True, hunt_threshold=25,
                                       is_bout=None, velocity_norms=None,
                                       is_adding_eyes_text=True, text_color=Colors.GREEN,
                                       # font etc
                                       row_left_side_text=50, col_left_side_text=20, space_between_text_rows=25,
                                       col_right_side_text=680, row_right_side_text=20, fontsize=0.6,
                                       text_font=cv2.FONT_HERSHEY_SIMPLEX, bold=2):
        """Add all elements on video (except for fish and eyes contours which are added before for debug

        :param redraw_fish_contours:
        :param is_bout:
        :param velocity_norms:
        :param bold:
        :param text_font:
        :param hunt_threshold:
        :param an_frame: input frame
        :param cleaned_non_fish: frame without fish
        :param output: fish analysis output struct for the annotations
        :param col_right_side_text, row_right_side_text: location of text beginning (right side)
        :param col_left_side_text, row_left_side_text: location of text beginning (left side)
        :param space_between_text_rows:
        :return: an_frame: frame with fish movie annotations
        """
        head_direction_angle = get_angle_to_horizontal(output.fish_head_origin_point,
                                                       output.fish_head_destination_point)
        if redraw_fish_contours:
            cv2.drawContours(an_frame, [output.fish_contour], -1, Colors.RED, 2)
            cv2.drawContours(an_frame, output.eyes_contour, -1, Colors.CYAN, 1)

        if output.eyes_data is not None and True:# is_adding_eyes_text:
            diff = output.eyes_data.diff_from_fish_direction_deg
            eye_angles = output.eyes_data.abs_angle_deg
            row_left_side_text += space_between_text_rows * 2
            diff = sorted(diff)
            cv2.putText(an_frame, "E: ({0:.2f},{1:.2f})".format(diff[0], diff[1]),
                        (col_left_side_text, row_left_side_text), text_font, fontsize, text_color, bold)
            # row_right_side_text += space_between_text_rows
            # row_left_side_text += space_between_text_rows * 2
            # cv2.putText(an_frame, "E: ({0:.2f},{1:.2f})".format(eye_angles[0], eye_angles[1]),
            #             (col_left_side_text, row_left_side_text), text_font, fontsize, text_color, bold)
            row_left_side_text += space_between_text_rows

        cv2.putText(an_frame, "Dir: {0:.2f}".format(head_direction_angle),
                    (col_left_side_text, row_left_side_text), text_font, fontsize, text_color, bold)
        row_left_side_text += space_between_text_rows

        if output.eyes_data is not None:
            # result, color = "No-hunt", Colors.GREEN
            # if cls.is_hunting(hunt_threshold, output):
            #     result, color = "Hunt", Colors.PINK
            # cv2.putText(an_frame, result, (col_left_side_text, row_left_side_text), text_font, fontsize, color, bold)
            # row_left_side_text += space_between_text_rows

            # draw ellipse majors
            ellipse: EllipseData
            for ellipse in output.eyes_data.ellipses:
                xc, yc = ellipse.ellipse_center
                angle, rmajor = ellipse.ellipse_direction, ellipse.ellipse_major
                angle = angle - 90 if angle > 90 else angle + 90
                xtop, ytop = xc + math.cos(math.radians(angle)) * rmajor, yc + math.sin(math.radians(angle)) * rmajor
                xbot, ybot = xc + math.cos(math.radians(angle + 180)) * rmajor, yc + math.sin(math.radians(angle + 180)) * rmajor
                cv2.line(an_frame, (int(xtop), int(ytop)), (int(xbot), int(ybot)), Colors.CYAN, 1)
                cv2.ellipse(an_frame, cls.point_to_int([xc, yc]),
                            cls.point_to_int([ellipse.ellipse_major, ellipse.ellipse_minor]), float(angle), 0.0, 360.0,
                            Colors.CYAN, 1)
            # draw ellipse centers
            # centers = [ellipse.ellipse_center for ellipse in output.eyes_data.ellipses]
            # cv2.circle(an_frame, cls.point_to_int(centers[0]), 1, Colors.CYAN, thickness=cv2.FILLED)
            # cv2.circle(an_frame, cls.point_to_int(centers[1]), 1, Colors.CYAN, thickness=cv2.FILLED)

        # Draw direction and points
        cv2.circle(an_frame, cls.point_to_int(output.fish_head_origin_point), 1, Colors.CYAN, thickness=cv2.FILLED)
        cv2.arrowedLine(an_frame, cls.point_to_int(output.fish_head_origin_point),
                        cls.point_to_int(output.fish_head_destination_point),
                        Colors.YELLOW if output.is_prediction else Colors.RED, 2)

        if output.tail_data is not None:
            tail: FrameAnalysisData.TailData = output.tail_data
            # cv2.circle(an_frame, cls.point_to_int(tail.tail_tip_point), 4, Colors.YELLOW, thickness=cv2.FILLED)
            # cv2.putText(an_frame, 'T: {0:.2f} {1:.2f}'.format(tail.tail_tip_point[0], tail.tail_tip_point[1]),
            #             (col_right_side_text, row_right_side_text), text_font, fontsize, Colors.GREEN, bold)
            # row_right_side_text += space_between_text_rows

            midline_path = tail.tail_path
            # for index in range(1, len(midline_path)):
            #     first_point = tuple(midline_path[index-1])
            #     second_point = tuple(midline_path[index])
                # cv2.line(an_frame, first_point, second_point, Colors.GREEN, 1)

        if is_bout is not None:
            result, color = "No-Bout", Colors.PINK
            if is_bout:
                result, color = "Bout", Colors.GREEN
            cv2.putText(an_frame, result, (col_left_side_text, row_left_side_text), text_font, fontsize, color, bold)
            row_left_side_text += space_between_text_rows

        if velocity_norms is not None:
            # cv2.putText(an_frame, 'V: {0:.2f}'.format(velocity_norms),
            #             (col_right_side_text, row_right_side_text), text_font, fontsize, text_color, bold)
            row_right_side_text += space_between_text_rows

    @staticmethod
    def is_hunting(hunt_threshold, output):
        if output.eyes_data is None:
            return False
        return len([d for d in output.eyes_data.diff_from_fish_direction_deg if abs(d) >= 10]) == 2 \
            and np.mean(np.abs(output.eyes_data.diff_from_fish_direction_deg)) >= hunt_threshold \
            and sum(np.sign(output.eyes_data.diff_from_fish_direction_deg)) == 0

    @staticmethod
    def validate_and_fix_head_data(analysis_data_outputs: FishOutput, output: FrameAnalysisData,
                                   frame_number):
        check_origin = analysis_data_outputs.origin_head_points_list.copy()
        check_origin[analysis_data_outputs.is_head_prediction_list, :] = np.nan
        check_dest = analysis_data_outputs.destination_head_points_list.copy()
        check_dest[analysis_data_outputs.is_head_prediction_list, :] = np.nan
        origin_head_points = ContourBasedTracking.fill_nan_2d_interpolate(check_origin.copy(), kind='linear')  #quadratic?
        dest_head_points = ContourBasedTracking.fill_nan_2d_interpolate(check_dest.copy(), kind='linear')
        if not np.isnan(output.fish_head_origin_point).any():
            origin_head_points[-1, :] = output.fish_head_origin_point
        if not np.isnan(output.fish_head_destination_point).any():
            dest_head_points[-1, :] = output.fish_head_destination_point

        head_direction_angles = [get_angle_to_horizontal(orig, dest)
                                 for (orig, dest) in zip(origin_head_points, dest_head_points)]

        is_head_origin_diff_norm_below_threshold = \
            abs(np.diff(ContourBasedTracking.calc_points_diff(origin_head_points))) < 12 # self.head_origin_deviation_threshold

        # angle can jump to 360 if cross the edge, but head flip should be around 180 degrees
        is_head_angle_diff_norm_below_threshold = np.bitwise_or(abs(360-abs(np.diff(head_direction_angles))) < 90,
                                                                abs(np.diff(head_direction_angles)) < 90)

        # quickfix: idenify both frame and after it as change: append zero and use diff equals 1 to identify location
        # (using int -1 marks false->true, and 1 marks true->false
        is_head_origin_diff_norm_below_threshold = \
            np.diff(np.concatenate([np.zeros(shape=(1,)), is_head_origin_diff_norm_below_threshold.astype(int)])) != -1
        is_head_angle_diff_norm_below_threshold = \
            np.diff(np.concatenate([np.zeros(shape=(1,)), is_head_angle_diff_norm_below_threshold.astype(int)])) != -1

        if not is_head_origin_diff_norm_below_threshold[-1] or not is_head_angle_diff_norm_below_threshold[-1] or \
           np.isnan(output.fish_head_origin_point[-1]) or np.isnan(output.fish_head_destination_point[-1]):
            reason = "empty eyes" if (np.isnan(output.fish_head_origin_point[-1]) or
                                      np.isnan(output.fish_head_destination_point[-1])) else "jump"
            logging.debug("Frame {0} using prediction due to {1}".format(frame_number, reason))

            origin_head_points = \
                ContourBasedTracking.fill_nan_2d_interpolate(analysis_data_outputs.origin_head_points_list.copy())
            dest_head_points = \
                ContourBasedTracking.fill_nan_2d_interpolate(analysis_data_outputs.destination_head_points_list.copy())
            output.fish_head_origin_point = origin_head_points[-1, :].copy()
            output.fish_head_destination_point = dest_head_points[-1, :].copy()
            output.is_prediction = True

    @staticmethod
    def __post_processing_is_diff_below_threshold_fix(is_diff_below_threshold, origin_head_points, deviation_threshold,
                                                      name_for_debug="Origin", use_start_end_segmentation=False):
        """Fix issues with simple boolean mark of threshold:
        1. when calculating diff (change), it is returned as twice for single point that jumps
        2. when there is a segment of "jumps" (continuous or with "holes" of correct values), the frames within are not
        identified as false (the diff from previous value is small).

        :param is_diff_below_threshold: list of boolean values to fix
        :param origin_head_points: used to make sure the segment is correctly found
        :param is_diff_below_threshold:
        :param is_diff_below_threshold:
        :return:
        """
        # due to the way diff is calculated, we always have False for increase and False for one index after decrease
        start_end_indices = np.where(np.bitwise_not(is_diff_below_threshold))[0]
        if use_start_end_segmentation and (len(start_end_indices) > 0 and len(start_end_indices) % 2 == 0):
            start_indices, end_indices = start_end_indices[0:len(start_end_indices):2], \
                                         start_end_indices[1:len(start_end_indices):2] - 1
            false_indices = np.concatenate([np.arange(s, e + 1) for (s, e) in zip(start_indices, end_indices)])
            is_ok = np.concatenate(
                [abs(ContourBasedTracking.calc_points_diff(origin_head_points[s:e + 1, :])) < deviation_threshold
                 for (s, e) in zip(start_indices, end_indices)])
            if np.array(is_ok).all():  # all differences within start-end are indeed the same
                is_diff_below_threshold[:] = True
                is_diff_below_threshold[false_indices] = False
                logging.debug(name_for_debug +
                              " start-end indices {0} => false indices {1}".format(start_end_indices + 1,
                                                                                   false_indices + 1))
            else:
                logging.error(name_for_debug +
                              " start-end indices have jumps within segment {0}".format(false_indices + 1))
                is_diff_below_threshold = \
                    np.diff(np.concatenate([is_diff_below_threshold[0:1],
                                            is_diff_below_threshold.astype(int)])) != -1
        elif len(start_end_indices) > 0:
            logging.debug(name_for_debug + " start-end indices have odd number. {0}".format(start_end_indices + 1))
            is_diff_below_threshold = \
                np.diff(np.concatenate([is_diff_below_threshold[0:1],
                                        is_diff_below_threshold.astype(int)])) != -1
        return is_diff_below_threshold

    def __post_processing_find_head_errors(self, analysis_data_outputs, use_start_end_segmentation=False):
        """Return list of 2 boolean values, one check origin jump and one angle jump (head flip)

        :param analysis_data_outputs: tracker output struct of the whole movie
        :param use_start_end_segmentation: default false, until this logic find better the segments
        :return: is_head_angle_diff_norm_below_threshold, is_head_origin_diff_norm_below_threshold - lists of booleans
        """
        # origin points and destination points used for both point jump and angle jump check.
        # Make sure false status has nan value (prediction has valid value since we check the prediction is ok)
        check_origin = analysis_data_outputs.origin_head_points_list.copy()
        check_origin[analysis_data_outputs.fish_status_list == False, :] = np.nan
        check_dest = analysis_data_outputs.destination_head_points_list.copy()
        check_dest[analysis_data_outputs.fish_status_list == False, :] = np.nan

        # Fill nan values with interpolation
        origin_head_points = ContourBasedTracking.fill_nan_2d_interpolate(check_origin.copy())
        dest_head_points = ContourBasedTracking.fill_nan_2d_interpolate(check_dest.copy())

        head_direction_angles = [get_angle_to_horizontal(orig, dest)
                                 for (orig, dest) in zip(origin_head_points, dest_head_points)]

        # origin can jump up to some threshold
        differences_origin = abs(ContourBasedTracking.calc_points_diff(origin_head_points))
        is_head_origin_diff_norm_below_threshold = differences_origin < self.head_origin_deviation_threshold

        # angle can jump to 360 if cross the edge, but head flip should be around 180 degrees
        head_direction_angles = np.concatenate([head_direction_angles[0:1], head_direction_angles])
        is_head_angle_diff_norm_below_threshold = np.bitwise_or(abs(360-np.diff(head_direction_angles)) < 90,
                                                                abs(np.diff(head_direction_angles)) < 90)

        # fine tune boolean marks by considering segments etc
        is_head_origin_diff_norm_below_threshold = \
            self.__post_processing_is_diff_below_threshold_fix(is_head_origin_diff_norm_below_threshold,
                                                               origin_head_points, self.head_origin_deviation_threshold,
                                                               name_for_debug="Origin",
                                                               use_start_end_segmentation=use_start_end_segmentation)
        is_head_angle_diff_norm_below_threshold = \
            self.__post_processing_is_diff_below_threshold_fix(is_head_angle_diff_norm_below_threshold,
                                                               head_direction_angles, 90, name_for_debug="Angle",
                                                               use_start_end_segmentation=use_start_end_segmentation)

        return is_head_angle_diff_norm_below_threshold, is_head_origin_diff_norm_below_threshold

    @staticmethod
    def fix_ellipse_angle(ellipse_angle, fish_dir_angle):  # todo move outside?
        """

        :param ellipse_angle: 0-180 relative to y axis (270 and 90 in x axis is 0 here)
        :param fish_dir_angle: 0-360 relative to x axis (counter-clockwise)
        :return: ellipse_angle is same direction relative to fish_direction_angle
        """
        result_ellipse_angle = 90 - ellipse_angle  # (-90) - 90 range - relative to x axis
        orig_result_ellipse_angle = result_ellipse_angle

        diff = result_ellipse_angle - fish_dir_angle
        if diff < 0:
            result_ellipse_angle += 180 * (round(abs(diff) / 180))
        elif diff > 180:
            result_ellipse_angle -= 180 * (round(abs(diff) / 180))

        if not (0 <= abs(result_ellipse_angle - fish_dir_angle) <= 90):
            logging.error('Error in ellipse angle. Bad result: {0}, fish-dir {1}, original ellipse angle {2]'.format(
                result_ellipse_angle, fish_dir_angle, orig_result_ellipse_angle))
        return result_ellipse_angle

    @staticmethod
    def get_fish_segment_and_masks(an_frame, cleaned, fish_contour):
        # mask contains fish only
        mask = np.full((an_frame.shape[0], an_frame.shape[1]), 0, dtype=np.uint8)
        cv2.drawContours(mask, [fish_contour], contourIdx=-1, color=Colors.WHITE, thickness=cv2.FILLED)
        segment = cv2.findNonZero(mask)  # shape: (# points, 1, 2)
        # Create cleaned figures only - expand mask with dilate + blurring
        d_mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)  # expand a little the fish
        ret, d_mask = cv2.threshold(cv2.medianBlur(d_mask, 9), 30, 255, cv2.THRESH_BINARY)
        cleaned_fish = cv2.bitwise_and(cleaned, cleaned, mask=d_mask)  # search objects within fish only
        cleaned_non_fish = cv2.bitwise_and(cleaned, cleaned, mask=np.bitwise_not(d_mask))
        return cleaned_fish, cleaned_non_fish, segment, mask, d_mask

    @staticmethod
    def get_contours(gray, threshold1=30, threshold2=200, is_blur=False, is_close=True, ctype=cv2.RETR_TREE,
                     close_kernel=(5, 5), min_area_size=None):  # todo scale?
        gray = cv2.Canny(gray, threshold1, threshold2)
        if is_blur:  # smear edges to have full fish contour
            gray = cv2.blur(gray, (3, 3))
        elif is_close:  # use close instead
            kkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kkernel)

        contours, hierarchy = cv2.findContours(gray, ctype, cv2.CHAIN_APPROX_NONE)
        if min_area_size is None:
            return [c for c in contours if c.shape[0] >= 5], hierarchy  # ellipse fit requires min 5 points
        return [c for c in contours if c.shape[0] >= 5 and cv2.contourArea(c) >= min_area_size], hierarchy

    @staticmethod
    def two_closest_shape_contours(contours_data, max_distance=25, scale=1):  # todo move outside?
        min_area_diff = math.inf
        eyes_contours = None
        for c1_d in contours_data:
            c1 = c1_d['contour']
            (xc_1, yc_1), _, _ = c1_d['ellipse']
            for c2_d in contours_data:
                c2 = c2_d['contour']
                curr_area_diff = abs(cv2.contourArea(c1) - cv2.contourArea(c2))
                (xc_2, yc_2), _, _ = c2_d['ellipse']
                if curr_area_diff < min_area_diff and np.all(c2 != c1) and \
                   0 < distance.euclidean([xc_1, yc_1], [xc_2, yc_2]) <= max_distance * scale:
                    eyes_contours = [c1_d, c2_d]
                    min_area_diff = curr_area_diff
        return eyes_contours

    @classmethod
    def get_eyes_no_eyes_within_fish_segment(cls, an_frame, frame, frame_number, scale=1,
                                             min_eyes_area=50, max_color=80, max_distance=25):  # todo refactor
        """Logic: find dark (<=max_color) contours within [min_eyes_size-min_eyes_size] area size (eyes are ~60).
        If more than 2 are found (for example, if fins create another dark contour), search for 2 closest to each other.

        min_eyes_area=50, max_eyes_area=200 - needed?

        :param is_new_movie:
        :param frame_number:
        :param scale: scale distance & area (based on zoom)
        :param max_distance:
        :param min_eyes_area:
        :param an_frame: frame to annotate eyes
        :param frame: input frame, masked by fish segment
        :param max_color: define how dark the eyes should be. ~70
        :return:
        """
        def search_possible_kernels(curr_frame=frame): # start with normal/most common. If not working try higher
            inner_cnts = search(curr_frame=curr_frame)
            found = False
            if 2 <= len(inner_cnts):  # no eyes were found
                eyes_sus = [c for c in inner_cnts if c[1] <= max_color]
                if len(eyes_sus) >= 2:
                    found = True
            if not found:
                inner_cnts2 = search(curr_frame=curr_frame, close_kernel=(3, 3))
                if 2 <= len(inner_cnts2):  # no eyes were found
                    eyes_sus = [c for c in inner_cnts2 if c[1] <= max_color]
                    if len(eyes_sus) >= 2:
                        found = True
                        inner_cnts = inner_cnts2
            if not found:
                inner_cnts2 = search(curr_frame=curr_frame, close_kernel=(7, 7))
                if 2 <= len(inner_cnts2):  # no eyes were found
                    eyes_sus = [c for c in inner_cnts2 if c[1] <= max_color]
                    if len(eyes_sus) >= 2:
                        found = True
                        inner_cnts = inner_cnts2
            return inner_cnts, found

        def search(curr_frame=frame, close_kernel=(5, 5)):
            contours, hier = cls.get_contours(curr_frame, close_kernel=close_kernel, ctype=cv2.RETR_CCOMP)
            inner_cnts = []
            for i, cont in enumerate(contours):
                if len(cont) >= 5:
                    mask_c = np.full((curr_frame.shape[0], curr_frame.shape[1]), 0, dtype=np.uint8)
                    cv2.drawContours(mask_c, [cont], -1, color=Colors.WHITE, thickness=cv2.FILLED)
                    mean_intensity = cv2.mean(curr_frame, mask=mask_c)[0]
                    inner_cnts.append([i, mean_intensity, cont, mask_c, cv2.fitEllipse(cont), cv2.contourArea(cont)])
            inner_cnts = sorted(inner_cnts, key=lambda r: -r[1])  # sort by mean intensity small to big
            inner_cnts = [c for c in inner_cnts if c[-1] >= min_eyes_area * scale and c[1] <= max_color]
            inds = set()
            for cont_1 in inner_cnts:
                (xc_1, yc_1), _, _ = cont_1[4]
                for cont_2 in inner_cnts:
                    (xc_2, yc_2), _, _ = cont_2[4]
                    if 0 < distance.euclidean([xc_1, yc_1], [xc_2, yc_2]) <= max_distance * scale:
                        inds.add(cont_1[0])
                        inds.add(cont_2[0])
            return [c for c in inner_cnts if c[0] in inds]

        eyes_data = []
        no_eyes_data = []

        inner_cnts, found = search_possible_kernels()
        if not found:
            frame_2 = frame.copy()
            # This is an corner case where eye is separated from body- try to use large contour to fix frame
            fish = cls.get_fish_contour(frame_2, scale=scale)
            if fish is None:
                return [], []  # this shouldn't happen
            cv2.drawContours(frame_2, [fish], -1, color=Colors.WHITE)
            inner_cnts2, found = search_possible_kernels(frame_2)
            if found:
                inner_cnts = inner_cnts2

        for i, mean_intensity, c, mask_c, ellipse, area in inner_cnts:
            # todo didn't find more efficient way to calculate mean within contour
            if mean_intensity <= max_color:  # Should be very dark
                eyes_data.append({'contour': c, 'mask': mask_c, 'area': cv2.contourArea(c),
                                  'mean_intensity': mean_intensity, 'ellipse': ellipse})
            else:
                no_eyes_data.append({'contour': c, 'mask': mask_c, 'area': cv2.contourArea(c),
                                     'mean_intensity': mean_intensity, 'ellipse': ellipse})
        if 0 < len(eyes_data) <= 2:  # one/two eyes were found
            cv2.drawContours(an_frame, [c['contour'] for c in eyes_data], -1, color=Colors.CYAN)
            return eyes_data, no_eyes_data
        elif len(eyes_data) > 2:
            eyes_data2 = cls.two_closest_shape_contours(eyes_data, max_distance=max_distance, scale=scale)
            if eyes_data2 is None:
                return [], []
            eyes_ellipses = [c['ellipse'] for c in eyes_data2]
            no_eyes_data.extend([c_d for c_d in eyes_data if c_d['ellipse'] not in eyes_ellipses])
            cv2.drawContours(an_frame, [c['contour'] for c in eyes_data2], -1, color=Colors.CYAN)
            return eyes_data2, no_eyes_data
        elif len(eyes_data) == 0:
            return [], []  # this shouldn't happen

    @classmethod
    def clean_plate_noise(cls, input_frame, frame_number, scale=1,
                          close_kernel=(9, 9), min_fish_size=2000, max_fish_size=10000, max_size_ratio=3, is_blur=True):
        """Remove contours which creates a plate.
        Use fish suspects & paramecia (based on size) estimates to remove plate (which can be "broken" to several
        pieces). is_blur=True adds few more pixels to make sureall is removed.
        :return: 'cleaned' frame
        """
        contours, _ = cls.get_contours(input_frame, ctype=cv2.RETR_EXTERNAL, close_kernel=close_kernel)
        hulls = [(cv2.convexHull(cnt), cnt) for cnt in contours]

        # Fish is limited both in its area and the ratio between this and the hull (small pieces of the plate have
        # larger ratio between the 2 even if the area is similar)
        fish_suspects = [(h, c) for (h, c) in hulls if
                         min_fish_size * scale <= cv2.contourArea(h) <= max_fish_size and
                         1 <= cv2.contourArea(h) / cv2.contourArea(c) <= max_size_ratio * scale]

        if len(fish_suspects) == 1:  # found exactly the fish - remove plate!
            fish_h, fish_c = fish_suspects[0]
            non_fish_contours = [c for (h, c) in hulls if 500 <= cv2.contourArea(h) and  # don't remove paramecia
                                 # Validate not fish
                                 cv2.contourArea(fish_h) != cv2.contourArea(h) and
                                 cv2.contourArea(fish_c) != cv2.contourArea(c)]
            result = input_frame.copy()
            mask = np.zeros(result.shape[:2], np.uint8)
            cv2.drawContours(mask, non_fish_contours, -1, Colors.WHITE, thickness=cv2.FILLED)
            if is_blur:  # smear edges to have full fish contour
                _, mask = cv2.threshold(cv2.blur(mask, (11, 11)), 10, 255, cv2.THRESH_BINARY)
            return cv2.bitwise_and(input_frame, input_frame, mask=cv2.bitwise_not(mask))

        # This should happen if fish is attached to plate (try to estimate plate as circle)
        result = clean_plate(input_frame)
        if result is not None:  # succeed -> check fish is found
            fish_contour = cls.get_fish_contour(result.astype(input_frame.dtype), scale=scale)  # get external only
            if fish_contour is not None:
                return result
            else:
                logging.error("Frame " + str(frame_number) + " find plate but didn't find fish.")
        else:
            logging.error("Frame " + str(frame_number) + " didn't find plate.")

        return None

    @classmethod
    def get_fish_contour(cls, gray, close_kernel=(9, 9), min_fish_size=50, max_fish_size=10000, scale=1):
        contours, _ = cls.get_contours(gray, ctype=cv2.RETR_EXTERNAL, close_kernel=close_kernel)  # get external only
        # remove paramecia
        contours = [c for c in contours if min_fish_size * scale <= cv2.contourArea(c) <= max_fish_size * scale]
        if len(contours) > 0:
            return max(contours, key=cv2.contourArea)  # if cleaned, fish is largest
        return None

    @staticmethod
    def get_fish_points(fish_contour):
        fish_ellipse = cv2.fitEllipse(fish_contour)
        (xc, yc), (d1, d2), angle = fish_ellipse
        rmajor = max(d1, d2) / 2
        if angle > 90:
            angle = angle - 90
        else:
            angle = angle + 90
        xtop = xc + math.cos(math.radians(angle)) * rmajor
        ytop = yc + math.sin(math.radians(angle)) * rmajor
        xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
        ybot = yc + math.sin(math.radians(angle + 180)) * rmajor
        return (xtop, ytop), (xbot, ybot)

    def calc_fish_direction_from_eyes(self, cleaned_fish_frame, eyes_data, frame_number, output, scale=1):
        """

        :param cleaned_fish_frame: input for current algorithm - original pixels of fish only (after mask)
        :param eyes_data: as calculated by searching for their contours and ellipses.
        :param frame_number: for debug use (you can break-point on specific frame)
        :param output: set to False if can't calculate direction properly
        :return: eyes_data, p_from, p_to - where p_to and p_from are dest and origin of fish direction
        """
        # Head contour around eyes - via bounding rectagle around both eyes
        min_x, min_y, max_x, max_y = [np.inf, np.inf, 0, 0]
        cleaned_fish_head = None
        for c in output.eyes_contour:
            (x, y, w, h) = cv2.boundingRect(c)
            min_x, max_x = min(x, min_x), max(x + w, max_x)
            min_y, max_y = min(y, min_y), max(y + h, max_y)

        # If didn't find this rectangle - proceed with fish contour
        if not (max_x - min_x > 0 and max_y - min_y > 0):
            p_to, p_from, eyes_data = \
                self.calc_0_approx_fish_direction_based_on_eyes(output.fish_contour, output.eyes_contour, eyes_data)
        else:
            # mask head - expand rectangle with add_head pixels
            add_head = 30 * max(1, int(scale / 2))
            top_left = (min_x - add_head, min_y - add_head)
            bottom_right = (max_x + add_head, max_y + add_head)
            # cv2.rectangle(an_frame, top_left, bottom_right, Colors.PINK)  # visualize rect

            # mask
            mask_head = np.full((cleaned_fish_frame.shape[0], cleaned_fish_frame.shape[1]), 0, dtype=np.uint8)
            cv2.rectangle(mask_head, top_left, bottom_right, Colors.WHITE, cv2.FILLED)
            cleaned_fish_head = cv2.bitwise_and(cleaned_fish_frame, cleaned_fish_frame,
                                                mask=mask_head)  # search objects within fish only
            # show("head", cleaned_fish_head)
            # Search fish again: At this point the eyes were found= dont expand fish too much (smaller kernel)
            fish_head_contour = self.get_fish_contour(cleaned_fish_head, close_kernel=(5, 5), scale=scale)

            # Approx.0 on head only
            p_to, p_from, eyes_data = \
                self.calc_0_approx_fish_direction_based_on_eyes(fish_head_contour, output.eyes_contour, eyes_data)

            # cv2.circle(an_frame, self.point_to_int(p_to), 5, Colors.BLUE, -1)

            # Override p_to with line through fish head
            # vx, vy, x0, y0 = self.get_head_line(fish_head_contour)  # this is an alternative algo.
            vx, vy, x0, y0 = self.get_line_perpendicular_to_eyes(eyes_data, p_from)

            # Fix vx vy based on 'stronger' direction (fix near 0 errors): the fix is since can be 180deg error
            length = add_head * 2
            if np.abs(vy) >= np.abs(vx) and vy != 0:
                t1 = length * np.sign((p_to[1] - y0) / vy)
            elif np.abs(vx) >= np.abs(vy) and vx != 0:
                t1 = length * np.sign((p_to[0] - x0) / vx)
            else:
                output.is_ok = False
                logging.error("Error: frame #{0} has 0 vx and vy!".format(frame_number))
                t1 = 0

            p_to = (float(x0 + t1 * vx), float(y0 + t1 * vy))
        return eyes_data, p_from, p_to, cleaned_fish_head, mask_head

    @staticmethod
    def calc_0_approx_fish_direction_based_on_eyes(head_contour, eyes_contours, eyes_data):
        """ Used to calc direction, by using shape fit around eyes as p_to, and mid-eyes point as p_from.
        This is 0th approx, since using fish direction it is fixed.

        :param head_contour:
        :param eyes_contours:
        :param eyes_data:
        :return:
        """
        p_to = ContourBasedTracking.closest_point_of_bound_shape_to_eyes(eyes_contours, head_contour)

        middle = [0, 0]
        for i in range(len(eyes_data)):
            (xc, yc), (d1, d2), angle = eyes_data[i]['ellipse']
            major = max(d1, d2) / 2
            minor = min(d1, d2) / 2
            eyes_data[i]['center'] = (xc, yc)
            eyes_data[i]['angle'] = angle
            eyes_data[i]['major'] = major
            eyes_data[i]['minor'] = minor
            middle[0] += xc
            middle[1] += yc
        p_from = (middle[0] / 2, middle[1] / 2)
        return p_to, p_from, eyes_data

    @staticmethod
    def closest_point_of_bound_shape_to_eyes(eyes_contours, head_contour):
        pts = ContourBasedTracking.get_fish_points(head_contour)
        distances = []
        for p in pts:
            distance_to_contours_top = [distance.euclidean((c[0][0][0], c[0][0][1]), (p[0], p[1])) for c in
                                        eyes_contours]
            distances.append({'dist': sum(distance_to_contours_top), 'point': (p[0], p[1])})
        distances = sorted(distances, key=lambda d: d['dist'])
        p_to = (distances[0]['point'][0], distances[0]['point'][1])  # 0 is closest
        return p_to

    @staticmethod
    def get_head_line(head_contour):
        [vx, vy, x, y] = cv2.fitLine(head_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        return vx[0], vy[0], x[0], y[0]  # todo cut with contour and not image

    @staticmethod
    def get_line_perpendicular_to_eyes(eyes_data, p_from):
        centers = [eye['center'] for eye in eyes_data]
        vx = centers[0][0] - centers[1][0]
        vy = centers[0][1] - centers[1][1]
        # Normal to line
        mag = math.sqrt(vx ** 2 + vy ** 2)
        temp = vx / mag
        vx = -(vy / mag)
        vy = temp
        return vx, vy, p_from[0], p_from[1]
