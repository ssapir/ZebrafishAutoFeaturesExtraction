import time
import traceback
from typing import List

import cv2  # pip install opencv-contrib-python, doc: https://pypi.org/project/opencv-contrib-python/
import sys
import numpy as np
from abc import ABCMeta, abstractmethod


SLEEP_BETWEEN_FRAMES = 1  # increase to 30 for slow motion view


class Colors:
    BLACK = (0, 0, 0)
    YELLOW = (0, 255, 255)
    PINK = (255, 0, 255)
    CYAN = (255, 255, 0)
    PURPLE = (120, 81, 169)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (50, 170, 50)  # (0, 255, 0) is hideous green


class ClassicCvAbstractTrackingAPI(metaclass=ABCMeta):
    """ All trackers should inherit from it (main pipeline usage) and adjust pipeline by booleans.

    Has the following process:
    - Pre-processing: calibration of tracker unique parameters based on X given frames. This step occurs before main
                      loop (analyse) since it can be slow, if needed.
      Note: static noise cleaning is done prior to this step. Functionality common to all trackers occurs outside.
    - Analyse: receive current frame (as is!), return this frame result (either needed by other trackers, or to be saved later).
               Should be fast, stateless/full algorithm (where state is aggregated within class).
    - Post-processing: fixing analyse step base on full event history. This is separated from analyse step since it can't
                       be used on real-time.
      Note: post-processing function gets (for now - todo memory) full frames and full previous calculation data lists,
            so no need to aggregate in memory (this is done to make sure all trackers work on full data).

    Example: using fish tracker -> paramecia tracker -> fish tracker.
    Example 1: fish tracker
    - Pre-processing: nothing (the generic noise cleaning is enough)
    - Analyse: input- frame + static noise. Output- direction points for current frame (to be saved),
               as well as fish contour & full-segment (for other trackers usage).
               First run in one iteration - paramecia data is None, fish contour only is calculated.
               Second run in one iteration- paramecia data is given, full output is returned to be saved.
    - Post-processing: refinement of points, after paramecia

    Example 2: paramecia tracker (called after fish tracker run)
    - Pre-processing: calibrate paramecia size.
    - Analyse: input - frame + static noise and fish output data. Output- paramecia center trajectories
               (to be saved), as well as full segments (for other trackers usage).
    - Post-processing: refinement of points, using predictions on full movie history.

    Common/simplest inheritance:
    - empty pre and post processing implementations
    - stateless (nothing is saved in memory) _analyse implementation
    """
    def __init__(self, visualize_movie=False):
        self.name = "Abstract"
        self.visualize_movie = visualize_movie
        self.analysis_duration_secs = []
        self.pre_processing_duration_secs = []
        self.post_processing_duration_secs = []

    # -------------- Override me --------------------

    @abstractmethod  # abstract methods must be overridden
    def _pre_process(self, dir_path, fish_name, event_id, noise_frame):
        """Calibration of inner properties unique to this tracker (explained before).
        This method is called between events.

        TODO adjust parameters
        :param dir_path: input directory for events
        :return: Nothing
        """
        raise Exception("Override me (reminder)")

    @abstractmethod  # abstract methods must be overridden
    def _analyse(self, input_frame: np.array, noise_frame: np.array, fps: int, frame_number: int, additional=None) \
            -> (np.array, bool, np.array):
        """Returns annotated frame (for debug-video) as well as data (class matching tracker's output).
        The annotated frame is optional (return empty np.array for no video creation at this stage).

        :param input_frame: current input from video. Also get noise_frame to clean data within tracker
        :param noise_frame: current input from video. Also get noise_frame to clean data within tracker
        :param fps: frames-per-second of the video
        :param frame_number:
        :param additional: additional data structs, from previous tracker processing. A list of any input from main.
        :return: annotated_frame, output class (depends on the specific tracker API)
        """
        raise Exception("Override me (reminder)")

    @abstractmethod  # abstract methods must be overridden
    def _post_process(self, input_frames_list: np.array, analysis_data_outputs=None) -> (np.array, bool, np.array):
        """Return frames list and outputs struct list, if needed, for fixes at the end of full event

        :param input_frames_list: list of input frames of full video, as well we full data
        :param analysis_data_outputs: struct holding analysis outputs
        :return: annotated_frames_list, outputs list (class depends on the specific tracker API)
        annotated_frames_list are saved as video for debug etc
        """
        raise Exception("Override me (reminder)")

    # -------------- Added Functionality --------------------

    def pre_process(self, dir_path, fish_name, event_id, noise_frame):
        """Wrapper of _pre_process with additional functionality of statistical measures.
        """
        outputs = None
        start = time.process_time()
        try:
            outputs = self._pre_process(dir_path, fish_name, event_id, noise_frame)
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
        end = time.process_time()
        self.pre_processing_duration_secs.append(end-start)
        return outputs

    def post_process(self, input_frames_list: np.array, analysis_data_outputs=None) -> (np.array, bool, np.array):
        """Wrapper of _post_process with additional functionality of statistical measures.
        """
        outputs = None
        start = time.process_time()
        # todo protect input frames list? Should it be a list?
        try:
            outputs = self._post_process(input_frames_list, analysis_data_outputs=analysis_data_outputs)
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
        end = time.process_time()
        self.post_processing_duration_secs.append(end-start)
        return outputs

    def analyse(self, input_frame: np.array, noise_frame: np.array, fps: int, frame_number: int, additional=None) \
            -> (np.array, bool, np.array):
        """Wrapper of _analyse with additional functionality of statistical measures, common frame annotations,
        and error handling.
        """
        start = time.process_time()

        # assign none values. These should be assigned if no error occurred
        annotated_frame = None
        output = None

        try:
            # Copy input image to make sure the trackers won't affect original (main) data by mistake
            outputs = self._analyse(input_frame=input_frame.copy(), noise_frame=noise_frame, fps=fps,
                                    frame_number=frame_number, additional=additional)
            if outputs is not None:  # None is an error!
                annotated_frame, output = outputs  # unpack outputs

                # allow empty annotated frame (if empty - will not be saved)
                if annotated_frame is not None:
                    annotated_frame = self._annotate_frame(annotated_frame, output.is_ok, fps, frame_number)
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)

        end = time.process_time()
        self.analysis_duration_secs.append(end-start)
        return annotated_frame, output

    def print_time_statistics(self):
        print("Analysis Time mean(s): ", np.mean(self.analysis_duration_secs),
              ', std ', np.std(self.analysis_duration_secs), ', min: ', np.min(self.analysis_duration_secs),
              ', max: ', np.max(self.analysis_duration_secs))
        if len(self.pre_processing_duration_secs) > 0:
            print("Pre-processing Time mean(s): ", np.mean(self.pre_processing_duration_secs),
                  ', std', np.std(self.pre_processing_duration_secs))
        if len(self.post_processing_duration_secs) > 0:
            print("Post-processing Time mean(s): ", np.mean(self.post_processing_duration_secs),
                  ', std', np.std(self.post_processing_duration_secs))
    @staticmethod
    def show(n, f):  # show small movie
        r = 1.5
        if f.shape[0] > 1000:
            r = 5
        return cv2.imshow(n, cv2.resize(f, (round(f.shape[0] / r), round(f.shape[1] / r))))

    @staticmethod
    def point_to_int(point):
        """For visualization of floating (x,y) point"""
        return round(point[0]), round(point[1])

    def _annotate_frame(self, input_frame: np.array, ok: bool, fps: float, frame_number: int) -> np.array:
        """Commonly used. Helps annotate similarly metadata as a frame for output video.

        If visualize_movie is True this will also show on live the output saved as video, therefore
        you should annotate any additional data before (like bbox and points).
        Use SLEEP_BETWEEN_FRAMES field to show the video fast/slow

        :param input_frame: frame to annotate
        :param ok: is tracking succeeded or not.
        :param fps: video measure
        :return: result frame for the video
        """
        column = 20
        row = 20
        # cv2.putText(input_frame, "# " + str(int(frame_number)), (column, row), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        #             Colors.GREEN, 2)
        row += 25

        if self.visualize_movie:  # Display result on live
            self.show("Tracking " + self.name, input_frame)
            cv2.waitKey(SLEEP_BETWEEN_FRAMES)
        return input_frame
