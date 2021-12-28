import math
import unittest
import numpy as np

from scipy.spatial import distance

from feature_analysis.fish_environment.fish_processed_data import rotate_data, ParameciumRelativeToFish, \
    pixels_mm_converters
from utils.video_utils import VideoFromRaw

FRAME_COLS = 900
FRAME_ROWS = 896


def point_to_int(point):
    return round(point[0]), round(point[1])


def frames_to_secs_converter(n_curr_frames):
    return n_curr_frames / float(VideoFromRaw.FPS)


def calc(point, prev_point, head_direction_angle, head_origin_point, ibi_frames):
    one_mm_in_pixels, _ = pixels_mm_converters()
    to_mm_per_seconds = 1.0 / (one_mm_in_pixels * frames_to_secs_converter(ibi_frames))
    per_seconds = 1.0 / (frames_to_secs_converter(ibi_frames))
    center_shifted = rotate_data(data=point, head_point=point_to_int(head_origin_point),
                                 direction=head_direction_angle)
    prev_center = rotate_data(data=prev_point, direction=head_direction_angle,
                              head_point=point_to_int(head_origin_point))

    _distance_from_fish_in_mm = [np.nan, np.nan]
    _distance_from_fish_in_mm[0] = distance.euclidean(prev_point, head_origin_point) / one_mm_in_pixels
    _distance_from_fish_in_mm[1] = distance.euclidean(point, head_origin_point) / one_mm_in_pixels

    _velocity_norm = distance.euclidean(prev_center, center_shifted)
    _velocity_towards_fish = prev_center[1] - center_shifted[1]
    _velocity_norm *= to_mm_per_seconds
    _velocity_towards_fish *= to_mm_per_seconds
    _velocity_orthogonal = math.sqrt(abs(_velocity_norm ** 2 - _velocity_towards_fish ** 2))
    _velocity_direction = ParameciumRelativeToFish.velocity2angle(_velocity_towards_fish, _velocity_orthogonal)
    return _velocity_norm, _velocity_direction, _velocity_towards_fish, _velocity_orthogonal, _distance_from_fish_in_mm


def get_fig_center(h = FRAME_COLS, w = FRAME_ROWS):
    (cX, cY) = (w // 2, h // 2)
    return cX, cY


class MyTestCase(unittest.TestCase):
    def test_pixel_mm_converter(self):
        img = np.zeros((FRAME_COLS, FRAME_ROWS))
        one_mm_in_pixels_v2 = int((img.shape[1] - 33) / 40)  # magic numbers are plate's edges
        one_pixel_in_mm_v2 = 1 / one_mm_in_pixels_v2

        one_mm_in_pixels, one_pixel_in_mm = pixels_mm_converters()

        self.assertEqual(one_mm_in_pixels_v2, one_mm_in_pixels / 2)
        self.assertEqual(one_pixel_in_mm_v2, one_pixel_in_mm * 2)

    def test_sec_converter(self):
        self.assertEqual(1, 1.0 / (frames_to_secs_converter(500)))
        self.assertEqual(5, 1.0 / (frames_to_secs_converter(100)))

    def test_simple_fish_in_center_move_to_fish(self):
        head_origin_point = get_fig_center()
        head_direction_angle = 90  # deg
        point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 3])  # 1mm distance
        prev_point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 4])  # 2mm distance

        ibi_frames = 500  # 1 sec

        _velocity_norm, _velocity_direction, _velocity_towards_fish, _velocity_orthogonal, _distance_from_fish_in_mm = \
            calc(point, prev_point, head_direction_angle, head_origin_point, ibi_frames)

        self.assertEqual(_distance_from_fish_in_mm[0], 4)
        self.assertEqual(_distance_from_fish_in_mm[1], 3)
        self.assertEqual(_velocity_norm, 1)
        self.assertEqual(_velocity_towards_fish, -1)
        self.assertEqual(_velocity_orthogonal, 0)
        self.assertEqual(_velocity_direction, 180)

    def test_simple_fish_in_center_faster_move_to_fish(self):
        head_origin_point = get_fig_center()
        head_direction_angle = 90  # deg
        point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 3])  # 1mm distance
        prev_point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 4])  # 2mm distance
        ibi_frames = 100  # 1 sec

        _velocity_norm, _velocity_direction, _velocity_towards_fish, _velocity_orthogonal, _distance_from_fish_in_mm = \
            calc(point, prev_point, head_direction_angle, head_origin_point, ibi_frames)

        self.assertEqual(_distance_from_fish_in_mm[0], 4)
        self.assertEqual(_distance_from_fish_in_mm[1], 3)
        self.assertEqual(_velocity_norm, 5)
        self.assertEqual(_velocity_towards_fish, -5)
        self.assertEqual(_velocity_orthogonal, 0)
        self.assertEqual(_velocity_direction, 180)

    def test_simple_fish_in_center_move_from_fish(self):
        head_origin_point = get_fig_center()
        head_direction_angle = 90  # deg
        point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 5])  # 1mm distance
        prev_point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 4])  # 2mm distance

        ibi_frames = 500  # 1 sec

        _velocity_norm, _velocity_direction, _velocity_towards_fish, _velocity_orthogonal, _distance_from_fish_in_mm = \
            calc(point, prev_point, head_direction_angle, head_origin_point, ibi_frames)

        self.assertEqual(_distance_from_fish_in_mm[0], 4)
        self.assertEqual(_distance_from_fish_in_mm[1], 5)
        self.assertAlmostEqual(_velocity_norm, 1)
        self.assertEqual(_velocity_direction, 0)
        self.assertEqual(_velocity_towards_fish, 1)
        self.assertEqual(_velocity_orthogonal, 0)

    def test_simple_fish_in_center_move_orthogonal_fish(self):
        head_origin_point = get_fig_center()
        head_direction_angle = 90  # deg
        point = np.array([head_origin_point[0] + 42 * 3, head_origin_point[1]])  # 1mm distance
        prev_point = np.array([head_origin_point[0] + 42 * 4, head_origin_point[1]])  # 2mm distance

        ibi_frames = 500  # 1 sec

        _velocity_norm, _velocity_direction, _velocity_towards_fish, _velocity_orthogonal, _distance_from_fish_in_mm = \
            calc(point, prev_point, head_direction_angle, head_origin_point, ibi_frames)

        self.assertEqual(_distance_from_fish_in_mm[0], 4)
        self.assertEqual(_distance_from_fish_in_mm[1], 3)
        self.assertEqual(_velocity_norm, 1)
        self.assertEqual(_velocity_towards_fish, 0)
        self.assertEqual(_velocity_orthogonal, 1)
        self.assertEqual(_velocity_direction, 90)

    def test_simple_fish_not_in_center_move_from_fish(self):
        head_origin_point = np.array(get_fig_center()) + [5, 5]
        head_direction_angle = 90  # deg
        point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 5])  # 1mm distance
        prev_point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 4])  # 2mm distance

        ibi_frames = 500  # 1 sec

        _velocity_norm, _velocity_direction, _velocity_towards_fish, _velocity_orthogonal, _distance_from_fish_in_mm = \
            calc(point, prev_point, head_direction_angle, head_origin_point, ibi_frames)

        self.assertEqual(_distance_from_fish_in_mm[0], 4)
        self.assertEqual(_distance_from_fish_in_mm[1], 5)
        self.assertAlmostEqual(_velocity_norm, 1)
        self.assertEqual(_velocity_direction, 0)
        self.assertEqual(_velocity_towards_fish, 1)
        self.assertEqual(_velocity_orthogonal, 0)

    def test_simple_fish_rotation_move_from_fish(self):
        head_origin_point = get_fig_center()
        head_direction_angle = 45  # deg
        point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 5])  # 1mm distance
        prev_point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 4])  # 2mm distance

        ibi_frames = 500  # 1 sec

        _velocity_norm, _velocity_direction, _velocity_towards_fish, _velocity_orthogonal, _distance_from_fish_in_mm = \
            calc(point, prev_point, head_direction_angle, head_origin_point, ibi_frames)

        self.assertEqual(_distance_from_fish_in_mm[0], 4)
        self.assertEqual(_distance_from_fish_in_mm[1], 5)
        self.assertAlmostEqual(_velocity_norm, 1)
        self.assertAlmostEqual(_velocity_direction, 45)
        self.assertAlmostEqual(_velocity_towards_fish, 0.70710678)
        self.assertAlmostEqual(_velocity_orthogonal, 0.70710678)


if __name__ == '__main__':
    unittest.main()