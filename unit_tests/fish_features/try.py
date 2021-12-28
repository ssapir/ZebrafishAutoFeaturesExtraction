import math
import numpy as np
from scipy.spatial import distance

from classic_cv_trackers.abstract_and_common_trackers import Colors
from feature_analysis.fish_environment.fish_processed_data import rotate_data, ParameciumRelativeToFish, \
    pixels_mm_converters

FRAME_COLS = 900
FRAME_ROWS = 896


def point_to_int(point):
    return round(point[0]), round(point[1])


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
    _velocity_towards_fish = distance.euclidean([0, prev_center[1]], [0, center_shifted[1]]) #_distance_from_fish_in_mm[1] - _distance_from_fish_in_mm[0]
    _velocity_norm *= to_mm_per_seconds
    _velocity_towards_fish *= per_seconds
    _velocity_orthogonal = math.sqrt(abs(_velocity_norm ** 2 - _velocity_towards_fish ** 2))
    _velocity_direction = ParameciumRelativeToFish.velocity2angle(_velocity_towards_fish, _velocity_orthogonal)
    return _velocity_norm, _velocity_direction, _velocity_towards_fish, _velocity_orthogonal, _distance_from_fish_in_mm


def get_fig_center(h = FRAME_COLS, w = FRAME_ROWS):
    (cX, cY) = (w // 2, h // 2)
    return cX, cY


if __name__ == '__main__':
    head_origin_point = get_fig_center()
    head_direction_angle = 45  # deg

    # point = np.array([head_origin_point[0] + 42 * 3, head_origin_point[1]])  # 1mm distance
    # prev_point = np.array([head_origin_point[0] + 42 * 4, head_origin_point[1]])  # 2mm distance
    point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 3])  # 1mm distance
    prev_point = np.array([head_origin_point[0], head_origin_point[1] - 42 * 4])  # 2mm distance
    print(head_origin_point, point, prev_point)

    center_shifted = rotate_data(data=point, head_point=point_to_int(head_origin_point),
                                 direction=head_direction_angle)
    prev_center = rotate_data(data=prev_point, direction=head_direction_angle,
                              head_point=point_to_int(head_origin_point))

    import cv2
    img = np.zeros((FRAME_COLS, FRAME_ROWS, 3))
    cv2.circle(img, point_to_int(head_origin_point), radius=3, color=Colors.WHITE)
    cv2.circle(img, point_to_int(point), radius=3,  color=Colors.RED, thickness=cv2.FILLED)
    cv2.circle(img, point_to_int(prev_point), radius=3,  color=Colors.CYAN, thickness=cv2.FILLED)
    cv2.circle(img, point_to_int(center_shifted), radius=6,  color=Colors.RED)
    cv2.circle(img, point_to_int(prev_center), radius=6,  color=Colors.CYAN)
    cv2.imshow("f", img), cv2.waitKey(10)
    import matplotlib.pyplot as plt
    plt.imshow(img), plt.show()