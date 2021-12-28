import numpy as np

from utils.geometric_functions import fix_angle_range


class EllipseData:
    # Ellipse params are ~original result of fitEllipse (major and minor are half (d1, d2))
    ellipse_center = None
    ellipse_major = None
    ellipse_minor = None
    ellipse_direction = None  # relative to y axis, 0-180 deg

    def __init__(self, ellipse_center, ellipse_major, ellipse_minor, ellipse_direction):
        self.ellipse_center = ellipse_center
        self.ellipse_major = ellipse_major
        self.ellipse_minor = ellipse_minor
        self.ellipse_direction = ellipse_direction

    @staticmethod
    def to_dict(ellipse_data_list):
        return {'ellipse_centers': [el.ellipse_center for el in ellipse_data_list],
                'ellipse_majors':  [el.ellipse_major for el in ellipse_data_list],
                'ellipse_minors':  [el.ellipse_minor for el in ellipse_data_list],
                'ellipse_dirs':    [el.ellipse_direction for el in ellipse_data_list]}


def absolute_angles_and_differences_in_deg(input_angles_list, main_direction_angle):
    """Fix angles to absolute values relative to horizontal & calculate the diff from a main direction angle (usually
    fish)

    :param input_angles_list:
    :param main_direction_angle: angle to compare difference from
    :return: abs_angle_deg, diff_from_main_dir_deg
    """
    diff_from_main_dir_deg = [fix_angle_range(abs(ang - main_direction_angle)) * np.sign(ang - main_direction_angle)
                              for ang in input_angles_list]  # fix after abs diff
    abs_angle_deg = [fix_angle_range(ang) for ang in input_angles_list]
    return abs_angle_deg, diff_from_main_dir_deg
