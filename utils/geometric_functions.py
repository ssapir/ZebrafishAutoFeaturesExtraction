import math
import numpy as np


def get_angle_to_horizontal(p_from, p_to):
    """Get two points, and calculate (rounded) angle in degrees relative to horizontal (x axis),
    counter-clockwise

    :param p_from: [x, y] origin - list/tuple/Point class
    :param p_to: [x, y] destination - list/tuple/Point class
    :return: angle in degrees - float.
    """
    if np.isnan(p_to).any() or np.isnan(p_from).any():
        angle_in_degrees = 0
    else:
        delta_y = np.round(float(p_to[1]) - float(p_from[1]))
        delta_x = np.round(float(p_to[0]) - float(p_from[0]))
        angle_in_degrees = math.degrees(math.atan2(delta_y, delta_x))

    angle_in_degrees = fix_angle_range(angle_in_degrees)
    return 360 - angle_in_degrees


def fix_angle_range(angle_in_degrees):
    """Return angle in degrees, between 0-360 (fix out of range)

    :param angle_in_degrees:
    :return:
    """
    # fix angle to be in range 0-360
    if angle_in_degrees < 0:
        angle_in_degrees += 360 * (math.ceil(abs(angle_in_degrees) / 360))
    elif angle_in_degrees > 360:
        angle_in_degrees = angle_in_degrees % 360
    return angle_in_degrees
