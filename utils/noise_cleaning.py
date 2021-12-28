import logging

import cv2
import os
import numpy as np

from classic_cv_trackers.abstract_and_common_trackers import Colors

# todo can read from txt?
FRAME_ROWS = 896
FRAME_COLS = 900


def get_plate(gray, threshold=0.01, min_area=100000, visualize_movie=False):  # full plate ~500k, fish ~4k
    # note: can use threshold to reduce widening the plate, but it causes half circle shapes, req. rewrite code below
    contours0, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hulls = [(cv2.convexHull(cnt), cv2.contourArea(cnt), cnt) for cnt in contours0]
    hulls = [h for h in hulls if h[1] >= min_area]
    hulls = sorted(hulls, key=lambda r: -r[1])  # external and internal of plate

    if visualize_movie:
        r = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(r,contours0, -1, Colors.GREEN, cv2.FILLED)
        cv2.drawContours(r, [cnt for cnt in contours0 if cv2.contourArea(cnt) > min_area/2], -1, Colors.CYAN, cv2.FILLED)
        cv2.drawContours(r, [h[0] for h in hulls], -1, Colors.RED)
        cv2.imshow("Error", resize(r))
        cv2.waitKey(120)

    if len(hulls) >= 1:
        answers = []
        for plate, area, plate_contour in hulls:
            if len(plate) >= 5:  # fitEllipse req this
                ellipse = cv2.fitEllipse(plate)
                poly_ellipse = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2),
                                                                                           int(ellipse[1][1] / 2)),
                                                int(ellipse[2]), 0, 360, 5)
                ans = cv2.matchShapes(poly_ellipse, plate, 1, 0.0)  # lower = better match
                if ans <= threshold:
                    answers.append((ans, plate, area, plate_contour, poly_ellipse))
            # new videos - allow external image frame with inner ellipse
            elif len(plate) == 4 and [0, 0] in plate and [gray.shape[1] - 1, gray.shape[0] - 1] in plate:
                answers.append((1, plate, area, plate_contour, plate))
        answers = sorted(answers, key=lambda r: -r[2])  # external and internal of plate are ordered by area
        if len(answers) >= 2:
            return answers[0], answers[1]
        elif len(answers) >= 1:
            return answers[0], None
    return None


def resize(f):
    return cv2.resize(f, (round(f.shape[0] / 1.5), round(f.shape[1] / 1.5)))


def threshold_to_emphasize_plate(gray, block_size=21, open_kernel_size=(5, 5)):
    # Use binarization with assigning 255 to threshold, k=-0.3
    gray = cv2.ximgproc.niBlackThreshold(gray, 255, cv2.THRESH_BINARY, block_size, -0.3,
                                         binarizationMethod=cv2.ximgproc.BINARIZATION_NICK)

    # remove small white noise in result image
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size))
    return gray


def clean_plate(input_frame, threshold=0.05, additional_thickness_remove=2):
    gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY).astype(input_frame.dtype)
    black_n_white = threshold_to_emphasize_plate(gray)
    # cv2.imshow('black_n_white', resize(black_n_white))

    results = get_plate(black_n_white, threshold=threshold)
    if results is None:
        results = get_plate(gray, threshold=threshold)
        if results is None:
            results = get_plate(cv2.equalizeHist(gray), threshold=threshold)
            if results is None:
                logging.info("Didnt find plate")
                return None

    ((ans_o, plate_o, area_o, plate_contour_o, poly_ellipse_o), r2) = results

    if r2 is None:  # only outer plate - search inner using these results
        # use contour to search inner hull - this is done to remove artifact when fish is near the plate and
        # plate_contour_o is connected to the fish (the inner hull will catch only ellipse shape)
        mask = np.zeros(gray.shape[:2], np.uint8)
        cv2.drawContours(mask, [poly_ellipse_o], -1, Colors.WHITE, thickness=cv2.FILLED)
        cv2.drawContours(mask, [plate_contour_o], -1, Colors.BLACK, thickness=cv2.FILLED)
        cv2.drawContours(mask, [poly_ellipse_o], -1, Colors.WHITE, thickness=additional_thickness_remove)
        results2 = get_plate(mask, threshold=0.2)
        if results2 is not None and results2[1] is None:  # found inner only
            (r2, _) = results2
        elif results2 is not None:  # take most inner
            (_, r2) = results2

    mask = np.zeros(gray.shape[:2], np.uint8)
    if r2 is not None:
        (ans_i, plate_i, area_i, plate_contour_i, poly_ellipse_i) = r2
        cv2.drawContours(mask, [plate_i], -1, Colors.WHITE, thickness=cv2.FILLED)
        cv2.drawContours(mask, [plate_i], -1, Colors.BLACK, thickness=additional_thickness_remove)

    clean = cv2.bitwise_and(input_frame, input_frame, mask=mask).astype(input_frame.dtype)

    # cv2.imshow('mask o-i', resize(mask))
    # res = input_frame.copy()
    # if r2 is None:  # only outer plate
    #     cv2.drawContours(res, [plate_o], -1, Colors.CYAN)
    #     cv2.drawContours(res, [plate_contour_o], -1, Colors.BLUE)
    # else:
    #     cv2.drawContours(res, [plate_i], -1, Colors.RED, thickness=additional_thickness_remove)
    #     cv2.drawContours(res, [plate_i], -1, Colors.PINK)
    #     cv2.drawContours(res, [plate_o], -1, Colors.CYAN)
    #     cv2.drawContours(res, [plate_contour_o], -1, Colors.BLUE)
    # cv2.imshow('result of circle detect', resize(res))
    # cv2.waitKey(120)

    if r2 is not None:
        return clean
    return None  # didn't find plate
