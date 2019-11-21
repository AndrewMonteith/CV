"""
    This file currently does not work.
"""

import cv2
import numpy as np

# COULD_TEST: CHANGING VALUE OF NEIGHBOURHOOD SIZE 21
# COULD_TEST: CHANGING VALUE OF MAX DISPARITY
# COULD_COMMENT: WHICH DISPARITY VALUE TO USE FOR CALCULATING DISTANCE

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21) # 21 is block size of neighbours

def take_subarray(arr, x, y, w, h):
    arr_h, arr_w = arr.shape

    return arr[max(y, 0):min(y+h+1, arr_h), max(x, 0):min(x+w+1, arr_w)]

def calculate_distance(left_image, right_image, box):
    grayL = cv2.cvtColor(left_image,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_image,cv2.COLOR_BGR2GRAY)

    grayL = np.power(grayL, 0.75).astype('uint8')
    grayR = np.power(grayR, 0.75).astype('uint8')

    disparity = stereoProcessor.compute(grayL,grayR)

    _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity = (disparity / 16.)

    feature_disparities = take_subarray(disparity, *box)

    if not np.any(feature_disparities):
        return -1

    # Mean of non-zero disparities. Need to think about improving this?
    feature_disparity = feature_disparities.sum() / np.count_nonzero(feature_disparities)

    focal_length = 399.9745178222656
    baseline_dist = 0.2090607502
    return (focal_length * baseline_dist) / feature_disparity

