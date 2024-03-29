import cv2
import numpy as np

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)  # 21 is block size of neighbours


def calc_disparity_map(gl_img, gr_img):
    disparity = stereoProcessor.compute(gl_img, gr_img)

    dispNoiseFilter = 8  # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

    _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity = (disparity / 16.)

    return disparity


def take_subarray(arr, x, y, w, h):
    arr_h, arr_w = arr.shape

    return arr[max(y, 0):min(y + h + 1, arr_h), max(x, 0):min(x + w + 1, arr_w)]


def non_zero_mean(disparities):
    return disparities.sum() / np.count_nonzero(disparities)


def kth_nonzero_percentile(disparities, k):
    return np.percentile(disparities[disparities != 0], k)


def calc_depth(disparity, box):
    feature_disparities = take_subarray(disparity, *box)

    if not np.any(feature_disparities):
        return 100_000_000  # too big for any featur

    feature_disparity = kth_nonzero_percentile(feature_disparities, 80)

    focal_length = 399.9745178222656
    baseline_dist = 0.2090607502
    return (focal_length * baseline_dist) / feature_disparity
