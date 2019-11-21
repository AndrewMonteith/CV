import cv2
import numpy as np

# CLAHE ~ Contrast Limited Adaptive Histogram Equalization
# Used to increase the contrast in the images so we can see more
# Pedestrians in the shadows and such whilst no extenuating
# the bright intensity regions
clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))

def prp_obj_detection_input(l_img):
    # input img is RGB, but equalization cannot be done on separate channels
    # so we switch to a colour space that is nicer
    ycrcb = cv2.cvtColor(l_img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    clahe.apply(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, l_img)
    return l_img

def prp_dist_calc_input(gl_img, gr_img):
    return np.power(gl_img, 0.75).astype('uint8'), np.power(gr_img, 0.75).astype('uint8')

