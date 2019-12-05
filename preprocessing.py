import cv2
import numpy as np

clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))


def prp_obj_detection_input(l_img, r_img):
    '''
        Preprocessing for object detection we apply CLAHE
        to both the left and right colour images
    '''

    # Need to convert to different colour space because normalization
    # cannot just be applied to each channel separately
    ycrcb = cv2.cvtColor(l_img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)

    clahe.apply(channels[0], channels[0])

    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, l_img)

    clahe.apply(r_img, r_img)

    return l_img


def prp_dist_calc_input(grey_l_img, r_img):
    '''
        Preprocess the images for the recovering distance step.
        We perform use laplacian filtering to sharpen the image
        We then use a pointwise operator (^0.75) to improve the quality of the disparity map
    '''

    laplace_l, laplace_r = cv2.Laplacian(grey_l_img, ddepth=cv2.CV_16S), cv2.Laplacian(r_img, ddepth=cv2.CV_16S)

    grey_l_img = grey_l_img - laplace_l
    r_img = r_img - laplace_r

    return np.power(grey_l_img.astype('uint8'), 0.75), np.power(r_img.astype('uint8'), 0.75)
