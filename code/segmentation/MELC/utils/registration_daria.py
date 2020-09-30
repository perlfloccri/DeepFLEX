from __future__ import print_function
import cv2
import numpy as np
from skimage.feature import register_translation


def register(ref_img: np.ndarray, phase_img: np.ndarray, fluor_img: np.ndarray):
    '''
    :param ref_img: reference image
    :param phase_img: phase contrast image of antibody image to be aligned
    :param fluor_img: fluorescence image to be aligned
    :return: registered fluorescence image
    '''

    # Calculate shift between ref and phase img
    # subpixel precision - 3rd parameter is factor that one pixel is devided by
    shift, error, diffphase = register_translation(ref_img, phase_img, 100)

    # shift ab image in x and y by calculated shift
    rows, cols = fluor_img.shape

    M = np.float32([[1, 0, round(shift[1])], [0, 1, round(shift[0])]])
    aligned_img = cv2.warpAffine(fluor_img, M, (cols, rows))

    return aligned_img

