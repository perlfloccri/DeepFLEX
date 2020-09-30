# --------------------------------------------------------
# Multi-Epitope-Ligand Cartography (MELC) phase-contrast image based segmentation pipeline
#
#
# Written by Filip Mivalt
# --------------------------------------------------------.

import copy
import numpy as np

def get_diff(x):
    """ Returns the edge represetnation
    Abs value of Edge representation of the image
    the simpliest way, only sum of differences in both directions

    Returns
    :param x: gets image
    :return: edge representation
    """
    return np.abs(np.diff(x, axis=0))[:, :-1] + np.abs(np.diff(x, axis=1))[:-1, :]

def _crop_idx(idx, shape):
    """ Image idx cropper
    Takes index in given format [x1, x2, y1, y3] = dict and shape of the images
    Cares about not to have indexes bigger then size of the images or negative indexes
    Both images are supposed to have same size

    :param idx: index of current window in image
    :param shape: shape of whole image
    :return: fixed indexes with respect to size of the image
    """
    idx = copy.deepcopy(idx)

    if idx['x2'][0] < 0:
        diff = np.abs(idx['x2'][0])

        idx['x1'][0] = 0
        idx['x1'][1] = shape[1] - diff

        idx['x2'][0] = diff
        idx['x2'][1] = shape[1]

    if idx['x2'][1] > shape[1]:
        diff =  idx['x2'][1] - shape[1]

        idx['x2'][0] = 0
        idx['x2'][1] = shape[1] - diff

        idx['x1'][0] = diff
        idx['x1'][1] = shape[1]

    if idx['y2'][0] < 0:
        diff = np.abs(idx['y2'][0])

        idx['y1'][0] = 0
        idx['y1'][1] = shape[0] - diff

        idx['y2'][0] = diff
        idx['y2'][1] = shape[0]

    if idx['y2'][1] > shape[0]:
        diff =  idx['y2'][1] - shape[0]

        idx['y2'][0] = 0
        idx['y2'][1] = shape[0] - diff

        idx['y1'][0] = diff
        idx['y1'][1] = shape[0]

    return idx

def _move_crop(idx, direction):
    """ Image idx mover
        Takes idx as standardized dictionary {x1, x2, y1, y2}, where each part is pair of numbers giving the area of image
        And moves this indexes about 1 pixel in given direction
        Shifted reference is stationary and only image no 2 is shifted.

        :param idx: index of both images region
        :param direction: direction of the image shift

        :return: fixed indexes with respect to size of the image
        """
    if direction == 'right':
        idx['x2'] += 1
    elif direction == 'left':
        idx['x2'] += -1
    elif direction == 'top':
        idx['y2'] += -1
    elif direction == 'down':
        idx['y2'] += 1
    return idx

def _get_regions(ref, img, idx):
    """
    Output are regions given by indexes in idx for images ref and img
    :param ref:  reference image
    :param img: registered image
    :param idx: standardized idx dictionary
    :return: choosen areas of images
    """
    ref_reg = ref[idx['y1'][0]:idx['y1'][1], idx['x1'][0]:idx['x1'][1]]
    img_reg = img[idx['y2'][0]:idx['y2'][1], idx['x2'][0]:idx['x2'][1]]
    return ref_reg, img_reg

def _get_region_cov(ref, img, idx):
    """
    Computes covariance cost for registration btwn images ref and img on given index idx
    :param ref:
    :param img:
    :param idx:
    :return:
    """
    def get_cov(x1, x2):
        x1 = x1 - np.mean(x1)
        x2 = x2 - np.mean(x2)
        cov = np.mean(x1 * x2)
        return cov

    ref_reg, img_reg = _get_regions(ref, img, idx)
    return get_cov(ref_reg, img_reg)


def register_images(ref, img, window_size, window_position, search_area):
    """
    Returns registering coeficients [y, x] of the shift for IMG against ref.
    Output of this function serves as an input into function register_from_idx, it takes care of itself. For registeration of two images
    :param ref: reference image
    :param img: registered image
    :param window_size: tuple of window size from image from registration
    :param window_position: Centre position of window in an images
    :param search_area: tuple of arguments giving the size where the algorithm will search for right position
    :return:
    """
    if len(window_size) == 1:
        window_size = (window_size, window_size)
    if len(search_area) == 1:
        window_size = (search_area, search_area)

    idxX1 = np.array([window_position[1] - 0.5 * window_size[1], window_position[1] + 0.5 * window_size[1]])
    idxY1 = np.array([window_position[0] - 0.5 * window_size[0], window_position[0] + 0.5 * window_size[0]])

    idxX2 = idxX1 + int(search_area[1] / 2)
    idxY2 = idxY1 + int(search_area [0]/ 2)

    idxX1 = idxX1.astype(np.int32)
    idxY1 = idxY1.astype(np.int32)
    idxX2 = idxX2.astype(np.int32)
    idxY2 = idxY2.astype(np.int32)

    idx = {'x1': idxX1, 'y1': idxY1, 'x2': idxX2, 'y2': idxY2}
    heat_map = np.zeros(ref.shape)

    losses = np.zeros((search_area[0]*search_area[1], 5))
    cntr = 0

    def do_loop(ref, img, idx, losses, heat_map, cntr):
        croped_index = _crop_idx(idx, ref.shape)
        cov_loss = _get_region_cov(ref, img, croped_index)

        losses[cntr, 0] = cov_loss
        losses[cntr, 1] = idx['x1'][0] - idx['x2'][0]
        losses[cntr, 2] = idx['x1'][1] - idx['x2'][1]
        losses[cntr, 3] = idx['y1'][0] - idx['y2'][0]
        losses[cntr, 4] = idx['y1'][1] - idx['y2'][1]
        heat_map[np.mean(idx['y2']).astype(np.uint16), np.mean(idx['x2']).astype(np.uint16)] = cov_loss
        return losses, heat_map




    for k in range(search_area[0]):
        if k > 0: idx = _move_crop(idx, 'top')

        if k % 2 == 1:
            for k2 in range(search_area[1]):
                idx = _move_crop(idx, 'right')
                losses, heat_map = do_loop(ref, img, idx, losses, heat_map, cntr)
                cntr += 1
        else:
            for k2 in range(search_area[1]):
                idx = _move_crop(idx, 'left')
                losses, heat_map = do_loop(ref, img, idx, losses, heat_map, cntr)
                cntr += 1

    max_idx = np.where(losses[:, 0] == np.max(losses[:, 0]))[0][0]
    registered_idx = losses[max_idx, 1:]

    y1 = int(window_position[0] - search_area[0] / 2)
    y2 = int(window_position[0] + search_area[0] / 2)
    x1 = int(window_position[1] - search_area[1] / 2)
    x2 = int(window_position[1] + search_area[1] / 2)

    return np.flip(registered_idx[1:-1]), heat_map[y1:y2, x1:x2]

def register_from_idx(ref, img, reg_idx):
    """
    Takes reference image, registered image and returns croped and registered image using reg_idx what is tuple or array
    of 2 shifts [y, x] direction of image

    :param ref:
    :param img:
    :param reg_idx:
    :return:
    """
    idxX1 = np.array([0, ref.shape[1]])
    idxY1 = np.array([0, ref.shape[0]])

    idxX2 = np.array([0, img.shape[1]])
    idxY2 = np.array([0, img.shape[0]])
    idx = {'x1': idxX1, 'y1': idxY1, 'x2': idxX2, 'y2': idxY2}

    idx['x2'][0] = idx['x2'][0] + reg_idx[1]
    idx['x2'][1] = idx['x2'][1] + reg_idx[1]
    idx['y2'][0] = idx['y2'][0] + reg_idx[0]
    idx['y2'][1] = idx['y2'][1] + reg_idx[0]
    print(idx)
    croped_index = _crop_idx(idx, ref.shape)
    print(croped_index)
    ref_regis, img_regis = _get_regions(ref, img, croped_index)
    return ref_regis, img_regis







