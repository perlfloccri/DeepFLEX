import os
import numpy as np
import cv2
from scipy.io import loadmat
from RoI_selection import RoI
import pandas as pd
import pickle
import sys
import argparse
import matplotlib.pyplot as plt


def get_parser():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Select regions of interest for single-cell analysis. User-guided elimination of artifacts and inhomogeneously illuminated areas.')

    parser.add_argument(
        '--path', dest='path', required=True,
        help='Path to folder of one sample.')

    #parser.add_argument('--roi_generated', dest='roi_generated', action='store_true', help='regions for cells to be considered in analysis already generated')
    parser.add_argument('--path_preselected_roi', help='If regions were preselected, pass path to rois.', default=None)
    return parser


def get_FID(x):
    temp = x['path'].split('/')
    _temp = temp[len(temp) - 1].split('.')[0].split('_')
    return _temp[0] + '_' + _temp[1]


def print_contours(cont):

    img = np.zeros((2018, 2018), dtype=np.uint8)

    masks = np.zeros_like(img, dtype=np.uint16)
    cntr256 = 0
    cntr_laps = 0

    temp_masks = np.zeros_like(img, dtype=np.uint8)
    print (cont)
    for contour in cont:
        contour = convert_contours_to_list(contour)
        cntr256 += 1
        temp_masks = cv2.drawContours(temp_masks, [contour], -1, cntr256, -1)

        if cntr256 == 255 or np.array_equal(contour, cont[cont.__len__() - 1]):
            temp_masks = temp_masks.astype(np.uint16)
            temp_masks[temp_masks > 0] = temp_masks[temp_masks > 0] + cntr_laps * 255
            cntr256 = 0
            cntr_laps += 1

            masks[temp_masks > 0] = temp_masks[temp_masks > 0]
            temp_masks = np.zeros_like(img, dtype=np.uint8)

    return masks

def convert_contours_to_list(cont):
    cont_list = []
    for cont_pair in cont[0]:
        cont_list.append(cont_pair.astype(np.uint8))
        print (cont_pair)
    return cont

class ROI_masks:

    def __init__(self, path: str, path_2_roi: str=''):

        self._path = path
        self._path_2_roi = path_2_roi

        if not os.path.exists(self._path):
            raise ValueError("Directory: " + self._path + " does not exist!")

        files = [os.path.join(root, name)
                 for root, dirs, files in os.walk(self._path + '/cut/fluor_cidre/')
                 for name in files
                 if name.endswith(('tif', 'TIF'))]

        paths_pd = pd.DataFrame(files)
        paths_pd = paths_pd.rename(columns={0: "path"})
        paths_ar = np.asarray(paths_pd)
        self._paths_ar = [_f[0] for _f in paths_ar if 'PBS' not in _f[0] and 'NONE' not in _f[0]]

        filenames = paths_pd.apply(lambda x: get_FID(x), axis=1)
        filenames_ar = np.asarray(filenames)
        self._filenames_ar = [_f for _f in filenames_ar if 'PBS' not in _f and 'NONE' not in _f]

    def generate_masks(self):

        fluor_images = []

        for i in range(0, len(self._paths_ar)):
            fluor_images.append(cv2.imread(self._paths_ar[i], cv2.IMREAD_UNCHANGED))

        predictions = loadmat(self._path + '/PREDICTIONS')#['predictions'][0]
        cell_mask = predictions['cells']
        nuclei_mask = predictions['nuclei']
        if self._path_2_roi == '':
            r = RoI(nuclei_mask, cell_mask, fluor_images, self._filenames_ar)
            roi = r.ROIselection()
        else:
            with open(self._path_2_roi, 'rb') as f:
                roi = pickle.load(f)
        # To remove cells as a whole instead of cutting them
        for i in range(0, nuclei_mask.max() + 1):
            n_coordinates = np.where(nuclei_mask == (i+1))
            c_coordinates = np.where(cell_mask == (i + 1))
            if 0 in roi[n_coordinates]:
                nuclei_mask[n_coordinates] = 0
                cell_mask[c_coordinates] = 0
            if 0 in roi[c_coordinates]:
                nuclei_mask[n_coordinates] = 0
                cell_mask[c_coordinates] = 0

        with open(self._path + '/nuclei_mask.pickle', 'wb') as f:
            pickle.dump(nuclei_mask, f)

        plt.imsave(self._path + '/nuclei_mask.png', nuclei_mask)

        with open(self._path + '/cell_mask.pickle', 'wb') as f:
            pickle.dump(cell_mask, f)

        plt.imsave(self._path + '/cell_mask.png', cell_mask)


def main(args):
    path = args.path
    print("Path to sample folder: " + path)
    roi_path = args.path_preselected_roi

    samples = [s for s in os.listdir(path) if [char for char in s][0] == 'B']
    for s in samples:
        fov_folders = os.listdir(path + '/' + s)

        for _p in fov_folders:
            if not roi_path:
                roi_mask = ''
            else:
                roi_mask = roi_path + '/' + s + '_' + _p.split('_')[0] + '.pickle'
                print (roi_mask)
            r = ROI_masks(path=path + '/' + s + '/' + _p + '/processed', path_2_roi=roi_mask)
            r.generate_masks()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

