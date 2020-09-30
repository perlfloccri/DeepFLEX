import os
import numpy as np
import cv2
from typing import Union
from scipy.io import loadmat
from image import Image
from multispectral_object import MSObject
#from b_Segmentation.RoI_selection import RoI
import pandas as pd
import pickle
import argparse


def get_parser():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Generate multi-channel single-cell objects')

    parser.add_argument(
        '--path', dest='path', required=True,
        help='Path to folder of one sample')

    return parser


def get_FID(x):
    temp = x['path'].split('/')
    _temp = temp[len(temp) - 1].split('.')[0].split('_')
    return _temp[0] + '_' + _temp[1]


def read_data(pan: list, folder: str) -> Union[list, np.ndarray]:
    fluor_images = []

    for i in range(0, len(pan)):
        fluor_images.append(cv2.imread(pan[i], cv2.IMREAD_UNCHANGED))

    with open(folder + '/nuclei_mask.pickle', 'rb') as f:
        nuclei_mask = pickle.load(f)

    with open(folder + '/cell_mask.pickle', 'rb') as f:
        cell_mask = pickle.load(f)

    return fluor_images, cell_mask, nuclei_mask


def identify_largest_cell(mask: np.ndarray) -> list:

    # calculate biggest width and height
    max_y = 0
    max_x = 0
    max_values = []

    for j in range(1, mask.max() + 1):
        cell_coordinates = np.where(mask == j)
        if len(cell_coordinates[0]) <= 1:
            continue
        else:
            current_max_y = cell_coordinates[0].max() - cell_coordinates[0].min()
            current_max_x = cell_coordinates[1].max() - cell_coordinates[1].min()
            if current_max_y > max_y:
                max_y = current_max_y

            if current_max_x > max_x:
                max_x = current_max_x

    max_values.append(max_y)
    max_values.append(max_x)

    return max_values


def cut_img(mask: np.ndarray, index: int, dim: list, raw: np.ndarray = None, is_raw: bool = False) -> np.ndarray:
    one_mask = mask * (mask == index)
    coordinates = np.where(mask == index)

    min_y = coordinates[0].min()
    max_y = coordinates[0].max()
    min_x = coordinates[1].min()
    max_x = coordinates[1].max()

    new_img = np.zeros((dim[0] + 7, dim[1] + 7), np.uint16)
    mask_cut = one_mask[min_y:max_y, min_x:max_x]
    lower_y = int(new_img.shape[0] / 2 - mask_cut.shape[0] / 2)
    upper_y = lower_y + mask_cut.shape[0]
    lower_x = int(new_img.shape[1] / 2 - mask_cut.shape[1] / 2)
    upper_x = lower_x + mask_cut.shape[1]

    if is_raw:
        one_raw = raw * (one_mask != 0)
        raw_cut = one_raw[min_y:max_y, min_x:max_x]
        raw_new = np.zeros((dim[0] + 7, dim[1] + 7), np.uint16)
        raw_new[lower_y:upper_y, lower_x:upper_x] = raw_cut

        img = raw_new

    else:
        mask_new = np.zeros((dim[0] + 7, dim[1] + 7), np.uint16)
        mask_new[lower_y:upper_y, lower_x:upper_x] = mask_cut
        co = np.where(mask_new != 0)
        mask_new[co] = 255

        img = mask_new

    return img, min_y, max_y, min_x, max_x


def generate_image_objects(panel: list, dataset: list, cell_mask: np.ndarray, nuclei_mask: np.ndarray, fov: int, pat_id: str) -> list:

    dim_largest_cell = identify_largest_cell(cell_mask)
    cells = []

    for j in range(1, nuclei_mask.max()+1):
        n_coordinates = np.where(nuclei_mask == j)
        c_coordinates = np.where(cell_mask == j)
        if len(np.unique(c_coordinates[0])) <= 2 or len(np.unique(n_coordinates[0])) <= 2 or len(np.unique(c_coordinates[1])) <= 2 or len(np.unique(n_coordinates[1])) <= 2:
            continue
        else:
            try:
                c_mask_img, y_min, y_max, x_min, x_max = cut_img(cell_mask, j, dim_largest_cell)
                n_mask_img, y_min, y_max, x_min, x_max = cut_img(nuclei_mask, j, dim_largest_cell)
                mask_tensor = np.zeros((dim_largest_cell[0] + 7, dim_largest_cell[1] + 7, 3), np.uint8)
                mask_tensor[:, :, 0] = n_mask_img
                mask_tensor[:, :, 1] = c_mask_img
                m_mask_img = np.copy(c_mask_img) # create membrane mask by cell_mask - nucleus_mask
                m_mask_img[np.where(n_mask_img != 0)] = 0
                mask_tensor[:, :, 2] = m_mask_img

                images = []

                for i in range(0, len(dataset)):

                    if 'Propidium' in panel[i]:

                        fl_img, y_min, y_max, x_min, x_max = cut_img(nuclei_mask, j, dim_largest_cell, dataset[i], True)
                        img = Image(fl_img, mask_tensor, panel[i], False, True) # change Image structure!!
                        images.append(img)

                    else:

                        fl_img, y_min, y_max, x_min, x_max = cut_img(cell_mask, j, dim_largest_cell, dataset[i], True)
                        img = Image(fl_img, mask_tensor, panel[i], True, True)

                        images.append(img)

                # Set label to zero
                cells.append(MSObject(images, pat_id=pat_id, idx_obj=j, label=0, idx_img=fov, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max))

                if j % 400 == 0:
                    print('Loaded object ' + str(j))
            except:
                print("Unequal mask sizes") 

    return cells


class FoV_MSObjects:

    def __init__(self, fov: int, pat_id: str, path: str):

        self._path = path
        self._fov = fov
        self._pat_id = pat_id

    def generate_dataset(self):

        f_dir = self._path

        if not os.path.exists(f_dir):
            raise ValueError("Directory: " + f_dir + " does not exist!")

        files = [os.path.join(root, name)
                for root, dirs, files in os.walk(f_dir + '/cut/fluor_cidre/')
                for name in files
                if name.endswith(('tif', 'TIF'))]

        paths_pd = pd.DataFrame(files)
        paths_pd = paths_pd.rename(columns={0: "path"})
        paths_ar = np.asarray(paths_pd)
        paths_ar = [_f[0] for _f in paths_ar if 'PBS' not in _f[0] and 'NONE' not in _f[0]]

        filenames = paths_pd.apply(lambda x: get_FID(x), axis=1)
        filenames_ar = np.asarray(filenames)
        filenames_ar = [_f for _f in filenames_ar if 'PBS' not in _f and 'NONE' not in _f]

        fluor_images, cell_mask, nuclei_mask = read_data(paths_ar, f_dir)
        cell_dataset = generate_image_objects(filenames_ar, fluor_images, cell_mask, nuclei_mask, fov=self._fov, pat_id = self._pat_id)

        return cell_dataset


def main(args):
    path = args.path

    patient_id = path.split('/')[-1]

    fov_folders = os.listdir(path)
    for _p in fov_folders:
        if "_FoV" in _p:
            fov = int(_p.split('_')[0])
            g = FoV_MSObjects(fov=fov, path=path + '/' + _p + '/processed', pat_id=patient_id)
            single_cells = g.generate_dataset()

            with open(path + '/' + _p + '/processed' + '/dataset.pickle', 'wb') as f:
                pickle.dump(single_cells, f)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
