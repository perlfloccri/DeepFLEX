from MELC_Datasets import generate_workingRaw_from_raw, MELCStructureDataset
import numpy as np
import tifffile as tiff
from registration import register
from MELC_Files import create_folder
from skimage import img_as_float, img_as_uint
import glob
import os
import argparse


def get_parser():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Image processing prior to segmentation')

    parser.add_argument(
        '--path', dest='path', required=True,
        help='Path to raw data of one field of view')

    parser.add_argument(
        '--image_directory_generated', dest='image_directory_generated', action='store_true')

    return parser


class MELCImageProcessing:

    def __init__(self, path: str, melc_structure_generated: bool = True):

        self._path = path
        self._path_registered_fluor = ''
        self._path_registered_bleach = ''
        self._path_registered_phase = ''
        self._path_registered_vis_fluor = ''
        self._path_registered_vis_bleach = ''
        self._path_registered_vis_phase = ''
        self._path_bg_corr = ''
        self._path_bg_corr_f = ''
        self._path_bg_corr_v_f = ''
        self._path_cut_f = ''
        self._path_cut_v_f = ''

        w_raw = self._path + '/w_raw'
        if not melc_structure_generated:
            print ("No MELC structure")
            '''
            Load raw data into structured MELC image folder
            '''
            generate_workingRaw_from_raw(self._path, w_raw)
        else:
            print ("MELC structure")
        print ("Pfad:" + path)
        melc_dataset = MELCStructureDataset(w_raw)

        '''
        Sort by creation date
        '''
        self._melc_fluor = melc_dataset.fluor_pd.sort_values('order_index', ascending=True)
        self._melc_phase = melc_dataset.phase_pd.sort_values('order_index', ascending=True)
        self._melc_bleach = melc_dataset.bleach_pd.sort_values('order_index', ascending=True)
        self._melc_phasebleach = melc_dataset.phasebleach_pd.sort_values('order_index', ascending=True)

        self.create_folders()
        self._corrected_bf_im = self.generate_bg_correction_img()
        self.process_images()

    def create_folders(self):
        '''
        Create folders for registered images
        '''
        path_processed = self._path + '/processed'
        path_registered = path_processed + '/registered'
        self._path_registered_fluor = path_registered + '/fluor'
        self._path_registered_bleach = path_registered + '/bleach'
        self._path_registered_phase = path_registered + '/phase'
        self._path_registered_vis_fluor = path_registered + '/vis_fluor'
        self._path_registered_vis_bleach = path_registered + '/vis_bleach'
        self._path_registered_vis_phase = path_registered + '/vis_phase'

        create_folder(path_processed)
        create_folder(path_registered)
        create_folder(self._path_registered_fluor)
        create_folder(self._path_registered_bleach)
        create_folder(self._path_registered_phase)
        create_folder(self._path_registered_vis_fluor)
        create_folder(self._path_registered_vis_bleach)
        create_folder(self._path_registered_vis_phase)

        '''
        Create folders for background corrected images
        '''

        self._path_bg_corr = self._path + '/processed/background_corr/'
        self._path_bg_corr_f = self._path_bg_corr + 'fluor/'
        self._path_bg_corr_v_f = self._path_bg_corr + 'vis_fluor/'
        self._path_bg_corr_p = self._path_bg_corr + 'phase/'
        self._path_bg_corr_v_p = self._path_bg_corr + 'vis_phase/'

        create_folder(self._path_bg_corr)
        create_folder(self._path_bg_corr_f)
        create_folder(self._path_bg_corr_v_f)
        create_folder(self._path_bg_corr_p)
        create_folder(self._path_bg_corr_v_p)

        '''
        Create folders for cut images
        '''
        path_cut = self._path + '/processed/cut/'
        self._path_cut_f = path_cut + 'fluor/'
        self._path_cut_v_f = path_cut + 'vis_fluor/'
        self._path_cut_p = path_cut + 'phase/'
        self._path_cut_v_p = path_cut + 'vis_phase/'

        create_folder(path_cut)
        create_folder(self._path_cut_f)
        create_folder(self._path_cut_v_f)
        create_folder(self._path_cut_p)
        create_folder(self._path_cut_v_p)

    def generate_bg_correction_img(self):

        '''
        Create correction image for fluorescence and bleaching images to be used in prospective illumination correction
        '''

        brightfield_im = []
        darkframe_im = []
        filter_names = ['XF116-2', 'XF111-2']
        calibration_path = self._path + '/w_raw/calibration/'
        brightfield_im.append(np.int16(tiff.imread(glob.glob(calibration_path + '*_cal_b001_5000_XF116-2_000.tif'))))
        brightfield_im.append(np.int16(tiff.imread(glob.glob(calibration_path + '*_cal_b001_5000_XF111-2_000.tif'))))
        darkframe_im.append(np.int16(tiff.imread(glob.glob(calibration_path + '*_cal_d001_5000_XF116-2_000.tif'))))
        darkframe_im.append(np.int16(tiff.imread(glob.glob(calibration_path + '*_cal_d001_5000_XF111-2_000.tif'))))

        corrected_brightfield_im = [(brightfield_im[i] - darkframe_im[i]) for i in range(len(filter_names))]
        corrected_brightfield_im[0][corrected_brightfield_im[0] <= 0] = 0
        corrected_brightfield_im[1][corrected_brightfield_im[1] <= 0] = 0

        return corrected_brightfield_im

    def process_images(self):
        '''
        Registration, background correction and cutting of images
        '''


        '''
        Registration
        '''
        ref_image = tiff.imread(glob.glob(self._path + '/w_raw/phase/*_Propidium iodide_200_XF116*.tif'))
        for i in range(0, (len(self._melc_fluor)-1)):

            pb_idx = np.where(self._melc_phasebleach['order_index'] == self._melc_bleach.iloc[i]['order_index'])[0][0]
            phasebleach_image = tiff.imread(self._melc_phasebleach.iloc[pb_idx]['path'])
            bleach_image = tiff.imread(self._melc_bleach.iloc[i]['path'])
            registered_bleach_image = register(ref_image, phasebleach_image, bleach_image)
            filename_bleach = '/' + str(int(self._melc_bleach.iloc[i]['order_index'])) + '_' + '_'.join(
                self._melc_bleach.iloc[i]['fid'].split('_')[:-1]) + '.tif'
            tiff.imsave(self._path_registered_bleach + filename_bleach, registered_bleach_image)

            save_vis_img(registered_bleach_image, self._path_registered_vis_bleach, filename_bleach)

            p_idx = np.where(self._melc_phase['order_index'] == self._melc_fluor.iloc[i+1]['order_index'])[0][0]
            phase_image = tiff.imread(self._melc_phase.iloc[p_idx]['path'])
            fluorescence_image = tiff.imread(self._melc_fluor.iloc[i+1]['path'])
            registered_phase_image = register(ref_image, phase_image, phase_image)
            registered_fluor_image = register(ref_image, phase_image, fluorescence_image)
            filename_fluor = '/' + str(int(self._melc_fluor.iloc[i+1]['order_index'])) + '_' + '_'.join(
                self._melc_fluor.iloc[i+1]['fid'].split('_')[:-1]) + '.tif'
            tiff.imsave(self._path_registered_fluor + filename_fluor, registered_fluor_image)
            tiff.imsave(self._path_registered_phase + filename_fluor, registered_fluor_image)

            save_vis_img(registered_fluor_image, self._path_registered_vis_fluor, filename_fluor)
            save_vis_img(registered_phase_image, self._path_registered_vis_phase, filename_fluor)

            '''
            Prospective illumination correction
            '''
            bleach = np.int16(registered_bleach_image)
            fluor = np.int16(registered_fluor_image)
            phase = np.int16(registered_phase_image)

            if self._melc_fluor.iloc[i+1]['filter'] == 'XF111-2':
                fluor -= self._corrected_bf_im[1]
                phase -= self._corrected_bf_im[1]
            else:
                fluor -= self._corrected_bf_im[0]
                phase -= self._corrected_bf_im[0]

            if self._melc_bleach.iloc[i]['filter'] == 'XF111-2':
                bleach -= self._corrected_bf_im[1]
            else:
                bleach -= self._corrected_bf_im[0]

            phase[phase < 0] = 0

            '''
            Subtraction of preceding bleached image
            '''
            fluor_wo_bg = fluor - bleach
            fluor_wo_bg[fluor_wo_bg < 0] = 0

            tiff.imsave(self._path_bg_corr_f + filename_fluor, fluor_wo_bg)
            save_vis_img(fluor_wo_bg, self._path_bg_corr_v_f, filename_fluor)

            tiff.imsave(self._path_bg_corr_p + filename_fluor, phase)
            save_vis_img(phase, self._path_bg_corr_v_p, filename_fluor)

            '''
            Cut off outliers/hot pixels
            '''

            fluor_wo_bg_cut = hot_pixel_removal(fluor_wo_bg)
            phase_bc_cut = hot_pixel_removal(phase)
            tiff.imsave(self._path_cut_f + filename_fluor, fluor_wo_bg_cut)
            save_vis_img(fluor_wo_bg_cut, self._path_cut_v_f, filename_fluor)
            tiff.imsave(self._path_cut_p + filename_fluor, phase_bc_cut)
            save_vis_img(phase_bc_cut, self._path_cut_v_p, filename_fluor)


def save_vis_img(img: np.ndarray, path: str, filename: str):
        '''
        Visualize raw image
        '''
        img_float = img_as_float(img.astype(int))
        img_float = img_float - np.percentile(img_float[20:-20, 20:-20], 0.135) # subtract background
        if not np.percentile(img_float[20:-20, 20:-20], 100 - 0.135) == 0.0:
            img_float /= np.percentile(img_float[20:-20, 20:-20], 100 - 0.135) # normalize to 99.865% of max value
        img_float[img_float < 0] = 0
        img_float[img_float > 1] = 1
        tiff.imsave(path + filename, img_as_uint(img_float))


def hot_pixel_removal(img: np.ndarray):

        sorted_img = np.sort(np.ravel(img))[::-1]
        img[img > sorted_img[3]] = sorted_img[3]

        return img[15:-15, 15:-15]


def main(args):
    path = args.path
    files_gen = args.image_directory_generated

    samples = [s for s in os.listdir(path) if  [char for char in s][0] == 'B']
    for s in samples:
        fovs = os.listdir(path + '/' + s)
        for f in fovs:
            melc_processed_data = MELCImageProcessing(path + '/' + s + '/' + f, melc_structure_generated=files_gen)
            melc_processed_data.process_images()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
