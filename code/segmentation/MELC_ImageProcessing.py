from MELC.utils.myDatasets import generate_workingRaw_from_raw, MELCStructureDataset
import numpy as np
import tifffile as tiff
from MELC.utils.registration_daria import register
import matplotlib.pyplot as plt
import cv2
from MELC.utils.Files import create_folder
from skimage import img_as_float, img_as_uint
from MELC.utils.f_transformations import filterLowFrequencies, visualize_frequencies
import glob
from os.path import join
from config import *
import argparse
SEPARATOR = '/'

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Run Training of Mask R-CNN')

    parser.add_argument(
        '--path', dest='path', required=True,
        help='Config file for training (and optionally testing)')

    return parser.parse_args()


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
        self._path_normalized_f = ''
        self._path_normalized_v_f = ''
        '''
        Extract MELC data and calibration data
        '''
        w_raw = self._path + SEPARATOR + 'w_raw'
        if not melc_structure_generated:
            generate_workingRaw_from_raw(self._path, w_raw)

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
        path_processed = join(self._path, 'processed')
        path_registered = join(path_processed, 'registered')
        self._path_registered_fluor = join(path_registered, 'fluor')
        self._path_registered_bleach = join(path_registered, 'bleach')
        self._path_registered_phase = join(path_registered, 'phase')
        self._path_registered_vis_fluor = join(path_registered, 'vis_fluor')
        self._path_registered_vis_bleach = join(path_registered, 'vis_bleach')
        self._path_registered_vis_phase = join(path_registered, 'vis_phase')

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

        self._path_bg_corr = self._path + SEPARATOR + 'processed' + SEPARATOR + 'background_corr' + SEPARATOR
        self._path_bg_corr_f = self._path_bg_corr + 'fluor' + SEPARATOR
        self._path_bg_corr_v_f = self._path_bg_corr + 'vis_fluor' + SEPARATOR
        self._path_bg_corr_p = self._path_bg_corr + 'phase' + SEPARATOR
        self._path_bg_corr_v_p = self._path_bg_corr + 'vis_phase' + SEPARATOR

        create_folder(self._path_bg_corr)
        create_folder(self._path_bg_corr_f)
        create_folder(self._path_bg_corr_v_f)
        create_folder(self._path_bg_corr_p)
        create_folder(self._path_bg_corr_v_p)

        '''
        Create folders for normalized images
        '''
        path_normalized = self._path + SEPARATOR + 'processed' + SEPARATOR + 'normalized'
        self._path_normalized_f = path_normalized + SEPARATOR + 'fluor' + SEPARATOR
        self._path_normalized_v_f = path_normalized + SEPARATOR + 'vis_fluor' + SEPARATOR
        self._path_normalized_p = path_normalized + SEPARATOR + 'phase' + SEPARATOR
        self._path_normalized_v_p = path_normalized + SEPARATOR + 'vis_phase' + SEPARATOR

        create_folder(path_normalized)
        create_folder(self._path_normalized_f)
        create_folder(self._path_normalized_v_f)
        create_folder(self._path_normalized_p)
        create_folder(self._path_normalized_v_p)

    def generate_bg_correction_img(self):

        '''
        Create correction image for fluorescence and bleaching images
        '''

        brightfield_im = []
        darkframe_im = []
        filter_names = ['XF116-2', 'XF111-2']
        calibration_path = self._path + SEPARATOR +'w_raw' + SEPARATOR + 'calibration' + SEPARATOR
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
        Registration, background correction and normalization of images
        '''


        '''
        Registration
        '''
        ref_image = tiff.imread(glob.glob(self._path + SEPARATOR + 'w_raw' + SEPARATOR + 'phase' + SEPARATOR + '*_Propidium iodide_200_XF116*.tif'))
        for i in range(0, (len(self._melc_fluor)-1)):

            pb_idx = np.where(self._melc_phasebleach['order_index'] == self._melc_bleach.iloc[i]['order_index'])[0][0]
            phasebleach_image = tiff.imread(self._melc_phasebleach.iloc[pb_idx]['path'])
            bleach_image = tiff.imread(self._melc_bleach.iloc[i]['path'])
            registered_bleach_image = register(ref_image, phasebleach_image, bleach_image)
            filename_bleach = SEPARATOR + str(int(self._melc_bleach.iloc[i]['order_index'])) + '_' + '_'.join(
                self._melc_bleach.iloc[i]['fid'].split('_')[:-1]) + '.tif'
            tiff.imsave(self._path_registered_bleach + filename_bleach, registered_bleach_image)

            save_vis_img(registered_bleach_image, self._path_registered_vis_bleach, filename_bleach)

            p_idx = np.where(self._melc_phase['order_index'] == self._melc_fluor.iloc[i+1]['order_index'])[0][0]
            phase_image = tiff.imread(self._melc_phase.iloc[p_idx]['path'])
            fluorescence_image = tiff.imread(self._melc_fluor.iloc[i+1]['path'])
            registered_phase_image = register(ref_image, phase_image, phase_image)
            registered_fluor_image = register(ref_image, phase_image, fluorescence_image)
            filename_fluor = SEPARATOR + str(int(self._melc_fluor.iloc[i+1]['order_index'])) + '_' + '_'.join(
                self._melc_fluor.iloc[i+1]['fid'].split('_')[:-1]) + '.tif'
            tiff.imsave(self._path_registered_fluor + filename_fluor, registered_fluor_image)
            tiff.imsave(self._path_registered_phase + filename_fluor, registered_fluor_image)

            save_vis_img(registered_fluor_image, self._path_registered_vis_fluor, filename_fluor)
            save_vis_img(registered_phase_image, self._path_registered_vis_phase, filename_fluor)

            '''
            Background Correction
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

            # Substraction of bleaching image
            fluor_wo_bg = fluor - bleach
            fluor_wo_bg[fluor_wo_bg < 0] = 0

            tiff.imsave(self._path_bg_corr_f + filename_fluor, fluor_wo_bg)
            save_vis_img(fluor_wo_bg, self._path_bg_corr_v_f, filename_fluor)

            tiff.imsave(self._path_bg_corr_p + filename_fluor, phase)
            save_vis_img(phase, self._path_bg_corr_v_p, filename_fluor)

            '''
            Normalization
            '''

            fluor_wo_bg_normalized = melc_normalization(fluor_wo_bg)
            phase_bc_normalized = melc_normalization(phase)
            tiff.imsave(self._path_normalized_f + filename_fluor, fluor_wo_bg_normalized)
            save_vis_img(fluor_wo_bg_normalized, self._path_normalized_v_f, filename_fluor)
            tiff.imsave(self._path_normalized_p + filename_fluor, phase_bc_normalized)
            save_vis_img(phase_bc_normalized, self._path_normalized_v_p, filename_fluor)


def save_vis_img(img: np.ndarray, path: str, filename: str):
        img_float = img_as_float(img.astype(int))
        img_float = img_float - np.percentile(img_float[20:-20, 20:-20], 0.135) # subtract background
        if not np.percentile(img_float[20:-20, 20:-20], 100 - 0.135) == 0.0:
            img_float /= np.percentile(img_float[20:-20, 20:-20], 100 - 0.135) # normalize to 99.865% of max value
        img_float[img_float < 0] = 0
        img_float[img_float > 1] = 1 # cut-off high intensities
        tiff.imsave(path + filename, img_as_uint(img_float))


def melc_normalization(img: np.ndarray):

        sorted_img = np.sort(np.ravel(img))[::-1]
        img[img > sorted_img[3]] = sorted_img[3] # cut off high intensities

        return img[15:-15, 15:-15]

''' 
For visualization and inspection of images

***Using normalization

registered_u8 = cv2.convertScaleAbs(registered_image, alpha=(255.0/65535.0))
kernel = np.ones((2, 2), np.float32)/4
mean_filtered_img = cv2.filter2D(registered_float, -1, kernel)
normalized_img = cv2.normalize(mean_filtered_img, None, 0, 255, cv2.NORM_MINMAX)


***Using FFT - cut 0.00001 percent of highest frequencies

images = []
images.append(registered_float)
visualize_frequencies(images)
pixels = registered_float.size
high_intensity_pixels = 3
percentage_non_artificial = 100-high_intensity_pixels/pixels
filtered_img = filterLowFrequencies(registered_float, percentage_non_artificial)
images.append(filtered_img)
visualize_frequencies(images)


***Plot histogram

hist = cv2.calcHist([registered_image], [0], None, [65535], [0, 65535])
plt.plot(hist)
plt.xticks(np.arange(0, 65535, step=2000))
plt.grid(True)
plt.yscale('log')        # plt.xlim([0, 65535])
plt.show()

'''
if __name__ == '__main__':
    args = parse_args()
    MELCImageProcessing(args.path, melc_structure_generated=False)

# raw_1 = r'G:\FORSCHUNG\LAB4\VISIOMICS\MELC\2019\3rdFinalPanel_18-6056\201912201349_1'

# melc_processed_data = MELCImageProcessing(raw_1, melc_structure_generated=False)


x = 0
