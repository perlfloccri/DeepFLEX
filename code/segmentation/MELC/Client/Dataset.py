# --------------------------------------------------------
# Multi-Epitope-Ligand Cartography (MELC) phase-contrast image based segmentation pipeline
#
#
# Written by Filip Mivalt
# --------------------------------------------------------



# Official libraries
import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
import json
import scipy.signal as signal

from numpy.random import randint
from scipy.signal import medfilt2d
from os.path import join, isfile, exists
from copy import deepcopy



# My libraries and packages

import MELC.utils.Files as myF
import MELC.Client.Registration as myReg

import os
if os.name == 'nt':
    from configWin import *
else:
    from config import *

from MELC.DLManager.Augmenter import CellObjectAugmenter, ImageAugmenter
from MELC.Client.Annotation import SVGAnnot
from numpy.random import randint, uniform

#visualisation

class MELCDataset:
    """
        A class representing one Run of MELC Dataset in raw format.

        Class for loading MELC Run data saved in the native raw format.
        It browses relevant folders, decodes filenames, reads data, registers images and subtracts background.
        Automatically performs image registration.
        Class stores META data for the MELC Run including the registration indexes.
        If loading next time, class reads META data, at first, and verifies them.
        If everything is fine, the class uses those META, instead compute evertyhing from the beginning.

        When loading fluorescence images, the class takes care about correct light field and background correction
        and also performes median filtration to prevent impulse noise present in most of the fluorescent images.
        Subtracted and filtered images are simultaneously saved into META data folder.
        If all META data are matching raw files, in the next class initialization, saved META images are used
        insted of the all computation again to save time.

        ...

        Attributes
        ----------
        PATH_DATA : str
            Path to the MELC Run raw data folder containing source and bleach folder with *.png files.
            Defined during initialization, by input parameter.


        PATH_META : str
            Path to the META folder

        PATH_META_IMG : int
            Path to the folder, where META images are stored. Generated automatically.

        FILE_META_RUN : str
            Path to the file, where META data about Run are stored. Generated automatically

        files_pd : pandas.DataFrame
            pandas.DataFrame containing META data about image files in the Run.

            Keys : 'path', 'fid', 'step_index', 'filter', 'integration_time', 'antibody',
                    'modality', 'type', 'registration'

        calib_pd : pandas.DataFrame
            pandas.DataFrame containing META data about calibration (light and dark field) images.

            Keys : 'integration_time', 'calib_type', 'filter', 'path'


        antibodies : numpy.array
            Array of all present antibodies in the Run

        border_min_y : int
            Field of view for all registred images. It is smaller, than actual size of the image
            because of the various shift caused by moving of the field of view.

        border_max_y : int
            Field of view for all registred images. It is smaller, than actual size of the image
            because of the various shift caused by moving of the field of view.

        border_min_x : int
            Field of view for all registred images. It is smaller, than actual size of the image
            because of the various shift caused by moving of the field of view.

        border_max_x : int
            Field of view for all registred images. It is smaller, than actual size of the image
            because of the various shift caused by moving of the field of view.

        Methods
        -------
        __init__(data_path)
            Initializes class object and calls all methods for META data reading, validation, or creation.

        __len__()
            Returns the number of antibodies present in Run

        __META_create(data_path)
            Browses given folder and creates META data to *.png all files.
            Decodes names and stores all information into the self.files_pd variable.
            The output contains calibration images as well.

        __META_write()
            Writes META data into *.csv file to the position defined by FILE_META_RUN variable.

        __META_read()
            Reads *.csv file from FILE_META_RUN filepath. And converts registration indexes from string to int.

        __META_verify()
            Checks if all files from META data exist.

        __META_remove_all()
            Removes meta for current run with all files and subfolders

        __register_dataset()
            Performs image coregistration. Two indexes are added to the files_pd.
            Coeficients are then translated in get_image_META() function into the coeficients

        __init_calibrations()
            Initialize calib_pd ready to use. Performs further decoding of the calibration file filenames.

        get_image_META(index)
            index: int
            Returns Dictionary with metadata for all files relating to the antibody with *index*.
            Metadata for fluorescence and background fluorescence with corresponding phase contrast images are provided.

        get_subtracted(index)
            index: int
            Returns phase contras, fluorescence image and metadata of antibody indexed by input variable.
            Background image and light field artifacts are already corrected for fluorescence.

        get_average_phase()
            Average phase image of registered phase contrast images from source (only fluorescence image corresponding phase
            contrast images, not bleaching) folder of original raw data.
    """

    def __init__(self, data_path):
        """
        Initializes MELC Run class object given by filepath data_path and calls all methods for META data reading, validation, or creation.

        :param: data_path: str
        """

        # generating paths
        self.PATH_DATA = data_path
        self.PATH_META = join(PATH_DATA_META, 'MELC_data', data_path.split(SEPARATOR)[-1])
        self.PATH_META_IMG = join(PATH_DATA_META, 'MELC_data', data_path.split(SEPARATOR)[-1], 'imgs')
        self.FILE_META_RUN = join(self.PATH_META, data_path.split(SEPARATOR)[-1] + '.csv')



        # check if self.FILE_META_RUN exists. if not, create neccesary folders and create metadata
        # do registration and save them
        if not isfile(self.FILE_META_RUN):
            if not exists(self.PATH_META):
                if not exists(join(PATH_DATA_META, 'MELC_data')):
                    myF.create_folder(join(PATH_DATA_META, 'MELC_data'))
                myF.create_folder(self.PATH_META)

            self.files_pd = self.__META_create(data_path)
            self.__register_dataset()
            self.__META_write()

        # if yes, then read them and validate. If files are NOT the same as metadata, create new metadata.
        # remove all old metadata and save the new ones
        else:
            try:
                self.__META_read()
            except:
                try: myF.remove_folder(data_path)
                except: pass
                self.__META_create(data_path)
                self.__register_dataset()
                self.__META_write()


            if not self.__META_verify():
                try: myF.remove_folder(data_path)
                except: pass
                self.__META_create(data_path)
                self.__register_dataset()
                self.__META_write()

        if not exists(self.PATH_META_IMG): # checks the if folder for saving images exists, if not -> create
            myF.create_folder(self.PATH_META_IMG)

        # separate DataFrame for antibody images and calibration images
        self.calib_pd = deepcopy(self.files_pd.loc[self.files_pd['step_index'] == 0].reset_index(drop='True'))
        self.files_pd = deepcopy(self.files_pd.loc[self.files_pd['step_index'] > 0].reset_index(drop='True'))
        self.__init_calibrations() # transform calibration dataframe to ready-to-use form

        # stores the antibodies of dataset into the var
        # self.antibody_indexes = np.array()

        # now I can ask for length of antibody array
        # checks all antibodies, which are not NONE or PBS
        # the aim is to exclude PBS files which can appear in the middle of the run
        self.antibody_indexes = np.array([])
        for idx in np.unique(self.files_pd['step_index']):
            temp = np.unique(self.files_pd.loc[self.files_pd['step_index'] == \
                                               np.unique(self.files_pd['step_index'])[idx - 1]]['antibody'])
            for staining in temp:
                if not (staining == 'PBS' or staining == 'NONE'):
                    self.antibody_indexes = np.append(self.antibody_indexes, idx)



        self.antibodies = np.empty(self.__len__(), dtype="<U20") # array allocation
        for k in range(0, self.__len__()): # takes antibody notation by its step_number
            temp = np.unique(self.files_pd.loc[self.files_pd['step_index'] == self.antibody_indexes[k]]['antibody'])
            self.antibodies[k] = temp[(temp!='PBS') & (temp!='NONE')][0]

            # takes the antibody of given step. there is k+2 because calibration (0) and PBS (1) do not count as an antibody

        # Decrease the size of field of view for run based on registration coeficients
        self.border_min_y = 0
        self.border_max_y = 0
        self.border_min_x = 0
        self.border_max_x = 0
        for row in self.files_pd.iterrows():
            temp = row[1]['registration']
            if temp[0] < self.border_min_y:
                self.border_min_y = temp[0]
            elif temp[0] > self.border_max_y:
                self.border_max_y = temp[0]

            if temp[1] < self.border_min_x:
                self.border_min_x = temp[1]
            elif temp[1] > self.border_max_x:
                self.border_max_x = temp[1]

    def __get_MELCDataset_len__(self):
        """
                Returns value of the number of valid antibodies in the run.

                :return length: int
                """
        return len(self.antibody_indexes)

    def __len__(self):
        """
        Returns value of the number of valid antibodies in the run.

        :return length: int
        """

        return self.__get_MELCDataset_len__()

    def __META_create(self, data_path):
        """
        Browses given folder and creates META data to *.png all files.
        Return pandas.DataFrame with decoded names and pointers.
        The output contains calibration coeficients as well.

        :param data_path: str
        :return files_pd: pandas.DataFrame
        """
        files_png = myF.get_files(data_path, ('png', 'PNG'))
        files_pd = pd.DataFrame(files_png)

        def get_FID(x):
            """
            separates file id from the MELC Run path

            :param x: pandas.DataFrame (row)
            :return fid: str
            """
            temp = x['path'].split(SEPARATOR)
            return temp[len(temp) - 1][0:-4]

        def get_order_index(x):
            """
            extracts index at which step (antibody) the image was acquired from data path

            :param x: pandas.DataFrame (row)
            :return step_index: str
            """
            return int(x['path'].split(SEPARATOR)[-1].split('_')[-1][0:-4])

        def get_filter(x):
            """
            extracts the filter used during this image acquisition

            :param x: pandas.DataFrame (row)
            :return filter: str
            """
            temp = x['path'].split(SEPARATOR)[-1].split('_')
            if len(temp) > 3:
                return temp[-2]
            else:
                return ''

        def get_integration_time(x):
            """
            extracts the time integration time of given image

            :param x: pandas.DataFrame (row)
            :return integration_time: str
            """
            temp = x['path'].split(SEPARATOR)[-1].split('_')
            if len(temp) > 3:
                return int(temp[-3])
            else:
                return 0

        def get_antibody(x):
            """
            extracts antibody of given image

            :param x: pandas.DataFrame (row)
            :return antibody: str
            """
            return x['fid'].split('_')[1]

        def get_modality(x):
            """
            extracts whether the image was phase-contrast of fluorescent image

            :param x: pandas.DataFrame (row)
            :return modality: str
            """
            temp = x['fid'].split('_')[0]
            if temp == 'o' or temp == 'b':
                return 'fluo'
            elif temp == 'p' or temp == 'pb':
                return 'phase'
            else:
                return ''

        def get_image_type(x):
            """
            extracts wheter image is acquired during source or bleaching phase of the cycle

            :param x: pandas.DataFrame (row)
            :return image_phase: str
            """
            temp = x['path'].split(SEPARATOR)[-2]
            if temp == 'source':
                return temp
            elif temp == 'bleach':
                return temp
            else:
                return ''
        # extract all neccesary information from the filenames and insert them into the DataFrame
        files_pd = files_pd.rename(columns={0: "path"})
        files_pd['fid'] = files_pd.apply(lambda x: get_FID(x), axis=1)
        files_pd['step_index'] = files_pd.apply(lambda x: get_order_index(x), axis=1)
        files_pd['filter'] = files_pd.apply(lambda x: get_filter(x), axis=1)
        files_pd['integration_time'] = files_pd.apply(lambda x: get_integration_time(x), axis=1)
        files_pd['antibody'] = files_pd.apply(lambda x: get_antibody(x), axis=1)
        files_pd['modality'] = files_pd.apply(lambda x: get_modality(x), axis=1)
        files_pd['type'] = files_pd.apply(lambda x: get_image_type(x), axis=1)

        files_pd = files_pd.loc[files_pd['integration_time'] > 0].reset_index(drop='True')
        return files_pd

    def __META_write(self):
        """
        Writes META data into the file given by self.FILE_META_RUN variable

        :return:
        """
        self.files_pd.to_csv(self.FILE_META_RUN)

    def __META_read(self):
        """
        Reads METADATA from the file given by self.FILE_META_RUN variable.

        :return:
        """

        def convert_registration_to_int(x):
            """
            Converts string array like '[ 0, -1]' into a numpy integer array. This solves the problem,
            that indexes are string array after reading META csv file.

            :param x:
            :return:
            """
            temp = np.array(x['registration'][1:-1].split(' '))
            temp_array = np.array([], dtype=temp.dtype)
            for ele in temp:
                if len(ele) > 0: temp_array = np.append(temp_array, ele)
            return temp_array.astype(np.int16)

        self.files_pd = pd.read_csv(self.FILE_META_RUN)
        self.files_pd = self.files_pd.drop(columns=self.files_pd.keys()[0]) # read_csv includes indexes as an column. Here is removed
        self.files_pd['registration'] = self.files_pd.apply(lambda x: convert_registration_to_int(x), axis=1)
        # converts registration from string like '[ 0, -1]' into int numpy array

    def __META_verify(self):
        """
        Checks if all files from META data exist. Returns True or False method.

        :return is_it_fine: bool
        """
        if len(self.files_pd) == len(self.__META_create(self.PATH_DATA)):
            for file in self.files_pd['path']:
                if not isfile(file):
                    return False
            return True
        else:
            return False

    def __META_remove_all(self):
        """
        Removes all meta folder with all files and subfolders

        :return:
        """
        myF.remove_folder(self.PATH_META)


    def __register_dataset(self):
        """
        Performs image coregistration. Two indexes are added to the files_pd. Registrates all images
        to the first phase-contrast source-image.
        Coeficients are then translated in get_image_META() function into the coeficients

        :return:
        """
        def do_register(ref_img, x):
            """
            reads image given by path in DataFrame (row) and calls registration functions from MELC.Client.Registration

            :param ref_img: numpy.array
            :param x: pandas.DataFrame (row)
            :return registration_index: numpy.array of int indexes
            """
            if x['modality'] == 'phase': # registers only phase contrast images
                temp = x['path']
                img = cv2.imread(temp, 2).astype(np.float32)
                img = myReg.get_diff(img) # cost function is computed from image representation, estimated using this func

                reg_idx, heatmap = myReg.register_images(ref_img, img,
                                                         (100, 100),
                                                         (750, 750), (26, 26))
                # registers img on referenc image ref_img with a window of size (100, 100)px at the position at (750,750)px
                # and browses raster of size (maximum shift) (26, 26)px

                reg_idx = np.array([reg_idx[0], reg_idx[1]]).astype(np.int16)
                return reg_idx
            else: return np.array([0, 0], dtype = np.int16)



        reference = deepcopy(self.files_pd.loc[self.files_pd['modality'] == 'phase']).reset_index(drop='True').iloc[0]
        ref_img = cv2.imread(reference['path'], 2).astype(np.float32)
        ref_img = myReg.get_diff(ref_img)

        print('Registration ' + self.PATH_DATA.split(SEPARATOR)[-1])
        self.files_pd['registration'] = self.files_pd.apply(lambda x: do_register(ref_img, x), axis=1)

        for index, row in self.files_pd.iterrows():
            if row['modality'] == 'fluo' and row['step_index'] > 0:
                temp = self.files_pd.loc[
                    (self.files_pd['step_index'] == row['step_index']) &
                    (self.files_pd['modality'] == 'phase') &
                    (self.files_pd['type'] == row['type'])
                ]
                row['registration'][0] = temp['registration'][temp.index[0]][0]
                row['registration'][1] = temp['registration'][temp.index[0]][1]


    def __init_calibrations(self):
        """
        Initialize calib_pd ready to use. Performs further decoding of the calibration file filenames.
        Transforms calib_pd DataFrame to another with relevant details for image correction, like integration time,
        dark or bright field and so on.

        :return:
        """
        def get_calib_type(x):
            """
            Returns b or d, depends on the file id. Represents bright and dark field.
            """
            return x['fid'].split('_')[2][0]

        self.calib_pd['calib_type'] = self.calib_pd.apply(lambda x: get_calib_type(x), axis=1)

        calib = np.unique(self.calib_pd['calib_type'])
        int_time = np.unique(self.calib_pd['integration_time'])
        filt = np.unique(self.calib_pd['filter'])

        calib_pd = {
            'integration_time': [],
            'calib_type': [],
            'filter': [],
            'path': []
        }

        to_erase = np.array([])
        cntr = 0
        for cal in calib:
            for int_t in int_time:
                for f in filt:
                    calib_pd['calib_type'].append(cal)
                    calib_pd['integration_time'].append(int_t)
                    calib_pd['filter'].append(f)

                    files = deepcopy(self.calib_pd.loc[
                        (self.calib_pd['calib_type'] == cal) &
                        (self.calib_pd['integration_time'] == int_t) &
                        (self.calib_pd['filter'] == f)
                    ]).reset_index(drop='True')

                    if len(files) > 0:
                        img_path = files['path'][0]
                        #img = cv2.imread(files['path'][0], 2).astype(np.float32)
                        #for k in range(1, len(files)):
                        #    img = img + cv2.imread(files['path'][k], 2).astype(np.float32)
                        #img = img / len(files)

                    else:
                        img_path = ''
                        #img_path = np.array([])
                        to_erase = np.append(to_erase, cntr)

                    cntr += 1
                    #calib_pd['image'].append(img)
                    calib_pd['path'].append(img_path)


        calib_pd = pd.DataFrame(calib_pd)
        calib_pd = calib_pd.drop(to_erase)
        self.calib_pd = calib_pd

    def get_image_META(self, index):
        """
        Returns Dictionary with metadata for all files relating to the antibody with *index*.
        Metadata for fluorescence and background fluorescence with corresponding phase contrast images are provided.


        :param index: int
        :return:
        """

        def consistence_check_bleach():
            pass_coef = True
            emsg = 'Following run is inconsistent: ' + self.PATH_DATA.split(SEPARATOR)[-1]
            if len(bleach_fluo) == 0:
                print('======================================')
                print(emsg)
                print('')
                print('File: ')
                print('Index:' + str(index))
                print('Step file index: ' + str(self.antibody_indexes[index]))
                print('Antibody: ' + source_fluo['antibody'])
                print('Filter: ' + source_fluo['filter'])
                print('Integration time: ' + str(source_fluo['integration_time']))
                print('Bleaching-fluorescence contrast file missing')
                print('======================================')
                pass_coef = False

            if len(bleach_phase) == 0:
                print('======================================')
                print(emsg)
                print('')
                print('File')
                print('Index:' + str(index))
                print('Step file index: ' + str(self.antibody_indexes[index]))
                print('Antibody: ' + source_fluo['antibody'])
                print('Bleaching-phase contrast file missing')
                print('======================================')
                pass_coef = False

            return pass_coef

        def consistence_check_source():
            pass_coef = True
            emsg = 'Following run is inconsistent: ' + self.PATH_DATA.split(SEPARATOR)[-1]
            if len(source_fluo) == 0:
                print(emsg)
                print('File with index/antibody: ' + str(index) + 'does not exist')
                pass_coef = False

            if len(source_phase) == 0:
                print(emsg)
                print('Following file')
                print('Index:' + str(index) + ' Antibody: ' + source_fluo['antibody'])
                print('Misses file source-phase-contrast file with')
                pass_coef = False
            return pass_coef

        if index >= self.__get_MELCDataset_len__(): return False


        source_fluo = self.files_pd.loc[
            (self.files_pd['step_index'] == self.antibody_indexes[index]) &
            (self.files_pd['modality'] == 'fluo') &
            (self.files_pd['type'] == 'source') &
            ((self.files_pd['antibody'] != 'NONE'))
            ]


        if len(source_fluo) == 1: source_fluo = source_fluo.iloc[0]
        else:
            source_fluo = []
            raise Exception('fuck')

        bleach_fluo = self.files_pd.loc[
            (self.files_pd['step_index'] == self.antibody_indexes[index]-1) &
            (self.files_pd['modality'] == 'fluo') &
            (self.files_pd['type'] == 'bleach') &
            (self.files_pd['filter'] == source_fluo['filter']) &
            (self.files_pd['integration_time'] == source_fluo['integration_time'])
            ]
        if len(bleach_fluo) == 1: bleach_fluo = bleach_fluo.iloc[0]
        else: bleach_fluo = []

        source_phase = self.files_pd.loc[
            (self.files_pd['step_index'] == self.antibody_indexes[index]) &
            (self.files_pd['modality'] == 'phase') &
            (self.files_pd['type'] == 'source')
            ]

        if len(source_phase) == 1: source_phase = source_phase.iloc[0]
        else: source_phase = []

        bleach_phase = self.files_pd.loc[
            (self.files_pd['step_index'] == self.antibody_indexes[index]-1) &
            (self.files_pd['modality'] == 'phase') &
            (self.files_pd['type'] == 'bleach')
            ]

        if len(bleach_phase) == 1: bleach_phase = bleach_phase.iloc[0]
        else: bleach_phase = []

        if consistence_check_source():
            source_fluo['indexes_1'] = np.array([self.border_max_y - source_fluo['registration'][0],
                                                 self.border_min_y - source_fluo['registration'][0]])
            source_fluo['indexes_2'] = np.array([self.border_max_x - source_fluo['registration'][1],
                                                 self.border_min_x - source_fluo['registration'][1]])

            source_phase['indexes_1'] = np.array([self.border_max_y - source_phase['registration'][0],
                                                  self.border_min_y - source_phase['registration'][0]])
            source_phase['indexes_2'] = np.array([self.border_max_x - source_phase['registration'][1],
                                                  self.border_min_x - source_phase['registration'][1]])
        else:
            return False

        if consistence_check_bleach():

            bleach_fluo['indexes_1'] = np.array([self.border_max_y - bleach_fluo['registration'][0],
                                                 self.border_min_y - bleach_fluo['registration'][0]])
            bleach_fluo['indexes_2'] = np.array([self.border_max_x - bleach_fluo['registration'][1],
                                                 self.border_min_x - bleach_fluo['registration'][1]])

            bleach_phase['indexes_1'] = np.array([self.border_max_y - bleach_phase['registration'][0],
                                                  self.border_min_y - bleach_phase['registration'][0]])
            bleach_phase['indexes_2'] = np.array([self.border_max_x - bleach_phase['registration'][1],
                                                  self.border_min_x - bleach_phase['registration'][1]])
        else:
            bleach_phase = False
            bleach_fluo = False
            print('Image is processed WITHOUT BACKGROUND SUBTRACTION')



        return {'source_fluo': source_fluo,
                'source_phase': source_phase,
                'bleach_fluo': bleach_fluo,
                'bleach_phase': bleach_phase
                }





    def get_subtracted(self, index):
        """
        Returns phase contrast, fluorescence image and metadata of antibody indexed by input variable.
        Background image and light field artifacts are already corrected for fluorescence.

        :param index: int
        :return:
        """
        temp = self.get_image_META(index)
        if temp == False: return False

        bleach_bool = True
        if type(temp['bleach_fluo']) == type(False):
            bleach_bool = False


        fluo_name = temp['source_fluo']['fid']
        phase_name = temp['source_phase']['fid']
        if exists(join(self.PATH_META_IMG, fluo_name + '.tif')) and exists(join(self.PATH_META_IMG, phase_name + '.tif')):
            phase = tiff.imread(join(self.PATH_META_IMG, phase_name + '.tif')).astype(np.float32)
            fluo = tiff.imread(join(self.PATH_META_IMG, fluo_name + '.tif')).astype(np.float32)

        else:
            try:
                calib = cv2.imread(self.calib_pd.loc[
                    (self.calib_pd['calib_type'] == 'b') &
                    (self.calib_pd['filter'] == temp['source_fluo']['filter']) &
                    (self.calib_pd['integration_time'] == temp['source_fluo']['integration_time'])
                ].iloc[0]['path'], 2)
            except:
                ## BCS THERE IS ONE RUN WITH INTEGRATION 8000ms and NO corresponding background file
                calib = np.zeros(cv2.imread(temp['source_fluo']['path'], 2).astype(np.float32).shape)
                print('There is missing calibration file to the file with index ' + str(index+2))
                print('Image will be processed without background. CONSIDER the using of this image for further analysis')


            source_fluo = cv2.imread(temp['source_fluo']['path'], 2).astype(np.float32)
            source_phase = cv2.imread(temp['source_phase']['path'], 2).astype(np.float32)
            if bleach_bool:
                bleach_fluo =cv2.imread(temp['bleach_fluo']['path'], 2).astype(np.float32)
                #bleach_phase = cv2.imread(temp['source_fluo']['path'], 2).astype(np.float32)

            source_fluo = source_fluo - calib
            if bleach_bool: bleach_fluo = bleach_fluo - calib

            source_fluo = source_fluo[
                          temp['source_fluo']['indexes_1'][0]: source_fluo.shape[0] + temp['source_fluo']['indexes_1'][1],
                          temp['source_fluo']['indexes_2'][0]: source_fluo.shape[1] + temp['source_fluo']['indexes_2'][1]
                          ]
            if bleach_bool:
                bleach_fluo = bleach_fluo[
                              temp['bleach_fluo']['indexes_1'][0]: bleach_fluo.shape[0] + temp['bleach_fluo']['indexes_1'][1],
                              temp['bleach_fluo']['indexes_2'][0]: bleach_fluo.shape[1] + temp['bleach_fluo']['indexes_2'][1]
                              ]
            source_phase = source_phase[
                          temp['source_phase']['indexes_1'][0]: source_phase.shape[0] + temp['source_phase']['indexes_1'][1],
                          temp['source_phase']['indexes_2'][0]: source_phase.shape[1] + temp['source_phase']['indexes_2'][1]
                          ]
            if bleach_bool:
                fluo = source_fluo - bleach_fluo
            else:
                fluo = source_fluo

            fluo = medfilt2d(fluo, 3)
            phase = source_phase

            fluo[fluo < 0] = 0

            tiff.imsave(join(self.PATH_META_IMG, phase_name + '.tif'), phase.astype(np.uint16))
            tiff.imsave(join(self.PATH_META_IMG, fluo_name + '.tif'), fluo.astype(np.uint16))

        return phase, fluo, temp

    def get_average_phase(self):
        """
        Average phase image of registered phase contrast images from source (only fluorescence image corresponding phase
        contrast images, not bleaching) folder of original raw data.

        :return:
        """
        print(self.PATH_META_IMG)
        if exists(join(self.PATH_META_IMG, 'average_phase.tif')):
            phase = tiff.imread(join(self.PATH_META_IMG, 'average_phase.tif')).astype(np.float32)
        else:
            temp_phase, fluo, temp = self.get_subtracted(0)
            for k in range(1, self.__get_MELCDataset_len__()):
                phase, fluo, temp = self.get_subtracted(k)
                temp_phase = temp_phase + phase
            phase = temp_phase / self.__get_MELCDataset_len__()
            phase = phase.round()
            phase[phase < 0] = 0
            tiff.imsave(join(self.PATH_META_IMG, 'average_phase.tif'), phase.astype(np.uint16))

        return phase

class MELCTiler(MELCDataset):

    def __init__(self, path_data, path_annotations):
        super(MELCTiler, self).__init__(path_data) # calls __init__ of MELCDataset to initialize all paths and run all the init. Everything is changed in this class then
        #MELCDataset.__init__(self, path_data)

        self.tile_size = 256
        self.tile_overlap = 128
        self.phase_average = self.get_average_phase()
        self.im_shape = self.phase_average.shape

        self.PATH_ANNOTATIONS = path_annotations
        if not ('_'.join(self.PATH_ANNOTATIONS.split(SEPARATOR)[-1].split('_')[1:]) == '_'.join(self.PATH_DATA.split(SEPARATOR)[-1].split('_')[:2])):
            error_str = 'Annotations ' '_'.join(self.PATH_ANNOTATIONS.split(SEPARATOR)[-1].split('_')[:2]) + ' do not match ' + \
                        '_'.join(self.PATH_DATA.split(SEPARATOR)[-1].split('_')[:2])  + ' data'
            raise Exception(error_str)

        self.annot_pd = self.__META_ANNOTATION_create()




    def __META_ANNOTATION_create(self):
       def get_FID(x):
            return x['path'].split(SEPARATOR)[-1]

       def get_X_idx(x):
           temp = np.array(x['fid'].split('X')[-1].split('.')[0].split('_'))
           temp_array = np.array([], dtype=temp.dtype)
           for ele in temp:
               if len(ele) > 0: temp_array = np.append(temp_array, ele)
           return temp_array.astype(np.int16)


       def get_Y_idx(x):
           temp = np.array(x['fid'].split('X')[0].split('Y')[-1].split('_'))
           temp_array = np.array([], dtype=temp.dtype)
           for ele in temp:
               if len(ele) > 0: temp_array = np.append(temp_array, ele)
           return temp_array.astype(np.int16)

       annotations_svg = myF.get_files(self.PATH_ANNOTATIONS, ('svg', 'SVG'))
       annot_pd = pd.DataFrame(annotations_svg)
       annot_pd = annot_pd.rename(columns={0: "path"})
       annot_pd['fid'] = annot_pd.apply(lambda x: get_FID(x), axis=1)
       annot_pd['Y_indexes'] = annot_pd.apply(lambda x: get_Y_idx(x), axis=1)
       annot_pd['X_indexes'] = annot_pd.apply(lambda x: get_X_idx(x), axis=1)

       return annot_pd

class MELCSynthesiser(MELCTiler):
    def __init__(self, path_data, path_annotations):
        self.annot_pd = []

        #super(MELCTiler, self).__init__(path_data)
        super(MELCSynthesiser, self).__init__(path_data, path_annotations)
        #MELCTiler.__init__(self, path_data, path_annotations)

        # take image of the background, initialise source files - annotations done by hand
        # employ what has already been done.
        self.background = self._get_background(self.phase_average)


        fid = open(join(path_annotations, 'meta.json'), 'r')
        self.annot_meta = json.loads(fid.read())
        fid.close()


        if self.annot_meta['original_file'].split('_')[0] == 'pb':
            refImg_path = \
            self.files_pd.loc[(self.files_pd['modality'] == 'phase') & (self.files_pd['type'] == 'bleach')][
                'path'].reset_index(drop='True')[0]
        else:
            refImg_path = \
            self.files_pd.loc[(self.files_pd['modality'] == 'phase') & (self.files_pd['type'] == 'source')][
                'path'].reset_index(drop='True')[0]

        ## TODO IMPLEMENT POSSIBILITY FOR PHASE_AVERAGE DATA

        refImg_path = refImg_path.split(SEPARATOR)
        refImg_path[-1] = self.annot_meta['original_file']
        refImg_path = SEPARATOR.join(refImg_path)

        refImg_whole = cv2.imread(refImg_path, 2)
        self.refImg_whole = refImg_whole
        cell_list = []
        cell_dict = {
            'image': np.array([]),
            'contour': np.array([])
        }
        frame = 25
        for k in range(self.annot_pd.__len__()):
            AnnotObj = SVGAnnot(self.annot_pd['path'][k])
            annotations = AnnotObj.get_contours()
            temp = self.annot_pd['path'][k].split(SEPARATOR)[-1].split('.')[0].split('_')
            idx1_IMG = np.array([int(temp[3]), int(temp[4])])
            idx2_IMG = np.array([int(temp[-2]), int(temp[-1])])

            idx1_IMG = idx1_IMG + self.annot_meta['idx1']
            idx2_IMG = idx2_IMG + self.annot_meta['idx2']

            refImg = refImg_whole[idx1_IMG[0]:idx1_IMG[1], idx2_IMG[0]:idx2_IMG[1]]

            for annot in annotations:
                if len(annot > 0):# bcs there may be faulty annotation, just one click
                    idx1 = np.array([annot[:, 0, 1].min() - frame, annot[:, 0, 1].max() + frame])
                    idx2 = np.array([annot[:, 0, 0].min() - frame, annot[:, 0, 0].max() + frame])


                    if idx1[0] > frame and idx2[0] > frame \
                        and idx1[1] < refImg.shape[1] - 1 - frame and idx2[1] < refImg.shape[1] - frame - 1:


                        idx1[idx1 < 0] = 0
                        idx2[idx2 < 0] = 0
                        idx1[idx1 >= refImg.shape[0]] = refImg.shape[0] - 1
                        idx2[idx2 >= refImg.shape[1]] = refImg.shape[1] - 1


                        annotImg = np.zeros(refImg.shape, dtype=np.uint8)
                        annotImg = cv2.drawContours(annotImg, [annot], -1, 255, -1)

                        temp_dict = deepcopy(cell_dict)
                        temp_dict['image'] = refImg[idx1[0]:idx1[1], idx2[0]:idx2[1]]
                        #temp_dict['mask'] = annotImg[idx1[0]:idx1[1], idx2[0]:idx2[1]]
                        temp_dict['contour'] = annot
                        temp_dict['contour'][:, 0, 0] = temp_dict['contour'][:, 0, 0] - idx2[0]
                        temp_dict['contour'][:, 0, 1] = temp_dict['contour'][:, 0, 1] - idx1[0]
                        #temp_dict['image'] = refImg
                        #temp_dict['mask'] = annotImg

                        cell_list.append(temp_dict)
        self.cell_list = cell_list
        self.CellAugmenter = CellObjectAugmenter()


        ## WORKS SO FAR TO HERE

    @classmethod
    def _extend(cls, x, extend):
        temp = np.array([])
        extend = int(extend)
        temp = np.append(temp, x[0, :].flatten())
        temp = np.append(temp, x[-1, :].flatten())
        temp = np.append(temp, x[:, 0].flatten())
        temp = np.append(temp, x[:, -1].flatten())

        y = np.zeros((x.shape[0] + 2 * extend, x.shape[1] + 2 * extend))
        y[extend:-extend, extend:-extend] = x
        return y

    @classmethod
    def _get_blurring_mask(self, size, std):
        winY = signal.gaussian(size[0], std)
        tempY = np.zeros((size[0], 1))
        tempY[:, 0] = winY[:]
        tempY = np.repeat(tempY, size[1], axis=1)

        winX = signal.gaussian(size[1], std)
        tempX = np.zeros((1, size[1]))
        tempX[0, :] = winX[:]
        tempX = np.repeat(tempX, size[0], axis=0)

        win = tempX * tempY
        win = win / win.sum()
        return win


    def _get_random_artificial_cell(self):
        temp = self._get_random_cell()
        img = temp['image']
        contour = temp['contour']
        img, contour = self.CellAugmenter.get_random_transform(img, [contour])
        img, contour = self.CellAugmenter.get_random_resize(img, contour)


        cntr2D = np.zeros(img.shape, dtype=np.uint8)
        mask = cv2.drawContours(cntr2D, contour, -1, 255, -1)
        halo_mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8),
                               iterations=2).astype(np.uint16)
        halo_mask = halo_mask - mask

        halo_max_value = img[halo_mask == 255].max()
        img_base_value = img[halo_mask + mask > 0].min()
        img = img - img_base_value

        synthImg = np.zeros(img.shape)
        synthImg[mask == 255] = halo_max_value
        synthImg[halo_mask == 255] = img[halo_mask == 255]

        win = self._get_blurring_mask((11, 11), 100)
        synthImg = signal.convolve2d(self._extend(synthImg, 5), win, 'valid')

        #figure()
        #imshow(synthImg, cmap='gray')
        #figure(999)
        #plot(synthImg[:, 35])




        win = self._get_blurring_mask((7, 7), 10)
        for k in range(2):
            synthImg[mask == 255] = halo_max_value
            synthImg[halo_mask == 255] = img[halo_mask == 255]

            #figure()
            #imshow(synthImg, cmap='gray')
            #figure(9999)
            #plot(synthImg[:, 35])

            synthImg = signal.convolve2d(self._extend(synthImg, 3), win, 'valid')

            #figure()
            #imshow(synthImg, cmap='gray')
            #figure(9999)
            #plot(synthImg[:, 35])


        win = self._get_blurring_mask((3, 3), 3)
        for k in range(2):
            synthImg[mask == 255] = halo_max_value
            synthImg[halo_mask == 255] = img[halo_mask == 255]

            #figure()
            #imshow(synthImg, cmap='gray')
            #figure(99999)
            #plot(synthImg[:, 35])

            synthImg = signal.convolve2d(self._extend(synthImg, 1), win, 'valid')

            #figure()
            #imshow(synthImg, cmap='gray')
            #figure(99999)
            #plot(synthImg[:, 35])

        synthImg[mask+halo_mask > 0] = img[mask+halo_mask > 0]
        win = self._get_blurring_mask((3, 3), 0.8)
        synthImg = signal.convolve2d(self._extend(synthImg, 1), win, 'valid')
#        synthImg = synthImg + img_base_value
        return synthImg, contour


    def _get_random_cell(self):
        return self.cell_list[randint(self.cell_list.__len__()-1)]

    def _get_random_background(self, size_tuple):
        s1 = size_tuple[0]
        s2 = size_tuple[1]
        beg0 = np.random.randint(100, self.background.shape[0] - 100 - s1)
        beg1 = np.random.randint(100, self.background.shape[1] - 100 - s2)
        return self.background[beg0:beg0 + s1, beg1:beg1 + s2]

    @classmethod
    def _get_background(cls, phase_contrast):
        # suppression
        background = deepcopy(phase_contrast).astype(np.float32)
        threshold_value = 30.
        background_value = np.median(background) * 0.96

        #figure()
        #plot(self.phase_average[:, 1000])
        #plot(np.ones(2000) + background_value)

        #figure()
        #imshow(self.background, cmap='gray')

        for k in range(5):
            temp_img = background.copy() - background_value
            temp1 = np.zeros(background.shape)
            temp2 = np.zeros(background.shape)
            temp1[temp_img > threshold_value] = 1
            temp2[temp_img < -2*threshold_value] = 1

            background[temp1 == 1] = background[temp1 == 1] - background_value
            background[temp1 == 1] = background[temp1 == 1] / 2 + background_value

            background[temp2 == 1] = background[temp2 == 1] - background_value
            background[temp2 == 1] = 2*background[temp2 == 1] / 3 + background_value

        return background
            #figure()
            #plot(self.phase_average[:, 1000])
            #plot(self.background[:, 1000])
            #plot(np.zeros(2000) + background_value)
            #plot(np.zeros(2000) + threshold_value + background_value)
            #plot(np.zeros(2000) - threshold_value + background_value)

            #figure()
            #temp = self.background.copy()
            #temp[-1, -1] = self.phase_average.max()
            #temp[0, 0] = self.phase_average.min()
            #imshow(temp, cmap='gray')


    def generate_synthetic_image(self, size_tuple):
        step_px = 40
        border_px = 25 # 30
        max_shift_px = 20

        probability_of_cell_placement = np.random.rand() * 0.2 + 0.7
        probability_of_cell_placement = 1
        background_multiply_ratio =  np.random.rand()*0.2+0.9
        annotations = list()

        Y = np.zeros(size_tuple)
        Y_background = self._get_random_background(size_tuple) * background_multiply_ratio
        Y_nobackground = np.zeros(Y.shape)
        Y_masks = np.zeros(Y.shape)
        grid1 = np.arange(max_shift_px + border_px, Y.shape[0] - border_px - max_shift_px, step_px).round().astype(np.int32)
        grid2 = np.arange(max_shift_px + border_px, Y.shape[0] - border_px - max_shift_px, step_px).round().astype(np.int32)



        cntr = 0
        for k1 in grid1:
            for k2 in grid2:
                luck = np.random.rand()
                if luck < probability_of_cell_placement:
                    cntr += 1

                    global_position_center_1 = k1
                    global_position_center_2 = k2

                    shift_1 = np.random.randint(-max_shift_px, +max_shift_px, 1)[0]
                    shift_2 = np.random.randint(-max_shift_px, +max_shift_px, 1)[0]

                    global_position_center_1 += shift_1
                    global_position_center_2 += shift_2


                    cell_image, annotation_contour = self._get_random_artificial_cell()
                    if cell_image.shape[0] % 2 == 0:
                        temp_cell_image = np.zeros((cell_image.shape[0] + 1, cell_image.shape[1]))
                        temp_cell_image[:-1, :] = cell_image
                        cell_image = temp_cell_image

                    if cell_image.shape[1] % 2 == 0:
                        temp_cell_image = np.zeros((cell_image.shape[0], cell_image.shape[1] + 1))
                        temp_cell_image[:, :-1] = cell_image
                        cell_image = temp_cell_image

                    cell_center = np.array(
                        [
                            np.floor((annotation_contour[0][:, 0, 1].max() + annotation_contour[0][:, 0, 1].min()) / 2),
                            np.floor((annotation_contour[0][:, 0, 0].max() + annotation_contour[0][:, 0, 0].min()) / 2)
                        ]
                    )

                    annotation_mask = np.zeros(cell_image.shape, dtype=np.uint8)
                    annotation_mask = cv2.drawContours(annotation_mask, annotation_contour, -1, 255, -1)

                    contour_to_img = deepcopy(annotation_contour[0])
                    contour_to_img[:, 0, 0] = contour_to_img[:, 0, 0] - cell_center[1] + global_position_center_2
                    contour_to_img[:, 0, 1] = contour_to_img[:, 0, 1] - cell_center[0] + global_position_center_1

                    temp_big_mask = np.zeros(Y.shape)
                    temp_big_mask = cv2.drawContours(temp_big_mask, [contour_to_img], -1, 255, -1)

                    temp_big_img = np.zeros(Y.shape)
                    idx1_small = np.array([0, cell_image.shape[0]])
                    idx2_small = np.array([0, cell_image.shape[1]])

                    idx1_big = idx1_small - cell_center[0] + global_position_center_1
                    idx2_big = idx2_small - cell_center[1] + global_position_center_2

                    if idx1_big[0] < 0:
                        idx1_small[0] = -1 * idx1_big[0]
                        idx1_big[0] = 0

                    if idx2_big[0] < 0:
                        idx2_small[0] = -1 * idx2_big[0]
                        idx2_big[0] = 0

                    if idx1_big[1] >= Y.shape[0]:
                        idx1_small[1] = idx1_small[1] - (idx1_big[1] - Y.shape[0])
                        idx1_big[1] = Y.shape[0]

                    if idx2_big[1] >= Y.shape[1]:
                        idx2_small[1] = idx2_small[1] - (idx2_big[1] - Y.shape[1])
                        idx2_big[1] = Y.shape[1]

                    idx1_big = idx1_big.astype(np.int16)
                    idx2_big = idx2_big.astype(np.int16)

                    temp_big_img[idx1_big[0] : idx1_big[1], idx2_big[0] : idx2_big[1]] = \
                        cell_image[idx1_small[0]: idx1_small[1], idx2_small[0]: idx2_small[1]]

                    #Y_nobackground[(Y_nobackground > 0) & (temp_big_img > 0)] = \
                     #   0.5 * Y_nobackground[(Y_nobackground > 0) & (temp_big_img > 0)]

                    #temp_big_img[(Y_nobackground > 0) & (temp_big_img > 0)] = \
                     #   0.5 * temp_big_img[(Y_nobackground > 0) & (temp_big_img > 0)]


                    Y_nobackground = Y_nobackground + temp_big_img
                    Y_masks = Y_masks + temp_big_mask
                    annotations.append(contour_to_img)


        Y = (Y_background-40) + (Y_nobackground - 40)

        img = Y
        tp = img.dtype
        img = img.astype(np.float64)
        img = img + randint(-100, 100)
        mn = img.mean()
        img = (img - mn) * uniform(0.95, 1.05) + mn
        img = img.round().astype(tp)

        Y = img
        return Y, annotations

        #orig_max = self.phase_average.max()
        #synth_backgr = np.median(Y_background)

        #idxes = np.zeros(Y.shape)
        #idxes[Y > 1.5 * orig_max] = 1
        #Y[idxes == 1] = Y[idxes == 1] - synth_backgr
        #Y[idxes == 1] = (orig_max-synth_backgr) * Y[idxes == 1] / Y[idxes == 1].max()
        #Y[idxes == 1] = Y[idxes == 1] + synth_backgr

        '''

        import seaborn as sns
        figure(0)
        imshow(self.phase_average, cmap='gray')
        figure(1)
        imshow(Y, cmap='gray')
        figure(2)
        imshow(Y + Y_masks, cmap='gray')
        figure(3)
        imshow(Y_masks, cmap='gray')
        figure(4)
        plot(self.phase_average[1000, 1000:1516])
        plot(Y[125, :])

        figure(5)
        sns.distplot(self.phase_average.flatten(), bins=60)
        figure(6)
        sns.distplot(Y.flatten(), bins=60)
        figure(7)
        sns.distplot(self.phase_average.flatten(), bins=60)
        sns.distplot(Y.flatten(), bins=60)

        '''

    def __len__(self):
        return self.annot_pd.__len__()

    def __getitem__(self, item):
        fid = self.annot_pd['fid'][item].split('.')[0]

        AnnotObj = SVGAnnot(self.annot_pd['path'][item])
        annotations = AnnotObj.get_contours()
        temp = self.annot_pd['path'][item].split(SEPARATOR)[-1].split('.')[0].split('_')

        idx1_IMG = np.array([int(temp[3]), int(temp[4])])
        idx2_IMG = np.array([int(temp[-2]), int(temp[-1])])
        #print('XXXXXXXXXXXXXXXXXXXXXXXXXXX')
        #print(idx2_IMG)
        self.annot_meta['idx2']
        idx1_IMG = idx1_IMG + self.annot_meta['idx1']
        idx2_IMG = idx2_IMG + self.annot_meta['idx2']

        idx1_IMG[idx1_IMG < 0] = 0
        idx2_IMG[idx2_IMG < 0] = 0

        idx1_IMG[idx1_IMG > self.refImg_whole.shape[0]] = self.refImg_whole.shape[0]
        idx2_IMG[idx2_IMG > self.refImg_whole.shape[1]] = self.refImg_whole.shape[1]


        #print(idx2_IMG)

        #print('XXXXXXXXXXXXXXXXXXXXXXXXXXX')
        refImg = self.refImg_whole[idx1_IMG[0]:idx1_IMG[1], idx2_IMG[0]:idx2_IMG[1]]

        return (refImg, annotations, fid)

    def generate_augmented_real(self):
        image, annotations, fid = self.__getitem__(randint(0, self.__len__() - 1))


        if image.shape[0] % 1 == 0:
            temp_image = np.zeros((image.shape[0] + 1, image.shape[1]))
            temp_image[:-1, :] = image
            temp_image[-1, :] = temp_image[-2, :]
            image = temp_image

        if image.shape[1] % 1 == 0:
            temp_image = np.zeros((image.shape[0], image.shape[1] + 1))
            temp_image[:, :-1] = image
            temp_image[:, -1] = temp_image[:, -2]
            image = temp_image


        image, annotations = ImageAugmenter.get_random_transform(image, annotations, randint(0, 2, 1, dtype=bool)[0])

        return (image, annotations, fid)



















