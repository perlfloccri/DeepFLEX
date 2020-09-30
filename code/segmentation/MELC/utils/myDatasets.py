import MELC.utils.myFiles as myF
import pandas as pd
from os.path import join
import cv2
import tifffile as tiff
from numpy import unique, where
from config import *
import sys
SEPARATOR = '/'

class RawDataset:
    """RawDataset loader.

            works with RAW folder structure of MELC images.
            Basicaly only one thing it does is that creates pandas DataFrame with the list of the files in the folder/bleach and folder/source
            This class is used in another function for converting raw data to the raw-melc structured folder into standardized uint16 tiff files

            Example:

            from myFiles import RawDataset
            path = "root/data/MELC"
            dataset = RawDataset(path)
            dataframe = dataset.merged()
    """
    def __init__(self, path_raw_data):
        f_source, c_source = self.get_dataFrame_MELCraw(join(path_raw_data, 'source'))
        f_bleach, c_bleach = self.get_dataFrame_MELCraw(join(path_raw_data, 'bleach'))
        source_raw_pd = pd.DataFrame(f_source)
        source_raw_pd[1] = pd.DataFrame(c_source)
        bleach_raw_pd = pd.DataFrame(f_bleach)
        bleach_raw_pd[1] = pd.DataFrame(c_bleach)
        self.merged_pd = pd.concat([source_raw_pd, bleach_raw_pd])
        self.merged_pd = self.merged_pd.reset_index(drop=True)

    def get_dataFrame_MELCraw(self, path_raw_data):
        files_png, creation_times = myF.get_files(path_raw_data, ('png'))
        files_pd = pd.DataFrame(files_png)
        creation_times = pd.DataFrame(creation_times)
        return files_pd, creation_times



def generate_workingRaw_from_raw(path_raw, path_wraw):
    """ Raw data convertor

            Converts raw MELC data from Christian into structured folder without any image transformation.
            Output files are in .tif format

            path_raw is a string, folder where are subfolders source and bleach

            path_wraw, is a string, folder workingRaw, where the MELC structure will be created and files will be converted there

            function recognizes o, p, b, pb files and converts into corresponding folder
            antibodies are coppied into bleach/phase-bleach/phase/fluor folder in path_wraw and files marked of different setup
             into the same filesystem in folder path_wraw/other
            These files which will be selected into other folder are given by variable bellow 'other'. You can modify it.
            There are 2 types of files now. 'NONE' and 'PBS'.
            Calibration files are selected and converted into path_wraw/calibration folder

            The structure is following:
            MAIN FOLDER
                fluor
                phase
                bleach
                phase-bleach
                calibration
                OTHERS
                    fluor
                    phase
                    bleach
                    phase-bleach
            Additional folders:
                calibration - calibration images from raw data
                other - subfolders with the structure same as main images folder: phase, fluor, bleach, phase-bleach
                        this folder is for images withou antibodies marked as NONE / PBS and so on.

        Caution:  BE CAREFUL. REMOVES EVERYTHING IN THE FOLDER path_wraw

        Example:

        from myDatasets import generate_workingRaw_from_raw

        pathRaw = "root/data/MELC/raw"
        path_MELC_struct = " root/data/MELC/working_raw"
        generate_workingRaw_from_raw(pathRaw, path_MELC_struct)
        """

    others = ['NONE']



    dat = RawDataset(path_raw)
    merged_pd = dat.merged_pd
    myF.clear_MELCraw_structured_folder(path_wraw)

    def get_FID(x):
        temp = x['path'].split(SEPARATOR)
        return temp[len(temp) - 1][0:-4]

    def get_acquisition_phase(x):
        acq_phase = x['path'].split(SEPARATOR)[-2] # acquisition phase raw/bleach
        return acq_phase

    def get_acquisition_channel(x):
        #acq_phase = x['path'].split(SEPARATOR)[-2] # acquisition phase raw/bleach
        acq_process_id = x['path'].split(SEPARATOR)[-1].split('_')[0] # o = fluor; p = phase; b = bleach; pb = phase-bleach
        if acq_process_id == 'o': return 'fluor'
        if acq_process_id == 'b': return 'bleach'
        if acq_process_id == 'p': return 'phase'
        if acq_process_id == 'pb': return 'phase-bleach'
        return acq_process_id

    def get_order_index(x):
        return int(x['path'].split(SEPARATOR)[-1].split('_')[-1][0:-4])

    def get_filter(x):
        temp = x['path'].split(SEPARATOR)[-1].split('_')
        if len(temp) > 3:
            return temp[-2]
        else: return ''

    def get_integration_time(x):
        temp = x['path'].split(SEPARATOR)[-1].split('_')
        if len(temp) > 3:
            return int(temp[-3])
        else: return 0

    def get_antibody(x):
        return x['path'].split(SEPARATOR)[-1].split('_')[1]

    merged_pd = merged_pd.rename(columns={0: "path"})
    merged_pd = merged_pd.rename(columns={1: "creation_time"})
    merged_pd['fid'] = merged_pd.apply(lambda x: get_FID(x), axis=1)
    merged_pd['acquisition_phase'] = merged_pd.apply(lambda x: get_acquisition_phase(x), axis=1)
    merged_pd['channel'] = merged_pd.apply(lambda x: get_acquisition_channel(x), axis=1)
    merged_pd['order_index'] = merged_pd.apply(lambda x: get_order_index(x), axis=1)
    merged_pd['filter'] = merged_pd.apply(lambda x: get_filter(x), axis=1)
    merged_pd['integration_time'] = merged_pd.apply(lambda x: get_integration_time(x), axis=1)
    merged_pd['antibody'] = merged_pd.apply(lambda x: get_antibody(x), axis=1)

    unique_antibodies = unique(merged_pd['antibody'])

    def png_to_tiff(path_png, path_tiff):
        file = tiff.imsave(path_tiff, cv2.imread(path_png, 2))
        return 0

    def rawMELC_to_wrawMELC(files_pd, path_wraw):
        files_pd = files_pd.reset_index(drop='True')
        for k in range(len(files_pd)):
            png_to_tiff(files_pd['path'][k], path_wraw + SEPARATOR + files_pd['channel'][k] + SEPARATOR + str(int(files_pd['order_index'][k])) + '_' + '_'.join(files_pd['fid'][k].split('_')[1:]) + '.tif')

    def rawCalib_to_wrawCalib(files_pd, path_to_save):
        files_pd = files_pd.reset_index(drop='True')
        print(path_to_save)
        for k in range(len(files_pd)):
            png_to_tiff(files_pd['path'][k], path_to_save + SEPARATOR + str(int(files_pd['order_index'][k])) + '_' + '_'.join(files_pd['fid'][k].split('_')[1:]) + '.tif')

    for antib in unique_antibodies:
        temp = merged_pd.loc[merged_pd['antibody'] == antib]
        if antib == 'cal':
            print('Parent: ' + path_wraw + SEPARATOR + 'calibration')
            rawCalib_to_wrawCalib(temp, path_wraw + SEPARATOR + 'calibration') #save calib
        #if antib in others:
            #rawMELC_to_wrawMELC(temp, path_wraw + SEPARATOR + 'others') # save others
        else:
            #if len(temp) >= 4 and len(unique(temp['channel'])) >= 4:
            if len(temp) >= 4:
                rawMELC_to_wrawMELC(temp, path_wraw)  # save others

class CalibrationDataset:
    """ Calibration files loader.

                Loader for loading calibration files from calibration folder

                calib_path is a string, path to the folder where the calibration files are. Intended for working_raw folder

            Caution:  can work mainly with calib files, where name of the files are standardized and produced by generate_workingRaw_from_raw function

            Example:
            from myDatasets import CalibrationDataset

            pathWRaw = "root/data/MELC/working_raw/calibration"
            calibDat = CalibrationDataset(pathWRaw)
            calibDat[0] #loads image from 1st position in dataframe from calibration folder
            calibDat.files_pd #returns pandas dataframe with calibration files with following columns
                                Columns:
                                    path
                                    fid - fileid
                                    integration_time # int
                                    order_index # int of order in which this calib was acquired. For calibration should be 0
                                    filter # filter code used for calibration measurement
            calib.get_file_idx(self, integration_time=5000, filter='XF111-2') # returns list of indexes of all files which match this criterium
            """

    def __init__(self, calib_path):
        files_tiff, _ = myF.get_files(calib_path, ('tif', 'TIF'))
        files_pd = pd.DataFrame(files_tiff)

        def get_FID(x):
            temp = x['path'].split(SEPARATOR)
            _temp = temp[len(temp) - 1].split('.')[0].split('_')
            return _temp[1] + '_' + _temp[2] + '_' + _temp[3]+'_'+_temp[4]

        def get_order_index(x):
            return int(x['path'].split(SEPARATOR)[-1].split('_')[-1][0:-4])

        def get_filter(x):
            temp = x['path'].split(SEPARATOR)[-1].split('_')
            if len(temp) > 3:
                return temp[-2]
            else:
                return ''

        def get_integration_time(x):
            temp = x['path'].split(SEPARATOR)[-1].split('_')
            if len(temp) > 3:
                return int(temp[-3])
            else:
                return 0

        #def get_creation_time(x):
            #return x['path'].split(SEPARATOR)[-1].split('_')[0]

        files_pd = files_pd.rename(columns={0: "path"})
        files_pd['fid'] = files_pd.apply(lambda x: get_FID(x), axis=1)
        files_pd['order_index'] = files_pd.apply(lambda x: get_order_index(x), axis=1)
        files_pd['filter'] = files_pd.apply(lambda x: get_filter(x), axis=1)
        files_pd['integration_time'] = files_pd.apply(lambda x: get_integration_time(x), axis=1)
        #files_pd['creation_time'] = files_pd.apply(lambda x: get_creation_time(x), axis=1)
        self.files_pd = files_pd

    def __len__(self):
        return len(self.files_pd)

    def __getitem__(self, item):
        if type(item) == type([]):
            return_list = list()
            for it in item:
                return_list.append(tiff.imread(self.files_pd['path'][it]))
            return return_list
        else:
            return tiff.imread(self.files_pd['path'][item])

    def get_file_idx(self, integration_time=5000, filter='XF111-2'):
        return self.files_pd.loc[(self.files_pd['integration_time'] == integration_time) & (self.files_pd['filter'] == filter)].index.tolist()


class MELCStructureDataset:
    """ MELC standardized file structure loader.

                    Loader for loading calibration files from calibration folder

                    MELC_path is a string, path to the folder where the are subfolders bleach, phase, fluor, phase-bleach
                                no file control is implemented, the correct structure generated by generate_workingRaw_from_raw function
                                is expected.

                    dataset[0] input is dictionary
                    output_dict = {
                                    'phase': phase image
                                    'fluor': fluor image
                                    'bleach': ...
                                    'phasebleach': ....
                                    'antibody': ...
                                    'order_index': ...
                                    'filter': ....
                                    'integration_time': ....
                                }

                Caution:  in each folder must be corresponding file specific by its order_index and antibody, required is only realization of antibody. in run


                Example:
                from myDatasets import MELCStructureDataset

                pathWRaw = "root/data/MELC/working_raw"
                melcDat = MELCStructureDataset(pathWRaw)
                melcDat[0] #loads image from 1st position of antibody in self.antibodies
                melcDat['CD44-PE'] #returns specific antibody file
                melcDat.antibodies

                """

    def __init__(self, MELC_path):
        path_phase = MELC_path + SEPARATOR + 'phase'
        path_bleach = MELC_path + SEPARATOR +'bleach'
        path_phasebleach = MELC_path + SEPARATOR +'phase-bleach'
        path_fluor = MELC_path + SEPARATOR +'fluor'

        self.phase_pd = self.get_MELC_files(path_phase)
        self.bleach_pd = self.get_MELC_files(path_bleach)
        self.fluor_pd = self.get_MELC_files(path_fluor)
        self.phasebleach_pd = self.get_MELC_files(path_phasebleach)

        self.antibodies = unique(self.fluor_pd['antibody'])

    def __len__(self):
        return len(self.antibodies)

    def __getitem__(self, item):
        if type(item) == type(''):
            item = where(self.antibodies == item)[0][0]

        antibody = self.antibodies[item]
        phase_row = self.phase_pd.iloc[[item]]
        fluor_row = self.fluor_pd.iloc[[item]]
        bleach_row = self.bleach_pd.iloc[[item]]
        phasebleach_row = self.phasebleach_pd.iloc[[item]]

        output_dict = {
            'phase': tiff.imread(phase_row['path'].iloc[0]),
            'fluor': tiff.imread(fluor_row['path'].iloc[0]),
            'bleach': tiff.imread(bleach_row['path'].iloc[0]),
            'phasebleach': tiff.imread(phasebleach_row['path'].iloc[0]),
            'antibody': antibody,
            'order_index': fluor_row['order_index'].iloc[0],
            'filter': fluor_row['filter'].iloc[0],
            'integration_time': fluor_row['integration_time'].iloc[0]
        }
        return output_dict

    def get_MELC_files(self, path):
        files, _ = myF.get_files(path, ('tif', 'TIF'))
        files_pd = pd.DataFrame(files)
        files_pd = files_pd.rename(columns={0: "path"})

        def get_FID(x):
            temp = x['path'].split(SEPARATOR)
            _temp = temp[len(temp) - 1].split('.')[0].split('_')
            return _temp[1]+'_' + _temp[2]+'_'+_temp[3]+'_'+_temp[4]

        def get_order_index(x):
            return int(x['path'].split(SEPARATOR)[-1].split('_')[-1][0:-4])

        def get_filter(x):
            temp = x['path'].split(SEPARATOR)[-1].split('_')
            if len(temp) > 3:
                return temp[-2]
            else:
                return ''

        def get_integration_time(x):
            temp = x['path'].split(SEPARATOR)[-1].split('_')
            if len(temp) > 3:
                return int(temp[-3])
            else:
                return 0

        def get_antibody(x):
            return x['path'].split(SEPARATOR)[-1].split('_')[1]

        #def get_creation_time(x):
            #return x['path'].split(SEPARATOR)[-1].split('_')[0]

        files_pd = files_pd.rename(columns={0: "path"})
        files_pd['fid'] = files_pd.apply(lambda x: get_FID(x), axis=1)
        files_pd['order_index'] = files_pd.apply(lambda x: get_order_index(x), axis=1)
        files_pd['filter'] = files_pd.apply(lambda x: get_filter(x), axis=1)
        files_pd['integration_time'] = files_pd.apply(lambda x: get_integration_time(x), axis=1)
        files_pd['antibody'] = files_pd.apply(lambda x: get_antibody(x), axis=1)
        #files_pd['creation_time'] = files_pd.apply(lambda x: get_creation_time(x), axis=1)
        return files_pd