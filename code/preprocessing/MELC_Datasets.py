import MELC_Files as myF
import pandas as pd
from os.path import join
import cv2
import tifffile as tiff
from numpy import unique, where


class RawDataset:
    """
    Generation of pandas dataframe from raw MELC images (found in folder /source and /bleach)
    to be used in another function for building a structured folder of tiff files
    """
    def __init__(self, path_raw_data):
        print ("Pfad: " + path_raw_data)
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
    """
    Converts raw MELC data into structured folder without any image transformation
    Output files are in .tif format
    """

    others = ['NONE']

    dat = RawDataset(path_raw)
    merged_pd = dat.merged_pd
    myF.clear_MELCraw_structured_folder(path_wraw)

    def get_FID(x):
        temp = x['path'].split('/')
        return temp[len(temp) - 1][0:-4]

    def get_acquisition_phase(x):
        acq_phase = x['path'].split('/')[-2]
        return acq_phase

    def get_acquisition_channel(x):
        acq_process_id = x['path'].split('/')[-1].split('_')[0] # o = fluor; p = phase; b = bleach; pb = phase-bleach
        if acq_process_id == 'o': return 'fluor'
        if acq_process_id == 'b': return 'bleach'
        if acq_process_id == 'p': return 'phase'
        if acq_process_id == 'pb': return 'phase-bleach'
        return acq_process_id

    def get_order_index(x):
        return int(x['path'].split('/')[-1].split('_')[-1][0:-4])

    def get_filter(x):
        temp = x['path'].split('/')[-1].split('_')
        if len(temp) > 3:
            return temp[-2]
        else: return ''

    def get_integration_time(x):
        temp = x['path'].split('/')[-1].split('_')
        if len(temp) > 3:
            return int(temp[-3])
        else: return 0

    def get_antibody(x):
        return x['path'].split('/')[-1].split('_')[1]

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
            png_to_tiff(files_pd['path'][k], path_wraw + '/' + files_pd['channel'][k] + '/' + str(int(files_pd['order_index'][k])) + '_' + '_'.join(files_pd['fid'][k].split('_')[1:]) + '.tif')

    def rawCalib_to_wrawCalib(files_pd, path_to_save):
        files_pd = files_pd.reset_index(drop='True')
        print(path_to_save)
        for k in range(len(files_pd)):
            png_to_tiff(files_pd['path'][k], path_to_save + '/' + str(int(files_pd['order_index'][k])) + '_' + '_'.join(files_pd['fid'][k].split('_')[1:]) + '.tif')

    for antib in unique_antibodies:
        temp = merged_pd.loc[merged_pd['antibody'] == antib]
        if antib == 'cal':
            print('Parent: ' + path_wraw + '/calibration')
            rawCalib_to_wrawCalib(temp, path_wraw + '/calibration')
        else:
            if len(temp) >= 4:
                rawMELC_to_wrawMELC(temp, path_wraw)

class CalibrationDataset:
    """
    Loading calibration files from calibration folder
    """

    def __init__(self, calib_path):
        files_tiff, _ = myF.get_files(calib_path, ('tif', 'TIF'))
        files_pd = pd.DataFrame(files_tiff)

        def get_FID(x):
            temp = x['path'].split('/')
            _temp = temp[len(temp) - 1].split('.')[0].split('_')
            return _temp[1] + '_' + _temp[2] + '_' + _temp[3]+'_'+_temp[4]

        def get_order_index(x):
            return int(x['path'].split('/')[-1].split('_')[-1][0:-4])

        def get_filter(x):
            temp = x['path'].split('/')[-1].split('_')
            if len(temp) > 3:
                return temp[-2]
            else:
                return ''

        def get_integration_time(x):
            temp = x['path'].split('/')[-1].split('_')
            if len(temp) > 3:
                return int(temp[-3])
            else:
                return 0

        files_pd = files_pd.rename(columns={0: "path"})
        files_pd['fid'] = files_pd.apply(lambda x: get_FID(x), axis=1)
        files_pd['order_index'] = files_pd.apply(lambda x: get_order_index(x), axis=1)
        files_pd['filter'] = files_pd.apply(lambda x: get_filter(x), axis=1)
        files_pd['integration_time'] = files_pd.apply(lambda x: get_integration_time(x), axis=1)
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
    """
    MELC standardized file structure loader
    """

    def __init__(self, MELC_path):
        path_phase = MELC_path + '/phase'
        path_bleach = MELC_path + '/bleach'
        path_phasebleach = MELC_path + '/phase-bleach'
        path_fluor = MELC_path + '/fluor'

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
            temp = x['path'].split('/')
            _temp = temp[len(temp) - 1].split('.')[0].split('_')
            return _temp[1]+'_' + _temp[2]+'_'+_temp[3]+'_'+_temp[4]

        def get_order_index(x):
            return int(x['path'].split('/')[-1].split('_')[-1][0:-4])

        def get_filter(x):
            temp = x['path'].split('/')[-1].split('_')
            if len(temp) > 3:
                return temp[-2]
            else:
                return ''

        def get_integration_time(x):
            temp = x['path'].split('/')[-1].split('_')
            if len(temp) > 3:
                return int(temp[-3])
            else:
                return 0

        def get_antibody(x):
            return x['path'].split('/')[-1].split('_')[1]

        files_pd = files_pd.rename(columns={0: "path"})
        files_pd['fid'] = files_pd.apply(lambda x: get_FID(x), axis=1)
        files_pd['order_index'] = files_pd.apply(lambda x: get_order_index(x), axis=1)
        files_pd['filter'] = files_pd.apply(lambda x: get_filter(x), axis=1)
        files_pd['integration_time'] = files_pd.apply(lambda x: get_integration_time(x), axis=1)
        files_pd['antibody'] = files_pd.apply(lambda x: get_antibody(x), axis=1)
        return files_pd
