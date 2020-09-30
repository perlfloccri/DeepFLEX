from config import *
SEPARATOR = '/'
def create_folder(PATH_TO_CREATE):
    """Creates a folder on PATH_TO_CREATE position.
        """
    from os import mkdir
    mkdir(PATH_TO_CREATE)


def remove_folder(PATH_TO_REMOVE):
    """Creates a folder on PATH_TO_CREATE position.
        """
    from shutil import rmtree
    from time import sleep
    try: rmtree(PATH_TO_REMOVE)
    except: pass
    sleep(0.05)


def get_files(path, endings_tuple):
    """File list generator.

        For each file in path directory and subdirectories with endings from endings_tuple returns path in a list variable

        path is a string, the path to the directory where you wanna search specific files.

        endings_tuple is a tuple of any length

        The function returns paths to all files in folder and files in all subfolders as well.

        Caution:  ending must be tuple, even for length of 1 ending. ('png')

        Example:

        from myFiles import get_files
        import pandas as pd

        path = "root/data"
        files_list = get_files(path)
        files_pd = pd.DataFrame(files_list)

        def get_FID(x):
            temp = x['path'].split('\\')
            return temp[len(temp) - 1][0:-4]
        files_pd = files_pd.rename(columns={0: "path"})
        files_pd['fid'] = files_pd.apply(lambda x: get_FID(x), axis=1)
        """

    import os
    import numpy as np

    data = [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith(endings_tuple)]

    creation_times = [os.stat(path).st_ctime for path in data]
    data = np.array(data)
    creation_times = np.array(creation_times)
    idx = creation_times.argsort()
    sorted_data = data[idx]
    sorted_creation_times = creation_times[idx]
    return sorted_data, sorted_creation_times


def clear_MELCraw_structured_folder(path_main_raw):
    """Cleaner for MELC structured folder.

            Removes all files and subfolders in path_main_raw and creates the MELC structure

            path_main_raw is a string, the path to the directory where you wanna remove and create again

            endings_tuple is a tuple of any length

            The function returns paths to all files in folder and files in all subfolders as well.

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

            With next functions you can create additional folders and it wont affect the functionality of other functions.
            Main images structure:
                fluor - fluorescent images
                phase - phase contrast images
                bleach - bleaching images
                phase-bleache - phase contrast images during bleaching

            Additional folders:
                calibration - calibration images from raw data
                other - subfolders with the structure same as main images folder: phase, fluor, bleach, phase-bleach
                        this folder is for images withou antibodies marked as NONE / PBS and so on.

            Caution:  BE CAREFUL. REMOVES EVERYTHING IN THE FOLDER

            Example:

            from myFiles import clear_MELCraw_structured_folder

            path = "root/data/MELC"
            clear_MELCraw_structured_folder(path)
            """



    path_data_phase = path_main_raw + SEPARATOR + 'phase'
    path_data_fluor = path_main_raw + SEPARATOR +  'fluor'
    path_data_bleach = path_main_raw +  SEPARATOR + 'bleach'
    path_data_phasebleach = path_main_raw +  SEPARATOR + 'phase-bleach'

    path_main_others = path_main_raw +  SEPARATOR + 'others'
    path_data_others_phase = path_main_others + SEPARATOR + 'phase'
    path_data_others_fluor = path_main_others + SEPARATOR + 'fluor'
    path_data_others_bleach = path_main_others + SEPARATOR + 'bleach'
    path_data_others_phasebleach = path_main_others + SEPARATOR + 'phase-bleach'

    path_main_calibration = path_main_raw + SEPARATOR + 'calibration'

    remove_folder(path_main_raw)

    create_folder(path_main_raw)
    create_folder(path_data_phase)
    create_folder(path_data_fluor)
    create_folder(path_data_bleach)
    create_folder(path_data_phasebleach)

    create_folder(path_main_others)
    create_folder(path_data_others_phase)
    create_folder(path_data_others_fluor)
    create_folder(path_data_others_bleach)
    create_folder(path_data_others_phasebleach)

    create_folder(path_main_calibration)
