def create_folder(PATH_TO_CREATE):
    """
    Creates a folder on PATH_TO_CREATE position.
    """
    #from os import mkdir
    #mkdir(PATH_TO_CREATE)
    import os
    if not os.path.exists(PATH_TO_CREATE):
        os.makedirs(PATH_TO_CREATE)

def remove_folder(PATH_TO_REMOVE):
    """
    Creates a folder on PATH_TO_CREATE position.
    """
    from shutil import rmtree
    from time import sleep
    try: rmtree(PATH_TO_REMOVE)
    except: pass
    sleep(0.05)


def get_files(path, endings_tuple):
    """
    Loading file from path with the ending stored in endings_tuple
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
    """
    Removes all files and subfolders in path_main_raw and creates the MELC file structure
    """

    path_data_phase = path_main_raw + '/phase'
    path_data_fluor = path_main_raw + '/fluor'
    path_data_bleach = path_main_raw + '/bleach'
    path_data_phasebleach = path_main_raw + '/phase-bleach'

    path_main_others = path_main_raw + '/others'
    path_data_others_phase = path_main_others + '/phase'
    path_data_others_fluor = path_main_others + '/fluor'
    path_data_others_bleach = path_main_others + '/bleach'
    path_data_others_phasebleach = path_main_others + '/phase-bleach'

    path_main_calibration = path_main_raw + '/calibration'

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
