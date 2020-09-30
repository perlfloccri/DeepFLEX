# --------------------------------------------------------
# Multi-Epitope-Ligand Cartography (MELC) phase-contrast image based segmentation pipeline
#
#
# Written by Filip Mivalt
# --------------------------------------------------------

from os import mkdir
from shutil import rmtree
from time import sleep
import os
if os.name == 'nt':
    from configWin import *
else:
    from config import *



def create_folder(PATH_TO_CREATE):
    """Creates a folder on PATH_TO_CREATE position.
        """
    try:
        mkdir(PATH_TO_CREATE)
    except OSError as e:
        e=1
    

def remove_folder(PATH_TO_REMOVE):
    """Creates a folder on PATH_TO_CREATE position.
        """
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

    return [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if name.endswith(endings_tuple)]


