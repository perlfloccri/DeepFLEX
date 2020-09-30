import argparse
import os
import shutil
from os.path import join
from MELC.utils.Files import create_folder

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Run Inference of Mask R-CNN')

    parser.add_argument(
        '--input_path', dest='input_path', required=True,
        help='Config file for training (and optionally testing)')

    parser.add_argument(
        '--output_path', dest='output_path', required=True)

    return parser.parse_args()


def main(): # Copies nuclei fluor file into separate folder to be able to run flo's segmentation
    args = parse_args()
    path_fluor = args.input_path
    pc_files = os.listdir(path=path_fluor)


    pc_folder = args.output_path
    create_folder(pc_folder)
    for pc in pc_files:
        pc_orig_file = join(path_fluor, pc)
        shutil.copy(pc_orig_file, pc_folder)


if __name__ == '__main__':
    main()


