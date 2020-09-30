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

def find_nuclei_file(predictions):
    for k in range(predictions.__len__()):
        if 'iodide' in predictions[k]:
            return k
    return None


def main(): # Copies nuclei fluor file into separate folder to be able to run flo's segmentation
    args = parse_args()
    path_fluor = args.input_path
    fluor_files = os.listdir(path=path_fluor)


    nuclei_folder = args.output_path
    create_folder(nuclei_folder)

    nuclei_orig_file = join(path_fluor, fluor_files[find_nuclei_file(fluor_files)])
    shutil.copy(nuclei_orig_file, nuclei_folder)


if __name__ == '__main__':
    main()


