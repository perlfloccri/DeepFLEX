import os
import argparse
import tifffile as tif
import glob


def get_parser():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Combine processed images of all FoVs of all samples into one folder as input for CIDRE')

    parser.add_argument(
        '--in_path', dest='in_path', required=True,
        help='path to all samples')

    parser.add_argument(
        '--out_path', dest='out_path', required=True,
        help='combined folder')

    return parser

def split(word):
    return [char for char in word]

def main(args):
    in_path = args.in_path
    out_path = args.out_path

    runs = os.listdir(in_path)

    for _r in runs:
        if (os.path.isdir(in_path + '/' + _r) and ('BM' in _r )):
            fov_folders = [_f for _f in os.listdir(in_path + '/' + _r)]
            for _f in fov_folders:
                images = []
                for img in glob.glob(in_path + '/' + _r + '/' + _f + '/processed/cut/fluor/*.tif'):
                    images.append(tif.imread(img))
                names = os.listdir(in_path + '/' + _r + '/' + _f + '/processed/cut/fluor')
                for i in range(0, len(images)):
                    tif.imwrite(out_path + '/' + _r + '|' + _f + '|' + names[i], images[i].astype('uint16'))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)