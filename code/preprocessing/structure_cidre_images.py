import os
import tifffile as tif
import argparse
import glob

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Load CIDRE-processed images into structured folder.')

    parser.add_argument(
        '--in_path', dest='in_path', required=True,
        help='Processed images by Cidre')

    parser.add_argument(
        '--out_path', dest='out_path', required=True,
        help='Folder of all samples')

    return parser.parse_args()

def split(word):
    return [char for char in word]


def main():
    args = parse_args()
    in_path = args.in_path
    out_path = args.out_path
    print ('in path: ' + in_path)
    print ('out path: ' + out_path)
    included_extensions = ['tif']
    filenames = [fn for fn in os.listdir(in_path) if any(fn.endswith(ext) for ext in included_extensions)]

    for _f in filenames:
        print (_f)
        if not os.path.exists(out_path + '/' + _f.split('|')[0] + '/' + _f.split('|')[1] + '/processed/cut/fluor_cidre'):
            os.mkdir(out_path + '/' + _f.split('|')[0] + '/' + _f.split('|')[1] + '/processed/cut/fluor_cidre')

        tif.imsave(out_path + '/' + _f.split('|')[0] + '/' + _f.split('|')[1] + '/processed/cut/fluor_cidre/' + '|'.join(_f.split('|')[2:]),tif.imread(in_path + '/' + _f))


if __name__ == '__main__':
    main()