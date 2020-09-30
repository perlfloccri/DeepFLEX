from restore.RESTORE import Normalization
import pandas as pd
import argparse
import numpy as np
import sys

def get_parser():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Correct batch effects by normalization using RESTORE (Chang et al., 2020)')

    parser.add_argument(
        '--path_feature_matrix', dest='path_feature_matrix', required=True,
        help='Path to csv file of extracted features')

    parser.add_argument(
        '--path_marker_status', dest='path_marker_status', required=True,
        help='Path to csv file of marker statuses: 0 if no positive cell present for respective marker, 1 otherwise')

    parser.add_argument(
        '--output_path', dest='output_path', required=True,
        help='Path for storing normalized data and predicted thresholds')

    parser.add_argument(
        '--thresholds_predicted', dest='thresholds_predicted', action='store_true')

    return parser


def main(args):
    path_features = args.path_feature_matrix
    path_marker_stats = args.path_marker_status
    thresh_gen = args.thresholds_predicted
    out_path = args.output_path

    features = pd.read_csv(path_features, sep=';', decimal=',')

    # Remove NaN or Inf values if there
    [x1,y] = np.where(np.isnan(features))
    [x2,y] = np.where(np.isinf(features))
    features = features.drop(np.unique(x1))
    features = features.drop(np.unique(x2))

    marker_status = pd.read_csv(path_marker_stats, sep=';')

    features = features.rename(columns={"pat_id": "batch", "FoV": "scene", "obj_idx": "cell", "batch": "FoV_counter"})
    features.head()
    
    location = ['n', 'c', 'm']
    i_feat = ['MXP', 'Me', 'To']

    marker_pairs = [    #mutually exclusive marker pairs
        ['CD14', 'CD3'],
        ['CD20', 'CD3'],
        ['CD24', 'CD3'],
        ['CD25', 'CD14'],
        ['CD276', 'CD14'],
        ['CD29', 'CD3'],
        ['CD3', 'CD20'],
        ['CD34', 'CD8'],
        ['CD4', 'CD8'],
        ['CD44', 'CD20'],
        ['CD45', 'CD34'],
        ['CD56', 'CD14'],
        ['CD8', 'CD4'],
        ['FAIM2', 'CD20'],
        ['GD2', 'CD45'],
        ['HLA-ABC', 'GD2'],
        ['HLA-DR', 'GD2'],
        ['PD-1', 'CD276'],
        ['Vimentin', 'CD8']
    ]

    feature_pairs = []
    for l in location:
        for f in i_feat:
            for m in marker_pairs:
                feature_pairs.append([m[0] + '_' + f + '_' + l, m[1] + '_' + f + '_' + l])
    norm = Normalization(features, marker_status, feature_pairs, out_path)
    norm.run(thresh_gen=thresh_gen)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)


