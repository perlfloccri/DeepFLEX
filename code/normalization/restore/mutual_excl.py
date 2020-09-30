import argparse
import pandas as pd
import numpy as np


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Calculate mutual exclusiveness between markers')

    parser.add_argument(
        '--in_path', dest='in_path', required=True,
        help='Path to extracted mean intensities')

    parser.add_argument(
        '--out_path', dest='out_path', required=True,
        help='Output path')

    parser.add_argument(
        '--wo_tumor', dest='wo_tumor', action='store_true',
        help='Should tumor infiltrated samples be included?')

    return parser.parse_args()


def split(word: str):
    return [char for char in word]


class MutualExclusiveness:

    def __init__(self, in_path: str, out_path: str, without_tumor_inf_samples: bool):
        self._in_path = in_path
        self._out_path = out_path
        self._wo_tum = without_tumor_inf_samples

    def load_data(self):
        matrix = pd.read_csv(self._in_path, sep=';')

        if self._wo_tum:
            patient_idx = matrix['pat_id'].values
            cells_2_drop = np.where(np.logical_or(patient_idx == 186056, np.logical_or(patient_idx == 192548,
                                        np.logical_or(patient_idx == 196196,
                                         np.logical_or(patient_idx == 190473, patient_idx == 191612)))))[0]
            matrix = matrix.drop(cells_2_drop, axis=0)

        headers_2_drop = [col for col in matrix.columns if (split(col)[0] == 'P' and split(col)[1] == 'r') or
                          split(col)[2] == 'V' or split(col)[0] == 'o' or split(col)[0] == 'b' or split(col)[0] == 'p'
                          or split(col)[0] == 'y' or split(col)[0] == 'x']
        matrix = matrix.drop(headers_2_drop, axis=1)

        return matrix

    def calc_correlation(self):
        data_matrix_df = self.load_data()
        feature_list = [feature.tolist() for feature in np.transpose(data_matrix_df.values)]
        correlation_matrix = np.corrcoef(feature_list)

        df_markers = [col for col in data_matrix_df.columns]
        correlation_matrix_df = pd.DataFrame(correlation_matrix, columns= df_markers, index=df_markers)

        correlation_matrix_df.to_csv(self._out_path + '\\correlation_matrix.csv', sep=';')

        return correlation_matrix_df

    def find_mutual_excl_markers(self):
        corr_matrix_df = self.calc_correlation()
        corr_matrix = corr_matrix_df.values
        corr_value_list = corr_matrix.tolist()
        markers = [col.split('_')[0] for col in corr_matrix_df.columns]

        mut_excl_markers = []
        markers_arr = np.asarray(markers)
        for row in corr_value_list:
            ind = np.asarray(row).argsort()
            sortedMarkers = markers_arr[ind]
            mut_excl_markers.append(sortedMarkers[0:5])

        mut_excl_markers = np.asarray(mut_excl_markers)
        column_names = [str(i+1) for i in range(5)]
        mut_excl_markers_df = pd.DataFrame(mut_excl_markers, index=markers, columns=column_names)

        mut_excl_markers_df.to_csv(self._out_path + '\\mutually_exclusive_markers.csv', sep=';')

        return mut_excl_markers_df

def main():
    args = parse_args()
    in_path = args.in_path
    out_path = args.out_path
    wo_tumor = args.wo_tumor

    ME = MutualExclusiveness(in_path, out_path, wo_tumor)
    ME.find_mutual_excl_markers()



if __name__ == '__main__':
    main()

