import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from fcsy.fcs import write_fcs
import os

def get_parser():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Extract features based on region-selected cell and nucleus mask')

    parser.add_argument(
        '--path', dest='path', required=True,
        help='Path to data of all samples')

    parser.add_argument(
        '--features_extracted', dest='features_extracted', action='store_true')

    return parser

def split(word: str):
    return [char for char in word]

class FeatureMatrix:

    def __init__(self, input_path: str):
        self._input_path = input_path
        self._counter = 0

    def load_data(self, fov_path: str):

        with open(fov_path + '/processed/dataset.pickle', 'rb') as f:
            msobject_dataset = pickle.load(f)

        return msobject_dataset

    def create_header(self, dataset: list):
        header = []
        for _i in dataset[0]._images:
            header.extend(_i.get_feature_names())

        return header

    def extract_feature_matrix(self, dataset: list):

        feature_matrix = []

        for i in range(0, len(dataset)):
            temp=dataset[i].get_pat_id().split('_')
            pat_id = int(temp[0].split('BM')[1]) * 10
            sample_id = int(temp[1])
            feature_matrix.append(np.append(dataset[i].get_features(True), float(pat_id+sample_id)))
            feature_matrix[i] = np.append(feature_matrix[i], float(dataset[i].get_mean_y()))
            feature_matrix[i] = np.append(feature_matrix[i], float(dataset[i].get_mean_x()))
            feature_matrix[i] = np.append(feature_matrix[i], float(dataset[i].get_idx_img()))
            feature_matrix[i] = np.append(feature_matrix[i], float(dataset[i].get_idx_obj()))
            feature_matrix[i] = np.append(feature_matrix[i], self._counter)

        return feature_matrix

    def create_dataframe(self, feat_ext: bool=False):
        if feat_ext:
            with open(self._input_path + '/analysis/features_large_df.pickle', 'rb') as f:
                features_df = pickle.load(f)
        else:
            runs = [_r for _r in os.listdir(self._input_path) if split(_r)[0] == 'B']

            if not os.path.exists(self._input_path + '/analysis'):
                os.mkdir(self._input_path + '/analysis')

            header = []
            features = []

            for _r in runs:
                fov_folders = os.listdir(self._input_path + '/' + _r)

                for _f in fov_folders:
                    data = self.load_data(self._input_path + '/' + _r + '/' + _f)
                    if not header:
                        header = self.create_header(data)
                    features.extend(self.extract_feature_matrix(data))
                    self._counter += 1
            
            header.extend(['pat_id', 'y_mean', 'x_mean', 'FoV', 'obj_idx', 'batch'])
            features_df = pd.DataFrame(features, columns=header)

            with open(self._input_path + '/analysis/features_large_df.pickle', 'wb') as f:
                pickle.dump(features_df, f)

            return features_df

    def devide_AB(self, secondary_ab: str, new_name: str, dataframe: pd.DataFrame):
        '''
        Devide 2nd secondary AB by first one (negative control) to increase signal to noise ratio
        '''
        df_with_header = dataframe
        header = [col for col in df_with_header.columns]

        patients = list(set(df_with_header.pat_id))

        features = [h for h in header if secondary_ab in h]
        features_1 = features[0: len(features) // 2]
        features_2 = features[len(features) // 2:]

        for i in range(0, len(features) // 2):
            if int(features_1[i].split('_')[0]) < int(features_2[i].split('_')[0]):
                feature = df_with_header[features_2[i]] / df_with_header[features_1[i]]
            else:
                feature = df_with_header[features_1[i]] / df_with_header[features_2[i]]

            column_name = new_name + '_' + features_1[i].split('_')[-2] + '_' + features_1[i].split('_')[-1]
            df_with_header[column_name] = feature

        headers_to_remove = [h for h in header if secondary_ab in h]
        df_with_header = df_with_header.drop(headers_to_remove, axis=1)

        return df_with_header

    def normalize_features(self, features: np.ndarray):
        f_normalized = RobustScaler(quantile_range=(1, 99)).fit_transform(features)

        return f_normalized

    def edit_feature_matrix(self, normalize: bool=False, features_extracted: bool=False):
        feat_df = self.create_dataframe(feat_ext = features_extracted)

        headers = [col for col in feat_df.columns]

        # Delete unconjugated antibodies
        headers_to_remove = [h for h in headers if 'FAIM2' in h or 'CD279' in h or 'Vim' in h]
        feat_df = feat_df.drop(headers_to_remove, axis=1)
        headers = [col for col in feat_df.columns]

        # Eliminate cells that have a bigger nuclear mask than cell mask
        # Eliminate incorrectly segmented cells (outliers)
        idx_nucleus_size = [i for i in range(0, len(headers)) if 'Propidium iodide_Size_nucleus' in headers[i]]
        idx_cell_size = [i for i in range(0, len(headers)) if 'Propidium iodide_Size_cell' in headers[i]]
        idx = []
        features = feat_df.values
        #lower_size_limit = np.percentile(features[:, idx_nucleus_size], 0.166)
        #upper_size_limit = np.percentile(features[:, idx_nucleus_size], 99.24)
        for i in range(0, features.shape[0]):
            if features[i, idx_nucleus_size] > features[i, idx_cell_size]:
                idx.append(i)
            #elif features[i, idx_nucleus_size] < lower_size_limit or features[i, idx_nucleus_size] > upper_size_limit:
            #    idx.append(i)
        features = np.delete(features, idx, axis=0)
        feat_df = pd.DataFrame(features, columns=headers)
        #feat_df.to_csv(r'/workspace/data/analysis/feature_matrix_debug.csv', sep=';', index=False, decimal=',')
        sec_abs = ['anti-Rabbit', 'Biotin', 'anti-Chicken']
        new_names = ['FAIM2', 'PD-1', 'Vimentin']
        for i in range(0, len(sec_abs)):
            feat_df = self.devide_AB(sec_abs[i], new_names[i], feat_df)

        if normalize:
            df = feat_df.copy()
            df = df.drop(['pat_id', 'y_mean', 'x_mean', 'FoV', 'obj_idx', 'batch'], axis=1)
            features_norm = self.normalize_features(df.values)

            df_short = pd.DataFrame(features_norm, columns=[col for col in df.columns])
            df_large = df_short.copy()
            df_large['pat_id'], df_large['y_mean'], df_large['x_mean'], df_large['FoV'], df_large['obj_idx'], \
            df_large['batch'] = feat_df['pat_id'], feat_df['y_mean'], feat_df['x_mean'], feat_df['FoV'], \
                                feat_df['obj_idx'], feat_df['batch']
            feat_df = df_large.copy()

        return feat_df


def reduce_header_length(headers: np.ndarray):
    new_headers = []
    for h in headers:

        if h == 'pat_id' or h == 'y_mean' or h == 'x_mean' or h == 'FoV' or h == 'obj_idx' or h == 'batch':
            new_headers.append(h)
        else:
            temp = h.split('_')
            if (temp[-2] == 'MeanXHighestPercent'):
                abbr = 'MXP'
            else:
                abbr = split(temp[-2])[0] + split(temp[-2])[1]
            if ('PD' in temp[-3].split('-')[0] or 'HLA' in temp[-3].split('-')[0]):
                new_headers.append(temp[-3].split('-')[0] + '-' + temp[-3].split('-')[1] + '_' + abbr + '_' + split(temp[-1])[0])
            else:
                new_headers.append(temp[-3].split('-')[0] + '_' + abbr + '_' + split(temp[-1])[0])

    return new_headers

def main(args):
    path = args.path
    print (path)
    features_ext = args.features_extracted

    f = FeatureMatrix(path)
    df_complete = f.edit_feature_matrix(normalize=False, features_extracted=features_ext)
    # Normalization over the whole dataset leads to batch effect, as shown in our publication.
    df = df_complete.copy()
    df = df.drop(['pat_id', 'y_mean', 'x_mean', 'FoV', 'obj_idx', 'batch'], axis=1)

    df.columns = reduce_header_length([col for col in df.columns])
    df_complete.columns = reduce_header_length([col for col in df_complete.columns])

    df_complete.to_csv(path + '/analysis/feature_matrix_large.csv', sep=';', index=False, decimal=',')
    df.to_csv(path + '/analysis/feature_matrix_short.csv', sep=';', index=False, decimal=',')
    #write_fcs(df_complete, path + '/analysis/feature_matrix_large.fcs')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)