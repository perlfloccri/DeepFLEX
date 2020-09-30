import os
import numpy as np
import holoviews as hv
hv.extension('bokeh')
from collections import defaultdict
from fcsy.fcs import write_fcs
from sklearn.preprocessing import MinMaxScaler
from sklearn import cluster
from sklearn import mixture
from scipy.stats import gaussian_kde
from ssc.cluster import selfrepresentation as sr
import pandas as pd
import random
import sys


def nested_dict():
    """
    A nested dictionary for hierarchical storage of thresholds.
    """
    return defaultdict(nested_dict)


class Normalization:
    """
    Automated intensity normalization framework for cycIF imaging datasets.

    Parameters
    ----------

    data: pandas DataFrame
        Contains batch ids, scene ids, cell ids, and marker intensities.
    marker_pairs: list of lists
        Each sub-list contains a marker and its exclusive counterpart marker.
    save_dir: string
        Path to directory for saving results
    manual_thresh: nested dict, optional
        Manual thresholds for adding to figures

    Attributes
    ----------
    threshs: pandas Dataframe

    """

    def __init__(self, data, marker_status, marker_pairs, save_dir, save_figs=False, manual_threshs=None):
        self.data = data
        self.marker_status = marker_status
        self.marker_pairs = marker_pairs

        self.threshs = pd.DataFrame({'batch': [], 'scene': [], 'model': []})
        for marker_pair in self.marker_pairs:
            self.threshs[marker_pair[0]] = np.nan

        self.save_dir = save_dir
        self.save_figs = save_figs
        self.manual_threshs = manual_threshs
        self._counter = 1

    def get_thresh_curve(self, data, marker_pair, thresh, color, batch):
        """
        Get holoview curve from thresh

        Parameters
        ----------
        data: pandas Dataframe
            Contains batch ids, scene ids, cell ids, and marker intensities.
        marker_pair: list
            A two-element list of marker names
        thresh: int (?)

        """

        p = int(round(data[:, 1].max()))
        x = np.array([thresh] * p)[:, np.newaxis]
        y = np.arange(0, p)[:, np.newaxis]

        xlabel = marker_pair[0]
        ylabel = marker_pair[1]

        xmax = data[:, 0].max()
        xlim = (0, max(thresh * 1.1, xmax))

        ymax = data[:, 1].max()
        ylim = (0, ymax)

        if batch == 'global':
            line_dash = 'solid'
        else:
            line_dash = 'dotted'

        curve = hv.Curve(np.hstack((x, y))).opts(xlabel=xlabel,
                                                 ylabel=ylabel,
                                                 xrotation=45,
                                                 xlim=xlim,
                                                 ylim=ylim,
                                                 tools=['hover'],
                                                 show_legend=False,
                                                 line_width=2,
                                                 color=color,
                                                 line_dash=line_dash)
        return curve

    def get_GMM_thresh(self, data, marker_pair, model, sigma_weight, color, batch):
        """
        sigma_weight: int, Weighting factor for sigma, where higher value == fewer cells classified as positive.
        """

        model.fit(data)

        neg_idx = np.argmax([np.diagonal(i)[1] for i in model.covariances_]) # Diagonal of covariance matrix = variance

        mu = model.means_[neg_idx, 0]

        # Extract sigma from covariance matrix
        sigma = np.sqrt(np.diagonal(model.covariances_[neg_idx])[0])

        thresh = mu + sigma_weight * sigma

        curve = self.get_thresh_curve(data, marker_pair, thresh, color, batch)

        return thresh, curve

    def get_clustering_thresh(self, data, marker_pair, model, sigma_weight, color, batch):
        """
        sigma_weight: int, Weighting factor for sigma, where higher value == fewer cells classified as positive.
        """
        model.fit(data)

        clusters = [
            data[model.labels_.astype('bool')],
            data[~model.labels_.astype('bool')]
        ]

        # Identify negative cluster based on maximum std on y-axis
        neg_cluster = clusters[np.argmax([i[:, 1].std() for i in clusters])]

        mu = neg_cluster[:, 0].mean()
        sigma = neg_cluster[:, 0].std()

        thresh = mu + sigma_weight * sigma

        curve = self.get_thresh_curve(data, marker_pair, thresh, color, batch)

        return thresh, curve

    def get_marker_pair_thresh(self, data, scene, marker_pair, batch):

        marker_pair_data = data[marker_pair].to_numpy()

        xlabel = marker_pair[0]
        ylabel = marker_pair[1]

        curves = []

        if batch != 'global' and marker_pair[0].split('_')[0] in self.marker_status[(self.marker_status.batch == str(batch)) & (self.marker_status.scene == scene)]:
            m_st = self.marker_status[(self.marker_status.batch == batch) & (self.marker_status.scene == scene)][marker_pair[0].split('_')[0]].values
        else:
            m_st = 1

        if m_st == 0:
            thresh = marker_pair_data[:, 0].max()

            self.threshs.loc[(self.threshs.scene == scene) & (self.threshs.batch == batch) & (
                    self.threshs.model == 'GMM'), xlabel] = thresh
            self.threshs.loc[(self.threshs.scene == scene) & (self.threshs.batch == batch) & (
                        self.threshs.model == 'KMeans'), xlabel] = thresh
            self.threshs.loc[(self.threshs.scene == scene) & (self.threshs.batch == batch) & (
                    self.threshs.model == 'SSC'), xlabel] = thresh
            curve = self.get_thresh_curve(marker_pair_data, marker_pair, thresh, 'black', batch)
            curves.append(curve)

        else:

            models = (
                ('KMeans', 'magenta', cluster.KMeans(n_clusters=2)),
                ('GMM', 'blue', mixture.GaussianMixture(n_components=2, n_init=10)),
                ('SSC', 'green', sr.SparseSubspaceClusteringOMP(n_clusters=2))
            )
            #models = (
            #    ('SSC', 'green', sr.SparseSubspaceClusteringOMP(n_clusters=2))
            #)

            sigma_weight = 0  # TODO: parameterize

            for name, color, model in models:

                if name == 'GMM':

                    thresh, curve = self.get_GMM_thresh(marker_pair_data,
                                                        marker_pair,
                                                        model,
                                                        sigma_weight,
                                                        color,
                                                        batch)

                else:

                    thresh, curve = self.get_clustering_thresh(marker_pair_data,
                                                            marker_pair,
                                                            model,
                                                            sigma_weight,
                                                            color,
                                                            batch)

                curves.append(curve)

                self.threshs.loc[(self.threshs.scene == scene) & (self.threshs.batch == batch) & (
                        self.threshs.model == name), xlabel] = thresh

        if batch == 'global':
            scatter = hv.Scatter(marker_pair_data).opts(xlabel=xlabel.split('_')[0],
                                                        ylabel=ylabel.split('_')[0],
                                                        xrotation=45,
                                                        show_legend=False,
                                                        alpha=0.2,
                                                        color='gray')
        else:

            if m_st == 1:
                # Get kde for plotting density of local scatter plot
                marker_pair_data_T = marker_pair_data.T
                z = gaussian_kde(marker_pair_data_T)(marker_pair_data_T)
                marker_pair_data = np.vstack((marker_pair_data_T, z)).T

                scatter = hv.Scatter(marker_pair_data, vdims=['y', 'z']).opts(title=str(batch),
                                                                            xlabel=xlabel.split('_')[0],
                                                                            ylabel=ylabel.split('_')[0],
                                                                            xrotation=45,
                                                                            show_legend=False,
                                                                            color='z')
            else:
                scatter = hv.Scatter(marker_pair_data).opts(title=str(batch),
                    xlabel=str(batch) + '_' + xlabel.split('_')[0],
                    ylabel=ylabel.split('_')[0],
                    xrotation=45,
                    show_legend=False,
                    color='red')

        return scatter, curves

    def save_thresh_figs(self, scene_scatters, scene, save_dir, feat):
        """
        Save hv figures for all marker pairs for a given scene as an hv.Layout()
        
        if 'To' in feat: #split html files to prevent memory error
            for i in range(len(scene_scatters)):
                final_fig = hv.Layout(scene_scatters[i: (i+1)]).opts(title=scene, shared_axes=False).cols(3)
                hv.save(final_fig, f'{save_dir}/{scene}_{feat}_gate_dist_plot_{i}.html')
        else:
            final_fig = hv.Layout(scene_scatters).opts(title=scene, shared_axes=False).cols(3)
            hv.save(final_fig, f'{save_dir}/{scene}_{feat}_gate_dist.html')
        """
    def visualize_scene(self, scene):
        """
        Threshold prediction and figure generation
        """

        scene_data = self.data[self.data.scene == scene]

        model_names = ['KMeans', 'GMM', 'SSC']
        batches = set(scene_data.batch)
        batches.add('global')
        for batch in batches:
            for m in model_names:
                self.threshs = self.threshs.append({'batch': batch, 'scene': scene, 'model': m},
                                                ignore_index=True)

        scene_scatters = []
        feature = (self.marker_pairs[0][0].split('_')[1] + '_' + self.marker_pairs[0][0].split('_')[2])

        for marker_pair in self.marker_pairs:

            if (marker_pair[0].split('_')[1] + '_' + marker_pair[0].split('_')[2]) != feature:

                if self.save_figs:
                    self.save_thresh_figs(scene_scatters, str(scene), self.save_dir, feature)
                scene_scatters = []

            print(str(self._counter) + '/' + str(437*9) + ': Processing scene = ' + str(scene) + ', batch = ' +
                  'global' + ' and marker_pair = ' + str(marker_pair))

            global_scatter, global_curves = self.get_marker_pair_thresh(scene_data,
                                                            scene,
                                                            marker_pair,
                                                            'global')

            for batch in set(scene_data.batch):
                print(str(self._counter)
                      + '/' + str(437*9) + ': Processing scene = ' + str(scene) + ', batch = ' +
                      str(batch) + ' and marker_pair = ' + str(marker_pair))
                self._counter += 1
                batch_scene_data = scene_data[scene_data.batch == batch]

                local_scatter, local_curves = self.get_marker_pair_thresh(batch_scene_data,
                                                                          scene,
                                                                          marker_pair,
                                                                          batch)

                scene_scatters.append(global_scatter * local_scatter * hv.Overlay(local_curves) * hv.Overlay(global_curves))

            feature = (marker_pair[0].split('_')[1] + '_' + marker_pair[0].split('_')[2])


    def predict_thresh(self):

        os.makedirs(self.save_dir, exist_ok=True)

        scenes = set(self.data.scene)
        for s in scenes:
            self.visualize_scene(s)

        self.threshs.to_csv(self.save_dir + '/threshs.csv', sep=';', index=False, decimal=',')

    def normalize_scene(self):
        '''
        normalize scene with predicted thresholds - all background values become < 1 and all foreground values > 1
        '''
        normalized_data = self.data.copy()
        scenes = set(self.data.scene)
        for s in scenes:
            scene_data = self.data[self.data.scene == s]

            for marker_pair in self.marker_pairs:
                for batch in set(scene_data.batch):
                    threshold = self.threshs[(self.threshs.batch == str(batch)) & (self.threshs.scene == s) & (self.threshs.model =='SSC')][marker_pair[0]]
                    threshold = threshold.values[0]
                    data = scene_data[scene_data.batch == batch][marker_pair[0]]
                    normalized_values = data/threshold
                    normalized_data.loc[(normalized_data.scene == s) & (normalized_data.batch == batch), marker_pair[0]] = normalized_values

        return normalized_data

    def scale_data(self, features_df: pd.DataFrame):
        '''
        All predicted background values are set to random values in the interval [0, 0.02] and foreground values are set to [0.02,1].
        Morphological features are scaled between 0 and 1.
        '''

        def scale(arr, new_min, new_max, lower_perc, upper_perc):
            arr_min = np.percentile(arr, lower_perc)
            arr_max = np.percentile(arr, upper_perc)
            if arr.size == 1:
                new_arr = arr/arr_max
            else:
                arr_delta = arr_max - arr_min
                new_delta = new_max - new_min
                tmp = (arr - arr_min)/arr_delta
                tmp[tmp < 0] = 0
                tmp[tmp > 1] = 1
                new_arr = tmp * new_delta + new_min

            return new_arr

        scenes = list(set(features_df.scene))
        list_df = []

        for s in scenes:
            df = features_df[features_df.scene == s].copy()
            df_short = df.drop(['batch', 'y_mean', 'x_mean', 'scene', 'cell', 'FoV_counter'], axis=1)
            df_ar = df_short.values
            for i in range(0, df_ar.shape[1]):
                if 'Propidium' not in df.columns[i]:
                    BG_co = np.where(df_ar[:, i] <= 1)
                    FG_co = np.where(df_ar[:, i] > 1)
                    df_ar[BG_co, i] = [random.uniform(0, 0.02) for i in range(len(BG_co[0]))]
                    if len(FG_co[0]) != 0:
                        df_ar[FG_co, i] = scale(df_ar[FG_co, i], 0.02, 1, 0, 100)
                else:
                    df_ar[:, i] = MinMaxScaler().fit_transform(df_ar[:, i].reshape(-1, 1))[:, 0]

            df_scaled = pd.DataFrame(df_ar, columns=[col for col in df_short.columns])
            df_scaled.index = df.index
            df_scaled[['batch', 'y_mean', 'x_mean', 'scene', 'cell', 'FoV_counter']] = df[
                ['batch', 'y_mean', 'x_mean', 'scene', 'cell', 'FoV_counter']]

            list_df.append(df_scaled)

        df_scaled_all = pd.concat(list_df)
        # Intensity features of nuclear stain PI are eliminated since these must not influence the result.
        headers_2_drop = [col for col in df_scaled_all.columns if col.split('_')[0] == 'Propidium iodide' and (col.split('_')[1] == 'MXP' or col.split('_')[1] == 'Me' or col.split('_')[1] == 'To' or (col.split('_')[1] == 'Ro' and col.split('_')[2] == 'c') or (col.split('_')[1] == 'So' and col.split('_')[2] == 'c'))]
        df_scaled_all = df_scaled_all.drop(headers_2_drop, axis=1)
        df_scaled_short = df_scaled_all.drop(['batch', 'y_mean', 'x_mean', 'cell', 'FoV_counter', 'scene'], axis=1)

        return df_scaled_all, df_scaled_short

    def scale_per_batch(self, batches: list, data: pd.DataFrame):

        data_all_batches = []
        for b in batches:
            df = data[data.batch == b]
            df_scaled, df_scaled_short = self.scale_data(df)

            df_scaled.to_csv(self.save_dir + '/' + str(b) + '_long.csv', sep=';', index=False, decimal=',')
            df_scaled_short.to_csv(self.save_dir + '/' + str(b) + '_short.csv', sep=';',
                             index=False, decimal=',')

            write_fcs(df_scaled, self.save_dir + '/' + str(b) + '_long.fcs')
            write_fcs(df_scaled_short, self.save_dir + '/' + str(b) + '_short.fcs')

            data_all_batches.append(df_scaled)

        scaled_data = pd.concat(data_all_batches)

        return scaled_data

    def run(self, thresh_gen: bool = False):
        os.makedirs(self.save_dir, exist_ok=True)

        #if thresh_gen:
        #    self.threshs = pd.read_csv(self.save_dir + '/threshs.csv', sep=';', decimal=',')
        #else:
        if not thresh_gen:
            self.predict_thresh()
        self.threshs = pd.read_csv(self.save_dir + '/threshs.csv', sep=';', decimal=',')

        norm_data = self.normalize_scene()
        batches = list(set(norm_data.batch))
        scaled_data = self.scale_per_batch(batches, norm_data)

        scaled_data.to_csv(self.save_dir + '/all_batches_long.csv', sep=';',
                           index=False, decimal=',')

        write_fcs(scaled_data, self.save_dir + '/all_batches_long.fcs')