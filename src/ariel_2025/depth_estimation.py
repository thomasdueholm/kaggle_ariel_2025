import os

#os.environ["JAX_PLATFORM_NAME"] = "cpu"

import matplotlib.pyplot as plt
import numpy as np
import jax
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
import pickle

from ariel_2025 import data_loader, util, spectrum_estimation, limb_darkening


def get_extended_airs_target(target, star_info, from_rows=0, to_rows=None):
    target = target.values[:, 1:]
    n_rows = np.sum(star_info['multiplicity'])
    extended_target = np.zeros((n_rows, target.shape[-1]))
    extended_weights = np.zeros(n_rows)
    extended_planet_ids = np.zeros(n_rows, dtype=np.int64)

    current_index = 0
    for i, (multiplicity, planet_id) in enumerate(zip(star_info['multiplicity'], star_info['planet_id'])):
        extended_target[current_index: current_index+multiplicity, :] = target[i, :]
        extended_weights[current_index: current_index+multiplicity] = 1 / multiplicity
        extended_planet_ids[current_index: current_index+multiplicity] = planet_id
        current_index += multiplicity

    if to_rows is None:
        to_rows = extended_target.shape[0]

    extended_target_airs = extended_target[from_rows: to_rows, 1:].flatten()
    extended_target_fgs1 = extended_target[from_rows: to_rows, 0]
    extended_weights = extended_weights[from_rows: to_rows]

    return extended_target_airs, extended_target_fgs1, extended_weights, extended_planet_ids


class DepthEstimator:

    def __init__(self, config, use_pca=False):
        self.config = config
        self.use_pca = use_pca

        self.pca_spectrum_estimator = None
        self.airs_estimator = None
        self.fgs1_estimator = None

    def get_data(self, data_dict):
        df_limb_darkening = data_dict['df_limb_darkening']
        limb_darkening_r_airs = data_dict['limb_darkening_r_airs']

        limb_darkening_spectrum = spectrum_estimation.get_simple_spectrum(
            limb_darkening_r_airs ** 2,
            smoothing_window=self.config['spectrum_smoothing_window'],
            level_change_scaler=self.config['spectrum_level_change_scaler'],
            trend_change_scaler=self.config['spectrum_trend_change_scaler'],
            noise_scaler=self.config['spectrum_noise_scaler']
        )

        limb_darkening_spectrum_mean = np.mean(limb_darkening_spectrum, axis=-1, keepdims=True)
        limb_darkening_spectrum = limb_darkening_spectrum - limb_darkening_spectrum_mean

        data_dict = {
            'r_airs_combined': df_limb_darkening['r_airs_combined'].values[:, None] ** 2,
            'r_fgs1': df_limb_darkening['r_fgs1'].values[:, None] ** 2,
            'r_airs': limb_darkening_spectrum,
            'limb_darkening_spectrum_mean': limb_darkening_spectrum_mean
        }

        if self.use_pca:
            r_airs_pca = self.pca_spectrum_estimator.predict(limb_darkening_spectrum)
            data_dict['r_airs_pca'] = r_airs_pca

        return data_dict

    def fit_predict_airs(self, data_dict, target_dict=None):
        ##### DATA ####################################################

        baseline_columns = ['limb_darkening_spectrum_mean', 'r_airs_combined'] #

        if self.use_pca:
            spectrum_columns = ['r_airs', 'r_airs_pca']#,
        else:
            spectrum_columns = ['r_airs']  # ,
        combined_columns = baseline_columns + spectrum_columns

        baseline_data = np.concatenate([data_dict[name] for name in baseline_columns], axis=-1)
        spectrum_data = np.stack([data_dict[name] for name in spectrum_columns], axis=-1)

        spectrum_length = spectrum_data.shape[1]
        baseline_data = np.repeat(baseline_data[:, None, :], spectrum_length, axis=1)

        baseline_data = baseline_data.reshape((-1, baseline_data.shape[-1]))
        spectrum_data = spectrum_data.reshape((-1, spectrum_data.shape[-1]))

        combined_data = np.concatenate([baseline_data, spectrum_data], axis=-1)

        ##### OPTIMIZATION ############################################

        if target_dict is not None:
            target_airs = target_dict['airs'].flatten()
            target_weights = target_dict['airs_weights'].flatten()

            self.airs_estimator = LinearRegression(positive=True)
            self.airs_estimator.fit(combined_data, target_airs, sample_weight=target_weights)

            print()
            for i, name in enumerate(combined_columns):
                print(name, self.airs_estimator.coef_[i])

        prediction = self.airs_estimator.predict(combined_data)
        prediction = prediction.reshape((-1, spectrum_length))

        return prediction

    def fit_predict_fgs1(self, data_dict, target_dict=None):
        ##### DATA ####################################################

        baseline_columns = ['r_fgs1'] #,

        baseline_data = np.concatenate([data_dict[name] for name in baseline_columns], axis=-1)

        ##### OPTIMIZATION ############################################

        if target_dict is not None:
            target_fgs1 = target_dict['fgs1']
            target_weights = target_dict['fgs1_weights']

            self.fgs1_estimator = LinearRegression()
            self.fgs1_estimator.fit(baseline_data, target_fgs1, sample_weight=target_weights)

            print()
            for i, name in enumerate(baseline_columns):
                print(name, self.fgs1_estimator.coef_[i])

        prediction = self.fgs1_estimator.predict(baseline_data)

        return prediction

    def fit(self, data_dict, target_dict):
        if self.use_pca:
            self.pca_spectrum_estimator = spectrum_estimation.PCASpectrumEstimator(self.config.get('pca_threshold', 0.99))
            self.pca_spectrum_estimator.fit(target_dict['airs'].reshape(len(data_dict['df_depth']), -1))

        data_dict = self.get_data(data_dict)

        self.fit_predict_airs(data_dict, target_dict)
        self.fit_predict_fgs1(data_dict, target_dict)

    def predict(self, data_dict):
        data_dict = self.get_data(data_dict)

        airs_prediction = self.fit_predict_airs(data_dict)
        fgs1_prediction = self.fit_predict_fgs1(data_dict)

        return airs_prediction, fgs1_prediction

    def save_checkpoint(self, checkpoint_name):
        airs_estimator_params = {
            'coef_': self.airs_estimator.coef_,
            'intercept_': self.airs_estimator.intercept_
        }

        fgs1_estimator_params = {
            'coef_': self.fgs1_estimator.coef_,
            'intercept_': self.fgs1_estimator.intercept_
        }

        params = {
            'airs_estimator_params': airs_estimator_params,
            'fgs1_estimator_params': fgs1_estimator_params
        }

        if self.use_pca:
            pca_params = self.pca_spectrum_estimator.get_params()
            params['pca_params'] = pca_params

        file_name = f'{checkpoint_name}.pkl'

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_checkpoint(self, checkpoint_name):
        file_name = f'{checkpoint_name}.pkl'

        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        self.airs_estimator = LinearRegression()
        self.airs_estimator.coef_ = params['airs_estimator_params']['coef_']
        self.airs_estimator.intercept_ = params['airs_estimator_params']['intercept_']

        self.fgs1_estimator = LinearRegression()
        self.fgs1_estimator.coef_ = params['fgs1_estimator_params']['coef_']
        self.fgs1_estimator.intercept_ = params['fgs1_estimator_params']['intercept_']

        if self.use_pca:
            self.pca_spectrum_estimator = spectrum_estimation.PCASpectrumEstimator(self.config.get('pca_threshold', 0.99))
            self.pca_spectrum_estimator.set_params(params['pca_params'])


def out_of_sample_evaluation(config, data_dict, target, use_pca=False, n_splits=5, n_repeats=1, random_state=42, outlier_planet_ids=None):
    if outlier_planet_ids is None:
        outlier_planet_ids = []

    def combine_data(is_incomplete, star_info):
        combined_is_incomplete = np.zeros(len(star_info), dtype=bool)

        for i, (multiplicity, first_index) in enumerate(zip(star_info['multiplicity'], star_info['first_index'])):
            combined_is_incomplete[i] = np.any(is_incomplete[first_index: first_index + multiplicity])

        return combined_is_incomplete

    def get_data(indices, data_dict, target, outlier_planet_ids=None):
        # indices are assumed to be sorted.

        if outlier_planet_ids is None:
            outlier_planet_ids = []

        star_info = data_dict['star_info']
        df_depth = data_dict['df_depth']

        outlier_planet_ids = np.array(outlier_planet_ids)
        selected_planet_ids = star_info.loc[indices, 'planet_id'].values

        outliers = np.isin(selected_planet_ids, outlier_planet_ids)
        indices = indices[~outliers]

        star_info = star_info.loc[indices]
        target = target.loc[indices]

        extension_mask = np.zeros(len(df_depth), dtype=bool)
        new_first_index = np.zeros(len(indices), dtype=np.int32)
        current_index = 0
        for i, (multiplicity, first_index) in enumerate(zip(star_info['multiplicity'], star_info['first_index'])):
            extension_mask[first_index: first_index + multiplicity] = True
            new_first_index[i] = current_index
            current_index += multiplicity

        star_info['first_index'] = new_first_index

        selected_data_dict = {
            'star_info': star_info
        }
        for key, value in data_dict.items():
            if key == 'star_info':
                continue
            if isinstance(value, np.ndarray):
                selected_data_dict[key] = value[extension_mask, ...]
            else:
                selected_data_dict[key] = value.loc[extension_mask]

        target_airs, target_fgs1, target_weights, planet_ids = get_extended_airs_target(target, star_info)

        spectrum_length = len(target_airs) // len(target_fgs1)

        target_dict = {
            'airs_weights': np.repeat(target_weights[:, None], spectrum_length, axis=1),
            'fgs1_weights': target_weights,
            'airs': target_airs,
            'fgs1': target_fgs1,
            'planet_ids': planet_ids
        }

        return selected_data_dict, target_dict, target

    star_info = data_dict['star_info']

    rskf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    fgs1_prediction_list = []
    airs_prediction_list = []
    target_list = []
    index_list = []

    config = config.copy()
    config['sigma_scale'] = 1.0

    for i, (train_index, test_index) in enumerate(rskf.split(star_info)):
        tick = datetime.datetime.now()

        data_dict_train, target_dict_train, _ = get_data(train_index, data_dict, target, outlier_planet_ids)
        data_dict_test, target_dict_test, _ = get_data(test_index, data_dict, target, outlier_planet_ids)

        depth_estimator = DepthEstimator(config, use_pca=use_pca)
        depth_estimator.fit(data_dict_train, target_dict_train)

        airs_prediction, fgs1_prediction = depth_estimator.predict(data_dict_test)

        fgs1_prediction_list.append(fgs1_prediction)
        airs_prediction_list.append(airs_prediction)
        target_list.append(target_dict_test)
        index_list.append(test_index)

        fgs1_error = np.average((fgs1_prediction - target_dict_test['fgs1']) ** 2, weights=target_dict_test['fgs1_weights'])
        airs_error = np.average((airs_prediction - target_dict_test['airs'].reshape(airs_prediction.shape)) ** 2, weights=target_dict_test['airs_weights'])

        print(f'Fold {i + 1} of {n_splits * n_repeats}:   {airs_error}   {fgs1_error}   {datetime.datetime.now() - tick}')

    combined_airs_prediction = np.concatenate(airs_prediction_list, axis=0)
    combined_fgs1_prediction = np.concatenate(fgs1_prediction_list, axis=0)
    combined_target = {}
    for key in target_list[0].keys():
        combined_target[key] = np.concatenate([t[key] for t in target_list], axis=0)

    combined_index = np.concatenate(index_list, axis=0)

    fgs1_error = combined_fgs1_prediction - combined_target['fgs1']
    airs_error = combined_airs_prediction - combined_target['airs'].reshape(combined_airs_prediction.shape)

    return fgs1_error, airs_error, combined_index, combined_airs_prediction, combined_fgs1_prediction, combined_target


def plot_prediction(use_pca=False, n_splits=5, n_repeats=3, random_state=17):
    outlier_planet_ids = [
        1843015807, 3649218579, 1349926825, 576917580, 3786449677, 2554492145,
        2270815333, 1597331512, 1473180683, 1124834224, 3176118959, 2609891029,
        1267010874, 3928265625, 2557736212, 360841026, 2104402723, 204264160
    ]

    print('n_outliers', len(outlier_planet_ids))

    outlier_planet_ids = None

    fgs1_signal, airs_signal, star_info, target = data_loader.load_data('train')

    cache_dir = util.get_dir('cache')
    df_depth = pd.read_csv(os.path.join(cache_dir, f'train_depth_info.csv'))

    df_limb_darkening, limb_darkening_r_airs, limb_darkening_u_airs = limb_darkening.load_limb_darkening('train')

    target_airs, target_fgs1, target_weights, planet_ids = get_extended_airs_target(target, star_info)

    config = data_loader.get_config()
    config['sigma_scale'] = 1.0

    data_dict_train = {
        'df_depth': df_depth,
        'star_info': star_info,
        'airs_signal': airs_signal,
        'df_limb_darkening': df_limb_darkening,
        'limb_darkening_r_airs': limb_darkening_r_airs,
        'limb_darkening_u_airs': limb_darkening_u_airs
    }

    spectrum_length = airs_signal.shape[1]

    target_dict = {
        'airs_weights': np.repeat(target_weights[:, None], spectrum_length, axis=1),
        'fgs1_weights': target_weights,
        'airs': target_airs,
        'fgs1': target_fgs1
    }

    ##### TEST ################################

    estimator = DepthEstimator(config, use_pca=use_pca)
    estimator.fit(data_dict_train, target_dict)

    airs_prediction, fgs1_prediction = estimator.predict(data_dict_train)

    target_airs = target_airs.reshape(airs_prediction.shape)

    airs_error = np.mean((airs_prediction - target_airs) ** 2)
    fgs1_error = np.mean((fgs1_prediction - target_fgs1) ** 2)

    print('airs_error', airs_error, 'fgs1_error', fgs1_error)

    plt.figure('Prediction scatter')

    plt.scatter(airs_prediction, target_airs, alpha=0.2, c='C0', label='airs')
    plt.scatter(fgs1_prediction, target_fgs1, alpha=0.2, c='C1', label='fgs1')

    plt.legend()

    airs_error = airs_prediction - target_airs
    fgs1_error = fgs1_prediction - target_fgs1

    util.draw_histogram(airs_error, 'C0', 'Airs', f'Error hist')
    util.draw_histogram(fgs1_error, 'C1', 'Fgs1', f'Error hist')

    #################################

    (
        fgs1_error, airs_error, combined_index, combined_airs_prediction, combined_fgs1_prediction, combined_target
    ) = out_of_sample_evaluation(config, data_dict_train, target, use_pca=use_pca, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state, outlier_planet_ids=outlier_planet_ids)

    mean_fgs1_error = np.average(fgs1_error ** 2, weights=combined_target['fgs1_weights'])
    mean_airs_error = np.average(airs_error ** 2, weights=combined_target['airs_weights'])

    print(f'Out-of-sample error: {mean_airs_error}   {mean_fgs1_error}')

    #################################

    plt.figure('Out-of-sample prediction scatter')

    plt.scatter(combined_airs_prediction, combined_target['airs'], alpha=0.2, c='C0', label='airs')
    plt.scatter(combined_fgs1_prediction, combined_target['fgs1'], alpha=0.2, c='C1', label='fgs1')

    plt.legend()

    util.draw_histogram(airs_error, 'C0', 'Airs', f'Out-of-sample error hist')
    util.draw_histogram(fgs1_error, 'C1', 'Fgs1', f'Out-of-sample error hist')

    error = np.sum(np.concatenate([57.846 * fgs1_error[:, None] ** 2, airs_error ** 2], axis=-1), axis=-1) / (57.846 + airs_error.shape[-1])

    outliers = np.argsort(error)[::-1]

    sorted_error = error[outliers]

    for i in range(30*n_repeats):
        idx = outliers[i]
        planet_id = combined_target['planet_ids'][idx]
        print(f'{idx}, {planet_id}: {sorted_error[i]}')

    #################################

    plt.show()


if __name__ == '__main__':
    print(jax.devices())
    print('num_devices:', jax.local_device_count())

    config = data_loader.get_config()

    print(config)

    plot_prediction(use_pca=True, n_splits=5, n_repeats=1, random_state=17)

