import os
import jax.random
import numpy as np
from sklearn.model_selection import RepeatedKFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import datetime
import pickle
import matplotlib.pyplot as plt

from ariel_2025 import util, data_loader, limb_darkening, depth_estimation, flax_training, score, spectrum_estimation


def get_extended_target(star_info, df_target):
    n_rows = np.sum(star_info['multiplicity'])
    extended_target = np.zeros((n_rows, df_target.values.shape[-1]))
    target_weights = np.zeros(n_rows)

    current_index = 0
    for i, multiplicity in enumerate(star_info['multiplicity']):
        extended_target[current_index: current_index+multiplicity, :] = df_target.values[i, :]
        target_weights[current_index: current_index+multiplicity] = 1 / multiplicity

        current_index += multiplicity

    target_dict = {name: extended_target[:, i].astype(df_target[name].values.dtype) for i, name in enumerate(df_target.columns)}
    df_target = pd.DataFrame(target_dict)

    return df_target, target_weights


def get_selected_data(data_dict, df_target, indices=None, outlier_planet_ids=None):
    star_info = data_dict['star_info']
    df_depth = data_dict['df_depth']

    if indices is None:
        indices = np.arange(len(star_info))
    else:
        indices = np.sort(indices)

    if outlier_planet_ids is None:
        outlier_planet_ids = []

    outlier_planet_ids = np.array(outlier_planet_ids)
    selected_planet_ids = star_info.loc[indices, 'planet_id'].values

    outliers = np.isin(selected_planet_ids, outlier_planet_ids)
    indices = indices[~outliers]

    star_info = star_info.loc[indices].reset_index(drop=True)

    extension_mask = np.zeros(len(df_depth), dtype=bool)
    new_first_index = np.zeros(len(indices), dtype=np.int32)
    current_index = 0
    for i, (multiplicity, first_index) in enumerate(zip(star_info['multiplicity'], star_info['first_index'])):
        extension_mask[first_index: first_index + multiplicity] = True
        new_first_index[i] = current_index
        current_index += multiplicity

    star_info['first_index'] = new_first_index

    df_target = df_target.loc[indices].reset_index(drop=True)
    df_target, target_weights = get_extended_target(star_info, df_target)

    selected_data_dict = {
        'star_info': star_info,
    }
    for key, value in data_dict.items():
        if key == 'star_info':
            continue
        if isinstance(value, np.ndarray):
            selected_data_dict[key] = value[extension_mask, ...]
        else:
            selected_data_dict[key] = value.loc[extension_mask]

    spectrum_length = df_target.values.shape[1] - 2

    target_dict = {
        'fgs1_weights': target_weights,
        'airs_weights': np.repeat(target_weights[:, None], spectrum_length, axis=1),
        'fgs1': df_target.values[:, 1],
        'airs': df_target.values[:, 2:],
    }

    return selected_data_dict, target_dict


def get_rolling_std(x):
    vmapped_centered_rolling_mean_and_std = jax.vmap(data_loader.centered_rolling_mean_and_std, in_axes=(0, None, None))
    rolling_std = vmapped_centered_rolling_mean_and_std(x, 29, 2)[1]

    return rolling_std

class StandardDeviationEstimator:

    def __init__(
            self,
            airs_std_estimator_factory,
            fgs1_std_estimator_factory,
            config,
            random_state=42,
            use_pca=False,
            save_analysis_cache=False,
            save_training_cache=False,
            load_analysis_cache=False,
            name='standard_deviation_estimator',
            verbose=True
    ):
        self.airs_std_estimator_factory = airs_std_estimator_factory
        self.fgs1_std_estimator_factory = fgs1_std_estimator_factory
        self.config = config
        self.use_pca = use_pca
        self.save_analysis_cache = save_analysis_cache
        self.save_training_cache = save_training_cache
        self.load_analysis_cache = load_analysis_cache
        self.name = name
        self.file_name = f'{name}.pkl'
        self.n_splits = config.get('standard_deviation_n_splits', 5)
        self.n_repeats = config.get('standard_deviation_n_repeats', 1)
        self.random_state = random_state
        self.verbose=verbose

        self.depth_estimator = None
        self.std_airs = None

        self.airs_scaler = None
        self.airs_std_estimator = None
        self.fgs1_scaler = None
        self.fgs1_std_estimator = None

    def save_data(self, data, file_prefix=None):
        if file_prefix is None:
            file_name = self.file_name
        else:
            file_name = f'{file_prefix}_{self.file_name}'

        file_name = os.path.join(util.get_dir('local_cache'), file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self, file_prefix=None):
        if file_prefix is None:
            file_name = self.file_name
        else:
            file_name = f'{file_prefix}_{self.file_name}'

        file_name = os.path.join(util.get_dir('local_cache'), file_name)
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        return data

    def define_data_columns(self, data_dict, train=False):
        n_rows = len(data_dict['df_depth'])
        spectrum_length = data_dict['mean_airs_prediction'].shape[-1]

        spectrum_std = np.std(data_dict['mean_airs_prediction'], axis=1, keepdims=True)

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

        data_dict_airs = {
            'limb_darkening_spectrum': limb_darkening_spectrum.reshape(-1, 1),
            'mean_airs_prediction': data_dict['mean_airs_prediction'].reshape(-1, 1),
            'airs_prediction': data_dict['airs_prediction'].reshape(-1, 1),
            'diff_airs_prediction': np.abs(data_dict['diff_airs_prediction'].reshape(-1, 1)),
            'std_baseline_airs': np.repeat(data_dict['std_airs'][None, :], n_rows, axis=0).reshape(-1, 1),
            'u_airs': data_dict['limb_darkening_u_airs'].reshape(-1, 1),
            'transit_length': np.repeat(data_dict['df_limb_darkening']['transit_length_airs'].values[:, None], spectrum_length, axis=1).reshape(-1, 1),
            'd_min': np.repeat(data_dict['df_limb_darkening']['d_min_airs'].values[:, None], spectrum_length, axis=1).reshape(-1, 1),
            'airs_nonlinearity': np.repeat(data_dict['df_limb_darkening']['airs_nonlinearity'].values[:, None], spectrum_length, axis=1).reshape(-1, 1),
            'multiplicity': np.clip(np.repeat(data_dict['multiplicity'][:, None], spectrum_length, axis=1).reshape(-1, 1), 1, 2),
            'spectrum_std': np.repeat(spectrum_std, spectrum_length, axis=1).reshape(-1, 1),
            'spectrum_placement': np.repeat(np.arange(spectrum_length)[None, :], n_rows, axis=0).reshape(-1, 1)
        }

        data_dict_fgs1 = {
            'mean_fgs1_prediction': data_dict['mean_fgs1_prediction'].reshape(-1, 1),
            'fgs1_prediction': data_dict['fgs1_prediction'].reshape(-1, 1),
            'diff_fgs1_prediction': np.abs(data_dict['diff_fgs1_prediction'].reshape(-1, 1)),
            'u_fgs1': data_dict['df_limb_darkening']['u_fgs1'].values.reshape(-1, 1),
            'transit_length': data_dict['df_limb_darkening']['transit_length_fgs1'].values.reshape(-1, 1),
            'd_min': data_dict['df_limb_darkening']['d_min_fgs1'].values.reshape(-1, 1),
            'multiplicity': np.clip(data_dict['multiplicity'].reshape(-1, 1), 1,2),
            'spectrum_std': spectrum_std,
        }

        if train or self.save_training_cache:
            data_dict_airs.update({
                'error_airs_mean': data_dict['error_airs_mean'].reshape(-1, 1),
                'error_airs': data_dict['error_airs'].reshape(-1, 1),
                'target_weight_airs': data_dict['target_weight_airs'].reshape(-1, 1),
            })

            data_dict_fgs1.update({
                'error_fgs1_mean': data_dict['error_fgs1_mean'].reshape(-1, 1),
                'error_fgs1': data_dict['error_fgs1'].reshape(-1, 1),
                'target_weight_fgs1': data_dict['target_weight_fgs1'].reshape(-1, 1),
            })

        return data_dict_airs, data_dict_fgs1

    def fit_predict_airs(self, data_dict, train=False):
        columns = ['mean_airs_prediction', 'diff_airs_prediction', 'u_airs', 'transit_length',
                   'd_min', 'multiplicity', 'spectrum_std', 'spectrum_placement', 'std_baseline_airs',
                   'airs_nonlinearity', 'limb_darkening_spectrum']

        data = np.concatenate([data_dict[name] for name in columns], axis=-1)

        if train:
            target_weights = data_dict['target_weight_airs'].flatten()

            self.airs_scaler = StandardScaler()
            self.airs_scaler.fit(data, sample_weight=target_weights)

        data = self.airs_scaler.transform((data))

        if self.save_training_cache:
            target_weights = data_dict['target_weight_airs'].flatten()
            error_airs_mean = data_dict['error_airs_mean'].flatten()

            save_dict = {
                'X': data,
                'y': error_airs_mean,
                'w': target_weights
            }

            if train:
                file_prefix = 'train_airs'
            else:
                file_prefix = 'test_airs'

            self.save_data(save_dict, file_prefix)

        if train:
            target_weights = data_dict['target_weight_airs'].flatten()
            error_airs_mean = data_dict['error_airs_mean'].flatten()

            self.airs_std_estimator = self.airs_std_estimator_factory()
            self.airs_std_estimator.fit(data, error_airs_mean, target_weights)

        standard_deviation = self.airs_std_estimator.predict(data)

        return standard_deviation

    def fit_predict_fgs1(self, data_dict, train=False):
        columns = ['mean_fgs1_prediction', 'diff_fgs1_prediction', 'u_fgs1', 'transit_length', 'd_min', 'multiplicity', 'spectrum_std']

        data = np.concatenate([data_dict[name] for name in columns], axis=-1)

        if train:
            target_weights = data_dict['target_weight_fgs1'].flatten()

            self.fgs1_scaler = StandardScaler()
            self.fgs1_scaler.fit(data, sample_weight=target_weights)

        data = self.fgs1_scaler.transform(data)

        if self.save_training_cache:
            target_weights = data_dict['target_weight_fgs1'].flatten()
            error_fgs1_mean = data_dict['error_fgs1_mean'].flatten()

            save_dict = {
                'X': data,
                'y': error_fgs1_mean,
                'w': target_weights
            }

            if train:
                file_prefix = 'train_fgs1'
            else:
                file_prefix = 'test_fgs1'

            self.save_data(save_dict, file_prefix)

        if train:
            target_weights = data_dict['target_weight_fgs1'].flatten()
            error_fgs1_mean = data_dict['error_fgs1_mean'].flatten()

            self.fgs1_std_estimator = self.fgs1_std_estimator_factory()
            self.fgs1_std_estimator.fit(data, error_fgs1_mean, target_weights)

        standard_deviation = self.fgs1_std_estimator.predict(data)

        return standard_deviation

    def fit(self, data_dict, df_target, outliers=None):
        if outliers is None:
            outliers = []

        data_dict_train, target_dict_train = get_selected_data(data_dict, df_target, outlier_planet_ids=outliers)

        self.depth_estimator = depth_estimation.DepthEstimator(self.config, use_pca=self.use_pca)
        self.depth_estimator.fit(data_dict_train, target_dict_train)

        if self.save_analysis_cache or not self.load_analysis_cache:
            rkf = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)

            star_info = data_dict['star_info']

            data_dict_list = []
            target_dict_list = []
            airs_prediction_list = []
            fgs1_prediction_list = []

            for i, (train_index, test_index) in enumerate(rkf.split(star_info)):
                tick = datetime.datetime.now()

                data_dict_train, target_dict_train = get_selected_data(data_dict, df_target, train_index, outlier_planet_ids=outliers)
                data_dict_test, target_dict_test = get_selected_data(data_dict, df_target, test_index, outlier_planet_ids=None)

                estimator = depth_estimation.DepthEstimator(self.config, use_pca=self.use_pca)
                estimator.fit(data_dict_train, target_dict_train)

                airs_prediction, fgs1_prediction = estimator.predict(data_dict_test)

                data_dict_list.append(data_dict_test)
                target_dict_list.append(target_dict_test)
                airs_prediction_list.append(airs_prediction)
                fgs1_prediction_list.append(fgs1_prediction)

                if self.verbose:
                    fgs1_error = np.average((fgs1_prediction - target_dict_test['fgs1']) ** 2, weights=target_dict_test['fgs1_weights'])
                    airs_error = np.average((airs_prediction - target_dict_test['airs']) ** 2, weights=target_dict_test['airs_weights'])

                    print(f'Fold {i + 1} of {self.n_splits * self.n_repeats}:   fgs1_error={fgs1_error}   {estimator.fgs1_estimator.coef_},   airs_error={airs_error}   {estimator.airs_estimator.coef_}   {datetime.datetime.now() - tick}')

            def concatenate_dict_list(dict_list):
                new_dict = {}
                for key, value in dict_list[0].items():
                    if isinstance(value, np.ndarray):
                        new_dict[key] = np.concatenate([x[key] for x in dict_list], axis=0)
                        #print(key, new_dict[key].shape)
                    else:
                        new_dict[key] = pd.concat([x[key] for x in dict_list], axis=0)
                        #print(key, new_dict[key].values.shape)

                return new_dict

            data_dict = concatenate_dict_list(data_dict_list)
            target_dict = concatenate_dict_list(target_dict_list)
            airs_prediction = np.concatenate(airs_prediction_list, axis=0)
            fgs1_prediction = np.concatenate(fgs1_prediction_list, axis=0)

            mean_airs_prediction = np.zeros_like(airs_prediction)
            mean_fgs1_prediction = np.zeros_like(fgs1_prediction)

            star_info = data_dict['star_info']
            star_info = star_info.reset_index(drop=True)
            first_index = np.zeros(len(star_info), dtype=np.int64)
            extended_planet_ids = np.zeros(len(data_dict['df_depth']), dtype=np.int64)
            extended_multiplicity = np.zeros(len(data_dict['df_depth']), dtype=np.int64)

            i = 0
            for j, (m, planet_id) in enumerate(zip(star_info['multiplicity'], star_info['planet_id'])):
                first_index[j] = i
                mean_airs_prediction[i: i + m, ...] = np.mean(airs_prediction[i: i + m, ...], axis=0, keepdims=True)
                mean_fgs1_prediction[i: i + m] = np.mean(fgs1_prediction[i: i + m], axis=0, keepdims=True)
                extended_planet_ids[i: i + m] = planet_id
                extended_multiplicity[i: i + m] = m

                i += m
            star_info['first_index'] = first_index

            data_dict['star_info'] = star_info
            data_dict['mean_airs_prediction'] = mean_airs_prediction
            data_dict['mean_fgs1_prediction'] = mean_fgs1_prediction
            data_dict['airs_prediction'] = airs_prediction
            data_dict['fgs1_prediction'] = fgs1_prediction
            data_dict['diff_airs_prediction'] = airs_prediction - mean_airs_prediction
            data_dict['diff_fgs1_prediction'] = fgs1_prediction - mean_fgs1_prediction
            data_dict['multiplicity'] = extended_multiplicity

            outlier_mask = np.isin(extended_planet_ids, np.array(outliers))

            error_airs_mean = mean_airs_prediction - target_dict['airs']
            error_airs = airs_prediction - target_dict['airs']
            std_airs = np.sqrt(np.mean(error_airs[~outlier_mask, ...] ** 2, axis=0))

            error_fgs1_mean = mean_fgs1_prediction - target_dict['fgs1']
            error_fgs1 = fgs1_prediction - target_dict['fgs1']

            self.std_airs = std_airs

            data_dict['error_airs'] = error_airs
            data_dict['error_airs_mean'] = error_airs_mean
            data_dict['error_fgs1'] = error_fgs1
            data_dict['error_fgs1_mean'] = error_fgs1_mean
            data_dict['target_weight_airs'] = target_dict['airs_weights']
            data_dict['target_weight_fgs1'] = target_dict['fgs1_weights']
            data_dict['std_airs'] = self.std_airs

            data_dict_airs, data_dict_fgs1 = self.define_data_columns(data_dict, train=True)

            if self.save_analysis_cache:
                save_dict = {
                    'std_airs': std_airs,
                    'data_dict_airs': data_dict_airs,
                    'data_dict_fgs1': data_dict_fgs1
                }

                self.save_data(save_dict, file_prefix='analysis')
        else:
            load_dict = self.load_data(file_prefix='analysis')
            self.std_airs = load_dict['std_airs']
            data_dict_airs = load_dict['data_dict_airs']
            data_dict_fgs1 = load_dict['data_dict_fgs1']

        self.fit_predict_airs(data_dict_airs, train=True)
        self.fit_predict_fgs1(data_dict_fgs1, train=True)

    def combine_predictions(self, airs_prediction, airs_standard_deviation, fgs1_prediction, fgs1_standard_deviation, star_info):
        combined_airs_prediction = np.zeros((len(star_info), airs_prediction.shape[-1]))
        combined_airs_standard_deviation = np.zeros((len(star_info), airs_standard_deviation.shape[-1]))
        combined_fgs1_prediction = np.zeros(len(star_info))
        combined_fgs1_standard_deviation = np.zeros(len(star_info))

        for i, (multiplicity, first_index) in enumerate(zip(star_info['multiplicity'], star_info['first_index'])):
            combined_airs_prediction[i, :] = np.mean(airs_prediction[first_index: first_index+multiplicity, :], axis=0)
            combined_airs_standard_deviation[i, :] = np.mean(airs_standard_deviation[first_index: first_index+multiplicity, :], axis=0)
            combined_fgs1_prediction[i] = np.mean(fgs1_prediction[first_index: first_index+multiplicity])
            combined_fgs1_standard_deviation[i] = np.mean(fgs1_standard_deviation[first_index: first_index+multiplicity])

        return combined_airs_prediction, combined_airs_standard_deviation, combined_fgs1_prediction, combined_fgs1_standard_deviation

    def predict(self, data_dict, df_target=None):
        data_dict = data_dict.copy()

        star_info = data_dict['star_info']

        airs_prediction, fgs1_prediction = self.depth_estimator.predict(data_dict)

        mean_airs_prediction = np.zeros_like(airs_prediction)
        mean_fgs1_prediction = np.zeros_like(fgs1_prediction)
        extended_multiplicity = np.zeros(len(data_dict['df_depth']), dtype=np.int64)

        i = 0
        for j, m in enumerate(star_info['multiplicity']):
            mean_airs_prediction[i: i + m, ...] = np.mean(airs_prediction[i: i + m, ...], axis=0, keepdims=True)
            mean_fgs1_prediction[i: i + m] = np.mean(fgs1_prediction[i: i + m], axis=0, keepdims=True)
            extended_multiplicity[i: i + m] = m

            i += m

        data_dict['mean_airs_prediction'] = mean_airs_prediction
        data_dict['mean_fgs1_prediction'] = mean_fgs1_prediction
        data_dict['airs_prediction'] = airs_prediction
        data_dict['fgs1_prediction'] = fgs1_prediction
        data_dict['diff_airs_prediction'] = airs_prediction - mean_airs_prediction
        data_dict['diff_fgs1_prediction'] = fgs1_prediction - mean_fgs1_prediction
        data_dict['multiplicity'] = extended_multiplicity

        if self.save_training_cache:
            _, target_dict = get_selected_data(data_dict, df_target)

            error_airs_mean = mean_airs_prediction - target_dict['airs']
            error_airs = airs_prediction - target_dict['airs']

            error_fgs1_mean = mean_fgs1_prediction - target_dict['fgs1']
            error_fgs1 = fgs1_prediction - target_dict['fgs1']

            data_dict['error_airs'] = error_airs
            data_dict['error_airs_mean'] = error_airs_mean
            data_dict['error_fgs1'] = error_fgs1
            data_dict['error_fgs1_mean'] = error_fgs1_mean
            data_dict['target_weight_airs'] = target_dict['airs_weights']
            data_dict['target_weight_fgs1'] = target_dict['fgs1_weights']

        data_dict['std_airs'] = self.std_airs

        data_dict_airs, data_dict_fgs1 = self.define_data_columns(data_dict, train=False)

        airs_standard_deviation = self.fit_predict_airs(data_dict_airs, train=False)
        fgs1_standard_deviation = self.fit_predict_fgs1(data_dict_fgs1, train=False)
        airs_standard_deviation = airs_standard_deviation.reshape(fgs1_standard_deviation.shape[0], -1)

        airs_prediction, airs_standard_deviation, fgs1_prediction, fgs1_standard_deviation = self.combine_predictions(
            airs_prediction,
            airs_standard_deviation,
            fgs1_prediction,
            fgs1_standard_deviation,
            data_dict['star_info']
        )

        prediction = np.concatenate([
            fgs1_prediction[:, None],
            airs_prediction
        ], axis=1)
        prediction = np.clip(prediction, 0, 1)

        standard_deviation = np.concatenate([
            fgs1_standard_deviation[:, None],
            airs_standard_deviation
        ], axis=1)
        standard_deviation = np.maximum(1e-7, standard_deviation)

        sigma_scale = self.config.get('sigma_scale', 1.0)
        standard_deviation *= sigma_scale

        return prediction, standard_deviation

    def save_checkpoint(self, name_suffix=''):
        models_dir = util.get_dir('models')
        checkpoint_dir = os.path.join(models_dir, f'{name_suffix}{self.name}')

        self.airs_std_estimator.save_checkpoint(f'{checkpoint_dir}_airs')
        self.fgs1_std_estimator.save_checkpoint(f'{checkpoint_dir}_fgs1')
        self.depth_estimator.save_checkpoint(f'{checkpoint_dir}_depth')

        params = {
            'std_airs': self.std_airs,
            'airs_scaler': {
                'mean_': self.airs_scaler.mean_,
                'scale_': self.airs_scaler.scale_,
                'n_samples_seen_': self.airs_scaler.n_samples_seen_,
                'var_': self.airs_scaler.var_
            },
            'fgs1_scaler': {
                'mean_': self.fgs1_scaler.mean_,
                'scale_': self.fgs1_scaler.scale_,
                'n_samples_seen_': self.fgs1_scaler.n_samples_seen_,
                'var_': self.fgs1_scaler.var_
            }
        }

        file_name = f'{checkpoint_dir}.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_checkpoint(self, name_suffix=''):
        models_dir = util.get_dir('models')
        checkpoint_dir = os.path.join(models_dir, f'{name_suffix}{self.name}')

        self.airs_std_estimator = self.airs_std_estimator_factory()
        self.airs_std_estimator.load_checkpoint(f'{checkpoint_dir}_airs')

        self.fgs1_std_estimator = self.fgs1_std_estimator_factory()
        self.fgs1_std_estimator.load_checkpoint(f'{checkpoint_dir}_fgs1')

        self.depth_estimator = depth_estimation.DepthEstimator(self.config, use_pca=self.use_pca)
        self.depth_estimator.load_checkpoint(f'{checkpoint_dir}_depth')

        file_name = f'{checkpoint_dir}.pkl'
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        self.std_airs = params['std_airs']

        self.airs_scaler = StandardScaler()
        self.airs_scaler.mean_ = params['airs_scaler']['mean_']
        self.airs_scaler.scale_ = params['airs_scaler']['scale_']
        self.airs_scaler.n_samples_seen_ = params['airs_scaler']['n_samples_seen_']
        self.airs_scaler.var_ = params['airs_scaler']['var_']

        self.fgs1_scaler = StandardScaler()
        self.fgs1_scaler.mean_ = params['fgs1_scaler']['mean_']
        self.fgs1_scaler.scale_ = params['fgs1_scaler']['scale_']
        self.fgs1_scaler.n_samples_seen_ = params['fgs1_scaler']['n_samples_seen_']
        self.fgs1_scaler.var_ = params['fgs1_scaler']['var_']


def out_of_sample_evaluation(
        airs_std_estimator_factory,
        fgs1_std_estimator_factory,
        config,
        data_dict,
        df_target,
        outlier_planet_ids=None,
        n_splits=5,
        n_repeats=1,
        random_state=42,
        save_analysis_cache=False,
        save_training_cache=False,
        load_analysis_cache=False,
        save_checkpoint=False,
        load_checkpoint=False,
        name_suffix=None,
        use_pca=False,
        plot_sigma_scale=False
):
    if name_suffix is None:
        name_suffix = ''
    else:
        name_suffix = name_suffix + '_'

    key = jax.random.PRNGKey(random_state)
    start_time = datetime.datetime.now()

    def get_data(indices, data_dict, df_target):
        # indices are assumed to be sorted.

        star_info = data_dict['star_info']
        df_depth = data_dict['df_depth']

        star_info = star_info.loc[indices].reset_index(drop=True)
        df_target = df_target.loc[indices].reset_index(drop=True)

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

        return selected_data_dict, df_target

    star_info = data_dict['star_info']

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    prediction_list = []
    standard_deviation_list = []
    target_list = []
    index_list = []

    for i, (train_index, test_index) in enumerate(rkf.split(star_info)):
        tick = datetime.datetime.now()

        data_dict_train, df_target_train = get_data(train_index, data_dict, df_target)
        data_dict_test, df_target_test = get_data(test_index, data_dict, df_target)

        name = f'{name_suffix}standard_deviation_estimator_{i}'

        key, seed = util.get_random_int(key)

        estimator = StandardDeviationEstimator(
            airs_std_estimator_factory,
            fgs1_std_estimator_factory,
            config,
            random_state=seed,
            use_pca=use_pca,
            save_analysis_cache=save_analysis_cache,
            save_training_cache=save_training_cache,
            load_analysis_cache=load_analysis_cache,
            name=name
        )

        if load_checkpoint:
            estimator.load_checkpoint()
        else:
            estimator.fit(data_dict_train, df_target_train, outlier_planet_ids)

        if save_checkpoint:
            estimator.save_checkpoint()

        prediction, standard_deviation = estimator.predict(data_dict_test, df_target_test)

        prediction_list.append(prediction)
        standard_deviation_list.append(standard_deviation)
        target_list.append(df_target_test)
        index_list.append(test_index)

        fgs1_error = prediction[:, 0] - df_target_test.values[:, 1]
        airs_error = prediction[:, 1:] - df_target_test.values[:, 2:]

        fgs1_std = standard_deviation[:, 0]
        airs_std = standard_deviation[:, 1:]

        fgs1_ll = flax_training.gaussian_log_likelihood(fgs1_std, fgs1_error)
        airs_ll = flax_training.gaussian_log_likelihood(airs_std, airs_error)

        print(f'Fold {i + 1} of {n_splits * n_repeats}:   {airs_ll}   {fgs1_ll}   {datetime.datetime.now() - tick}')

    prediction = np.concatenate(prediction_list, axis=0)
    standard_deviation = np.concatenate(standard_deviation_list, axis=0)
    df_target_test = pd.concat(target_list, axis=0)

    test_index = np.concatenate(index_list, axis=0)

    fgs1_error = prediction[:, 0] - df_target_test.values[:, 1]
    airs_error = prediction[:, 1:] - df_target_test.values[:, 2:]

    fgs1_std = standard_deviation[:, 0]
    airs_std = standard_deviation[:, 1:]

    fgs1_ll = flax_training.gaussian_log_likelihood(fgs1_std, fgs1_error)
    airs_ll = flax_training.gaussian_log_likelihood(airs_std, airs_error)

    print(f'Combined score:   {airs_ll}   {fgs1_ll}   {datetime.datetime.now() - start_time}')

    combined_ll = (57.846 * fgs1_ll + 282 * airs_ll) / (57.846 + 282)

    if plot_sigma_scale:
        sigma_scale_array = np.linspace(0.8, 1.2, 100)
        log_loss_list = []

        for sigma_scale in sigma_scale_array:
            scaled_fgs1_std = sigma_scale * standard_deviation[:, 0]
            scaled_airs_std = sigma_scale * standard_deviation[:, 1:]

            fgs1_ll = flax_training.gaussian_log_likelihood(scaled_fgs1_std, fgs1_error)
            airs_ll = flax_training.gaussian_log_likelihood(scaled_airs_std, airs_error)

            log_loss_list.append((57.846 * fgs1_ll + 282 * airs_ll) / (57.846 + 282))

        log_loss_list = np.array(log_loss_list)

        idx = np.argmax(log_loss_list)

        print('Best sigma_scale:', sigma_scale_array[idx], log_loss_list[idx])

        plt.figure('sigma_scale')

        plt.plot(sigma_scale_array, log_loss_list, label='log_loss')

        plt.legend()

        plt.show()

    return combined_ll


def generate_submission(
        config=None,
        ensemble_size=5,
        file_prefix='test',
        std_estimator_seed=17,
        save_checkpoint=False,
        load_checkpoint=False,
        use_pca=True,
        plot_sigma_scale=False
):
    global outlier_planet_ids

    key = jax.random.PRNGKey(std_estimator_seed)

    if config is None:
        config = data_loader.get_config()

    airs_mlp_config = config['airs_mlp_config']
    fgs1_mlp_config = config['fgs1_mlp_config']

    airs_std_estimator_factory = flax_training.FlaxModelTrainerFactory(config=airs_mlp_config)
    fgs1_std_estimator_factory = flax_training.FlaxModelTrainerFactory(config=fgs1_mlp_config)

    ##### TRAIN ###############################

    cache_dir = util.get_dir('cache')

    if not load_checkpoint:
        fgs1_signal, airs_signal, star_info, df_target_train = data_loader.load_data('train')
        df_depth = pd.read_csv(os.path.join(cache_dir, f'train_depth_info.csv'))
        df_limb_darkening, limb_darkening_r_airs, limb_darkening_u_airs = limb_darkening.load_limb_darkening('train')

        data_dict_train = {
            'df_depth': df_depth,
            'star_info': star_info,
            'df_limb_darkening': df_limb_darkening,
            'limb_darkening_r_airs': limb_darkening_r_airs,
            'limb_darkening_u_airs': limb_darkening_u_airs,
        }

    ##### TEST ################################

    fgs1_signal, airs_signal, star_info, _ = data_loader.load_data(file_prefix)
    df_depth = pd.read_csv(os.path.join(cache_dir, f'{file_prefix}_depth_info.csv'))
    df_limb_darkening, limb_darkening_r_airs, limb_darkening_u_airs = limb_darkening.load_limb_darkening(file_prefix)

    data_dict_test = {
        'df_depth': df_depth,
        'star_info': star_info,
        'df_limb_darkening': df_limb_darkening,
        'limb_darkening_r_airs': limb_darkening_r_airs,
        'limb_darkening_u_airs': limb_darkening_u_airs,
    }

    prediction_list = []
    standard_deviation_list = []

    for i in range(ensemble_size):
        key, random_int = util.get_random_int(key)

        name = f'submission_{i}'

        estimator = StandardDeviationEstimator(
            airs_std_estimator_factory,
            fgs1_std_estimator_factory,
            config,
            random_state=random_int,
            name=name
        )

        if load_checkpoint:
            estimator.load_checkpoint()
        else:
            estimator.fit(data_dict_train, df_target_train, outlier_planet_ids)

        if save_checkpoint:
            estimator.save_checkpoint()

        prediction, standard_deviation = estimator.predict(data_dict_test)

        prediction_list.append(prediction)
        standard_deviation_list.append(standard_deviation)

    prediction = np.mean(np.stack(prediction_list, axis=-1), axis=-1)
    standard_deviation = np.mean(np.stack(standard_deviation_list, axis=-1), axis=-1)

    if use_pca:
        prediction_list = []
        standard_deviation_list = []

        for i in range(ensemble_size):
            key, random_int = util.get_random_int(key)

            name = f'submission_pca_{i}'

            estimator = StandardDeviationEstimator(
                airs_std_estimator_factory,
                fgs1_std_estimator_factory,
                config,
                random_state=random_int,
                use_pca=True,
                name=name
            )

            if load_checkpoint:
                estimator.load_checkpoint()
            else:
                estimator.fit(data_dict_train, df_target_train, outlier_planet_ids)

            if save_checkpoint:
                estimator.save_checkpoint()

            pca_prediction, pca_standard_deviation = estimator.predict(data_dict_test)

            prediction_list.append(pca_prediction)
            standard_deviation_list.append(pca_standard_deviation)

        pca_prediction = np.mean(np.stack(prediction_list, axis=-1), axis=-1)
        pca_standard_deviation = np.mean(np.stack(standard_deviation_list, axis=-1), axis=-1)

        if load_checkpoint:
            df_target_train = data_loader.load_target()

        pca_mask = spectrum_estimation.get_pca_mask(config, data_dict_test, df_target_train)

        prediction = np.where(pca_mask[:, None], pca_prediction, prediction)
        standard_deviation = np.where(pca_mask[:, None], pca_standard_deviation, standard_deviation)

    dict_submission = {'planet_id': star_info['planet_id']}
    dict_submission.update({f'wl_{i+1}': prediction[:, i] for i in range(prediction.shape[-1])})
    dict_submission.update({f'sigma_{i+1}': standard_deviation[:, i] for i in range(standard_deviation.shape[-1])})
    df_submission = pd.DataFrame(dict_submission)

    submissions_dir = util.get_dir('submissions')
    df_submission.to_csv(os.path.join(submissions_dir, f'submission.csv'), index=False)

    if file_prefix == 'train':
        ll_score = score.score(
            df_target_train.copy(),
            df_submission.copy(),
            row_id_column_name='planet_id',
            naive_mean=np.mean(df_target_train.values[:, 1:]),
            naive_sigma=np.std(df_target_train.values[:, 1:]),
            fsg_sigma_true=1e-6,
            airs_sigma_true=1e-5,
            fgs_weight=57.846
        )

        print(f'll_score: {ll_score}')

    if plot_sigma_scale:
        sigma_scale_array = np.linspace(0.8, 1.2, 100)
        ll_score_list = []

        for sigma_scale in sigma_scale_array:
            dict_submission = {'planet_id': star_info['planet_id']}
            dict_submission.update({f'wl_{i + 1}': prediction[:, i] for i in range(prediction.shape[-1])})
            dict_submission.update({f'sigma_{i + 1}': sigma_scale * standard_deviation[:, i] for i in range(standard_deviation.shape[-1])})
            df_submission = pd.DataFrame(dict_submission)

            ll_score = score.score(
                df_target_train.copy(),
                df_submission.copy(),
                row_id_column_name='planet_id',
                naive_mean=np.mean(df_target_train.values[:, 1:]),
                naive_sigma=np.std(df_target_train.values[:, 1:]),
                fsg_sigma_true=1e-6,
                airs_sigma_true=1e-5,
                fgs_weight=57.846
            )

            ll_score_list.append(ll_score)

        ll_score_list= np.array(ll_score_list)

        idx = np.argmax(ll_score_list)

        print('Best sigma_scale:', sigma_scale_array[idx], ll_score_list[idx])

        plt.figure('sigma_scale')

        plt.plot(sigma_scale_array, ll_score_list, label='ll_score')

        plt.legend()

        plt.show()


def data_analysis():
    file_name = f'standard_deviation_estimator_{0}.pkl'

    std_estimator = StandardDeviationEstimator(None, None, {}, file_name=file_name)

    load_dict = std_estimator.load_data('analysis')

    data_dict_airs = load_dict['data_dict_airs']
    data_dict_fgs1 = load_dict['data_dict_fgs1']

    def plot_std(x, error, label_name, fig_name):
        print(f'plot_std: {label_name}, {fig_name}')
        x = x.flatten()
        error = error.flatten()

        idx = np.argsort(x)
        x = x[idx]
        error = error[idx]

        unique = np.unique(x)

        if len(unique) < 10:
            for i, y in enumerate(unique):
                mask = (x == y)
                util.draw_histogram(error[mask], f'C{i}', y, f'{fig_name}')
        else:
            n_intervals = 300
            mean_x = []
            std = []
            for i in range(n_intervals):
                start = int(i * len(x) / n_intervals)
                end = int((i+1) * len(x) / n_intervals)

                mean_x.append(np.mean(x[start: end]))
                std.append(np.sqrt(np.mean(error[start: end] ** 2)))

            plt.figure(fig_name)

            plt.scatter(mean_x, std, label=label_name)

            estimator = LinearRegression()
            estimator.fit(np.array(mean_x)[:, None], std)
            x_range = np.array([np.min(mean_x), np.max(mean_x)])[:, None]
            y_range = estimator.predict(x_range)

            plt.plot(x_range, y_range, c='k', linestyle='dashed')

            plt.legend()

    error_fgs1 = data_dict_fgs1.pop('error_fgs1')
    error_airs = data_dict_airs.pop('error_airs')

    print('data_dict_airs')
    for key, value in data_dict_airs.items():
        print(key, value.shape)
        plot_std(value, error_airs, 'airs', f'airs {key}')

    print('data_dict_fgs1 fgs1')
    for key, value in data_dict_fgs1.items():
        print(key, value.shape)
        plot_std(value, error_fgs1, 'fgs1', f'fgs1 {key}')

    n_rows = error_fgs1.shape[0]

    print('data_dict_fgs1 airs')
    for key, value in data_dict_airs.items():
        print(key, value.shape)
        plot_std(np.mean(value.reshape(n_rows, -1), axis=-1), error_fgs1, 'airs', f'fgs1 {key}')

    #######################################################

    std_baseline = data_dict_airs['std_baseline_airs'][:282, 0]

    plt.figure('airs std_baseline')

    plt.plot(np.arange(len(std_baseline)), std_baseline, label='std_baseline')

    plt.legend()

    #######################################################

    multiplicity_mask = (data_dict_airs['multiplicity'].flatten() == 2)

    error_airs_2 = error_airs[multiplicity_mask, :].reshape(-1, 2)
    error_airs_2_min = np.min(error_airs_2, axis=-1)
    error_airs_2_max = np.max(error_airs_2, axis=-1)
    error_airs_2_mean = np.mean(error_airs_2, axis=-1)

    util.draw_histogram(error_airs_2_min, f'C0', 'error_airs_2_min', f'multiplicity 2')
    util.draw_histogram(error_airs_2_max, f'C1', 'error_airs_2_max', f'multiplicity 2')
    util.draw_histogram(error_airs_2_mean, f'C2', 'error_airs_2_mean', f'multiplicity 2')

    plt.show()


outlier_planet_ids = [
    1843015807, 3649218579, 1349926825, 576917580, 3786449677, 2554492145,
    2270815333, 1597331512, 1473180683, 1124834224, 3176118959, 2609891029,
    1267010874, 3928265625, 2557736212, 360841026, 2104402723, 204264160
]


if __name__ == '__main__':
    #data_analysis()
    #exit()

    print('n_outliers', len(outlier_planet_ids))
    #outlier_planet_ids = None

    fgs1_signal, airs_signal, star_info, df_target = data_loader.load_data('train')

    cache_dir = util.get_dir('cache')
    df_depth = pd.read_csv(os.path.join(cache_dir, f'train_depth_info.csv'))

    df_limb_darkening, limb_darkening_r_airs, limb_darkening_u_airs = limb_darkening.load_limb_darkening('train')

    # print(df_depth.columns)

    config = data_loader.get_config()
    config['sigma_scale'] = 1.0

    data_dict_train = {
        'df_depth': df_depth,
        'star_info': star_info,
        'df_limb_darkening': df_limb_darkening,
        'limb_darkening_r_airs': limb_darkening_r_airs,
        'limb_darkening_u_airs': limb_darkening_u_airs,
    }

    use_pca = True
    if use_pca:
        name_suffix = 'pca'
        airs_mlp_config = {
            'input_clipping_threshold': 7.0,
            'n_hidden': (256,),
            'dropout': (0.4,),
            'log_scale': -9.0,
            'batch_size': 256,
            'n_epochs': 5,
            'warmup_epochs': 1,
            'learning_rate': 0.0004,
            'weight_decay': 1e-5,
            'gradient_clipping_threshold': 0.15,
            'silent': False,
            'seed': 5,
            'input_shape': (1, 11)
        }
        fgs1_mlp_config = {
            'input_clipping_threshold': 5.0,
            'n_hidden': (32,),
            'dropout': (0.1,),
            'log_scale': -8.0,
            'batch_size': 64,
            'n_epochs': 150,
            'warmup_epochs': 10,
            'learning_rate': 0.0004,
            'weight_decay': 1e-7,
            'gradient_clipping_threshold': 0.1,
            'silent': False,
            'seed': 4,
            'input_shape': (1, 7)
        }
    else:
        name_suffix = None
        airs_mlp_config = {
            'input_clipping_threshold': 7.0,
            'n_hidden': (256,),
            'dropout': (0.4,),
            'log_scale': -9.0,
            'batch_size': 256,
            'n_epochs': 5,
            'warmup_epochs': 1,
            'learning_rate': 0.0004,
            'weight_decay': 1e-5,
            'gradient_clipping_threshold': 0.15,
            'silent': False,
            'seed': 5,
            'input_shape': (1, 11)
        }
        fgs1_mlp_config = {
            'input_clipping_threshold': 5.0,
            'n_hidden': (32,),
            'dropout': (0.1,),
            'log_scale': -8.0,
            'batch_size': 64,
            'n_epochs': 150,
            'warmup_epochs': 10,
            'learning_rate': 0.0004,
            'weight_decay': 1e-7,
            'gradient_clipping_threshold': 0.1,
            'silent': False,
            'seed': 4,
            'input_shape': (1, 7)
        }

    #airs_mlp_config = config['airs_mlp_config']
    #fgs1_mlp_config = config['fgs1_mlp_config']

    print('airs_mlp_config:', airs_mlp_config)
    print('fgs1_mlp_config:', fgs1_mlp_config)

    airs_std_estimator_factory = flax_training.FlaxModelTrainerFactory(config=airs_mlp_config)
    fgs1_std_estimator_factory = flax_training.FlaxModelTrainerFactory(config=fgs1_mlp_config)

    #std_estimator = StandardDeviationEstimator(airs_std_estimator_factory, fgs1_std_estimator_factory, config, save_analysis_cache=True, save_training_cache=True, load_analysis_cache=False, n_splits=5, n_repeats=3)
    #std_estimator = StandardDeviationEstimator(airs_std_estimator_factory, fgs1_std_estimator_factory, config, load_analysis_cache=True, n_splits=5, n_repeats=3)
    #std_estimator.fit(data_dict_train, df_target, outlier_planet_ids)

    out_of_sample_evaluation(
        airs_std_estimator_factory,
        fgs1_std_estimator_factory,
        config,
        data_dict_train,
        df_target,
        outlier_planet_ids=outlier_planet_ids,
        n_splits=5,
        n_repeats=3,
        random_state=71,
        load_analysis_cache=True,
        #save_analysis_cache=True,
        #save_training_cache=True,
        #save_checkpoint=True,
        #load_checkpoint=True,
        name_suffix=name_suffix,
        use_pca=use_pca,
        plot_sigma_scale=True
    )

    # PCA
    # Combined score:   7.981698089695622   7.499569716913966   0:15:30.203879
    # Combined score:   7.966913990594598   7.499569716913966   0:05:27.997089
    # Combined score:   7.977910430774623   7.498671252976158   0:15:34.545345
    # Combined score:   7.982429660296191   7.499369241055654   0:15:37.402813
    # Combined score:   7.982506591711768   7.4993318487676115   0:05:29.371774

    # Combined score:   7.908196074413703   7.500726876985368   0:14:41.978415
    # Combined score:   7.91041045466766   7.500726876985368   0:05:14.989923