import os

#os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from ariel_2025 import util, data_loader, depth_estimation, kalman_smoother, limb_darkening


def get_simple_spectrum(
        airs_signal,
        smoothing_window=29,
        level_change_scaler=1.0,
        trend_change_scaler=0.0001,
        noise_scaler=1.3
):
    airs_spectrum_smoothed, _ = kalman_smoother.apply_kalman_smoother(airs_signal[..., ::-1], window=smoothing_window, level_change_scaler=level_change_scaler, trend_change_scaler=trend_change_scaler, noise_scaler=noise_scaler)
    airs_spectrum_smoothed = airs_spectrum_smoothed[..., 0][..., ::-1]

    return airs_spectrum_smoothed


class PCASpectrumEstimator:

    def __init__(self, threshold):
        self.threshold = threshold
        self.eigenvectors = None

    def fit(self, X):
        X = X - np.mean(X, axis=-1, keepdims=True)

        covariance_matrix = np.cov(X, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalue_proportion = eigenvalues / np.sum(eigenvalues)
        cumulative_eigenvalue = np.cumsum(eigenvalue_proportion)
        cumulative_eigenvalue[1:] = cumulative_eigenvalue[:-1]
        cumulative_eigenvalue[0] = 0

        n_eigenvectors = np.argmax(cumulative_eigenvalue >= self.threshold)
        self.eigenvectors = eigenvectors[:, :n_eigenvectors]

    def predict(self, X):
        X = X - np.mean(X, axis=-1, keepdims=True)

        def linear_regression(X, y):
            XTX = X.mT @ X
            XTy = X.mT @ y

            coefficients = jnp.linalg.solve(XTX, XTy)
            prediction = X @ coefficients

            return prediction

        vmapped_linear_regression = jax.jit(jax.vmap(linear_regression, in_axes=(None, 0)))

        prediction = vmapped_linear_regression(self.eigenvectors, X)

        return prediction

    def get_params(self):
        params = {
            'threshold': self.threshold,
            'eigenvectors': self.eigenvectors
        }

        return params

    def set_params(self, params):
        self.threshold = params['threshold']
        self.eigenvectors = params['eigenvectors']


def get_pca_mask(config, data_dict, df_target, show_plot=False):
    pca_threshold = config['pca_threshold']
    pca_distance_threshold = config['pca_distance_threshold']

    limb_darkening_r_airs = data_dict['limb_darkening_r_airs']

    limb_darkening_spectrum = get_simple_spectrum(
        limb_darkening_r_airs ** 2,
        smoothing_window=config['spectrum_smoothing_window'],
        level_change_scaler=config['spectrum_level_change_scaler'],
        trend_change_scaler=config['spectrum_trend_change_scaler'],
        noise_scaler=config['spectrum_noise_scaler']
    )
    limb_darkening_spectrum = limb_darkening_spectrum - np.mean(limb_darkening_spectrum, axis=-1, keepdims=True)

    target = df_target.values[:, 2:]
    target = target - np.mean(target, axis=-1, keepdims=True)

    target_pca_spectrum_estimator = PCASpectrumEstimator(pca_threshold)
    target_pca_spectrum_estimator.fit(target)
    target_prediction = target_pca_spectrum_estimator.predict(limb_darkening_spectrum)

    distance = np.sqrt(np.sum((target_prediction - limb_darkening_spectrum) ** 2, axis=-1))

    pca_mask = (distance <= pca_distance_threshold)

    star_info = data_dict['star_info']

    pca_mask_shortened = np.zeros(len(star_info), dtype=bool)
    for i, (multiplicity, first_index) in enumerate(zip(star_info['multiplicity'], star_info['first_index'])):
        pca_mask_shortened[i] = np.all(pca_mask[first_index: first_index + multiplicity])

    if show_plot:
        print('pca_mask', pca_mask.shape, pca_mask.dtype, np.sum(pca_mask) / len(pca_mask))
        print('pca_mask_shortened', pca_mask_shortened.shape, pca_mask_shortened.dtype, np.sum(pca_mask_shortened) / len(pca_mask_shortened))

        distance_shortened = np.zeros(len(star_info), dtype=distance.dtype)
        for i, (multiplicity, first_index) in enumerate(zip(star_info['multiplicity'], star_info['first_index'])):
            distance_shortened[i] = np.max(distance[first_index: first_index + multiplicity])

        util.draw_histogram(distance, f'C0', f'L2 distance', 'get_pca_mask')
        util.draw_histogram(distance_shortened, f'C1', f'L2 distance_shortened', 'get_pca_mask')

    return pca_mask_shortened


def data_analysis_PCA():
    fgs1_signal, airs_signal_raw, star_info, df_target = data_loader.load_data('train')

    target_airs, target_airs_combined, target_fgs1, target_weights = depth_estimation.get_extended_airs_target(df_target, star_info)

    cache_dir = util.get_dir('cache')
    df_depth = pd.read_csv(os.path.join(cache_dir, f'train_depth_info.csv'))

    target = target_airs.reshape((len(df_depth), -1))

    df_limb_darkening, limb_darkening_r_airs, limb_darkening_u_airs = limb_darkening.load_limb_darkening('train')

    config = data_loader.get_config()

    data_dict = {
        'star_info': star_info,
        'limb_darkening_r_airs': limb_darkening_r_airs
    }

    get_pca_mask(config, data_dict, df_target, show_plot=True)

    smoothing_window = 29
    level_change_scaler = 1.0
    trend_change_scaler = 0.0001
    noise_scaler = 1.3

    ld_spectrum_airs = limb_darkening_r_airs ** 2
    lb_spectrum_smoothed = get_simple_spectrum(ld_spectrum_airs, smoothing_window=smoothing_window, level_change_scaler=level_change_scaler, trend_change_scaler=trend_change_scaler, noise_scaler=noise_scaler)
    lb_spectrum_smoothed = lb_spectrum_smoothed - np.mean(lb_spectrum_smoothed, axis=-1, keepdims=True)

    target = target - np.mean(target, axis=-1, keepdims=True)

    lb_spectrum_error = lb_spectrum_smoothed - target

    util.draw_histogram(lb_spectrum_error, 'C0', 'lb_spectrum_error', 'spectrum_error')

    for i, threshold in enumerate([0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]):
        pca_spectrum_estimator = PCASpectrumEstimator(threshold)
        pca_spectrum_estimator.fit(lb_spectrum_smoothed)
        prediction = pca_spectrum_estimator.predict(lb_spectrum_smoothed)

        pca_error = prediction - target

        util.draw_histogram(pca_error, f'C{i+1}', f'pca_error {threshold}', 'spectrum_error')

    util.draw_histogram(lb_spectrum_error, 'C0', 'lb_spectrum_error', 'target spectrum_error')

    for i, threshold in enumerate([0.98, 0.985, 0.99, 0.9925, 0.994, 0.995, 0.996, 0.9975]):
        pca_spectrum_estimator = PCASpectrumEstimator(threshold)
        pca_spectrum_estimator.fit(target)
        prediction = pca_spectrum_estimator.predict(lb_spectrum_smoothed)

        pca_error = prediction - target

        util.draw_histogram(pca_error, f'C{i+1}', f'pca_error {threshold}', 'target spectrum_error')

    pca_spectrum_estimator = PCASpectrumEstimator(0.98)
    pca_spectrum_estimator.fit(lb_spectrum_smoothed)
    prediction = pca_spectrum_estimator.predict(lb_spectrum_smoothed)

    pca_error = prediction - target

    pca_error_mean = np.mean(pca_error, axis=0)
    pca_error_std = np.std(pca_error, axis=0)

    lb_spectrum_error_mean = np.mean(lb_spectrum_error, axis=0)
    lb_spectrum_error_std = np.std(lb_spectrum_error, axis=0)

    target_std = np.std(target, axis=0)

    target_pca_spectrum_estimator = PCASpectrumEstimator(0.99)
    target_pca_spectrum_estimator.fit(target)
    target_prediction = target_pca_spectrum_estimator.predict(lb_spectrum_smoothed)

    target_pca_error = target_prediction - target

    target_pca_error_mean = np.mean(target_pca_error, axis=0)
    target_pca_error_std = np.std(target_pca_error, axis=0)

    plt.figure('spectrum_error spectrum')

    plt.axhline(0, c='k', linestyle='dashed')

    plt.plot(np.arange(len(pca_error_mean)), pca_error_mean, c='C0', label='pca_error_mean')
    plt.plot(np.arange(len(pca_error_mean)), pca_error_mean + 2*pca_error_std, c='C0')
    plt.plot(np.arange(len(pca_error_mean)), pca_error_mean - 2*pca_error_std, c='C0')

    plt.plot(np.arange(len(lb_spectrum_error_mean)), lb_spectrum_error_mean, c='C1', label='lb_spectrum_smoothed')
    plt.plot(np.arange(len(lb_spectrum_error_mean)), lb_spectrum_error_mean + 2*lb_spectrum_error_std, c='C1')
    plt.plot(np.arange(len(lb_spectrum_error_mean)), lb_spectrum_error_mean - 2*lb_spectrum_error_std, c='C1')

    plt.plot(np.arange(len(target_pca_error_mean)), target_pca_error_mean, c='C2', label='target_pca_error')
    plt.plot(np.arange(len(target_pca_error_mean)), target_pca_error_mean + 2*target_pca_error_std, c='C2')
    plt.plot(np.arange(len(target_pca_error_mean)), target_pca_error_mean - 2*target_pca_error_std, c='C2')

    plt.plot(np.arange(len(target_std)), 2*target_std, c='C3', label='constant')
    plt.plot(np.arange(len(target_std)), - 2*target_std, c='C3')

    plt.legend()

    distance = np.sqrt(np.sum((target_prediction - lb_spectrum_smoothed) ** 2, axis=-1))
    #distance /= np.max(distance)

    util.draw_histogram(distance, f'C0', f'euclidean', 'pca target distance')

    util.plot_correlation(target, 1.1, pca_threshold=0.993, title='target')
    util.plot_correlation(lb_spectrum_smoothed, 1.1, pca_threshold=0.993, title='lb_spectrum_smoothed')

    plt.show()


if __name__ == '__main__':
    data_analysis_PCA()
