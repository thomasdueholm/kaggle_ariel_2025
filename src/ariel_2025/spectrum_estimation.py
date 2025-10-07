import os

#os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from ariel_2025 import util, data_loader, depth_estimation, kalman_smoother, limb_darkening


def get_raw_spectrum(
        airs_signal,
        df_depth,
        data_window=128
):
    extended_airs_signal = np.concatenate([
        np.full((airs_signal.shape[0], airs_signal.shape[1], data_window), np.nan, dtype=airs_signal.dtype),
        airs_signal,
        np.full((airs_signal.shape[0], airs_signal.shape[1], data_window), np.nan, dtype=airs_signal.dtype)
    ], axis=2)

    transit_start_index = df_depth['transit_start_index'].values.astype(np.int32) + data_window  # data_window is added because of the additional nan entries.
    transit_middle_index = df_depth['transit_middle_index'].values.astype(np.int32) + data_window
    transit_end_index = df_depth['transit_end_index'].values.astype(np.int32) + data_window

    def dynamic_nanmean(y, start, end):
        if not isinstance(start, list):
            start = [start]
            end = [end]

        selection_range = jnp.arange(y.shape[-1])
        selection_mask = jnp.zeros(y.shape, dtype=bool)

        for start_index, end_index in zip(start, end):
            selection_mask = jnp.logical_or(
                selection_mask,
                jnp.logical_and(
                    selection_range >= start_index,
                    selection_range < end_index
                )
            )

        selection_mask = jnp.where(
            jnp.isnan(y),
            False,
            selection_mask
        ).astype(jnp.float32)

        sum = jnp.nansum(y * selection_mask)
        interval_length = jnp.sum(selection_mask)
        mean = sum / interval_length

        return mean

    vmapped_dynamic_nanmean = dynamic_nanmean
    for i in range(len(extended_airs_signal.shape) - 2, -1, -1):
        if i < len(transit_start_index.shape):
            vmapped_dynamic_nanmean = jax.vmap(vmapped_dynamic_nanmean, in_axes=(0, 0, 0))
        else:
            vmapped_dynamic_nanmean = jax.vmap(vmapped_dynamic_nanmean, in_axes=(0, None, None))
    vmapped_dynamic_nanmean = jax.jit(vmapped_dynamic_nanmean)

    airs_transit_level = vmapped_dynamic_nanmean(
        extended_airs_signal,
        transit_middle_index - data_window,
        transit_middle_index + data_window
    )
    airs_no_transit_level = vmapped_dynamic_nanmean(
        extended_airs_signal,
        [transit_start_index - data_window, transit_end_index - data_window],
        [transit_start_index + data_window, transit_end_index + data_window]
    )

    airs_spectrum = 1 - airs_transit_level / airs_no_transit_level

    return airs_spectrum


def get_spectrum(
        airs_signal,
        df_depth,
        data_window=128,
        smoothing_window=21,
        level_change_scaler=1.0,
        trend_change_scaler=0.0001,
        noise_scaler=1.0,
        return_plot_data=False
):
    extension = 0

    airs_spectrum = get_raw_spectrum(
        airs_signal,
        df_depth,
        data_window=data_window
    )

    airs_spectrum_smoothed, _ = kalman_smoother.apply_kalman_smoother(airs_spectrum, window=smoothing_window, level_change_scaler=level_change_scaler, trend_change_scaler=trend_change_scaler, noise_scaler=noise_scaler)
    airs_spectrum_smoothed = airs_spectrum_smoothed[..., extension: airs_spectrum_smoothed.shape[1]-extension, 0]

    if return_plot_data:
        return airs_spectrum_smoothed, airs_spectrum[:, extension: airs_spectrum.shape[1]-extension, ...]

    return airs_spectrum_smoothed


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


def data_analysis():
    fgs1_signal, airs_signal_raw, fgs1_foreground_noise, airs_foreground_noise, star_info, target = data_loader.load_data('train')

    cache_dir = util.get_dir('cache')
    df_depth = pd.read_csv(os.path.join(cache_dir, f'train_depth_info.csv'))

    df_limb_darkening, limb_darkening_r_airs, limb_darkening_u_airs = limb_darkening.load_limb_darkening('train')

    spectrum_smoothed, spectrum = get_spectrum(
        airs_signal_raw,
        df_depth,
        return_plot_data=True
    )

    extended_target, extended_target_airs_combined, extended_target_fgs1, extended_weights = depth_estimation.get_extended_airs_target(target, star_info)

    target = target.values[:, 2:]

    extended_target = extended_target.reshape((-1, target.shape[1]))

    ld_spectrum_airs = limb_darkening_r_airs ** 2

    smoothing_window = 29
    level_change_scaler = 1.0
    trend_change_scaler = 0.0001
    noise_scaler = 1.3

    # 0.0001241, 0.0001920

    # 0.0002359

    # 0.0003334

    # 0.0001170
    # 0.0001171
    # 0.0001212
    # 0.0001279
    # 0.0001274
    # 0.0001427
    # 0.0001330
    # 0.0001338
    # 0.0001215

    ld_spectrum_airs_smoothed = get_simple_spectrum(ld_spectrum_airs, smoothing_window=smoothing_window, level_change_scaler=level_change_scaler, trend_change_scaler=trend_change_scaler, noise_scaler=noise_scaler)

    spectrum_error = spectrum_smoothed - extended_target
    ld_spectrum_airs_error = ld_spectrum_airs_smoothed - extended_target
    ld_spectrum_airs_raw_error = ld_spectrum_airs - extended_target

    util.draw_histogram(ld_spectrum_airs_raw_error, 'C0', 'raw ld_spectrum_airs_error', 'spectrum_error')
    util.draw_histogram(ld_spectrum_airs_error, 'C1', 'ld_spectrum_airs_error', 'spectrum_error')
    util.draw_histogram(spectrum_error, 'C2', 'raw spectrum_error', 'spectrum_error')

    target_shifted = extended_target - np.nanmean(extended_target, axis=-1, keepdims=True)
    spectrum_shifted = spectrum_smoothed - np.nanmean(spectrum_smoothed, axis=-1, keepdims=True)
    ld_spectrum_airs_shifted = ld_spectrum_airs_smoothed - np.nanmean(ld_spectrum_airs_smoothed, axis=-1, keepdims=True)
    ld_spectrum_airs_raw_shifted = ld_spectrum_airs - np.nanmean(ld_spectrum_airs, axis=-1, keepdims=True)

    spectrum_error = spectrum_shifted - target_shifted
    ld_spectrum_airs_error = ld_spectrum_airs_shifted - target_shifted
    ld_spectrum_airs_raw_error = ld_spectrum_airs_raw_shifted - target_shifted

    util.draw_histogram(ld_spectrum_airs_raw_error, 'C0', 'raw ld_spectrum_airs_error', 'spectrum_error shifted')
    util.draw_histogram(ld_spectrum_airs_error, 'C1', 'ld_spectrum_airs_error', 'spectrum_error shifted')
    util.draw_histogram(spectrum_error, 'C2', 'raw spectrum_error', 'spectrum_error shifted')

    spectrum_error_mean = np.nanmean(spectrum_error, axis=0)
    spectrum_std = np.nanstd(spectrum_error, axis=0)

    ld_spectrum_airs_raw_error_mean = np.nanmean(ld_spectrum_airs_raw_error, axis=0)
    ld_spectrum_airs_raw_std = np.nanstd(ld_spectrum_airs_raw_error, axis=0)

    ld_spectrum_airs_error_mean = np.nanmean(ld_spectrum_airs_error, axis=0)
    ld_spectrum_airs_std = np.nanstd(ld_spectrum_airs_error, axis=0)

    plt.figure('spectrum_error spectrum')

    plt.axhline(0, c='k', linestyle='dashed')

    plt.plot(np.arange(len(ld_spectrum_airs_raw_error_mean)), ld_spectrum_airs_raw_error_mean, c='C0', label='raw ld_spectrum_airs_smoothed')
    plt.plot(np.arange(len(ld_spectrum_airs_raw_error_mean)), ld_spectrum_airs_raw_error_mean + 2*ld_spectrum_airs_raw_std, c='C0')
    plt.plot(np.arange(len(ld_spectrum_airs_raw_error_mean)), ld_spectrum_airs_raw_error_mean - 2*ld_spectrum_airs_raw_std, c='C0')

    plt.plot(np.arange(len(ld_spectrum_airs_error_mean)), ld_spectrum_airs_error_mean, c='C1', label='ld_spectrum_airs_smoothed')
    plt.plot(np.arange(len(ld_spectrum_airs_error_mean)), ld_spectrum_airs_error_mean + 2*ld_spectrum_airs_std, c='C1')
    plt.plot(np.arange(len(ld_spectrum_airs_error_mean)), ld_spectrum_airs_error_mean - 2*ld_spectrum_airs_std, c='C1')

    plt.plot(np.arange(len(spectrum_error_mean)), spectrum_error_mean, c='C2', label='spectrum_smoothed')
    plt.plot(np.arange(len(spectrum_error_mean)), spectrum_error_mean + 2*spectrum_std, c='C2')
    plt.plot(np.arange(len(spectrum_error_mean)), spectrum_error_mean - 2*spectrum_std, c='C2')

    plt.legend()

    plt.show()

    ##########################################

    row_ids = [549, 1067, 362, 393, 758, 759, 394, 1203, 736, 738, 476, 735, 128, 511, 103, 898, 251, 141, 492, 514]
    planet_ids = [496, 968, 329, 357, 689, 689, 357, 1094, 668, 670, 429, 668, 120, 461, 95, 818, 231, 132, 444, 464]

    #for planet_id in range(40):
    #    row_id = star_info['first_index'].values[planet_id]

    for row_id, planet_id in zip(row_ids, planet_ids):
        fig, axs = plt.subplots(2, sharex=True)
        fig.canvas.manager.set_window_title(f'level_diff {row_id} {planet_id}')

        data = spectrum[row_id, :]
        data = data - np.nanmean(data, axis=-1, keepdims=True)
        data_smoothed = spectrum_smoothed[row_id, :]
        data_smoothed = data_smoothed - np.nanmean(data_smoothed, axis=-1, keepdims=True)

        ld_data = ld_spectrum_airs[row_id, :]
        ld_data = ld_data - np.nanmean(ld_data, axis=-1, keepdims=True)

        ld_data_smoothed = ld_spectrum_airs_shifted[row_id, :]

        true_level = target[planet_id, :]
        true_level = true_level - np.nanmean(true_level, axis=-1, keepdims=True)

        x_range = np.arange(len(data))

        axs[0].axhline(0, c='k', linestyle='dashed')

        axs[0].plot(x_range, data, label='level')
        axs[0].plot(x_range, ld_data, label='ld_level')
        axs[0].plot(x_range, ld_data_smoothed, label='ld_level_smoothed')
        axs[0].plot(x_range, data_smoothed, label='smoothed level')

        print('x_range', len(x_range), len(x_range[5: -5]))

        axs[0].plot(x_range, true_level, label='target')

        axs[0].legend()

        axs[1].axhline(0, c='k', linestyle='dashed')

        error = data_smoothed - true_level
        ld_error = ld_data_smoothed - true_level
        axs[1].plot(x_range, error, label='error')
        axs[1].plot(x_range, ld_error, label='ld_error')

        axs[1].legend()

    plt.show()


def data_analysis_PCA():
    fgs1_signal, airs_signal_raw, fgs1_foreground_noise, airs_foreground_noise, star_info, df_target = data_loader.load_data('train')

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

    spectrum_smoothed, spectrum = get_spectrum(
        airs_signal_raw,
        df_depth,
        return_plot_data=True
    )
    spectrum_smoothed = spectrum_smoothed[:, 5: -5]
    spectrum_smoothed = spectrum_smoothed - np.mean(spectrum_smoothed, axis=-1, keepdims=True)

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

    #distance = np.mean(np.abs(target_prediction - lb_spectrum_smoothed), axis=-1)
    #distance /= np.max(distance)

    #util.draw_histogram(distance, f'C1', f'abs', 'pca target distance')

    #distance = np.max(np.abs(target_prediction - lb_spectrum_smoothed), axis=-1)
    #distance /= np.max(distance)

    #util.draw_histogram(distance, f'C2', f'max', 'pca target distance')

    plt.show()

    util.plot_correlation(target, 1.1, title='target')
    util.plot_correlation(spectrum_smoothed, 1.1, title='spectrum_smoothed')
    util.plot_correlation(lb_spectrum_smoothed, 1.1, title='lb_spectrum_smoothed')

    plt.show()


def data_analysis_foreground_noise():
    fgs1_signal, airs_signal_raw, fgs1_foreground_noise, airs_foreground_noise, star_info, target = data_loader.load_data('train')

    extension = 0

    airs_weights = np.nansum(airs_signal_raw[:, extension: airs_signal_raw.shape[1]-extension, :], axis=-1)
    airs_weights = airs_weights / np.sum(airs_weights, axis=-1, keepdims=True)

    foreground_noise_spectrum = get_simple_spectrum(
        airs_foreground_noise
    )

    airs_foreground_noise = airs_foreground_noise[:, extension: airs_foreground_noise.shape[1]-extension]
    airs_foreground_noise_weighted = np.sum(airs_foreground_noise * airs_weights, axis=-1)

    for i in range(100, 110):
        plt.figure(f'{i} {fgs1_foreground_noise[i]} {airs_foreground_noise_weighted[i]}')

        plt.plot(np.arange(airs_foreground_noise.shape[1]), airs_foreground_noise[i, :])
        plt.plot(np.arange(airs_foreground_noise.shape[1]), foreground_noise_spectrum[i, :])

    plt.show()


if __name__ == '__main__':
    #data_analysis_foreground_noise()
    data_analysis()
    #data_analysis_PCA()
