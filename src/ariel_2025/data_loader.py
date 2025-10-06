import os
import jax
import pandas as pd
import jax.numpy as jnp
import numpy as np
from functools import partial
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
from jax.sharding import PartitionSpec, NamedSharding
import gc

from ariel_2025 import util

jax.config.update('jax_enable_x64', True)


# https://www.kaggle.com/code/gordonyip/calibrating-and-binning-ariel-data

def ADC_convert(signal, gain=0.4369, offset=-1000):
    signal = signal.astype(jnp.float64) / gain + offset
    return signal


def mask_hot_dead(signal, dead, dark, outlier_threshold=5.0):
    dark_mean = jnp.nanmean(dark)
    dark_std = jnp.nanstd(dark)

    nan_mask = jnp.logical_or(
        (dark > dark_mean + outlier_threshold * dark_std),
        (dark < dark_mean - outlier_threshold * dark_std)
    )
    nan_mask = jnp.logical_or(
        nan_mask,
        dead
    )
    nan_mask = jnp.logical_or(
        nan_mask,
        jnp.isnan(dark)
    )
    nan_mask = jnp.logical_or(
        nan_mask,
        jnp.isnan(dead)
    )

    while len(nan_mask.shape) < len(signal.shape):
        nan_mask = jnp.expand_dims(nan_mask, 0)

    signal = jnp.where(nan_mask, jnp.nan, signal)

    return signal


def apply_linear_corr(linear_corr, clean_signal):
    linear_corr = jnp.flip(linear_corr, axis=0)

    linear_corr = jnp.transpose(linear_corr, (1, 2, 0))
    clean_signal = jnp.transpose(clean_signal, (2, 3, 0, 1))

    vmapped_polyval = jax.vmap(jax.vmap(jnp.polyval))

    clean_signal = vmapped_polyval(linear_corr, clean_signal)
    clean_signal = jnp.transpose(clean_signal, (2, 3, 0, 1))

    return clean_signal


def clean_dark(signal, dark, dt):
    while len(dark.shape) < len(signal.shape):
        dark = jnp.expand_dims(dark, 0)
    while len(dt.shape) < len(signal.shape):
        dt = jnp.expand_dims(dt, -1)

    signal = signal - dark * dt

    return signal


def correct_flat_field(flat, signal):
    while len(flat.shape) < len(signal.shape):
        flat = jnp.expand_dims(flat, 0)

    signal = signal / flat

    return signal


@partial(jax.jit, static_argnames=['window', 'min_periods'])
def centered_rolling_mean_and_std(x, window, min_periods):
    x = jnp.asarray(x, dtype=jnp.float64)

    nan_x = jnp.isnan(x)
    x = jnp.where(nan_x, 0, x)
    x_squared = x ** 2

    cumsum_x = jnp.cumsum(x, axis=0)
    cumsum_x_squared = jnp.cumsum(x_squared, axis=0)
    cumsum_nan_x = jnp.cumsum(1 - nan_x, axis=0)

    window_sum = cumsum_x[window:] - cumsum_x[:len(cumsum_x)-window]
    window_sum_squared = cumsum_x_squared[window:] - cumsum_x_squared[:len(cumsum_x) - window]
    window_nan_sum = cumsum_nan_x[window:] - cumsum_nan_x[:len(cumsum_nan_x)-window]

    initial_window_sum = cumsum_x[(window-1) // 2: window]
    initial_window_sum_squared = cumsum_x_squared[(window-1) // 2: window]
    initial_window_nan_sum = cumsum_nan_x[(window-1) // 2: window]

    last_window_sum = cumsum_x[-1] - cumsum_x[-window: -window // 2]
    last_window_sum_squared = cumsum_x_squared[-1] - cumsum_x_squared[-window: -window // 2]
    last_window_nan_sum = cumsum_nan_x[-1] - cumsum_nan_x[-window: -window // 2]

    window_sum = jnp.concat([initial_window_sum, window_sum, last_window_sum])
    window_sum_squared = jnp.concat([initial_window_sum_squared, window_sum_squared, last_window_sum_squared])
    window_nan_sum = jnp.concat([initial_window_nan_sum, window_nan_sum, last_window_nan_sum])

    mean = jnp.where(
        window_nan_sum >= min_periods,
        window_sum / jnp.maximum(1, window_nan_sum),
        jnp.nan
    )

    second_moment = jnp.where(
        window_nan_sum >= min_periods,
        window_sum_squared / jnp.maximum(1, window_nan_sum),
        jnp.nan
    )

    std = jnp.sqrt(jnp.maximum(0, second_moment - mean ** 2))

    return mean, std


def get_outliers(signal, outlier_threshold=5.0, window=255):
    mean, std = centered_rolling_mean_and_std(signal, window=window, min_periods=window//3)

    outlier_mask = jnp.logical_or(
        signal >= mean + outlier_threshold * std,
        signal <= mean - outlier_threshold * std
    )

    return outlier_mask, mean, std


def remove_foreground_noise_fgs1(signal):
    # See https://www.kaggle.com/competitions/ariel-data-challenge-2024/writeups/c-number-daiwakun-1st-place-solution

    original_heatmap = jnp.nanmean(signal, axis=0)

    def interpolate_single_nan_values(y):
        y = y.at[..., 1: -1, 1: -1].set(jnp.where(
            jnp.isnan(y[..., 1: -1, 1: -1]),
            (y[..., :-2, 1: -1] + y[..., 2:, 1: -1] + y[..., 1: -1, :-2] + y[..., 1: -1, 2:]) / 4,
            y[..., 1: -1, 1: -1]
        ))

        return y

    signal = interpolate_single_nan_values(signal)

    pixel_mean_original = jnp.nanmean(signal, axis=0)
    not_nan_pixels = 1 - jnp.isnan(pixel_mean_original)

    n_removed = 6

    pixel_mean_original = pixel_mean_original
    not_nan_pixels = not_nan_pixels

    row_sum = jnp.nansum(pixel_mean_original, axis=0)
    row_not_nan = jnp.sum(not_nan_pixels, axis=0)

    column_order = jnp.argsort(row_sum)

    foreground_sum = jnp.nansum(row_sum[column_order[:n_removed]])
    foreground_not_nan = jnp.sum(row_not_nan[column_order[:n_removed]])

    signal = signal[..., :, column_order[n_removed:]]
    pixel_mean = pixel_mean_original[:, column_order[n_removed:]]
    not_nan_pixels = not_nan_pixels[:, column_order[n_removed:]]

    column_sum = jnp.nansum(pixel_mean, axis=1)
    column_not_nan = jnp.sum(not_nan_pixels, axis=1)

    row_order = jnp.argsort(column_sum)

    foreground_sum += jnp.nansum(column_sum[row_order[:n_removed]])
    foreground_not_nan += jnp.sum(column_not_nan[row_order[:n_removed]])

    signal = signal[..., row_order[n_removed:], :]

    nan_mask = jnp.isin(jnp.arange(32), row_order[n_removed:])[:, None] * jnp.isin(jnp.arange(32), column_order[n_removed:])[None, :]
    rim_heatmap = jnp.where(nan_mask, jnp.nan, pixel_mean_original)

    nan_mask = jnp.logical_or(jnp.isin(jnp.arange(32), row_order[:n_removed])[:, None], jnp.isin(jnp.arange(32), column_order[:n_removed])[None, :])
    center_heatmap = jnp.where(nan_mask, jnp.nan, pixel_mean_original)

    foreground_noise = foreground_sum / foreground_not_nan

    signal = signal - foreground_noise

    return signal, original_heatmap, rim_heatmap, center_heatmap


def remove_foreground_noise_airs(signal):
    # See https://www.kaggle.com/competitions/ariel-data-challenge-2024/writeups/c-number-daiwakun-1st-place-solution

    original_heatmap = jnp.nanmean(signal, axis=0)

    n_removed = 6

    column_sum = jnp.nansum(original_heatmap, axis=1)

    row_order = jnp.argsort(column_sum)

    broken_pixels = jnp.any(jnp.isnan(original_heatmap[row_order[-4:]]), axis=0)

    def interpolate_single_nan_values(y):
        y = y.at[..., 1: -1, :].set(jnp.where(
            jnp.isnan(y[..., 1: -1, :]),
            (y[..., :-2, :] + y[..., 2:, :]) / 2,
            y[..., 1: -1, :]
        ))

        return y

    pixel_mean_original = interpolate_single_nan_values(original_heatmap)
    not_nan_pixels = 1 - jnp.isnan(pixel_mean_original)

    foreground_sum = jnp.nansum(pixel_mean_original[row_order[:n_removed]], axis=0)
    foreground_not_nan = jnp.sum(not_nan_pixels[row_order[:n_removed]], axis=0)

    signal = interpolate_single_nan_values(signal)
    signal = signal[..., row_order[n_removed:], :]

    pixel_mean_original = jnp.where(broken_pixels[None, :], jnp.nan, pixel_mean_original)

    nan_mask = jnp.isin(jnp.arange(32), row_order[n_removed:])[:, None]
    rim_heatmap = jnp.where(nan_mask, jnp.nan, pixel_mean_original)

    nan_mask = jnp.isin(jnp.arange(32), row_order[:n_removed])[:, None]
    center_heatmap = jnp.where(nan_mask, jnp.nan, pixel_mean_original)

    foreground_noise = foreground_sum / foreground_not_nan

    signal = jnp.where(broken_pixels[None, None, :], jnp.nan, signal - foreground_noise)

    return signal, original_heatmap, rim_heatmap, center_heatmap


def calibrate_data(
        raw_data,
        dt_airs,
        dark_outlier_threshold,
        outlier_threshold,
        outlier_window,
        return_plot_data=False
):
    (
        fgs1_signal,
        fgs1_dead,
        fgs1_dark,
        fgs1_linear_corr,
        fgs1_flat,
        airs_signal,
        airs_dead,
        airs_dark,
        airs_linear_corr,
        airs_flat
    ) = raw_data

    fgs1_signal = ADC_convert(fgs1_signal)
    fgs1_signal = mask_hot_dead(fgs1_signal, fgs1_dead, fgs1_dark, outlier_threshold=dark_outlier_threshold)
    fgs1_signal = apply_linear_corr(fgs1_linear_corr, fgs1_signal)

    dt_fgs1 = np.ones(fgs1_signal.shape[:3]) * 0.1
    dt_fgs1[..., 1] += 0.1

    fgs1_signal = clean_dark(fgs1_signal, fgs1_dark, dt_fgs1)
    fgs1_signal = fgs1_signal[:, 1, ...] - fgs1_signal[:, 0, ...]
    fgs1_signal = correct_flat_field(fgs1_flat, fgs1_signal)

    naive_mean_fgs1_signal = jnp.nanmean(fgs1_signal, axis=(1, 2))
    outlier_mask, fgs1_mean, fgs1_std = get_outliers(naive_mean_fgs1_signal, outlier_threshold=outlier_threshold, window=outlier_window)

    fgs1_signal = jnp.where(outlier_mask[:, None, None], jnp.nan, fgs1_signal)
    fgs1_signal, fgs1_original_heatmap, fgs1_rim_heatmap, fgs1_center_heatmap = remove_foreground_noise_fgs1(fgs1_signal)
    fgs1_signal = jnp.nanmean(fgs1_signal, axis=(1, 2))

    ################

    airs_signal = ADC_convert(airs_signal)
    airs_signal = mask_hot_dead(airs_signal, airs_dead, airs_dark, outlier_threshold=dark_outlier_threshold)
    airs_signal = apply_linear_corr(airs_linear_corr, airs_signal)

    airs_signal = clean_dark(airs_signal, airs_dark, dt_airs)
    airs_signal = airs_signal[:, 1, ...] - airs_signal[:, 0, ...]
    airs_signal = correct_flat_field(airs_flat, airs_signal)

    naive_mean_airs_signal = jnp.nanmean(airs_signal, axis=1)
    outlier_mask, mean_airs, std_airs = get_outliers(naive_mean_airs_signal, outlier_threshold=outlier_threshold, window=outlier_window)
    airs_signal = jnp.where(outlier_mask[:, None, :], np.nan, airs_signal)

    airs_signal, airs_original_heatmap, airs_rim_heatmap, airs_center_heatmap = remove_foreground_noise_airs(airs_signal)
    airs_signal = jnp.nanmean(airs_signal, axis=1)

    airs_signal = airs_signal.mT

    if return_plot_data:
        output = (
            fgs1_signal,
            airs_signal,
            naive_mean_fgs1_signal,
            fgs1_mean,
            fgs1_std,
            naive_mean_airs_signal.mT,
            mean_airs.mT,
            std_airs.mT,
            fgs1_original_heatmap,
            fgs1_rim_heatmap,
            fgs1_center_heatmap,
            airs_original_heatmap,
            airs_rim_heatmap,
            airs_center_heatmap
        )
    else:
        output = (
            fgs1_signal,
            airs_signal
        )

    return output


def get_raw_data(planet_id, instance=0, file_prefix='test', airs_expansion=0):
    data_dir = util.get_dir('data')
    data_dir = os.path.join(data_dir, f'{file_prefix}/{planet_id}')

    fgs1_signal = pd.read_parquet(os.path.join(data_dir, f'FGS1_signal_{instance}.parquet'), engine='pyarrow').values.reshape((135000 // 2, 2, 32, 32))
    fgs1_dead = pd.read_parquet(os.path.join(data_dir, f'FGS1_calibration_{instance}/dead.parquet'), engine='pyarrow').values
    fgs1_dark = pd.read_parquet(os.path.join(data_dir, f'FGS1_calibration_{instance}/dark.parquet'), engine='pyarrow').values
    fgs1_linear_corr = pd.read_parquet(os.path.join(data_dir, f'FGS1_calibration_{instance}/linear_corr.parquet'), engine='pyarrow').values.reshape((6, 32, 32))
    fgs1_flat = pd.read_parquet(os.path.join(data_dir, f'FGS1_calibration_{instance}/flat.parquet'), engine='pyarrow').values

    cut_inf, cut_sup = 39, 321

    airs_signal = pd.read_parquet(os.path.join(data_dir, f'AIRS-CH0_signal_{instance}.parquet'), engine='pyarrow').values.reshape((11250 // 2, 2, 32, 356))
    airs_signal = np.flip(airs_signal[..., cut_inf-airs_expansion: cut_sup+airs_expansion], axis=-1)

    airs_dead = pd.read_parquet(os.path.join(data_dir, f'AIRS-CH0_calibration_{instance}/dead.parquet'), engine='pyarrow').values
    airs_dead = np.flip(airs_dead[..., cut_inf - airs_expansion: cut_sup + airs_expansion], axis=-1)

    airs_dark = pd.read_parquet(os.path.join(data_dir, f'AIRS-CH0_calibration_{instance}/dark.parquet'), engine='pyarrow').values
    airs_dark = np.flip(airs_dark[..., cut_inf - airs_expansion: cut_sup + airs_expansion], axis=-1)

    airs_linear_corr = pd.read_parquet(os.path.join(data_dir, f'AIRS-CH0_calibration_{instance}/linear_corr.parquet'), engine='pyarrow').values.reshape((6, 32, 356))
    airs_linear_corr = np.flip(airs_linear_corr[..., cut_inf-airs_expansion: cut_sup+airs_expansion], axis=-1)

    airs_flat = pd.read_parquet(os.path.join(data_dir, f'AIRS-CH0_calibration_{instance}/flat.parquet'), engine='pyarrow').values
    airs_flat = np.flip(airs_flat[..., cut_inf - airs_expansion: cut_sup + airs_expansion], axis=-1)

    output = (
        fgs1_signal,
        fgs1_dead,
        fgs1_dark,
        fgs1_linear_corr,
        fgs1_flat,
        airs_signal,
        airs_dead,
        airs_dark,
        airs_linear_corr,
        airs_flat
    )

    return output


def get_star_info(file_prefix):
    data_dir = util.get_dir('data')

    star_info_file = os.path.join(data_dir, f'{file_prefix}_star_info.csv')
    star_info = pd.read_csv(star_info_file)
    star_info['planet_id'] = star_info['planet_id'].values.astype(np.int64)

    data_dir = os.path.join(data_dir, f'{file_prefix}')
    multiplicity = []

    for planet_id in star_info['planet_id']:
        planet_dir = os.path.join(data_dir, f'{planet_id}')
        n_copies = len(os.listdir(planet_dir)) // 4
        multiplicity.append(n_copies)

    star_info['multiplicity'] = multiplicity
    star_info['first_index'] = [0] + list(np.cumsum(multiplicity)[:-1])

    return star_info


def generate_data(config, file_prefix, dark_outlier_threshold=5.0, outlier_threshold=4.0, outlier_window=511, plot_data=False):
    batch_size = config['data_loader_batch_size']
    n_devices = config['n_devices']

    data_dir = util.get_dir('data')

    axis_info = pd.read_parquet(os.path.join(data_dir, f'axis_info.parquet'), engine='pyarrow')
    star_info = get_star_info(file_prefix)

    axis_info = axis_info['AIRS-CH0-integration_time'].dropna().values.reshape((11250 // 2, 2))
    axis_info[:, 1] += 0.1

    mesh = jax.make_mesh((n_devices,), ('batch',))
    sharding = NamedSharding(mesh, PartitionSpec('batch'))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    axis_info = jax.device_put(axis_info, replicated_sharding)

    sample_data = get_raw_data(planet_id=34983, instance=0, file_prefix='train')

    batched_data = []
    for data in sample_data:
        batched_data.append(np.repeat(data[None, ...], batch_size, axis=0))
    batched_data = tuple(batched_data)
    current_batch_index = 0

    sample_data = None

    calibrate_data_fn = partial(
        calibrate_data,
        dark_outlier_threshold=dark_outlier_threshold,
        outlier_threshold=outlier_threshold,
        outlier_window=outlier_window,
        return_plot_data=plot_data
    )
    calibrate_data_fn = jax.vmap(calibrate_data_fn, in_axes=(0, None))
    calibrate_data_fn = jax.jit(calibrate_data_fn)
    calibrate_data_fn = calibrate_data_fn.lower(
        jax.device_put(batched_data, sharding),
        axis_info
    ).compile()

    sample_output = calibrate_data_fn(
        jax.device_put(batched_data, sharding),
        axis_info
    )

    if plot_data:
        n_rows = 1 * batch_size
    else:
        n_rows = star_info['multiplicity'].sum()

    print('n_rows', n_rows)
    print('batch_size', batch_size)

    output_data = []
    for data in sample_output:
        output_data.append(np.zeros((n_rows,) + data.shape[1:], dtype=np.float32))

    sample_output = None

    current_row = 0

    start_time = datetime.datetime.now()
    print('Start time:', start_time)
    tick = start_time
    for planet_id, multiplicity in zip(star_info['planet_id'], star_info['multiplicity']):
        print(planet_id, multiplicity)

        for i in range(multiplicity):
            raw_data = get_raw_data(planet_id=planet_id, instance=i, file_prefix=file_prefix)

            for j, data in enumerate(raw_data):
                batched_data[j][current_batch_index, ...] = data
            current_batch_index += 1

            if current_batch_index >= batch_size:
                #jax.clear_caches()
                gc.collect()

                calibrated_data = calibrate_data_fn(
                    jax.device_put(batched_data, sharding),
                    axis_info
                )

                for j, data in enumerate(calibrated_data):
                    output_data[j][current_row: current_row+batch_size, ...] = np.asarray(data, dtype=np.float32)

                current_batch_index = 0
                current_row += batch_size

                next_tick = datetime.datetime.now()
                print(f'{current_row} / {n_rows}: {next_tick - tick}')
                tick = next_tick

            if current_row >= n_rows:
                break
        if current_row >= n_rows:
            break

    remaining_rows = n_rows - current_row
    if remaining_rows > 0:
        calibrated_data = calibrate_data_fn(
            jax.device_put(batched_data, sharding),
            axis_info
        )

        for j, data in enumerate(calibrated_data):
            output_data[j][current_row:, ...] = np.asarray(data[:remaining_rows, ...], dtype=np.float32)

    print('Time spent:', datetime.datetime.now() - start_time)

    if plot_data:
        print('fgs1', output_data[0].shape)
        print('airs', output_data[1].shape)

        for plot_index in range(output_data[0].shape[0]):

            show_scatter_plot = False
            if show_scatter_plot:
                for i in range(5):
                    plt.figure(f'{plot_index} airs {41+i}')

                    y = output_data[5][plot_index, 41+i, :]
                    x = np.arange(len(y))

                    plt.scatter(x, y, alpha=0.8, s=0.5)

                    plt.plot(x, output_data[6][plot_index, 41+i, :], c='k', linestyle='dashed')
                    plt.plot(x, output_data[6][plot_index, 41+i, :] + outlier_threshold * output_data[7][plot_index, 41+i, :], c='r', linestyle='dashed')
                    plt.plot(x, output_data[6][plot_index, 41+i, :] - outlier_threshold * output_data[7][plot_index, 41+i, :], c='r', linestyle='dashed')

                plt.figure(f'{plot_index} fgs1')
                y = output_data[2][plot_index, :]
                x = np.arange(len(y))

                plt.scatter(x, y, alpha=0.8, s=0.5)

                plt.plot(x, output_data[3][plot_index, :], c='k', linestyle='dashed')
                plt.plot(x, output_data[3][plot_index, :] + outlier_threshold * output_data[4][plot_index, :], c='r', linestyle='dashed')
                plt.plot(x, output_data[3][plot_index, :] - outlier_threshold * output_data[4][plot_index, :], c='r', linestyle='dashed')

            show_heatmap = True
            if show_heatmap:
                ###### FGS1 ######

                fig, axs = plt.subplots(1, 3)
                fig.canvas.manager.set_window_title(f'{plot_index} fgs1 heatmap')
                fig.suptitle(f'FGS1', fontsize=16)

                fig.set_size_inches(12.5, 4)
                plt.subplots_adjust(bottom=0.15, top=0.85, left=0.03, right=0.97, wspace=0.15)

                axs[0].title.set_text('Raw data')
                axs[1].title.set_text('Foreground noise')
                axs[2].title.set_text('Processed data')

                fgs1_original_heatmap = output_data[8][plot_index, ...]
                fgs1_rim_heatmap = output_data[9][plot_index, ...]
                fgs1_center_heatmap = output_data[10][plot_index, ...]

                sns.heatmap(fgs1_original_heatmap, cmap="coolwarm", ax=axs[0])
                sns.heatmap(fgs1_rim_heatmap, cmap="coolwarm", ax=axs[1])
                sns.heatmap(fgs1_center_heatmap, cmap="coolwarm", ax=axs[2])

                ###### AIRS ######

                fig, axs = plt.subplots(1, 3)
                fig.canvas.manager.set_window_title(f'{plot_index} airs heatmap')
                fig.suptitle(f'AIRS', fontsize=16)

                fig.set_size_inches(12.5, 4)
                plt.subplots_adjust(bottom=0.15, top=0.85, left=0.03, right=0.97, wspace=0.15)

                axs[0].title.set_text('Raw data')
                axs[1].title.set_text('Foreground noise')
                axs[2].title.set_text('Processed data')

                airs_original_heatmap = output_data[11][plot_index, ...]
                airs_rim_heatmap = output_data[12][plot_index, ...]
                airs_center_heatmap = output_data[13][plot_index, ...]

                sns.heatmap(airs_original_heatmap, cmap="coolwarm", ax=axs[0])
                sns.heatmap(airs_rim_heatmap, cmap="coolwarm", ax=axs[1])
                sns.heatmap(airs_center_heatmap, cmap="coolwarm", ax=axs[2])

        plt.show()

        exit()

    cache_dir = util.get_dir('cache')

    star_info.to_csv(os.path.join(cache_dir, f'{file_prefix}_star_info.csv'), index=False)
    np.save(os.path.join(cache_dir, f'{file_prefix}_fgs1_data.npy'), output_data[0])
    np.save(os.path.join(cache_dir, f'{file_prefix}_airs_data.npy'), output_data[1])


def load_target():
    data_dir = util.get_dir('data')
    df_target = pd.read_csv(os.path.join(data_dir, f'train.csv'))

    return df_target


def load_data(file_prefix):
    data_dir = util.get_dir('data')
    cache_dir = util.get_dir('cache')

    fgs1_signal = np.load(os.path.join(cache_dir, f'{file_prefix}_fgs1_data.npy'))
    airs_signal = np.load(os.path.join(cache_dir, f'{file_prefix}_airs_data.npy'))
    star_info = pd.read_csv(os.path.join(cache_dir, f'{file_prefix}_star_info.csv'))

    if file_prefix == 'train':
        target = pd.read_csv(os.path.join(data_dir, f'train.csv'))
    else:
        target = None

    return fgs1_signal, airs_signal, star_info, target


def get_config(use_kaggle_config=True):
    parameters_dir = util.get_dir('parameters')
    if use_kaggle_config:
        parameter_file = os.path.join(parameters_dir, 'kaggle_config.json')
    else:
        parameter_file = os.path.join(parameters_dir, 'config.json')

    config = {}

    if os.path.isfile(parameter_file):
        with open(parameter_file, 'r') as json_file:
            config = json.load(json_file)
    else:
        print(f'Missing file: {parameter_file}')

    return config


if __name__ == '__main__':
    print(jax.devices())
    print('num_devices:', jax.local_device_count())
    print(datetime.datetime.now())

    config = get_config(use_kaggle_config=False)

    generate_data(config, 'train', plot_data=True)
    fgs1_signal, airs_signal, star_info, target = load_data('train')

    print('fgs1_signal', fgs1_signal.shape, fgs1_signal.dtype)
    print('airs_signal', airs_signal.shape, airs_signal.dtype)
