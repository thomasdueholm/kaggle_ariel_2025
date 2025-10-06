import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jax
import pandas as pd
import datetime

from ariel_2025 import data_loader, transit_identification, kalman_smoother, util


def get_depth(
        y_fgs1,
        y_airs,
        config,
        fig_name=None
):
    (
        transit_start,
        transit_middle,
        transit_end
    ) = transit_identification.find_breakpoints(
        y_fgs1,
        y_airs,
        window=config['transit_identification_window'],
        level_change_scaler=config['transit_identification_level_change_scaler'],
        trend_change_scaler=config['transit_identification_trend_change_scaler'],
        noise_scaler=config['transit_identification_noise_scaler']
    )

    def get_first_and_last_non_nan_index(x):
        nan_x = jnp.isnan(x)
        first = jnp.argmin(nan_x)
        last = x.shape[0] - 1 - jnp.argmin(nan_x[::-1])

        return first, last

    vmapped_get_first_and_last_non_nan_index = get_first_and_last_non_nan_index
    for i in range(len(y_fgs1.shape) - 1):
        vmapped_get_first_and_last_non_nan_index = jax.vmap(vmapped_get_first_and_last_non_nan_index)
    vmapped_get_first_and_last_non_nan_index = jax.jit(vmapped_get_first_and_last_non_nan_index)

    airs_range = jnp.arange(y_airs.shape[-1])[None, ...]

    first_non_nan, last_non_nan = vmapped_get_first_and_last_non_nan_index(y_airs)

    airs_transit_mask = jnp.logical_and(
        airs_range > jnp.maximum(transit_start[..., None], first_non_nan[..., None]),
        airs_range < jnp.minimum(transit_end[..., None], last_non_nan[..., None])
    )
    while len(airs_transit_mask.shape) < len(y_airs.shape):
        airs_transit_mask = jnp.expand_dims(airs_transit_mask, axis=0)
    y_airs_no_transit = jnp.where(airs_transit_mask, jnp.nan, y_airs)

    mu_airs, _ = kalman_smoother.apply_kalman_smoother(
        y_airs,
        window=config['window'],
        level_change_scaler=config['level_change_scaler'],
        trend_change_scaler=config['trend_change_scaler'],
        noise_scaler=config['noise_scaler']
    )
    mu_airs_no_transit, _ = kalman_smoother.apply_kalman_smoother(
        y_airs_no_transit.astype(jnp.float64),
        window=config['no_transit_window'],
        level_change_scaler=config['no_transit_level_change_scaler'],
        trend_change_scaler=config['no_transit_trend_change_scaler'],
        noise_scaler=config['no_transit_noise_scaler']
    )

    if len(mu_airs_no_transit.shape) != len(mu_airs.shape):
        mu_airs_no_transit = mu_airs_no_transit[0, ...]

    ###################################

    def get_incomplete_transit(
            mu,
            transit_start,
            transit_middle,
            transit_end,
            window,
            binning,
            data_for_plotting
    ):
        expanded_transit_start = (binning*transit_start).astype(jnp.int64)
        expanded_transit_middle = (binning*transit_middle).astype(jnp.int64)
        expanded_transit_end = (binning*transit_end).astype(jnp.int64)

        # The slope of the transit valley is estimated by averaging multiple step lengths across the middle.
        stable_length = jnp.minimum(expanded_transit_middle - expanded_transit_start, expanded_transit_end - expanded_transit_middle) // 2

        selection_range = jnp.arange(mu.shape[0])
        distance = 2 * jnp.maximum(1, jnp.abs(selection_range - expanded_transit_middle))

        from_selection_mask = jnp.logical_and(
            selection_range >= expanded_transit_middle - stable_length,
            selection_range < expanded_transit_middle - stable_length // 2
        ).astype(jnp.float32)
        to_selection_mask = jnp.logical_and(
            selection_range >= expanded_transit_middle + 1 + stable_length // 2,
            selection_range < expanded_transit_middle + stable_length + 1
        ).astype(jnp.float32)
        selection_length = stable_length - stable_length // 2

        transit_slope = jnp.sum((to_selection_mask - from_selection_mask) * mu[:, 0] / distance, axis=-1) / selection_length

        start_index = jnp.maximum(0, expanded_transit_start - 3 * binning * window)
        level_start = mu[start_index, 0]
        end_index = jnp.minimum(mu.shape[-2] - 1, expanded_transit_end + 3 * binning * window)
        level_end = mu[end_index, 0]

        middle_start = level_start + (expanded_transit_middle - start_index) * transit_slope
        middle_end = level_end + (expanded_transit_middle - end_index) * transit_slope

        if data_for_plotting:
            return expanded_transit_start, expanded_transit_middle, expanded_transit_end, middle_start, middle_end, transit_slope, start_index, end_index

        middle = jnp.maximum(middle_start, middle_end)

        return middle, transit_slope, (middle_start >= middle_end), (start_index == 0), (end_index == mu.shape[-2] - 1)

    vmapped_get_incomplete_transit = get_incomplete_transit
    for i in range(len(mu_airs.shape) - 2):
        vmapped_get_incomplete_transit = jax.vmap(vmapped_get_incomplete_transit, in_axes=(0, 0, 0, 0, None, None, None))
    vmapped_get_incomplete_transit = jax.jit(vmapped_get_incomplete_transit, static_argnames=['window', 'binning', 'data_for_plotting'])

    incomplete_middle_airs, transit_slope_airs, left_max, left_border_airs, right_border_airs = vmapped_get_incomplete_transit(
        mu_airs,
        transit_start,
        transit_middle,
        transit_end,
        config['window'],
        1,
        False
    )

    is_incomplete_airs = jnp.logical_or(left_border_airs, right_border_airs)

    vmapped_take = jnp.take
    for i in range(len(mu_airs.shape) - 2):
        vmapped_take = jax.vmap(vmapped_take, in_axes=(0, 0, None))

    middle_airs = jnp.where(
        is_incomplete_airs,
        incomplete_middle_airs,
        vmapped_take(mu_airs_no_transit[..., 0], transit_middle.astype(jnp.int64), 0)
    )

    transit_low_airs = vmapped_take(mu_airs[..., 0], transit_middle.astype(jnp.int64), 0)

    if fig_name is not None:
        plt.figure(fig_name)

        idx = 511 - 256

        normalization_factor_airs, shift_airs = draw_figure(
            y_airs[idx, ...],
            mu_airs[idx, ...],
            mu_airs_no_transit[idx, ...],
            'airs',
            1,
            transit_middle[idx],
            draw_scatter_plot=True
        )

        plt.axvline(transit_start[idx], c='k', linestyle='dashed')
        plt.axvline(transit_middle[idx], c='k', linestyle='dashed')
        plt.axvline(transit_end[idx], c='k', linestyle='dashed')

        print('mu_airs', mu_airs.shape)
        print('transit_start', transit_start.shape)
        print('transit_middle', transit_middle.shape)
        print('transit_end', transit_end.shape)

        (
            airs_transit_start,
            airs_transit_middle,
            airs_transit_end,
            airs_middle_start,
            airs_middle_end,
            transit_slope_airs,
            start_index,
            end_index
        ) = get_incomplete_transit(
            mu_airs[idx, ...],
            transit_start[idx],
            transit_middle[idx],
            transit_end[idx],
            config['window'],
            1,
            True
        )

        print('draw transit_slope_airs', transit_slope_airs)

        line_start = np.arange(y_airs.shape[1]) * transit_slope_airs
        line_start = line_start - line_start[airs_transit_middle] + airs_middle_start
        line_start /= normalization_factor_airs

        plt.plot(np.arange(len(line_start)), line_start - shift_airs, c='orange', linestyle='dashed')

        line_start = np.arange(y_airs.shape[1]) * transit_slope_airs
        line_start = line_start - line_start[airs_transit_middle] + airs_middle_end
        line_start /= normalization_factor_airs

        plt.plot(np.arange(len(line_start)), line_start - shift_airs, c='green', linestyle='dashed')

        plt.legend()

        plt.show()

    output = (
        middle_airs,
        transit_low_airs,
        transit_start,
        transit_end
    )

    return output


def draw_figure(y, mu, mu_no_transit, label, binning, transit_middle, draw_scatter_plot=False):
    print('draw_figure', transit_middle)
    print('mu', mu.shape)
    print('mu_no_transit', mu_no_transit.shape)

    transit_middle = int(binning * transit_middle)

    level_no_transit = mu_no_transit[..., 0]

    normalization_factor = level_no_transit[transit_middle]
    print('normalization_factor', normalization_factor)

    if normalization_factor < 0:
        normalization_factor = 1

    level_no_transit = level_no_transit / normalization_factor
    level = mu[:, 0] / normalization_factor

    shift = level[transit_middle]

    level_no_transit = level_no_transit - shift
    level = level - shift

    x_range = np.arange(len(level))

    if draw_scatter_plot:
        plt.scatter(x_range / binning, y / normalization_factor - shift, alpha=0.4, s=0.5)

    plt.plot(x_range / binning, level_no_transit, c='purple', label=f'{label} no transit')
    plt.plot(x_range / binning, level, c='red', label=f'{label} with transit')

    return normalization_factor, shift


def generate_depth_estimation(file_prefix, config=None, fig_name=None):
    if config is None:
        config = data_loader.get_config()

    batch_size = config['depth_estimation_batch_size']

    fgs1_signal, airs_signal, star_info, target = data_loader.load_data(file_prefix)

    extension = 0

    airs_signal = airs_signal[:, extension: airs_signal.shape[1]-extension, ...]
    y_airs = np.nansum(airs_signal, axis=1)

    n_rows = star_info['multiplicity'].sum()

    selected_columns = ['Rs', 'Mp']
    extended_data = np.zeros((n_rows, len(selected_columns)))
    selected_data = star_info[selected_columns].values
    current_index = 0
    for i, multiplicity in enumerate(star_info['multiplicity']):
        extended_data[current_index: current_index + multiplicity, :] = selected_data[i, :]
        current_index += multiplicity

    result_list = []

    tick = datetime.datetime.now()
    start_time = tick
    print('Start time:', start_time)
    current_index = 0
    while current_index < n_rows:
        end_index = min(n_rows, current_index+batch_size)

        result = get_depth(
            fgs1_signal[current_index: end_index, ...],
            y_airs[current_index: end_index, ...],
            config,
            fig_name=fig_name
        )

        result_list.append(result)

        print(f'{current_index} / {n_rows}: {datetime.datetime.now() - tick}')
        tick = datetime.datetime.now()

        current_index += batch_size

    print('Time spent:', datetime.datetime.now() - start_time)

    column_names = [
        'middle_airs',
        'transit_low_airs',
        'transit_start_index',
        'transit_end_index'
    ]

    result_dict = {name: np.concatenate([x[i] for x in result_list], axis=0) for i, name in enumerate(column_names)}

    result_dict.update({name: extended_data[:, i] for i, name in enumerate(selected_columns)})

    df_result = pd.DataFrame(result_dict)

    cache_dir = util.get_dir('cache')
    df_result.to_csv(os.path.join(cache_dir, f'{file_prefix}_depth_info.csv'), index=False)

    return df_result


if __name__ == '__main__':
    print(jax.devices())
    print('num_devices:', jax.local_device_count())

    config = data_loader.get_config(use_kaggle_config=False)

    print(config)

    generate_depth_estimation('train', config, fig_name='Incomplete transit')
    #generate_depth_estimation('test', config)
