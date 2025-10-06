import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import datetime

from ariel_2025 import data_loader, kalman_smoother


def find_individual_breakpoints(mu, plot_config):
    trend = mu[:, 1]
    trend_diff = trend[1:] - trend[:-1]

    transit_start = jnp.argmin(trend)
    transit_end = jnp.argmax(trend)
    transit_mid = (transit_start + transit_end) / 2

    trend_diff_threshold = jnp.maximum(trend_diff[jnp.minimum(transit_start, len(trend_diff)-1)], trend_diff[jnp.minimum(transit_end, len(trend_diff)-1)])

    # Trend plot:
    #   0  1     2     3  4
    #   |  |  |  |  | /|\ |
    # --|\ | /|--|--|/ | \|--
    #   | \|/ |  |  |  |  |
    # Breakpoints:
    # 0) Trend goes from positive to negative, or trend_diff above trend_diff_threshold.
    # 1) Argmin of trend.
    # 2) Middle of transit. Halfway between 2 and 6.
    # 3) Argmax of trend.
    # 4) Trend goes from positive to negative, or trend_diff above trend_diff_threshold.

    positive_to_negative = jnp.logical_and(trend[:-1] > 0, trend[1:] < 0)
    negative_trend_normalization = trend_diff > trend_diff_threshold
    breakpoint_positive_to_negative = jnp.logical_or(positive_to_negative, negative_trend_normalization)

    index_range = jnp.arange(trend_diff.shape[0])

    breakpoints = (
        trend_diff.shape[0] - 1 - jnp.argmax(jnp.flip(jnp.logical_and(index_range < transit_start, breakpoint_positive_to_negative), axis=0)),
        transit_start,
        transit_mid,
        transit_end,
        jnp.argmax(jnp.logical_and(index_range > transit_end, breakpoint_positive_to_negative))
    )

    if plot_config is not None:
        axs = plot_config['axs']
        label = plot_config['label']
        color = plot_config['color']
        binning = plot_config['binning']

        mean_level = (np.nanmean(mu[:binning * 512, 0]) + np.nanmean(mu[-binning * 512:, 0])) / 2

        axs[0].plot(np.arange(mu.shape[0]) / binning, mu[:, 0] / mean_level, c=color, label=f'level {label}')

        axs[1].axhline(0, c='k', linestyle='dashed')
        axs[1].plot(np.arange(len(trend)) / binning, trend * binning / mean_level, c=color, label=f'trend {label}')

        axs[2].axhline(0, c='k', linestyle='dashed')
        axs[2].plot(np.arange(len(trend_diff)) / binning, trend_diff * binning ** 2 / mean_level, c=color, label=f'trend_diff {label}')

        axs[2].axhline(trend_diff_threshold * binning ** 2 / mean_level, linestyle='dotted', c=color, label=f'trend_diff_threshold {label}')

        for i in range(3):
            for b in breakpoints:
                axs[i].axvline(b / binning, c=color, linestyle='dashed')
            axs[i].legend()

    return breakpoints


def find_breakpoints(
        y_fgs1,
        y_airs,
        window=127,
        level_change_scaler=0.1,
        trend_change_scaler=0.8,
        noise_scaler=3.0,
        plot_config_fgs1=None,
        plot_config_airs=None,
        plot_combined_breakpoints=True
):
    vmapped_find_individual_breakpoints = find_individual_breakpoints
    for i in range(len(y_fgs1.shape) - 1):
        vmapped_find_individual_breakpoints = jax.vmap(vmapped_find_individual_breakpoints, in_axes=(0, None))

    if plot_config_fgs1 is None or plot_config_airs is None:
        vmapped_find_individual_breakpoints = jax.jit(vmapped_find_individual_breakpoints)

    mu_fgs1, _ = kalman_smoother.apply_kalman_smoother(
        y_fgs1,
        window=12*window+1,
        level_change_scaler=level_change_scaler,
        trend_change_scaler=trend_change_scaler,
        noise_scaler=noise_scaler
    )
    breakpoints_fgs1 = vmapped_find_individual_breakpoints(
        mu_fgs1,
        plot_config_fgs1
    )

    mu_airs, _ = kalman_smoother.apply_kalman_smoother(
        y_airs,
        window=window,
        level_change_scaler=level_change_scaler,
        trend_change_scaler=trend_change_scaler,
        noise_scaler=noise_scaler
    )
    breakpoints_airs = vmapped_find_individual_breakpoints(
        mu_airs,
        plot_config_airs
    )

    transit_middle = breakpoints_fgs1[2] / 24 + breakpoints_airs[2] / 2

    breakpoint_1 = breakpoints_fgs1[1] / 24 + breakpoints_airs[1] / 2
    breakpoint_3 = breakpoints_fgs1[3] / 24 + breakpoints_airs[3] / 2

    breakpoint_0_fgs1 = breakpoints_fgs1[0] / 12
    breakpoint_0_fgs1 = jnp.where(breakpoint_0_fgs1 >= breakpoint_1, 0, breakpoint_0_fgs1)
    breakpoint_0_airs = jnp.where(breakpoints_airs[0] >= breakpoint_1, 0, breakpoints_airs[0])
    transit_start = jnp.maximum(breakpoint_0_fgs1, breakpoint_0_airs)

    breakpoint_4_fgs1 = breakpoints_fgs1[4] / 12
    breakpoint_4_fgs1 = jnp.where(breakpoint_4_fgs1 <= breakpoint_3, mu_airs.shape[-2]-1, breakpoint_4_fgs1)
    breakpoint_4_airs = jnp.where(breakpoints_airs[4] <= breakpoint_3, mu_airs.shape[-2]-1, breakpoints_airs[4])
    transit_end = jnp.minimum(breakpoint_4_fgs1, breakpoint_4_airs)

    if plot_config_fgs1 is not None and plot_combined_breakpoints:
        axs = plot_config_fgs1['axs']

        mean_level = np.nanmean(mu_fgs1[:, 0])
        axs[0].scatter(np.arange(len(mu_fgs1)) / 12, y_fgs1 / mean_level - (mu_fgs1[:, 0] / mean_level)[int(12 * transit_middle)], alpha=0.4, s=0.5, c='C0')

        for i in range(3):
            axs[i].axvline(transit_start, linestyle='dashed', c='green')
            axs[i].axvline(transit_middle, linestyle='dashed', c='green')
            axs[i].axvline(transit_end, linestyle='dashed', c='green')

    output = (
        transit_start,
        transit_middle,
        transit_end
    )

    return output


if __name__ == '__main__':
    fgs1_signal, airs_signal, star_info, target = data_loader.load_data('train')

    print('fgs1_signal', fgs1_signal.shape)
    print('airs_signal', airs_signal.shape)
    print('target', target.values.shape)
    print('star_info', len(star_info))

    target = target.values[:, 1:]

    window = 127  #255

    tick = datetime.datetime.now()

    row_ids = [0, 549, 1067, 362, 393, 758, 759, 394, 1203, 736, 738, 476, 735, 128, 511, 103, 898, 251, 141, 492, 514]
    planet_ids = [0, 496, 968, 329, 357, 689, 689, 357, 1094, 668, 670, 429, 668, 120, 461, 95, 818, 231, 132, 444, 464]

    for target_id, signal_id in zip(planet_ids, row_ids):
        fig_name = f'{target_id} fgs1: {target[target_id, 0]:.7f} airs: {np.mean(target[target_id, 1:]):.7f}'
        print(fig_name)

        #signal_id = star_info.loc[target_id, 'first_index'] + (star_info.loc[target_id, 'multiplicity'] - 1)

        y_fgs1 = fgs1_signal[signal_id, :]
        y_airs = airs_signal[signal_id, ...]

        fig, axs = plt.subplots(3, sharex=True)
        fig.canvas.manager.set_window_title(fig_name)
        plt.subplots_adjust(bottom=0.06, top=0.96, left=0.085, right=0.96)

        plot_config_fgs1 = {
            'axs': axs,
            'binning': 12,
            'color': 'blue',
            'label': 'fgs1'
        }

        plot_config_airs = {
            'axs': axs,
            'binning': 1,
            'color': 'red',
            'label': 'airs'
        }

        find_breakpoints(
            y_fgs1.astype(np.float64),
            np.nansum(y_airs, axis=-2).astype(np.float64),
            window=window,
            level_change_scaler=0.0,
            trend_change_scaler=0.8,
            noise_scaler=5.0,
            plot_config_fgs1=plot_config_fgs1,
            plot_config_airs=plot_config_airs,
            plot_combined_breakpoints=False
        )

    print('Time spent:', datetime.datetime.now() - tick)

    plt.show()
