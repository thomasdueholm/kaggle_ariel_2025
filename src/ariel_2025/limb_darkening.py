import os

# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import jax.numpy as jnp
import jax
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import optax
from jax.sharding import PartitionSpec, NamedSharding
import gc
import seaborn as sns
import optuna

from ariel_2025 import util, data_loader, depth_estimation, standard_deviation

jax.config.update("jax_enable_x64", True)


def discrete_integral(r, d, u, v, n_steps):
    # https://www.astro.uvic.ca/~tatum/stellatm/atm6.pdf

    R_start = jnp.maximum(d - r, 0.0)
    R_stop = jnp.minimum(d + r, 1.0)

    eps = (R_stop - R_start) / n_steps
    t = jnp.linspace(R_start + eps / 2, R_stop - eps / 2, n_steps, dtype=jnp.float64)
    t = jax.lax.stop_gradient(t)
    t_squared = t ** 2

    inner = jnp.clip((t_squared + d[None, :] ** 2 - r ** 2) / (2 * t * d[None, :]), -1 + 1e-12, 1 - 1e-12)
    arccos = jnp.arccos(inner)

    limb_darkening = 1 - u * ((1 - v) * (1 - jnp.sqrt(jnp.maximum(1 - t_squared, 0))) - v * t_squared) # The last minus should have been plus for a more meaningful interpretation.

    sum = jnp.sum(2 * t * arccos * eps * limb_darkening, axis=0)

    return sum


def discrete_integral_no_transit(u, n_steps=10000):
    eps = 1 / n_steps

    t = jnp.linspace(0, 1, n_steps)
    limb_darkening = 1 - u * (1 - jnp.sqrt(1 - t ** 2))
    sum = jnp.sum(2 * jnp.pi * t * eps * limb_darkening)

    return sum


@partial(jax.jit, static_argnames=['data_length', 'n_steps'])
def get_ideal_transit(
        u,
        v,
        r,
        d_min,
        transit_middle,
        transit_length,
        data_length,
        n_steps
):
    x_start = jnp.sqrt(jnp.maximum(0, (r + 1) ** 2 - d_min ** 2))
    x_range = jnp.linspace(0, 2 * x_start, data_length)

    d = jnp.sqrt(jnp.maximum(0, d_min ** 2 + ((x_range - 2 * transit_middle * x_start) / transit_length) ** 2))

    transit = jax.checkpoint(discrete_integral, static_argnums=4)(r, d, u, v, n_steps)
    no_transit = discrete_integral_no_transit(u)

    ideal_transit = (no_transit - transit) / no_transit

    return ideal_transit


@partial(jax.jit, static_argnames=['n_steps', 'fit_all', 'polynomial_degree', 'ridge_regularization', 'normalize_error'])
def evaluate_error(
        signal,
        u,
        v,
        r,
        d_min,
        transit_middle,
        transit_length,
        transit_start,
        transit_end,
        n_steps,
        fit_all,
        polynomial_degree,
        ridge_regularization,
        normalize_error
):
    if fit_all:
        w = jnp.ones(signal.shape[-1], dtype=bool)
    else:
        selection_range = jnp.arange(signal.shape[-1])
        w = jnp.logical_and(
            selection_range >= transit_start,
            selection_range <= transit_end
        )
    w = jnp.where(jnp.isnan(signal), False, w)
    signal = jnp.nan_to_num(signal)

    transit = get_ideal_transit(
        u,
        v,
        r,
        d_min,
        transit_middle,
        transit_length,
        signal.shape[-1],
        n_steps
    )

    sum_w = jnp.maximum(1, jnp.sum(w))
    signal_mean = jnp.maximum(1e-7, jnp.sum(w * signal) / sum_w)
    no_dip_signal = signal / transit

    x_range = 2 * jnp.arange(no_dip_signal.shape[-1], dtype=jnp.float64) / no_dip_signal.shape[-1]
    X = x_range[:, None] ** jnp.arange(polynomial_degree + 1)[None, :]

    if polynomial_degree > 1:
        ridge = jnp.concatenate([jnp.zeros(2), ridge_regularization * jnp.ones(polynomial_degree - 1)])
    else:
        ridge = jnp.zeros(polynomial_degree + 1)

    ridge = jnp.diag(ridge)

    XTWX = X.mT @ (w[:, None] * X) + ridge
    XTWy = X.mT @ (w * no_dip_signal)

    XTWX = jnp.where(jnp.full(XTWX.shape, jnp.sum(w) == 0), jnp.identity(XTWX.shape[0]), XTWX)

    linear_regression_coefficients = jnp.linalg.solve(XTWX, XTWy)

    no_dip_prediction = X @ linear_regression_coefficients
    prediction = no_dip_prediction * transit

    if normalize_error:
        error = jnp.sum(w * ((prediction - signal) / signal_mean) ** 2) / sum_w
    else:
        error = jnp.sum(w * (prediction - signal) ** 2) / sum_w

    return error, linear_regression_coefficients, prediction, transit


@partial(jax.jit, static_argnames=['static_params'])
def evaluate_limb_darkening(
        params,
        data_dict,
        static_params
):
    v_fgs1, v_airs, ridge_regularization, fit_all, polynomial_degree, return_plot_data = static_params

    transit_start = data_dict['transit_start']
    transit_end = data_dict['transit_end']

    fgs1_signal = data_dict['fgs1_signal']
    airs_combined_signal = data_dict['airs_combined_signal']
    airs_sum_signal = data_dict['airs_sum_signal']

    u_fgs1 = jnp.clip(params['u_fgs1'], 0, 1)
    r_fgs1 = jnp.clip(params['r_fgs1'], 0, 1)
    d_min = jnp.clip(params['d_min'], 0, 1)

    fgs1_error, fgs1_linear_regression_coefficients, fgs1_prediction, fgs1_transit = evaluate_error(
        fgs1_signal,
        u_fgs1,
        v_fgs1,
        r_fgs1,
        d_min,
        params['transit_middle'],
        params['transit_length'],
        12 * transit_start,
        12 * transit_end,
        128,
        fit_all,
        polynomial_degree,
        ridge_regularization,
        True
    )

    u_airs_combined = jnp.clip(params['u_airs_combined'], 0, 1)
    r_airs_combined = jnp.clip(params['r_airs_combined'], 0, 1)

    airs_combined_error, airs_linear_regression_coefficients, airs_combined_prediction, airs_combined_transit = evaluate_error(
        airs_combined_signal,
        u_airs_combined,
        v_airs,
        r_airs_combined,
        d_min,
        params['transit_middle'],
        params['transit_length'],
        transit_start,
        transit_end,
        512,
        fit_all,
        polynomial_degree,
        ridge_regularization,
        True
    )

    u_airs_sum = jnp.clip(params['u_airs_sum'], 0, 1)
    r_airs_sum = jnp.clip(params['r_airs_sum'], 0, 1)

    airs_sum_error, airs_sum_linear_regression_coefficients, airs_sum_prediction, airs_sum_transit = evaluate_error(
        airs_sum_signal,
        u_airs_sum,
        v_airs,
        r_airs_sum,
        d_min,
        params['transit_middle'],
        params['transit_length'],
        transit_start,
        transit_end,
        512,
        fit_all,
        polynomial_degree,
        ridge_regularization,
        True
    )

    output_dict = {}

    if return_plot_data:
        output_dict.update({
            'airs_combined_linear_regression_coefficients': airs_linear_regression_coefficients,
            'airs_sum_linear_regression_coefficients': airs_sum_linear_regression_coefficients,
            'fgs1_linear_regression_coefficients': fgs1_linear_regression_coefficients,
            'airs_combined_prediction': airs_combined_prediction.astype(jnp.float32),
            'airs_sum_prediction': airs_sum_prediction.astype(jnp.float32),
            'fgs1_prediction': fgs1_prediction.astype(jnp.float32),
            'airs_combined_transit': airs_combined_transit.astype(jnp.float32),
            'airs_sum_transit': airs_sum_transit.astype(jnp.float32),
            'fgs1_transit': fgs1_transit.astype(jnp.float32),
        })

    # The constants 0.056 and 0.45 are chosen such that fgs1_error and airs_combined_error contribute
    # somewhat similar amounts to the error. The constant 10 assigns higher importance to airs_sum_error,
    # since it is less noisy, and the shared error is mainly used to estimate d_min, transit_middle, and
    # transit_length.

    combined_error = 0.056 * 57.846 * fgs1_error + 0.45 * 282 * airs_combined_error + 10 * 282 * airs_sum_error

    return combined_error, output_dict


@partial(jax.jit, static_argnames=['static_params'])
def evaluate_limb_darkening_spectrum(
        params,
        data_dict,
        static_params
):
    v_airs, ridge_regularization, u_airs_regularization, fit_all, polynomial_degree, return_plot_data = static_params

    transit_start = data_dict['transit_start']
    transit_end = data_dict['transit_end']

    airs_signal = data_dict['airs_signal']

    r_airs = jnp.clip(params['r_airs'], 0, 1)
    u_airs = jnp.clip(params['u_airs'], 0, 1)
    d_min = jnp.clip(params['d_min'], 0, 1)
    transit_middle = params['transit_middle']
    transit_length = params['transit_length']

    vmapped_evaluate_error = jax.vmap(
        evaluate_error,
        in_axes=(0, 0, None, 0, None, None, None, None, None, None, None, None, None, None)
    )

    error, linear_regression_coefficients, prediction, transit = vmapped_evaluate_error(
        airs_signal,
        u_airs,
        v_airs,
        r_airs,
        d_min,
        transit_middle,
        transit_length,
        transit_start,
        transit_end,
        32,
        fit_all,
        polynomial_degree,
        ridge_regularization,
        False
    )

    error = jnp.nanmean(error)

    if u_airs.shape[0] > 1:
        error += u_airs_regularization * jnp.var((u_airs[1:] - u_airs[:-1]) / jnp.mean(u_airs))

    intensity = jnp.nanmean(prediction / transit, axis=-1)

    output_dict = {
        'airs_linear_regression_coefficients': linear_regression_coefficients,
        'airs_transit': transit.astype(jnp.float32),
        'airs_intensity': intensity
    }

    if return_plot_data:
        output_dict.update({
            'airs_prediction': prediction.astype(jnp.float32),
        })

    return error, output_dict


def get_initial_params(df_depth, airs_combined_signal_shape):
    data_length = airs_combined_signal_shape[-1]

    airs_no_transit = df_depth['middle_airs'].values
    airs_transit = df_depth['transit_low_airs'].values

    airs_prediction = 1 - airs_transit / airs_no_transit

    transit_start = df_depth['transit_start_index'].values
    transit_end = df_depth['transit_end_index'].values

    transit_middle = (transit_start + transit_end) / (2 * data_length)
    transit_length = (transit_end - transit_start) / data_length

    params = {
        'u_fgs1': 0.5 * np.ones(airs_combined_signal_shape[:1]),
        'r_fgs1': np.sqrt(0.95 * airs_prediction),
        'u_airs_combined': 0.25 * np.ones(airs_combined_signal_shape[:1]),
        'r_airs_combined': np.sqrt(0.95 * airs_prediction),
        'u_airs_sum': 0.25 * np.ones(airs_combined_signal_shape[:1]),
        'r_airs_sum': np.sqrt(0.95 * airs_prediction),
        'd_min': 0.45 * np.ones(airs_combined_signal_shape[:1]),
        'transit_middle': transit_middle,
        'transit_length': 0.95 * transit_length
    }

    return params, transit_start, transit_end


def get_initial_spectrum_params(params, spectrum_length):
    # https://www.kaggle.com/competitions/ariel-data-challenge-2024/writeups/c-number-daiwakun-1st-place-solution

    params = {key: value.copy() for key, value in params.items()}

    u_airs = np.repeat(params['u_airs_combined'][:, None], spectrum_length, axis=1)
    r_airs = np.repeat(params['r_airs_combined'][:, None], spectrum_length, axis=1)

    spectrum_params = {
        'd_min': params['d_min'],
        'transit_middle': params['transit_middle'],
        'transit_length': params['transit_length'],
        'u_airs': u_airs,
        'r_airs': r_airs
    }

    return spectrum_params


def get_optimizer(params, config, steps_per_epoch, frozen_parameters=None):
    if frozen_parameters is None:
        frozen_parameters = []

    n_epochs = config.get('epochs', 1)
    learning_rate = config.get(f'learning_rate', 0.1)
    gradient_clipping_threshold = config.get('gradient_clipping_threshold', None)

    warmup_fn = optax.linear_schedule(
        init_value=learning_rate / 100,
        end_value=learning_rate,
        transition_steps=5 * steps_per_epoch
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=max(1, (n_epochs - 5) * steps_per_epoch)
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[5 * steps_per_epoch]
    )

    def zero_grads():
        # https://github.com/deepmind/optax/issues/159#issuecomment-896459491

        def init_fn(_):
            return ()

        def update_fn(updates, state, params=None):
            return jax.tree.map(jnp.zeros_like, updates), ()

        return optax.GradientTransformation(init_fn, update_fn)

    zero_grad = zero_grads()
    adam = optax.adam(b1=0.8, learning_rate=schedule_fn)

    optimizer_choice = {key: 'zero_grad' if key in frozen_parameters else 'adam' for key in params.keys()}

    optimizer = optax.multi_transform(
        {'zero_grad': zero_grad, 'adam': adam},
        optimizer_choice
    )

    if gradient_clipping_threshold is not None:
        optimizer = optax.chain(
            optax.clip(gradient_clipping_threshold),
            optimizer
        )

    optimizer_state = optimizer.init(params)

    return optimizer, optimizer_state


def train_limb_darkening(
        evaluation_fn,
        params,
        static_params,
        data_dict,
        config,
        frozen_parameters=None,
        return_plot_data=False
):
    print('train_limb_darkening')
    print(config)

    evaluation_fn = partial(evaluation_fn, static_params=static_params)

    n_epochs = config.get(f'epochs', 1)
    n_devices = config['n_devices']
    batch_size = config[f'batch_size']

    mesh = jax.make_mesh((n_devices,), ('batch',))
    sharding = NamedSharding(mesh, PartitionSpec('batch'))

    n_rows = next(iter(data_dict.values())).shape[0]
    steps_per_epoch = (n_rows + batch_size - 1) // batch_size

    def evaluate_batch(
            params,
            **data_dict
    ):
        vmapped_evaluate_limb_darkening = jax.vmap(evaluation_fn)
        batch_error, plot_data = vmapped_evaluate_limb_darkening(
            params,
            data_dict
        )
        batch_error_mean = jnp.mean(batch_error)

        return batch_error_mean, (batch_error, plot_data)

    dummy_params = {key: value[:1, ...] for key, value in params.items()}
    dummy_data_dict = {key: value[:1, ...] for key, value in data_dict.items()}

    dummy_error, (error_vector, dummy_plot_data) = evaluate_batch(dummy_params, **dummy_data_dict)
    plot_data_dict = {key: np.zeros((n_rows,) + value.shape[1:], dtype=value.dtype) for key, value in dummy_plot_data.items()}

    dummy_params = None
    dummy_data_dict = None
    dummy_error = None
    dummy_plot_data=None

    error_list = []

    batch_tick = datetime.datetime.now()

    # JAX jitted functions are cached in a dictionary since jax.clear_caches() is called to improve memory stability.
    jitted_functions = {}

    current_index = 0
    while current_index < n_rows:
        gc.collect()
        jax.clear_caches()

        next_index = min(current_index + batch_size, n_rows)
        if next_index - current_index > n_devices:
            next_index -= (next_index - current_index) % n_devices

        next_batch_tick = datetime.datetime.now()
        print(f'Current range: {current_index} -- {next_index},   {next_batch_tick - batch_tick}')
        batch_tick = next_batch_tick

        best_error = np.full((next_index - current_index,), 999999)
        best_params = {key: value[current_index: next_index, ...] for key, value in params.items()}

        if (next_index - current_index) % n_devices == 0:
            selected_params = {key: jax.device_put(value[current_index: next_index, ...], sharding) for key, value in params.items()}
            selected_data_dict = {key: jax.device_put(value[current_index: next_index, ...], sharding) for key, value in data_dict.items()}
        else:
            selected_params = {key: value[current_index: next_index, ...] for key, value in params.items()}
            selected_data_dict = {key: value[current_index: next_index, ...] for key, value in data_dict.items()}

        optimizer, optimizer_state = get_optimizer(selected_params, config, steps_per_epoch, frozen_parameters=frozen_parameters)

        if next_index - current_index not in jitted_functions:
            jitted_grad_fn = jax.value_and_grad(evaluate_batch, has_aux=True, argnums=0)
            jitted_grad_fn = jax.jit(jitted_grad_fn).lower(selected_params, **selected_data_dict).compile()
            jitted_functions[next_index - current_index] = jitted_grad_fn
        else:
            jitted_grad_fn = jitted_functions[next_index - current_index]

        epoch_start_time = datetime.datetime.now()
        for i in range(n_epochs):

            (error, (error_vector, plot_data)), grads = jitted_grad_fn(selected_params, **selected_data_dict)

            improved_params = (error_vector < best_error)
            best_error = np.where(improved_params, error_vector, best_error)
            best_params = {
                key: np.where(
                    np.expand_dims(improved_params, tuple(np.arange(1, len(value.shape)))),
                    selected_params[key],
                    value
                )
                for key, value in best_params.items()
            }

            if np.sum(improved_params) > 0:
                best_str = f'   NEW BEST! {np.sum(improved_params, dtype=np.int32)}'
            else:
                best_str = ''

            updates, optimizer_state = optimizer.update(grads, optimizer_state)
            selected_params = optax.apply_updates(selected_params, updates)

            for key in ['d_min', 'u_fgs1', 'r_fgs1', 'u_airs_combined', 'r_airs_combined', 'u_airs', 'r_airs']:
                if key in selected_params:
                    selected_params[key] = jnp.clip(selected_params[key], 1e-7, 1 - 1e-7)

            if i == n_epochs - 1:
                for key, value in plot_data.items():
                    plot_data_dict[key][current_index: next_index] = np.asarray(value)

                error_list.append(np.mean(best_error))

            if return_plot_data:
                print(f'epoch: {i + 1} ### ERROR: {error:.9f}, time: {datetime.datetime.now() - epoch_start_time},   {datetime.datetime.now()}{best_str}')
                epoch_start_time = datetime.datetime.now()

        for key, value in best_params.items():
            params[key][current_index: next_index, ...] = value

        current_index = next_index

    print('Combined error:', np.mean(error_list))

    return params, plot_data_dict


def remove_outliers(signal, window=255, outlier_threshold=4.0):
    for i in range(signal.shape[0]):
        series = pd.Series(signal[i, :])

        mean = series.rolling(window, min_periods=window // 3, center=True).mean().values
        std = series.rolling(window, min_periods=window // 3, center=True).std().values

        outlier_mask = (
                (signal[i, :] >= mean + outlier_threshold * std) |
                (signal[i, :] <= mean - outlier_threshold * std)
        )

        signal[i, :] = np.where(outlier_mask, np.nan, signal[i, :])

    return signal


def get_combined_deviation(signal, prediction, transit_start, transit_end, window=255):
    deviation = np.zeros(signal.shape[0])

    selection_range = np.arange(signal.shape[-1])

    for i in range(signal.shape[0]):
        series = pd.Series(signal[i, :])

        mean = series.rolling(window, min_periods=window // 3, center=True).mean().values
        std = series.rolling(window, min_periods=window // 3, center=True).std().values

        error = np.abs(prediction[i, :] - mean) / std

        selection_mask = (selection_range >= transit_start[i]) & (selection_range <= transit_end[i])
        deviation[i] = np.nanmean(error[selection_mask])

    return deviation


@partial(jax.jit, static_argnames=['static_params'])
def evaluate_limb_darkening_gain_drift(
        params,
        data_dict,
        static_params
):
    v_airs, u_airs_regularization, return_prediction, return_transit = static_params

    wavelengths = data_dict['wavelengths']
    airs_signal = data_dict['airs_signal']

    w = 1.0 - jnp.isnan(airs_signal)
    sum_w = jnp.maximum(1, jnp.sum(w))
    airs_signal = jnp.nan_to_num(airs_signal)

    r_airs = jnp.clip(params['r_airs'], 0, 1)
    u_airs = jnp.clip(params['u_airs'], 0, 1)
    d_min = jnp.clip(params['d_min'], 0, 1)
    transit_middle = params['transit_middle']
    transit_length = params['transit_length']
    f_coefficients = params['f_coefficients']
    g_coefficients = params['g_coefficients']

    time_range = 2 * jnp.arange(airs_signal.shape[-1], dtype=jnp.float64) / airs_signal.shape[-1]

    polynomial_degree = f_coefficients.shape[-1] - 1
    X = time_range[:, None] ** jnp.arange(polynomial_degree+1)[None, :]
    f_values = X @ f_coefficients

    polynomial_degree = g_coefficients.shape[-1] - 1
    X = wavelengths[:, None] ** jnp.arange(polynomial_degree + 1)[None, :]
    g_values = X @ g_coefficients

    fg_product = g_values[:, None] * f_values[None, :]

    vmapped_get_ideal_transit = jax.vmap(get_ideal_transit, in_axes=(0, None, 0, None, None, None, None, None))

    transit = vmapped_get_ideal_transit(
        u_airs,
        v_airs,
        r_airs,
        d_min,
        transit_middle,
        transit_length,
        airs_signal.shape[-1],
        32
    )

    scaled_transit = transit * (1 + fg_product)

    XTWX = jnp.sum(w * scaled_transit ** 2, axis=-1)
    XTWy = jnp.sum(w * scaled_transit * airs_signal, axis=-1)

    XTWX = jnp.where(jnp.sum(w, 1) == 0, jnp.ones_like(XTWX), XTWX)

    intensity = XTWy / XTWX

    prediction = intensity[:, None] * scaled_transit

    error = jnp.sum(w * (prediction - airs_signal) ** 2) / sum_w

    if u_airs.shape[0] > 1:
        error += u_airs_regularization * jnp.var((u_airs[1:] - u_airs[:-1]) / jnp.mean(u_airs))

    output_dict = {}

    if return_prediction:
        output_dict.update({
            'airs_prediction': prediction.astype(jnp.float32)
        })

    if return_transit:
        output_dict.update({
            'airs_transit': transit.astype(jnp.float32),
            'airs_intensity': intensity
        })

    return error, output_dict


def get_gain_error(params, data_dict, static_params):
    f_coefficients = params['f_coefficients']
    f_coefficients = f_coefficients / jnp.maximum(1e-7, jnp.max(jnp.abs(f_coefficients)))

    signal = data_dict['airs_signal']
    intensity = data_dict['intensity']
    transit = data_dict['transit']
    wavelengths = data_dict['wavelengths']

    return_plot_data = static_params[0]

    time_range = 2 * jnp.arange(signal.shape[-1], dtype=jnp.float64) / signal.shape[-1]

    polynomial_degree = f_coefficients.shape[-1] - 1
    X = time_range[:, None] ** jnp.arange(polynomial_degree+1)[None, :]
    f_values = X @ f_coefficients

    w = 1.0 - jnp.isnan(signal)
    sum_w = jnp.maximum(1, jnp.sum(w))
    signal = jnp.nan_to_num(signal)

    X = wavelengths[:, None] ** jnp.arange(polynomial_degree+1)[None, :]
    X = X[:, None, :] * f_values[None, :, None]

    intensified_transit = transit * intensity[:, None]
    X *= intensified_transit[:, :, None]

    y = signal - intensified_transit

    X = X.reshape((-1, X.shape[-1]))
    y = y.flatten()
    w = w.flatten()

    XTWX = X.mT @ (w[:, None] * X)
    XTWy = X.mT @ (w * y)

    XTWX = jnp.where(jnp.full(XTWX.shape, jnp.sum(w) == 0), jnp.identity(XTWX.shape[0]), XTWX)

    linear_regression_coefficients = jnp.linalg.solve(XTWX, XTWy)

    prediction = X @ linear_regression_coefficients

    error = jnp.sum(w * (prediction - y) ** 2) / sum_w

    prediction = prediction.reshape((wavelengths.shape[0], -1))
    prediction = prediction + intensified_transit

    output_dict = {
        'linear_regression_coefficients': linear_regression_coefficients,
        'f_coefficients': f_coefficients
    }

    if return_plot_data:
        output_dict.update({
            'prediction': prediction.astype(jnp.float32)
        })

    return error, output_dict


def estimate_gain_drift(config, airs_signal, transit, intensity, f_coefficients, wavelengths, show_plot=False, verbose=False):
    params = {
        'f_coefficients': f_coefficients
    }

    static_params = (show_plot,)

    data_dict = {
        'airs_signal': airs_signal,
        'intensity': intensity,
        'transit': transit,
        'wavelengths': np.repeat(wavelengths[None, :], airs_signal.shape[0], axis=0)
    }

    params, output_dict = train_limb_darkening(
        get_gain_error,
        params,
        static_params,
        data_dict,
        config,
        return_plot_data=verbose
    )

    f_coefficients = output_dict['f_coefficients']
    g_coefficients = output_dict['linear_regression_coefficients']

    if show_plot:
        for i in range(airs_signal.shape[0]):
            plt.figure(f'{i} {g_coefficients[i, ...]}')

            selected_f_coefficients = f_coefficients[i, ...]
            selected_g_coefficients = g_coefficients[i, ...]

            time_range = 2 * jnp.arange(airs_signal.shape[-1], dtype=jnp.float64) / airs_signal.shape[-1]

            polynomial_degree = selected_f_coefficients.shape[-1] - 1
            X = time_range[:, None] ** jnp.arange(polynomial_degree + 1)[None, :]
            f_values = X @ selected_f_coefficients

            spectrum_range = 2 * jnp.arange(wavelengths.shape[0], dtype=jnp.float64) / wavelengths.shape[0]
            X = spectrum_range[:, None] ** jnp.arange(polynomial_degree + 1)[None, :]
            g_values = X @ selected_g_coefficients

            product = f_values[:, None] * g_values[None, :]

            max_val = np.max(np.abs(product))

            sns.heatmap(product, cmap="coolwarm", vmin=-max_val, vmax=max_val)

    return f_coefficients, g_coefficients


def generate_limb_darkening(
        file_prefix,
        config,
        plot_data=False,
        from_rows=0,
        to_rows=None,
        save_data=True,
        outlier_planet_ids=None
):
    fgs1_signal, airs_signal, star_info, target = data_loader.load_data(file_prefix)

    cache_dir = util.get_dir('cache')
    df_depth = pd.read_csv(os.path.join(cache_dir, f'{file_prefix}_depth_info.csv'))

    #####################################

    extended_planet_ids = np.zeros(len(df_depth), dtype=np.int64)
    for planet_id, multiplicity, first_index in zip(star_info['planet_id'], star_info['multiplicity'], star_info['first_index']):
        extended_planet_ids[first_index: first_index+multiplicity] = planet_id

    if outlier_planet_ids is not None:
        outlier_planet_ids = np.array(outlier_planet_ids)
        selected_planet_ids = star_info['planet_id'].values

        outliers = np.isin(selected_planet_ids, outlier_planet_ids)

        extended_mask = np.zeros(star_info['multiplicity'].sum(), dtype=bool)

        star_info = star_info.loc[outliers]
        target = target.loc[outliers]

        for multiplicity, first_index in zip(star_info['multiplicity'], star_info['first_index']):
            extended_mask[first_index: first_index+multiplicity] = True

        fgs1_signal = fgs1_signal[extended_mask, ...]
        airs_signal = airs_signal[extended_mask, ...]
        df_depth = df_depth.loc[extended_mask]
        extended_planet_ids = extended_planet_ids[extended_mask]
    elif to_rows is not None:
        fgs1_signal = fgs1_signal[from_rows: to_rows, ...]
        airs_signal = airs_signal[from_rows: to_rows, ...]
        df_depth = df_depth.loc[from_rows: to_rows - 1, ...]
        extended_planet_ids = extended_planet_ids[from_rows: to_rows]

    #####################################

    airs_sum_signal = np.nansum(airs_signal, axis=1)

    airs_combined_weights = np.nanmean(airs_signal, axis=-1, keepdims=True)
    airs_combined_signal = np.nansum(airs_signal / airs_combined_weights, axis=1)
    gc.collect()

    airs_sum_signal = remove_outliers(airs_sum_signal)
    airs_combined_signal = remove_outliers(airs_combined_signal)
    params, transit_start, transit_end = get_initial_params(df_depth, airs_combined_signal.shape)

    extra_data = config.get('limb_darkening_extra_data', 0)
    transit_start = transit_start - extra_data
    transit_end = transit_end + extra_data

    data_dict = {
        'fgs1_signal': fgs1_signal,
        'airs_combined_signal': airs_combined_signal,
        'airs_sum_signal': airs_sum_signal,
        'transit_start': transit_start,
        'transit_end': transit_end
    }

    training_config = {
        'epochs': config.get(f'limb_darkening_epochs', 1),
        'n_devices': config.get(f'n_devices_limb_darkening', 1),
        'batch_size': config['limb_darkening_batch_size_1'],
        'gradient_clipping_threshold': config['limb_darkening_gradient_clipping_threshold']
    }

    v_fgs1 = config['limb_darkening_v_fgs1']
    v_airs = config['limb_darkening_v_airs']
    ridge_regularization = config['limb_darkening_ridge_regularization']

    learning_rates_list = [0.011, 0.006, 0.001]
    ridge_regularization_list = [3000, 3000, ridge_regularization]
    fit_all_list = [False, True, True]
    polynomial_degree_list = [1, 4, 4]

    for i in range(len(learning_rates_list)):
        if i == len(learning_rates_list) - 1:
            static_params = (v_fgs1, v_airs, ridge_regularization_list[i], fit_all_list[i], polynomial_degree_list[i], plot_data)
        else:
            static_params = (v_fgs1, v_airs, ridge_regularization_list[i], fit_all_list[i], polynomial_degree_list[i], False)

        print()
        print(datetime.datetime.now(), f'{i}: Training with learning rate {learning_rates_list[i]:.5f} and polynomial_degree = {polynomial_degree_list[i]}.')

        training_config['learning_rate'] = learning_rates_list[i]

        params, data_for_plotting = train_limb_darkening(
            evaluate_limb_darkening,
            params,
            static_params,
            data_dict,
            training_config,
            return_plot_data=plot_data
        )

    for key, value in params.items():
        print(key, np.mean(value), np.std(value))

    if not plot_data:
        fgs1_signal = None
        airs_combined_signal = None
        data_dict = None
        data_for_plotting = None

        gc.collect()

    ######################################################################

    print()
    print(datetime.datetime.now(), 'Initial training with airs_signal.')

    spectrum_params = get_initial_spectrum_params(params, 282)

    spectrum_data_dict = {
        'airs_signal': airs_signal,
        'transit_start': transit_start,
        'transit_end': transit_end
    }

    u_airs_regularization = config[f'limb_darkening_u_airs_regularization']

    fit_all = fit_all_list[-1]
    polynomial_degree = polynomial_degree_list[-1]

    static_params = (v_airs, ridge_regularization, u_airs_regularization, fit_all, polynomial_degree, plot_data)

    training_config.update({
        'batch_size': config['limb_darkening_batch_size_2'],
        'learning_rate': 0.007,
        'epochs': 2 * config.get('fine_tuning_epochs', 30)
    })

    spectrum_params, spectrum_data_for_plotting = train_limb_darkening(
        evaluate_limb_darkening_spectrum,
        spectrum_params,
        static_params,
        spectrum_data_dict,
        training_config,
        frozen_parameters=['d_min', 'transit_middle', 'transit_length', 'u_airs'],#
        return_plot_data=plot_data
    )

    f_coefficients = spectrum_data_for_plotting['airs_linear_regression_coefficients']
    f_coefficients = np.mean(f_coefficients, axis=1)
    f_coefficients[:, 0] -= 1

    spectrum_params['f_coefficients'] = f_coefficients

    gain_config = training_config.copy()
    gain_config.update({
        'epochs': config.get('gain_epochs', 60),
        'batch_size': config.get('gain_drift_batch_size', 50),
        'gradient_clipping_threshold': None
    })

    gain_learning_rates = [200.0, 0.1, 0.03]

    fine_tune_config = training_config.copy()
    fine_tune_config.update({
        'epochs': config.get('fine_tuning_epochs', 30),
        'batch_size': config.get('limb_darkening_batch_size_2', 50),
        'gradient_clipping_threshold': 0.1
    })

    fine_tune_learning_rates = [0.005, 0.002, 0.001]

    n_repeats = 3

    for i in range(n_repeats):
        transit = spectrum_data_for_plotting['airs_transit']
        intensity = spectrum_data_for_plotting['airs_intensity']

        f_coefficients = spectrum_params['f_coefficients']

        data_dir = util.get_dir('data')
        wavelengths = pd.read_csv(os.path.join(data_dir, f'wavelengths.csv')).values[0, 1:]

        plot_gain = False
        if i == n_repeats-1:
            plot_gain = False

        print()
        print(datetime.datetime.now(), f'{i}: Drift estimation.')
        print('f_coefficients', f_coefficients.shape)

        gain_config['learning_rate'] = gain_learning_rates[i]
        if i == 0:
            gain_config['epochs'] = 2*config.get('gain_epochs', 60)
        else:
            gain_config['epochs'] = config.get('gain_epochs', 60)

        f_coefficients, g_coefficients = estimate_gain_drift(
            gain_config,
            airs_signal,
            transit,
            intensity,
            f_coefficients,
            wavelengths,
            show_plot=(plot_data and plot_gain),
            verbose=plot_data
        )

        if not plot_data:
            transit = None
            intensity = None
            spectrum_data_for_plotting = None
            gc.collect()

        #############################################################

        spectrum_data_dict = {
            'airs_signal': airs_signal,
            'wavelengths': np.repeat(wavelengths[None, :], airs_signal.shape[0], axis=0)
        }

        spectrum_params['f_coefficients'] = f_coefficients
        spectrum_params['g_coefficients'] = g_coefficients

        if i < n_repeats-1:
            static_params = (v_airs, u_airs_regularization, False, True)
        else:
            static_params = (v_airs, u_airs_regularization, True, plot_data)

        print()
        print(datetime.datetime.now(), f'{i}: Correction of parameters other than f_coefficients and g_coefficients.')

        fine_tune_config['learning_rate'] = fine_tune_learning_rates[i]

        spectrum_params, spectrum_data_for_plotting = train_limb_darkening(
            evaluate_limb_darkening_gain_drift,
            spectrum_params,
            static_params,
            spectrum_data_dict,
            fine_tune_config,
            frozen_parameters=['d_min', 'transit_middle', 'transit_length', 'f_coefficients', 'g_coefficients'],#, 'u_airs'
            return_plot_data=plot_data
        )

    airs_nonlinearity = np.sum(np.abs(spectrum_params['f_coefficients'][:, 2:]), axis=-1) * np.sum(np.abs(spectrum_params['g_coefficients'][:, 1:]), axis=-1)
    airs_nonlinearity = airs_nonlinearity / np.nanmean(airs_signal, axis=(1, 2))

    gc.collect()

    valid_pixels = ~np.all(np.isnan(airs_signal), axis=-1)

    r_airs = spectrum_params['r_airs']
    r_airs = np.where(valid_pixels, r_airs, np.nan)

    if not plot_data:
        airs_signal = None
        spectrum_data_for_plotting = None
        gc.collect()

    data_dict = {
        'u_fgs1': params['u_fgs1'],
        'r_fgs1': params['r_fgs1'],
        'u_airs_combined': params['u_airs_combined'],
        'r_airs_combined': params['r_airs_combined'],
        'u_airs_sum': params['u_airs_sum'],
        'r_airs_sum': params['r_airs_sum'],
        'd_min_fgs1': params['d_min'],
        'transit_length_fgs1': params['transit_length'],
        'd_min_airs': spectrum_params['d_min'],
        'transit_length_airs': spectrum_params['transit_length'],
        'airs_nonlinearity': airs_nonlinearity
    }

    df_limb_darkening = pd.DataFrame(data_dict)

    if save_data and not plot_data:
        cache_dir = util.get_dir('cache')
        df_limb_darkening.to_csv(os.path.join(cache_dir, f'{file_prefix}_limb_darkening.csv'), index=False)

        np.save(os.path.join(cache_dir, f'{file_prefix}_u_airs_limb_darkening.npy'), spectrum_params['u_airs'])
        np.save(os.path.join(cache_dir, f'{file_prefix}_r_airs_limb_darkening.npy'), r_airs)

    plot_params = False
    if plot_params and plot_data:
        for key, value in params.items():
            util.draw_histogram(value, 'C0', key, key)

    plot_scatter = False
    if plot_scatter and plot_data:
        fgs1_prediction = data_for_plotting['fgs1_prediction']

        fgs1_transit = data_for_plotting['fgs1_transit']

        transit_middle = np.round(params['transit_middle'] * airs_combined_signal.shape[-1]).astype(np.int32)
        transit_length = np.round(params['transit_length'] * airs_sum_signal.shape[-1]).astype(np.int32)
        transit_start = transit_middle - transit_length // 2
        transit_end = transit_middle + transit_length // 2

        for i in range(len(df_depth)):
            u_fgs1 = params['u_fgs1'][i]
            d_min = params['d_min'][i]

            fgs1_x_range = np.arange(fgs1_signal.shape[-1])

            plt.figure(f'{i} fgs1 {extended_planet_ids[i]}, u_fgs1={u_fgs1}, d_min={d_min}')

            plt.scatter(fgs1_x_range, fgs1_signal[i, ...], alpha=0.3, s=0.4)

            plt.plot(fgs1_x_range, fgs1_prediction[i, ...] / fgs1_transit[i, ...], c='g')
            plt.plot(fgs1_x_range, fgs1_prediction[i, ...], c='r')

            plt.axvline(transit_start[i] * 12, c='k', linestyle='dashed')
            plt.axvline(transit_end[i] * 12, c='k', linestyle='dashed')

    plot_scatter_airs_combined = True
    if plot_scatter_airs_combined and plot_data:
        airs_combined_prediction = data_for_plotting['airs_combined_prediction']

        transit_middle = np.round(params['transit_middle'] * airs_combined_signal.shape[-1]).astype(np.int32)
        transit_length = np.round(params['transit_length'] * airs_sum_signal.shape[-1]).astype(np.int32)
        transit_start = transit_middle - transit_length // 2
        transit_end = transit_middle + transit_length // 2

        airs_combined_transit = data_for_plotting['airs_combined_transit']

        linear_regression_coefficients = data_for_plotting['airs_combined_linear_regression_coefficients']

        for i in range(len(df_depth)):
            u_airs_combined_x_range = np.arange(airs_combined_signal.shape[-1])

            plt.figure(f'{i} {extended_planet_ids[i]}, airs_combined {linear_regression_coefficients[i, 2:]}')

            plt.scatter(u_airs_combined_x_range, airs_combined_signal[i, ...], alpha=0.4, s=0.6)

            plt.plot(u_airs_combined_x_range, airs_combined_prediction[i, ...] / airs_combined_transit[i, ...], c='g')
            plt.plot(u_airs_combined_x_range, airs_combined_prediction[i, ...], c='r')

            plt.axvline(transit_start[i], c='k', linestyle='dashed')
            plt.axvline(transit_end[i], c='k', linestyle='dashed')
            plt.axvline(transit_middle[i], c='k', linestyle='dashed')

    plot_scatter_airs_sum = True
    if plot_scatter_airs_sum and plot_data:
        airs_sum_prediction = data_for_plotting['airs_sum_prediction']

        transit_middle = np.round(params['transit_middle'] * airs_sum_signal.shape[-1]).astype(np.int32)
        transit_length = np.round(params['transit_length'] * airs_sum_signal.shape[-1]).astype(np.int32)
        transit_start = transit_middle - transit_length // 2
        transit_end = transit_middle + transit_length // 2

        airs_sum_transit = data_for_plotting['airs_sum_transit']

        linear_regression_coefficients = data_for_plotting['airs_sum_linear_regression_coefficients']

        for i in range(len(df_depth)):
            u_airs_sum_x_range = np.arange(airs_sum_signal.shape[-1])

            plt.figure(f'{i} {extended_planet_ids[i]}, airs_sum {linear_regression_coefficients[i, 2:]}')

            plt.scatter(u_airs_sum_x_range, airs_sum_signal[i, ...], alpha=0.4, s=0.6)

            plt.plot(u_airs_sum_x_range, airs_sum_prediction[i, ...] / airs_sum_transit[i, ...], c='g')
            plt.plot(u_airs_sum_x_range, airs_sum_prediction[i, ...], c='r')

            plt.axvline(transit_start[i], c='k', linestyle='dashed')
            plt.axvline(transit_end[i], c='k', linestyle='dashed')
            plt.axvline(transit_middle[i], c='k', linestyle='dashed')

    plot_scatter_airs = False
    if plot_scatter_airs and plot_data:
        transit_middle = np.round(params['transit_middle'] * airs_combined_signal.shape[-1]).astype(np.int32)
        transit_length = np.round(params['transit_length'] * airs_sum_signal.shape[-1]).astype(np.int32)
        transit_start = transit_middle - transit_length // 2
        transit_end = transit_middle + transit_length // 2

        idx = 0

        airs_transit = spectrum_data_for_plotting['airs_transit']
        airs_transit = airs_transit[idx, ...]

        airs_prediction = spectrum_data_for_plotting['airs_prediction']
        airs_prediction = airs_prediction[idx, ...]

        selected_signal = airs_signal[idx, ...]

        for i in range(215, 225):
            x_range = np.arange(selected_signal.shape[-1])

            plt.figure(f'airs_spectrum {idx} {extended_planet_ids[idx]} {i}')

            plt.scatter(x_range, selected_signal[i, ...], alpha=0.4, s=0.6)

            plt.plot(x_range, airs_prediction[i, ...] / airs_transit[i, ...], c='g')
            plt.plot(x_range, airs_prediction[i, ...], c='r')

            plt.axvline(transit_start[idx], c='k', linestyle='dashed')
            plt.axvline(transit_end[idx], c='k', linestyle='dashed')
            plt.axvline(transit_middle[idx], c='k', linestyle='dashed')

    plot_error = False
    if plot_error and plot_data:
        extended_target_airs, extended_target_fgs1, _, _ = depth_estimation.get_extended_airs_target(target, star_info, from_rows, to_rows)
        extended_target_airs = extended_target_airs.reshape((extended_target_fgs1.shape[0], -1))

        print('extended_target_airs', extended_target_airs.shape)

        from ariel_2025 import spectrum_estimation

        limb_darkening_spectrum = spectrum_estimation.get_simple_spectrum(
            spectrum_params['r_airs'] ** 2,
            smoothing_window=config['spectrum_smoothing_window'],
            level_change_scaler=config['spectrum_level_change_scaler'],
            trend_change_scaler=config['spectrum_trend_change_scaler'],
            noise_scaler=config['spectrum_noise_scaler']
        )

        util.draw_histogram(spectrum_params['r_airs'] ** 2 - extended_target_airs, 'C0', 'r_airs', 'error')
        util.draw_histogram(params['r_fgs1'] ** 2 - extended_target_fgs1, 'C1', 'r_fgs1', 'error')
        util.draw_histogram(limb_darkening_spectrum - extended_target_airs, 'C3', 'r_airs smoothed', 'error')

        r_airs_shifted = spectrum_params['r_airs'] ** 2
        r_airs_shifted = r_airs_shifted - np.mean(r_airs_shifted, axis=1, keepdims=True)
        target_shifted = extended_target_airs - np.mean(extended_target_airs, axis=1, keepdims=True)
        util.draw_histogram(r_airs_shifted - target_shifted, 'C0', 'r_airs_shifted', 'shifted error')

        limb_darkening_spectrum_shifted = limb_darkening_spectrum - np.mean(limb_darkening_spectrum, axis=1, keepdims=True)
        util.draw_histogram(limb_darkening_spectrum_shifted - target_shifted, 'C3', 'r_airs_shifted smoothed', 'shifted error')

        util.draw_histogram(-target_shifted, 'C4', 'constant', 'shifted error')

    plot_airs = False
    if plot_airs and plot_data:
        extended_target_airs, extended_target_fgs1, _, _ = depth_estimation.get_extended_airs_target(target, star_info, from_rows, to_rows)
        extended_target_airs = extended_target_airs.reshape((extended_target_fgs1.shape[0], -1))

        from ariel_2025 import spectrum_estimation

        limb_darkening_spectrum = spectrum_estimation.get_simple_spectrum(
            spectrum_params['r_airs'] ** 2,
            smoothing_window=config['spectrum_smoothing_window'],
            level_change_scaler=config['spectrum_level_change_scaler'],
            trend_change_scaler=config['spectrum_trend_change_scaler'],
            noise_scaler=config['spectrum_noise_scaler']
        )

        u_airs = spectrum_params['u_airs']

        for i in range(len(df_depth)):
            r_airs = spectrum_params['r_airs'][i, :] ** 2

            r_airs_smoothed = pd.Series(r_airs).rolling(11, min_periods=1, center=True).mean().values

            r_airs_smoothed2 = limb_darkening_spectrum[i, ...]

            selected_u = u_airs[i, ...]

            fig, axs = plt.subplots(2, sharex=True)
            fig.canvas.manager.set_window_title(f'plot_airs {i}')

            axs[0].plot(np.arange(len(r_airs)), extended_target_airs[i, ...], label='target')

            axs[0].plot(np.arange(len(r_airs)), r_airs, label='r')
            axs[0].plot(np.arange(len(r_airs)), r_airs_smoothed, label='r_smoothed')
            axs[0].plot(np.arange(len(r_airs)), r_airs_smoothed2, label='r_airs_smoothed2')

            axs[1].plot(np.arange(selected_u.shape[0]), selected_u, label='u', c='C0')
            axs[1].axhline(0, c='k', linestyle='dashed')

            axs[0].legend()
            axs[1].legend()

    if plot_data:
        plt.show()


def load_limb_darkening(file_prefix):
    cache_dir = util.get_dir('cache')

    df_limb_darkening = pd.read_csv(os.path.join(cache_dir, f'{file_prefix}_limb_darkening.csv'))

    limb_darkening_u_airs = np.load(os.path.join(cache_dir, f'{file_prefix}_u_airs_limb_darkening.npy'))
    limb_darkening_r_airs = np.load(os.path.join(cache_dir, f'{file_prefix}_r_airs_limb_darkening.npy'))

    return df_limb_darkening, limb_darkening_r_airs, limb_darkening_u_airs


def hyperparameter_search(config=None):
    if config is None:
        config = data_loader.get_config()
    print(config)

    file_prefix = 'train'

    fgs1_signal, airs_signal, star_info, df_target = data_loader.load_data(file_prefix)

    cache_dir = util.get_dir('cache')
    df_depth = pd.read_csv(os.path.join(cache_dir, f'{file_prefix}_depth_info.csv'))

    df_target, target_weights = standard_deviation.get_extended_target(star_info, df_target)

    fgs1_target = df_target.values[:, 1]
    airs_target = np.mean(df_target.values[:, 2:], axis=-1)

    #####################################

    airs_sum_signal = np.nansum(airs_signal, axis=1)

    airs_combined_weights = np.nanmean(airs_signal, axis=-1, keepdims=True)
    airs_combined_signal = np.nansum(airs_signal / airs_combined_weights, axis=1)
    gc.collect()

    airs_sum_signal = remove_outliers(airs_sum_signal)
    airs_combined_signal = remove_outliers(airs_combined_signal)
    params, transit_start, transit_end = get_initial_params(df_depth, airs_combined_signal.shape)

    extra_data = config.get('limb_darkening_extra_data', 0)
    transit_start = transit_start - extra_data
    transit_end = transit_end + extra_data

    data_dict = {
        'fgs1_signal': fgs1_signal,
        'airs_combined_signal': airs_combined_signal,
        'airs_sum_signal': airs_sum_signal,
        'transit_start': transit_start,
        'transit_end': transit_end
    }

    training_config = {
        'epochs': config.get(f'limb_darkening_epochs', 1),
        'n_devices': config.get(f'n_devices_limb_darkening', 1),
        'batch_size': config['limb_darkening_batch_size_1'],
        'gradient_clipping_threshold': config['limb_darkening_gradient_clipping_threshold']
    }

    def objective(trial, initial_params, data_dict, fgs1_target, airs_target, training_config):
        params = {key: value.copy() for key, value in initial_params.items()}

        hyperparameters = {
            'ridge_regularization': trial.suggest_float('ridge_regularization', 20, 500),  # 75, #
            'v_fgs1': trial.suggest_float('v_fgs1', 0.01, 0.025),  # 0.025, #
            'v_airs': trial.suggest_float('v_airs', 0.06, 0.12)  # 0.1 #
        }

        print('hyperparameters', hyperparameters)

        v_fgs1 = hyperparameters['v_fgs1']
        v_airs = hyperparameters['v_airs']
        ridge_regularization = hyperparameters['ridge_regularization']

        learning_rates_list = [0.011, 0.006, 0.001]
        ridge_regularization_list = [3000, 3000, ridge_regularization]
        fit_all_list = [False, True, True]
        polynomial_degree_list = [1, 4, 4]

        for i in range(len(learning_rates_list)):
            static_params = (v_fgs1, v_airs, ridge_regularization_list[i], fit_all_list[i], polynomial_degree_list[i], False)

            print()
            print(datetime.datetime.now(), f'{i}: Training with learning rate {learning_rates_list[i]:.5f} and polynomial_degree = {polynomial_degree_list[i]}.')

            training_config['learning_rate'] = learning_rates_list[i]

            params, _ = train_limb_darkening(
                evaluate_limb_darkening,
                params,
                static_params,
                data_dict,
                training_config,
                return_plot_data=False
            )

            fgs1_prediction = params['r_fgs1'] ** 2
            airs_prediction = params['r_airs_combined'] ** 2

            fgs1_error = np.std(fgs1_prediction - fgs1_target)
            airs_error = np.std(airs_prediction - airs_target)

            score = (57.846 * fgs1_error + 282 * airs_error) / (57.846 + 282)

            print(f'{i}: airs_error initial: {airs_error}, fgs1_error: {fgs1_error}, score: {score}')

        return score

    obj = partial(objective, initial_params=params, data_dict=data_dict, fgs1_target=fgs1_target, airs_target=airs_target, training_config=training_config)

    study_name = f'##### HYPERPARAMETER SEARCH #####'
    print(study_name)

    initial_seed = 1337
    n_trials = 6

    sampler = optuna.samplers.TPESampler(seed=initial_seed)
    study = optuna.create_study(study_name=study_name, sampler=sampler)

    study.optimize(obj, n_trials=n_trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_value, study.best_trial.params)


if __name__ == '__main__':
    print(jax.devices())
    print('num_devices:', jax.local_device_count())

    search_for_hyperparameters = False
    if search_for_hyperparameters:
        tick = datetime.datetime.now()
        print('Start time:', tick)

        hyperparameter_search()

        print('Time spent:', datetime.datetime.now() - tick)

        exit()

    plot_limb_darkening = True
    if plot_limb_darkening:
        tick = datetime.datetime.now()
        print('Start time:', tick)

        config = data_loader.get_config(use_kaggle_config=False)
        print(config)

        from_rows = 100 #124
        to_rows = from_rows + 1*config['limb_darkening_batch_size_2']

        outlier_planet_ids = [
            158006264,
            1843015807,
            #3649218579,
            #1349926825,
            ##576917580, 3786449677,
            ##2554492145,
            #2270815333, 1597331512,
            #1473180683, 1124834224, 3176118959, 2609891029,
            #1267010874, 3928265625, 2557736212, 360841026, 2104402723, 204264160
        ]

        generate_limb_darkening(
            'train',
            config,
            plot_data=True,
            #from_rows=from_rows,
            #to_rows=to_rows,
            save_data=False,
            outlier_planet_ids=outlier_planet_ids
        )

        print('Time spent:', datetime.datetime.now() - tick)

        exit()
