import numpy as np
from scipy.linalg import solve_discrete_are
import jax
import jax.numpy as jnp
from functools import partial

from ariel_2025 import data_loader

jax.config.update("jax_enable_x64", True)


@jax.jit
def kalman_filter(A, Q, C, R, y, mu_0, Sigma_0):
    # The math is based on Machine Learning: A Probabilistic Perspective by Kevin P. Murphy (2012), Section 18.3.1.

    def forward_pass(carry, y_t):
        mu_t_minus_1, Sigma_t_minus_1, A, Q, C, R = carry

        mu_prediction = A @ mu_t_minus_1
        Sigma_prediction = A @ Sigma_t_minus_1 @ A.mT + Q

        S_t = C @ Sigma_prediction @ C.mT + R

        if S_t.shape[-1] == 1:
            K_t = Sigma_prediction @ C.mT / S_t
        else:
            K_t = Sigma_prediction @ C.mT @ jnp.linalg.inv(S_t)

        y_hat_t = C @ mu_prediction
        r_t = jnp.where(jnp.isnan(y_t), 0, y_t - y_hat_t)

        mu_t = mu_prediction + K_t @ r_t
        Sigma_t = jnp.where(jnp.isnan(y_t), Sigma_prediction, (jnp.identity(C.shape[-1]) - K_t @ C) @ Sigma_prediction)

        carry = mu_t, Sigma_t, A, Q, C, R
        result = mu_t, Sigma_t, mu_prediction, Sigma_prediction

        return carry, result

    init = mu_0, Sigma_0, A, Q, C, R
    _, result = jax.lax.scan(forward_pass, init, y)

    return result


@jax.jit
def kalman_smoother(A, Q, C, R, y, mu_0, Sigma_0):
    # The math is based on Machine Learning: A Probabilistic Perspective by Kevin P. Murphy (2012), Section 18.3.2.

    def backward_pass(carry, x):
        mu_smoothed_next, Sigma_smoothed_next, A = carry
        mu, Sigma, mu_prediction, Sigma_prediction, Sigma_prediction_inv = x

        J_t = Sigma @ A.mT @ Sigma_prediction_inv

        mu_smoothed = mu + J_t @ (mu_smoothed_next - mu_prediction)
        Sigma_smoothed = Sigma + J_t @ (Sigma_smoothed_next - Sigma_prediction) @ J_t.mT

        carry = mu_smoothed, Sigma_smoothed, A
        result = mu_smoothed, Sigma_smoothed

        return carry, result

    mu, Sigma, mu_prediction, Sigma_prediction = kalman_filter(A, Q, C, R, y, mu_0, Sigma_0)

    mu = jnp.flip(mu, axis=0)
    Sigma = jnp.flip(Sigma, axis=0)
    mu_prediction = jnp.flip(mu_prediction, axis=0)
    Sigma_prediction = jnp.flip(Sigma_prediction, axis=0)

    Sigma_prediction_inv = jax.vmap(jnp.linalg.inv)(Sigma_prediction)

    init = mu[0, ...], Sigma[0, ...], A
    xs = mu[1:, ...], Sigma[1:, ...], mu_prediction[:-1, ...], Sigma_prediction[:-1, ...], Sigma_prediction_inv[:-1, ...]

    _, (mu_smoothed, Sigma_smoothed) = jax.lax.scan(backward_pass, init, xs)

    mu_smoothed = jnp.concat([jnp.flip(mu_smoothed, axis=0), mu[0: 1, ...]], axis=0)
    Sigma_smoothed = jnp.concat([jnp.flip(Sigma_smoothed, axis=0), Sigma[0: 1, ...]], axis=0)

    return mu_smoothed, Sigma_smoothed


@partial(jax.jit, static_argnames=['window', 'min_periods'])
def centered_rolling_mean(x, window, min_periods):
    nan_x = jnp.isnan(x)
    x = jnp.where(nan_x, 0, x)

    cumsum_x = jnp.cumsum(x)
    cumsum_nan_x = jnp.cumsum(1 - nan_x)

    window_sum = cumsum_x[window:] - cumsum_x[:len(cumsum_x)-window]
    window_nan_sum = cumsum_nan_x[window:] - cumsum_nan_x[:len(cumsum_nan_x)-window]

    initial_window_sum = cumsum_x[(window-1) // 2: window]
    initial_window_nan_sum = cumsum_nan_x[(window-1) // 2: window]

    last_window_sum = cumsum_x[-1] - cumsum_x[-window: -window // 2]
    last_window_nan_sum = cumsum_nan_x[-1] - cumsum_nan_x[-window: -window // 2]

    window_sum = jnp.concat([initial_window_sum, window_sum, last_window_sum])
    window_nan_sum = jnp.concat([initial_window_nan_sum, window_nan_sum, last_window_nan_sum])

    mean = jnp.where(
        window_nan_sum >= min_periods,
        window_sum / jnp.maximum(1, window_nan_sum),
        jnp.nan
    )

    return mean


@partial(jax.jit, static_argnames=['window', 'level_change_scaler', 'trend_change_scaler', 'noise_scaler'])
def get_kalman_initialization(x, window, level_change_scaler, trend_change_scaler, noise_scaler):
    initial_level = jnp.nanmean(x[:window])
    last_level = jnp.nanmean(x[-window:])
    initial_trend = (last_level - initial_level) / (len(x) - window)

    mu_0 = jnp.array([initial_level, initial_trend])

    mean = centered_rolling_mean(x, window, window//3)

    level_change = mean[1:] - mean[:-1]
    max_level_change = jnp.nanmax(jnp.abs(level_change))

    mean_level_change = centered_rolling_mean(level_change, window, window//3)
    trend_change = mean_level_change[1:] - mean_level_change[:-1]
    max_trend_change = jnp.nanmax(jnp.abs(trend_change))

    Q = jnp.diag(jnp.array([max_level_change * level_change_scaler, max_trend_change * trend_change_scaler]) ** 2)
    Q = jnp.nan_to_num(Q, nan=1e-4)

    var = jnp.nanvar(x - mean)

    R = jnp.full((1, 1), var * noise_scaler)
    R = jnp.nan_to_num(R, nan=1e-4)

    return Q, R, mu_0


def apply_kalman_smoother(x, window=255, level_change_scaler=0.1, trend_change_scaler=1.0, noise_scaler=3.0):
    x = x.astype(np.float64)

    vmapped_get_kalman_initialization = get_kalman_initialization
    vmapped_kalman_smoother = kalman_smoother

    for i in range(len(x.shape) - 1):
        vmapped_get_kalman_initialization = jax.vmap(vmapped_get_kalman_initialization, in_axes=(0, None, None, None, None))
        vmapped_kalman_smoother = jax.vmap(vmapped_kalman_smoother, in_axes=(None, 0, None, 0, 0, 0, 0))

    vmapped_get_kalman_initialization = jax.jit(vmapped_get_kalman_initialization, static_argnames=['window', 'level_change_scaler', 'trend_change_scaler', 'noise_scaler'])
    vmapped_kalman_smoother = jax.jit(vmapped_kalman_smoother)

    A = jnp.array([
        [1, 1],
        [0, 1]
    ])
    C = jnp.array([
        [1, 0]
    ])

    Q, R, mu_0 = vmapped_get_kalman_initialization(x, window, level_change_scaler, trend_change_scaler, noise_scaler)

    # Sigma_0 is set to the steady state for A, Q, C, R. See, e.g., Section 2.2.1 of https://www.robots.ox.ac.uk/~ian/Teaching/Estimation/LectureNotes2.pdf
    # I did not find a simple way to solve Discrete Algebraic Riccati Equations (DARE) with JAX. Otherwise, the entire function could be vmapped.
    if len(Q.shape) == 2:
        Sigma_0 = solve_discrete_are(A.T, C.T, Q, R).astype(np.float64)
    else:
        Sigma_0 = np.zeros(Q.shape, dtype=np.float64)
        for index in np.ndindex(Q.shape[:-2]):
            extended_index = index + (slice(None), slice(None))
            Q_selected = Q[extended_index]
            for i in range(10):
                try:
                    Sigma_0[extended_index] = solve_discrete_are(A.T, C.T, Q_selected, R[extended_index])
                    break
                except Exception as e:
                    if i == 9:
                        raise e
                    else:
                        increase =np.power(10.0, -10.0+i)
                        print(f'Failed to solve discrete Ricatti equation. Increasing diagonal noise for Q by {increase}.')
                        print(Q_selected)
                        Q_selected = Q_selected + np.identity(Q_selected.shape[0]) * increase

    mu, V = vmapped_kalman_smoother(A, Q, C, R, x, mu_0, Sigma_0)

    return mu, V


if __name__ == '__main__':
    fgs1_signal, airs_signal, fgs1_foreground_noise, airs_foreground_noise, star_info, target = data_loader.load_data('train')

    print('fgs1_signal', fgs1_signal.shape)
    print('airs_signal', airs_signal.shape)
    print('target', target.values.shape)

    y = fgs1_signal[:20, :]
    y = airs_signal[:20, :]

    print('y', y.shape)

    mu, V = apply_kalman_smoother(y)

    print('mu', mu.shape)
    print('V', V.shape)
