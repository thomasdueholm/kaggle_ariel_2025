import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
import matplotlib.animation as animation

jax.config.update("jax_enable_x64", True)


def get_limb_darkening_intensity(t, u, v):
    limb_darkening_intensity = 1.0 - u * ((1 - v) * (1 - jnp.sqrt(jnp.maximum(1 - t ** 2, 0))) + v * t ** 2)

    return limb_darkening_intensity


def discrete_integral(r, d, u, v, n_steps):
    # https://www.astro.uvic.ca/~tatum/stellatm/atm6.pdf

    R_start = jnp.maximum(d - r, 0.0)
    R_stop = jnp.minimum(d + r, 1.0)

    eps = (R_stop - R_start) / n_steps
    t = jnp.linspace(R_start + eps / 2, R_stop - eps / 2, n_steps, dtype=jnp.float64)
    t = jax.lax.stop_gradient(t)

    inner = jnp.clip((t ** 2 + d[None, :] ** 2 - r ** 2) / (2 * t * d[None, :]), -1 + 1e-12, 1 - 1e-12)
    arccos = jnp.arccos(inner)

    limb_darkening_intensity = get_limb_darkening_intensity(t, u, v)

    sum = jnp.sum(2 * t * arccos * eps * limb_darkening_intensity, axis=0)

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


def draw_transit(
        r_squared,
        u,
        v,
        d_min,
        transit_middle,
        transit_length,
        time=0.0,
        ideal_transit=None,
        fig_name='transit',
        n_layers=500
):
    r = np.sqrt(r_squared)

    if ideal_transit is None:
        ideal_transit = get_ideal_transit(
            u,
            v,
            r,
            d_min,
            transit_middle,
            transit_length,
            1000,
            512
        )

    radius_list = np.linspace(1 / n_layers, 1, n_layers)

    limb_darkening_intensity = get_limb_darkening_intensity(radius_list, u, v)

    fig, axs = plt.subplots(1, 3)
    fig.canvas.manager.set_window_title(fig_name)
    fig.suptitle(f'r ** 2 = {r_squared},   u = {u},   v = {v},   d_min = {d_min},   transit_middle = {transit_middle},   transit_length = {transit_length}', fontsize=12)
    fig.set_size_inches(12.5, 4)

    plt.subplots_adjust(bottom=0.15, top=0.8, left=0.03, right=0.97)

    axs[0].set_title('Simulation')

    # Draw star
    for star_radius, c in zip(radius_list[::-1], limb_darkening_intensity[::-1]):
        circle = plt.Circle((0, 0), star_radius, color=(1.0, float(c), 0.0))
        axs[0].add_patch(circle)

    # Draw planet trajectory
    true_transit_length = 2 * np.sqrt((1 + r) ** 2 - d_min ** 2)
    data_length = true_transit_length / transit_length
    start = -transit_middle * data_length
    end = (1 - transit_middle) * data_length

    axs[0].plot([start, end], [-d_min, -d_min], linestyle='dashed', color='black')

    # Draw planet
    planet = plt.Circle((time * end + (1 - time) * start, -d_min), r, color='black')
    axs[0].add_patch(planet)

    axs[0].set_axis_off()
    axs[0].set_xlim(-2, 2)
    axs[0].set_ylim(-1.55, 1.55)

    # Transit graph
    axs[1].set_title('Transit')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Total observed light')

    axs[1].axhline(1, linestyle='dotted', c='green')
    axs[1].axhline(1 - r_squared, linestyle='dotted', c='green')

    axs[1].plot(np.linspace(0, 1, ideal_transit.shape[0]), ideal_transit)
    axvline = axs[1].axvline(time, linestyle='dashed', c='black')
    axs[1].set_ylim(min(np.min(ideal_transit), 1 - r_squared) - 0.01, 1.01)

    # Limb darkening intensity graph
    axs[2].set_title('Limb darkening intensity')
    axs[2].set_xlabel('Distance from center')
    axs[2].set_ylabel('Intensity')

    radius_list = np.linspace(0, 1, n_layers+1)

    limb_darkening_intensity = get_limb_darkening_intensity(radius_list, u, v)

    axs[2].plot(radius_list, limb_darkening_intensity)
    axs[2].axhline(0, linestyle='dashed', c='black')

    return fig, planet, axvline, start, end


def draw_animated_transit(
        r_squared,
        u,
        v,
        d_min,
        transit_middle,
        transit_length,
        fig_name='transit',
        n_layers=500,
        n_frames=200
):
    r = np.sqrt(r_squared)

    ideal_transit = get_ideal_transit(
        u,
        v,
        r,
        d_min,
        transit_middle,
        transit_length,
        1000,
        512
    )

    fig, planet, axvline, start, end = draw_transit(
        r_squared,
        u,
        v,
        d_min,
        transit_middle,
        transit_length,
        0.0,
        ideal_transit,
        fig_name=fig_name,
        n_layers=n_layers
    )

    def animate(i):
        time = i / (n_frames - 1)
        planet.set(center=(time * end + (1 - time) * start, -d_min))
        axvline.set(xdata=[time])

    ani = animation.FuncAnimation(fig, animate, repeat=True, frames=n_frames, interval=50)

    return ani


if __name__ == '__main__':
    figure_params_list = [
        {
            'r_squared': 0.04,
            'u': 0.0,
            'v': 0.0,
            'd_min': 0.6,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.4
        },
        {
            'r_squared': 0.04,
            'u': 1.0,
            'v': 0.0,
            'd_min': 0.6,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.4
        },
        {
            'r_squared': 0.04,
            'u': 0.0,
            'v': 0.0,
            'd_min': 0.95,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.2
        },
        {
            'r_squared': 0.04,
            'u': 1.0,
            'v': 1.0,
            'd_min': 0.6,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.2
        },
        {
            'r_squared': 0.04,
            'u': 1.0,
            'v': -1.0,
            'd_min': 0.6,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.2
        },
        {
            'r_squared': 0.1,
            'u': 0.4,
            'v': 0.0,
            'd_min': 1.1,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.2
        },
        {
            'r_squared': 0.1,
            'u': 0.4,
            'v': 0.0,
            'd_min': 0.5,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.2
        },
        {
            'r_squared': 0.1,
            'u': 0.4,
            'v': 0.0,
            'd_min': 0.0,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.2
        },
        {
            'r_squared': 0.01,
            'u': 0.4,
            'v': 0.0,
            'd_min': 0.9,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.2
        },
        {
            'r_squared': 0.01,
            'u': 0.4,
            'v': 0.0,
            'd_min': 0.5,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.2
        },
        {
            'r_squared': 0.01,
            'u': 0.4,
            'v': 0.0,
            'd_min': 0.0,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.2
        },
        {
            'r_squared': 0.04,
            'u': 0.5,
            'v': 0.025,
            'd_min': 0.4,
            'transit_middle': 0.5,
            'transit_length': 0.7,
            'time': 0.4
        }
    ]

    animation_params_list = [
        {
            'r_squared': 0.04,
            'u': 0.0,
            'v': 0.0,
            'd_min': 0.6,
            'transit_middle': 0.5,
            'transit_length': 0.7
        },
        {
            'r_squared': 0.04,
            'u': 1.0,
            'v': 0.0,
            'd_min': 0.6,
            'transit_middle': 0.5,
            'transit_length': 0.7
        },
        {
            'r_squared': 0.04,
            'u': 0.0,
            'v': 0.0,
            'd_min': 0.95,
            'transit_middle': 0.5,
            'transit_length': 0.7
        },
        {
            'r_squared': 0.04,
            'u': 0.5,
            'v': 0.025,
            'd_min': 0.4,
            'transit_middle': 0.5,
            'transit_length': 0.7
        }
    ]

    figure_list = []
    for i, params in enumerate(figure_params_list):
        fig_name = f'transit_{i}'

        print(fig_name)

        fig, _, _, _, _ = draw_transit(
            params['r_squared'],
            params['u'],
            params['v'],
            params['d_min'],
            params['transit_middle'],
            params['transit_length'],
            params['time'],
            fig_name=fig_name
        )

        plt.savefig(f'{fig_name}.png', bbox_inches='tight')

        #plt.show()

    animation_list = []
    for i, params in enumerate(animation_params_list):
        fig_name = f'animated_transit_{i}'

        print(fig_name)

        ani = draw_animated_transit(
            params['r_squared'],
            params['u'],
            params['v'],
            params['d_min'],
            params['transit_middle'],
            params['transit_length'],
            fig_name=fig_name,
            n_frames=100
        )

        animation_list.append(ani)

        writer = animation.PillowWriter(fps=15)
        ani.save(f'{fig_name}.gif', writer=writer)

        #plt.show()

    #from IPython.display import HTML
    #HTML(ani.to_jshtml())

    plt.show()
