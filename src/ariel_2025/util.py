import os
import json
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import optuna
from jax import random
import scipy.cluster.hierarchy as sch
import seaborn as sns


def get_dir(dir_name):
    util_dir_path = os.path.dirname(os.path.realpath(__file__))
    settings_file = os.path.join(util_dir_path, f'../../SETTINGS.json')

    if os.path.isfile(settings_file):
        with open(settings_file, 'r') as json_file:
            settings_dict = json.load(json_file)
    else:
        raise Exception(f'Error while producing dir {dir_name}: Unable to find SETTINGS.json. Make sure that it is the location "../../SETTINGS.json" relative to util.py.')

    if dir_name == 'data':
        data_dir = settings_dict['RAW_DATA_DIR']
    elif dir_name == 'cache':
        data_dir = settings_dict['CACHE_DIR']
    elif dir_name == 'local_cache':
        data_dir = settings_dict['LOCAL_CACHE_DIR']
    elif dir_name == 'parameters':
        data_dir = settings_dict['PARAMETERS_DIR']
    elif dir_name == 'models':
        data_dir = settings_dict['MODEL_CHECKPOINT_DIR']
    elif dir_name == 'logs':
        data_dir = settings_dict['LOGS_DIR']
    elif dir_name == 'submissions':
        data_dir = settings_dict['SUBMISSION_DIR']
    else:
        raise Exception(f'Unknown directory name {dir_name} requested from util.get_dir.')

    data_dir = os.path.abspath(data_dir)

    #print(data_dir)

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    return data_dir


def delete_dir(directory):
    for temp_file in os.listdir(directory):
        file_to_be_deleted = os.path.join(directory, temp_file)
        os.remove(file_to_be_deleted)

    os.rmdir(directory)


class KeyFactory:

    def __init__(self, key):
        self.key = key
        self.initial_key = key

    def reset(self):
        self.key = self.initial_key

    def get_key(self):
        self.key, subkey = jax.random.split(self.key)

        return subkey


def get_random_int(key):
    key, subkey = random.split(key)
    random_int = int(random.randint(subkey, (1,), 0, 1e7, dtype=int)[0])

    return key, random_int


def get_dummy_trial(key):
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    key, seed = get_random_int(key)
    sampler = optuna.samplers.TPESampler(seed=seed)
    dummy_study = optuna.create_study(study_name="dummy", sampler=sampler)
    trial = dummy_study.ask()
    optuna.logging.set_verbosity(verbosity)

    return key, trial


def draw_histogram(data, color, label, fig_name, n_bins=100, mask=None, target=None):
    print('draw_histogram:', fig_name, data.shape)

    if target is None:
        target = np.ones(data.shape)

    plt.figure(fig_name)

    data = data.flatten()
    target = target.flatten()

    if mask is not None:
        mask = mask.flatten()
        data = data[mask]
        target = target[mask]

    target = target[~np.isnan(data)]
    data = data[~np.isnan(data)]

    target_values = np.sort(np.unique(target))
    cmap = plt.get_cmap('gist_ncar')
    colors = cmap(np.linspace(0, 0.9, len(target_values)))

    for i, value in enumerate(target_values):
        selected_data = data[target == value]

        mean = np.mean(selected_data, dtype=np.float32)
        std = np.std(selected_data, dtype=np.float32)
        selected_data = np.clip(selected_data, mean - 5 * std, mean + 5 * std)

        counts, bins = np.histogram(selected_data, bins=n_bins, density=True)

        if color is None:
            c = colors[i]
        else:
            c = color

        if len(target_values) > 1:
            plt.stairs(counts, bins, label=f'{i} {label} {mean:.7f} {std:.7f}', color=c)
        else:
            plt.stairs(counts, bins, label=f'{label} {mean:.7f} {std:.7f}', color=c)

        plt.axvline(mean, linestyle='dashed', color=c)

    plt.legend()


def plot_correlation(data, hierarchy_threshold, pca_threshold=0.9, title='', labels=None):
    print('plot_correlation:', title, data.shape)

    data = data.reshape((-1, data.shape[-1]))
    data = data[~np.any(np.isnan(data), axis=1), :]

    corr = np.corrcoef(data, rowvar=False)

    pairwise_distances = (np.maximum(0, 1 - corr))[np.triu_indices(corr.shape[0], k=1)]
    linkage = sch.linkage(pairwise_distances, method='average')
    cluster_array = sch.fcluster(linkage, hierarchy_threshold * np.max(linkage[:, 2]), criterion='distance')

    plt.figure(f'Dendogram {title}')
    sch.dendrogram(linkage, color_threshold=hierarchy_threshold * np.max(linkage[:, 2]), labels=labels)

    plt.figure(f'Correlation heatmap {title}')
    idx = np.argsort(cluster_array)
    sns.heatmap(corr[idx, :][:, idx], cmap="coolwarm", vmin=-1, vmax=1)

    plt.figure(f'Scree plot {title}')

    plt.axhline(0, color='k')
    plt.axhline(pca_threshold, color='k', linestyle='dashed')

    covariance_matrix = np.cov(data, rowvar=False)

    for cluster in np.sort(np.unique(cluster_array)):
        cluster_mask = (cluster_array == cluster)

        if cluster_mask.sum() <= 1:
            continue
        else:
            cluster_covariance = covariance_matrix[:, cluster_mask][cluster_mask, :]
            cluster_covariance = (cluster_covariance + cluster_covariance.T) / 2

            eigenvalues, eigenvectors = np.linalg.eigh(cluster_covariance)

            unresolved_error = False
            error_correction = -6
            while np.any(np.iscomplex(eigenvalues)):
                if error_correction > -2:
                    print(f'Warning: np.linalg.eig produced complex values. Error correction = {10 ** error_correction}')
                eigenvalues, eigenvectors = np.linalg.eigh(cluster_covariance + (10 ** error_correction) * np.identity(cluster_covariance.shape[0]))
                error_correction += 1
                if error_correction >= 10:
                    print('Unresolved error: Failed to produced eigenvalues with np.linalg.eig.')
                    unresolved_error = True
                    break

            if unresolved_error:
                continue

            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]

            eigenvalue_proportion = eigenvalues / np.sum(eigenvalues)
            cumulative_eigenvalue = np.cumsum(eigenvalue_proportion)
            cumulative_eigenvalue[1:] = cumulative_eigenvalue[:-1]
            cumulative_eigenvalue[0] = 0

            pca_number = [x + 1 for x in range(len(eigenvalue_proportion))]
            plt.plot(pca_number, cumulative_eigenvalue, label=f'{cluster}')

    plt.legend()

