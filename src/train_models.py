import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import jax
import datetime

from ariel_2025 import standard_deviation, data_loader


print(jax.devices())
print('num_devices:', jax.local_device_count())
tick = datetime.datetime.now()
print(tick)

config = data_loader.get_config(use_kaggle_config=False)

standard_deviation.generate_submission(
    config=config,
    ensemble_size=5,
    file_prefix='train',
    save_checkpoint=True,
    use_pca=True,
    plot_sigma_scale=False,
    std_estimator_seed=711
)

print('Time spent:', datetime.datetime.now() - tick)

