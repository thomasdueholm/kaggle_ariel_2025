import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import sys
import jax
import datetime

from ariel_2025 import data_loader, initial_depth_estimation

print(jax.devices())
print('num_devices:', jax.local_device_count())
tick = datetime.datetime.now()
print(tick)

if len(sys.argv) > 1:
    file_prefix = sys.argv[1]
else:
    file_prefix = 'test'

print(datetime.datetime.now(), 'depth_estimation.generate_depth_estimation')

if file_prefix != 'test':
    config = data_loader.get_config(use_kaggle_config=False)
else:
    config = data_loader.get_config()

print(config)

initial_depth_estimation.generate_depth_estimation(file_prefix, config=config)
