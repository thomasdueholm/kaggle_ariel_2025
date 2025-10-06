import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import sys
import jax
import datetime

from ariel_2025 import data_loader

print(jax.devices())
print('num_devices:', jax.local_device_count())
tick = datetime.datetime.now()
print(tick)

if len(sys.argv) > 1:
    file_prefix = sys.argv[1]
else:
    file_prefix = 'test'

print(datetime.datetime.now(), 'data_loader.generate_data')

if file_prefix != 'test':
    config = data_loader.get_config(use_kaggle_config=False)
else:
    config = data_loader.get_config()

print(config)

data_loader.generate_data(config, file_prefix)
