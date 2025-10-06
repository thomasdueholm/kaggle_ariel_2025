import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import sys
import jax
import datetime

from ariel_2025 import data_loader, limb_darkening

print(jax.devices())
print('num_devices:', jax.local_device_count())
tick = datetime.datetime.now()
print(tick)

if len(sys.argv) > 1:
    file_prefix = sys.argv[1]
else:
    file_prefix = 'test'

if file_prefix != 'test':
    config = data_loader.get_config(use_kaggle_config=False)
else:
    config = data_loader.get_config()

print(config)

tick = datetime.datetime.now()

print(tick, 'limb_darkening.generate_limb_darkening')
limb_darkening.generate_limb_darkening(file_prefix, config)

print('Time spent:', datetime.datetime.now() - tick)