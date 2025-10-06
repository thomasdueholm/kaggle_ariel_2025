import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import sys
import jax
import datetime

from ariel_2025 import standard_deviation, data_loader


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

standard_deviation.generate_submission(
    config=config,
    ensemble_size=5,
    file_prefix=file_prefix,
    load_checkpoint=True
)

print('Time spent:', datetime.datetime.now() - tick)
