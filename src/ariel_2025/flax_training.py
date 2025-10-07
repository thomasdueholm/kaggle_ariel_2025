import os
import datetime
import numpy as np
import jax
from functools import partial
import optax
from flax.training import train_state
from typing import Any
import jax.numpy as jnp
import orbax.checkpoint as ocp
import flax.linen as nn
from typing import Tuple


@partial(jax.jit, static_argnames=['nn'])
def jitted_apply_model(nn, params, batch_stats, X):
    return nn.apply({'params': params, 'batch_stats': batch_stats}, X, train=False)


def batched_apply(nn, state, X, batch_size):
    params = state.params
    batch_stats = state.batch_stats
    if batch_stats is None:
        batch_stats = {}

    result = []

    X_shape = X.shape

    current_index = 0
    while current_index < X_shape[0]:
        next_index = min(current_index + batch_size, X_shape[0])
        X_selected = X[current_index: next_index, ...]
        result_selected = jitted_apply_model(nn, params, batch_stats, X_selected)
        result.append(result_selected)
        current_index = next_index

    result = np.concatenate(result, axis=0)

    return result


def gaussian_log_likelihood(standard_deviation, prediction_minus_y, sample_weight=None):
    if sample_weight is None:
        sample_weight = jnp.ones_like(standard_deviation)

    standard_deviation = jnp.maximum(standard_deviation, 1e-7)

    gll = -0.5 * (jnp.log(2 * jnp.pi) + 2 * jnp.log(standard_deviation) + prediction_minus_y ** 2 / standard_deviation ** 2)
    gll = jnp.sum(gll * sample_weight) / jnp.sum(sample_weight)

    return gll


@partial(jax.jit, static_argnames=['train'])
def apply_model(state, X, y, w, train: bool, dropout_key):

    def evaluate_batch(params, batch_stats, X, y, w, train: bool, dropout_key):
        prediction, updates = state.apply_fn({'params': params, 'batch_stats': batch_stats}, X, train=train, mutable=['batch_stats'], rngs={'dropout': dropout_key})

        score = -gaussian_log_likelihood(prediction, y, w)

        return score, updates

    grad_fn = jax.value_and_grad(evaluate_batch, has_aux=True, argnums=0)

    batch_stats = state.batch_stats
    if batch_stats is None:
        batch_stats = {}

    (score, updates), grads = grad_fn(
        state.params,
        batch_stats,
        X,
        y,
        w,
        train,
        dropout_key
    )

    return -score, grads, updates


@jax.jit
def update_model(state, grads, updates):
    state = state.apply_gradients(grads=grads)
    if 'batch_stats' in updates:
        state = state.replace(batch_stats=updates['batch_stats'])
    return state


def train_epoch(state, X, y, w, config, rng):
    """Train for a single epoch."""
    batch_size = config['batch_size']

    train_size = X.shape[0]
    steps_per_epoch = train_size // batch_size

    rng, perm_rng = jax.random.split(rng)

    perms = jax.random.permutation(perm_rng, train_size)
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    score_list = []

    for i, perm in enumerate(perms):
        rng, dropout_key = jax.random.split(rng)

        batch_X = X[perm, ...]
        batch_y = y[perm, ...]
        batch_w = w[perm, ...]

        score, grads, updates = apply_model(state, batch_X, batch_y, batch_w, train=True, dropout_key=dropout_key)

        state = update_model(state, grads, updates)
        score_list.append(score)

    score = np.mean(score_list)

    return state, score


class CustomTrainState(train_state.TrainState):
    batch_stats: Any
    key: jax.Array


def get_state(config, key, nn, checkpoint_dir=None, learning_rate_fn=None):

    key, params_rng = jax.random.split(key)

    dummy_input = np.zeros(tuple(config['input_shape']), dtype=np.float64)

    variables = nn.init({'params': params_rng}, dummy_input, train=False)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    if checkpoint_dir is not None and not os.path.isdir(checkpoint_dir):
        print('Not a checkpoint_dir:', checkpoint_dir)
        checkpoint_dir = None

    if checkpoint_dir is not None:
        data_dict = {
            'params': params,
            'batch_stats': batch_stats
        }

        abstract_data_dict = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, data_dict)

        checkpointer = ocp.StandardCheckpointer()
        variables = checkpointer.restore(checkpoint_dir, abstract_data_dict)

        params = variables['params']
        batch_stats = variables.get('batch_stats', {})

    if learning_rate_fn is None:
        learning_rate = config.get('learning_rate', 0.01)
    else:
        learning_rate = learning_rate_fn

    adamw = optax.adamw(
        learning_rate=learning_rate,
        b1=config.get('adam_b1', 0.9),
        b2=config.get('adam_b2', 0.999),
        eps=config.get('adam_eps', 1e-8),
        weight_decay=config.get('weight_decay', 0.0)
    )

    adam = optax.adam(
        learning_rate=learning_rate,
        b1=config.get('adam_b1', 0.9),
        b2=config.get('adam_b2', 0.999),
        eps=config.get('adam_eps', 1e-8)
    )

    def zero_grads():
        #https://github.com/deepmind/optax/issues/159#issuecomment-896459491

        def init_fn(_):
            return ()

        def update_fn(updates, state, params=None):
            return jax.tree_util.tree_map(jnp.zeros_like, updates), ()

        return optax.GradientTransformation(init_fn, update_fn)

    def apply_weight_decay(key_path, value):
        key = str(key_path[-1])[2: -2]
        return key != 'bias' and key != 'scale' and 'no_weight_decay' not in key

    def create_mask(params, weight_decay_mask):
        # https://colab.research.google.com/drive/1g_pt2Rc3bv6H6qchvGHD-BpgF-Pt4vrC#scrollTo=TqDvTL_tIQCH&uniqifier=1

        def _map(params, mask, weight_decay):
            for k in params:
                if 'zero_grad' in k:
                    mask[k] = 'zero_grad'
                elif isinstance(params[k], dict):
                    mask[k] = {}
                    _map(params[k], mask[k], weight_decay[k])
                else:
                    if weight_decay[k]:
                        mask[k] = 'adamw'
                    else:
                        mask[k] = 'adam'

        mask = {}
        _map(params, mask, weight_decay_mask)
        return mask

    weight_decay_mask = jax.tree_util.tree_map_with_path(apply_weight_decay, params)
    optimizer_mask = create_mask(params, weight_decay_mask)

    optimizer = optax.multi_transform(
        {'adamw': adamw, 'adam': adam, 'zero_grad': zero_grads()},
        optimizer_mask
    )
    tx = optax.chain(
        optax.clip(config.get('gradient_clipping_threshold', 1)),
        optimizer
    )

    state = CustomTrainState.create(apply_fn=nn.apply, params=params, batch_stats=batch_stats, key=key, tx=tx)

    return state


def create_learning_rate_fn(config, steps_per_epoch):
    """Creates learning rate schedule."""
    schedule_type = config.get('learning_rate_schedule_type', 'cosine')

    if schedule_type == 'cosine':
        warmup_epochs = config.get('warmup_epochs', 0)

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=config['learning_rate'],
            transition_steps=warmup_epochs * steps_per_epoch
        )

        cosine_epochs = max(config['n_epochs'] - warmup_epochs, 1)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=config['learning_rate'],
            decay_steps=cosine_epochs * steps_per_epoch
        )

        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_epochs * steps_per_epoch]
        )
    elif schedule_type == 'constant':
        schedule_fn = optax.constant_schedule(config['learning_rate'])
    else:
        schedule_fn = None

    return schedule_fn


def train_and_evaluate(config, nn, rng, X_train, y_train, w_train, X_test=None, y_test=None, w_test=None):
    batch_size = config['batch_size']

    rng, init_rng = jax.random.split(rng)

    if config.get('warmup_epochs', -1) < 0:
        learning_rate_fn = None
    else:
        steps_per_epoch = X_train.shape[0] // batch_size
        learning_rate_fn = create_learning_rate_fn(config, steps_per_epoch)

    state = get_state(config, init_rng, nn, learning_rate_fn=learning_rate_fn)

    for epoch in range(1, config['n_epochs'] + 1):
        now = datetime.datetime.now()

        rng, input_rng = jax.random.split(rng)

        state, train_score = train_epoch(
            state,
            X_train,
            y_train,
            w_train,
            config,
            input_rng
        )

        if not config.get('silent', True):
            if X_test is not None:
                prediction = batched_apply(nn, state, X_test, batch_size)
                test_score = gaussian_log_likelihood(prediction, y_test, w_test)
            else:
                test_score = 0.0

            if learning_rate_fn is not None:
                lr = learning_rate_fn(state.step)
            else:
                lr = config['learning_rate']

            if 'log_scale no_weight_decay' in state.params:
                log_scale_str = f", log_scale: {state.params['log_scale no_weight_decay']:.5f}"
            else:
                log_scale_str = ''

            print(f'epoch: {epoch}, train_score: {train_score:.5f}, test_score: {test_score:.5f}, time: {datetime.datetime.now() - now}, learning rate: {lr:.5f}{log_scale_str}')

    return state


class MLP(nn.Module):

    input_clipping_threshold: float
    n_hidden: Tuple[int]
    dropout: Tuple[float]
    log_scale: float

    @nn.compact
    def __call__(self, x, train: bool):
        x = x.astype(jnp.float64)
        x = jnp.clip(x, -self.input_clipping_threshold, self.input_clipping_threshold)

        for n_hidden, dropout in zip(self.n_hidden, self.dropout):
            x_sigmoid = nn.BatchNorm(use_running_average=(not train))(x)
            x_sigmoid = nn.Dense(features=n_hidden, use_bias=True)(x_sigmoid)
            x_sigmoid = nn.sigmoid(x_sigmoid)

            x = nn.Dense(features=n_hidden, use_bias=True)(x)
            x = nn.BatchNorm(use_running_average=(not train))(x)
            x = nn.relu(x)

            x = x * x_sigmoid

            x = nn.Dropout(rate=dropout, deterministic=not train)(x)

        x = nn.Dense(features=1, use_bias=True)(x)[..., 0]
        x = 1 + nn.elu(x)

        log_scale = self.param('log_scale no_weight_decay', nn.initializers.constant(self.log_scale), (), jnp.float32)

        x = jnp.exp(log_scale) * x

        return x


class FlaxModelTrainer:

    def __init__(self, config, key=jax.random.PRNGKey(42)):
        self.config = config
        self.key = key

        self.nn = None
        self.state = None

    def fit(self, X_train, y_train, w_train, X_test=None, y_test=None, w_test=None):
        self.nn = MLP(
            input_clipping_threshold=self.config['input_clipping_threshold'],
            n_hidden=tuple(self.config['n_hidden']),
            dropout=tuple(self.config['dropout']),
            log_scale=self.config['log_scale']
        )

        self.key, rng = jax.random.split(self.key)
        self.state = train_and_evaluate(self.config, self.nn, rng, X_train, y_train, w_train, X_test=X_test, y_test=y_test, w_test=w_test)

    def predict(self, X):
        batch_size = self.config['batch_size']
        prediction = batched_apply(self.nn, self.state, X, batch_size)

        return prediction

    def save_checkpoint(self, checkpoint_dir):
        data_dict = {
            'params': self.state.params,
            'batch_stats': self.state.batch_stats
        }

        orbax_checkpointer = ocp.StandardCheckpointer()
        orbax_checkpointer.save(checkpoint_dir, data_dict, force=True)
        orbax_checkpointer.wait_until_finished()

    def load_checkpoint(self, checkpoint_dir):
        self.nn = MLP(
            input_clipping_threshold=self.config['input_clipping_threshold'],
            n_hidden=tuple(self.config['n_hidden']),
            dropout=tuple(self.config['dropout']),
            log_scale=self.config['log_scale']
        )

        self.key, state_key = jax.random.split(self.key)
        self.state = get_state(self.config, state_key, nn=self.nn, checkpoint_dir=checkpoint_dir)


class FlaxModelTrainerFactory:

    def __init__(self, config):
        self.config = config
        self.key = jax.random.PRNGKey(config.get('seed', 42))
        self.initial_key = self.key

    def reset(self):
        self.key = self.initial_key

    def __call__(self, config=None):
        if config is None:
            config = self.config

        self.key, subkey = jax.random.split(self.key)
        flax_model_trainer = FlaxModelTrainer(config, key=subkey)
        return flax_model_trainer

