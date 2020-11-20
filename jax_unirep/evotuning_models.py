"""
Models from the original unirep paper.

These exist as convenience functions to import into a notebook or script.

Generally, you would use these functions in the following fashion:

```python
from jax_unirep.evotuning_models import mlstm256
from jax_unirep.evotuning import fit
from jax.random import PRNGKey

init_func, model_func = mlstm256()
_, params = init_func(PRNGKey(42), input_shape=(-1, 10))

tuned_params = fit(
    sequences,  # we assume you've got them prepped!
    n_epochs=1,
    model_func=model_func,
    params=params,
)
```
"""
from jax.experimental.stax import Dense, Softmax, serial

from .layers import AAEmbedding, mLSTM, mLSTMHiddenStates


def mlstm1900():
    """mLSTM1900 model functions."""
    model_layers = (
        AAEmbedding(10),
        mLSTM(1900),
        mLSTMHiddenStates(),
        Dense(26),
        Softmax,
    )
    init_fun, apply_fun = serial(*model_layers)
    return init_fun, apply_fun


mlstm1900_init_fun, mlstm1900_apply_fun = mlstm1900()


def mlstm256():
    """mLSTM256 model functions."""
    model_layers = (
        AAEmbedding(10),
        mLSTM(256),
        mLSTMHiddenStates(),
        mLSTM(256),
        mLSTMHiddenStates(),
        mLSTM(256),
        mLSTMHiddenStates(),
        mLSTM(256),
        mLSTMHiddenStates(),
        Dense(26),
        Softmax,
    )
    init_fun, apply_fun = serial(*model_layers)
    return init_fun, apply_fun


def mlstm64():
    """mLSTM64 model functions."""
    model_layers = (
        AAEmbedding(10),
        mLSTM(64),
        mLSTMHiddenStates(),
        mLSTM(64),
        mLSTMHiddenStates(),
        mLSTM(64),
        mLSTMHiddenStates(),
        mLSTM(64),
        mLSTMHiddenStates(),
        Dense(26),
        Softmax,
    )
    init_fun, apply_fun = serial(*model_layers)
    return init_fun, apply_fun
