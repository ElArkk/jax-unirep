from functools import partial

import pytest
from jax.experimental import stax
from jax.random import PRNGKey

from jax_unirep.evotuning import evotune, fit
from jax_unirep.evotuning_models import mlstm64

from .test_layers import validate_mLSTM_params

"""Evolutionary tuning function tests."""


@pytest.fixture
def params():
    """Return randomly initialized params."""
    init_fun, _ = mlstm64()
    _, parameters = init_fun(key=PRNGKey(0), input_shape=(-1, 10))
    return parameters


def test_evotune():
    """Simple execution test for evotune."""
    seqs = ["MTN", "BDD"] * 5
    n_epochs_config = {"high": 1}
    init_func, model_func = mlstm64()
    _, params = init_func(PRNGKey(42), (-1, 10))
    fit_func = partial(fit, model_func=model_func, params=params)
    _, _ = evotune(
        sequences=seqs,
        fit_func=fit_func,
        model_func=model_func,
        n_trials=1,
        n_epochs_config=n_epochs_config,
    )


@pytest.mark.parametrize("holdout_seqs", (["ASDV", None]))
@pytest.mark.parametrize("batch_method", (["length", "random"]))
def test_fit(holdout_seqs, batch_method):
    """Execution test for ``jax_unirep.evotuning.fit``."""
    sequences = ["ASDFGHJKL", "ASDYGHTKW", "HSKS", "HSGL", "ER"]

    key = PRNGKey(42)

    init_func, model_func = mlstm64()
    _, params = init_func(key, input_shape=(-1, 10))

    tuned_params = fit(
        model_func=model_func,
        params=params,
        sequences=sequences,
        n_epochs=1,
        batch_method=batch_method,
        batch_size=2,
        holdout_seqs=holdout_seqs,
    )

    validate_mLSTM_params(tuned_params[0], 64)
