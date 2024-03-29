from functools import partial

import pytest
from jax.example_libraries import stax
from jax.random import PRNGKey

from jax_unirep.evotuning import evotune, fit
from jax_unirep.evotuning_models import mlstm64

from .test_layers import validate_mLSTM_params

"""Evolutionary tuning function tests."""


@pytest.fixture
def model():
    """Return mlstm with randomly initialized parameters."""
    init_fun, apply_fun = mlstm64()
    _, params = init_fun(rng=PRNGKey(0), input_shape=(-1, 26))
    return apply_fun, params


@pytest.mark.slow
def test_evotune(model):
    """Simple execution test for evotune."""
    seqs = ["MTN", "BDD"] * 5
    n_epochs_config = {"high": 1}

    model_func, params = model
    _, _ = evotune(
        sequences=seqs,
        model_func=model_func,
        params=params,
        n_trials=1,
        n_epochs_config=n_epochs_config,
    )
    # now test using all defaults
    _, _ = evotune(
        sequences=seqs,
        n_trials=1,
        n_epochs_config=n_epochs_config,
    )


@pytest.mark.slow
@pytest.mark.parametrize("holdout_seqs", (["ASDV", None]))
@pytest.mark.parametrize("batch_method", (["length", "random"]))
def test_fit(model, holdout_seqs, batch_method):
    """Execution test for ``jax_unirep.evotuning.fit``."""
    sequences = ["ASDFGHJKL", "ASDYGHTKW", "HSKS", "HSGL", "ER"]

    model_func, params = model
    tuned_params = fit(
        model_func=model_func,
        params=params,
        sequences=sequences,
        n_epochs=1,
        batch_method=batch_method,
        batch_size=2,
        holdout_seqs=holdout_seqs,
    )

    validate_mLSTM_params(tuned_params[1], 64)

    # now test using all defaults
    tuned_params = fit(
        sequences=sequences,
        n_epochs=1,
        batch_method=batch_method,
        batch_size=2,
        holdout_seqs=holdout_seqs,
    )

    validate_mLSTM_params(tuned_params[1], 1900)
