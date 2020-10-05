"""Evolutionary tuning function tests."""
from functools import partial

import numpy as np
import pytest

from jax.experimental import stax
from jax.random import PRNGKey
from jax_unirep.evotuning import evotune, evotuning_layers, fit

from .test_layers import validate_mLSTM_params


@pytest.fixture
def params():
    """Return randomly initialized params."""
    model_layers = evotuning_layers(mlstm_size=64)
    init_fun, _ = stax.serial(*model_layers)
    _, parameters = init_fun(PRNGKey(0), (-1, 10))
    return parameters


def test_evotune():
    """Simple execution test for evotune."""
    seqs = ["MTN", "BDD"] * 5
    n_epochs_config = {"high": 1}
    fit_func = partial(fit, mlstm_size=256, rng=PRNGKey(0))
    _, _ = evotune(
        sequences=seqs,
        fit_func=fit_func,
        n_trials=1,
        n_epochs_config=n_epochs_config,
    )


@pytest.mark.parametrize("holdout_seqs", (["ASDV", None]))
@pytest.mark.parametrize("batch_method", (["length", "random"]))
def test_fit(holdout_seqs, batch_method):
    """Execution test for ``jax_unirep.evotuning.fit``."""
    sequences = ["ASDFGHJKL", "ASDYGHTKW", "HSKS", "HSGL", "ER"]

    key = PRNGKey(42)

    params = fit(
        mlstm_size=64,
        rng=key,
        sequences=sequences,
        n_epochs=1,
        batch_method=batch_method,
        batch_size=2,
        holdout_seqs=holdout_seqs,
    )

    validate_mLSTM_params(params[0], 64)
