from functools import partial

import numpy as np
import pytest

from jax import vmap
from jax.experimental import stax
from jax.random import PRNGKey
from jax_unirep.evotuning import evotune, evotuning_layers, fit
from jax_unirep.utils import input_output_pairs

from .test_layers import validate_mLSTM_params

"""Evolutionary tuning function tests."""


@pytest.fixture
def params():
    model_layers = evotuning_layers(mlstm_size=64)
    init_fun, _ = stax.serial(*model_layers)
    _, parameters = init_fun(PRNGKey(0), (-1, 10))
    return parameters


@pytest.mark.skip(reason="Maybe deprecate?")
def test_evotune():
    """
    Simple execution test for evotune.
    """
    seqs = ["MTN", "BDD"] * 5
    n_epochs_config = {"high": 1}
    _, params_new = evotune(
        sequences=seqs,
        n_trials=1,
        params=None,
        n_epochs_config=n_epochs_config,
    )


@pytest.mark.skip(reason="Not needed.")
def test_predict(params):
    """
    Unit test for ``jax_unirep.evotuning.predict``.

    We test that the shape of the output of ``predict``
    is identical to the shape of the ys to predict.

    We also test that the evotune `predict` function gives us bounded values
    that are between 0 and 1.
    """
    sequences = ["ASDFGHJKL", "ASDYGHTKW"]
    xs, ys = input_output_pairs(sequences)
    res = vmap(partial(predict, params))(xs)

    assert res.shape == ys.shape
    assert res.min() >= 0
    assert res.max() <= 1


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


@pytest.mark.skip(reason="Execution test already done in ``test_evotune``.")
def test_objective():
    """
    Placeholder test for the future.

    ``objective()`` gets called as part of ``test_evotune``,
    and the original idea here was to design an execution-based test
    for the ``objective()`` function call.
    Since it's already executed as part of ``test_evotune``,
    we are skipping this test until we have time to implement it.
    """
    assert False
