"""Evolutionary tuning function tests."""
from contextlib import suppress as does_not_raise
from functools import partial
from typing import Dict

import numpy as np
import pytest
from jax import vmap
from jax.experimental.optimizers import adam
from jax.random import PRNGKey
from jax_unirep.evotuning import (
    evotune,
    evotune_loss,
    evotuning_pairs,
    fit,
    init_fun,
    input_output_pairs,
    length_batch_input_outputs,
    predict,
)
from jax_unirep.utils import load_dense_1900, load_params_1900


@pytest.fixture
def params():
    _, params = init_fun(PRNGKey(0), (-1, 10))
    return params


@pytest.mark.parametrize(
    "seqs, expected",
    [
        ([], pytest.raises(ValueError)),
        (["MT", "MTN"], pytest.raises(ValueError)),
        (["MT", "MB", "MD"], does_not_raise()),
    ],
)
def test_input_output_pairs(seqs, expected):

    with expected:
        assert input_output_pairs(seqs) is not None

    if expected == does_not_raise():
        xs, ys = input_output_pairs(seqs)
        assert xs.shape == (len(seqs), len(seqs[0]) + 1, 10)
        assert ys.shape == (len(seqs), len(seqs[0]) + 1, 25)


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


def test_length_batch_input_outputs():
    """Example test for ``length_batch_input_outputs``."""
    sequences = ["ASDF", "GHJKL", "PILKN"]
    xs, ys = length_batch_input_outputs(sequences)
    assert len(xs) == len(set([len(x) for x in sequences]))
    assert len(ys) == len(set([len(x) for x in sequences]))


def test_get_batch_len():
    batched_seqs = [["ABC", "ACD"], ["AABC", "EKQJ"], ["QWLRJK", "QJEFLK"]]
    mean_batch_length, batch_lengths = get_batch_len(batched_seqs)
    assert mean_batch_length == 2
    assert batch_lens == [2, 2, 2]


def test_evotuning_pairs():
    """Unit test for evotuning_pairs function."""
    sequence = "ACGHJKL"
    x, y = evotuning_pairs(sequence)
    assert x.shape == (len(sequence) + 1, 10)  # embeddings ("x") are width 10
    assert y.shape == (len(sequence) + 1, 25)  # output is one of 25 chars


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


def test_fit(params):
    """
    Execution test for ``jax_unirep.evotuning.fit``.
    """
    sequences = ["ASDFGHJKL", "ASDYGHTKW"]

    fitted_params = fit(params, sequences, n=1)


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
