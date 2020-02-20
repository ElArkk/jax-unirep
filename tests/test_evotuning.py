from contextlib import suppress as does_not_raise
from typing import Dict

import numpy as np
import pytest
from jax.experimental.optimizers import adam

from jax_unirep.evotuning import (
    evotune_loss_funcs,
    evotune_step,
    evotuning_pairs,
    fit,
    input_output_pairs,
    length_batch_input_outputs,
    predict,
)
from jax_unirep.utils import load_dense_1900, load_params_1900


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


def test_evotune_step():
    params = dict()
    params["dense"] = load_dense_1900()
    params["mlstm1900"] = load_params_1900()

    seqs = ["MTB", "MBD", "MDT"]
    xs, ys = length_batch_input_outputs(seqs)

    init, update, get_params = adam(step_size=0.005)
    optimizer_funcs = update, get_params

    state = init(params)
    for x, y in zip(xs, ys):
        state = evotune_step(
            0, state, optimizer_funcs, evotune_loss_funcs, x, y
        )
    params_new = get_params(state)

    assert_param_shapes_equal(params, params_new)


def test_length_batch_input_outputs():
    """Example test for ``length_batch_input_outputs``."""
    sequences = ["ASDF", "GHJKL", "PILKN"]
    xs, ys = length_batch_input_outputs(sequences)
    assert len(xs) == len(set([len(x) for x in sequences]))
    assert len(ys) == len(set([len(x) for x in sequences]))


def test_evotuning_pairs():
    sequence = "ACGHJKL"
    x, y = evotuning_pairs(sequence)
    assert x.shape == (len(sequence) + 1, 10)  # embeddings ("x") are width 10
    assert y.shape == (len(sequence) + 1, 25)  # output is one of 25 chars


def test_predict():
    """
    Unit test for ``jax_unirep.evotuning.predict``.

    We test that the shape of the output of ``predict``
    is identical to the shape of the ys to predict.
    """
    params = dict()
    params["mlstm1900"] = load_params_1900()
    params["dense"] = load_dense_1900()

    sequences = ["ASDFGHJKL", "ASDYGHTKW"]
    xs, ys = input_output_pairs(sequences)
    res = predict(params, xs)

    assert res.shape == ys.shape


def test_fit():
    """
    Execution test for ``jax_unirep.evotuning.fit``.

    Basically ensuring that the output arrays have the same shape,
    and that there are no errors executing the function
    on a gold-standard test.
    """
    params = dict()
    params["mlstm1900"] = load_params_1900()
    params["dense"] = load_dense_1900()

    sequences = ["ASDFGHJKL", "ASDYGHTKW"]

    fitted_params = fit(params, sequences, n=1)

    assert_param_shapes_equal(params, fitted_params)
    # assert len(state) == len(final_state)

    # for state_el, final_state_el in zip(state[0], final_state[0]):
    #     for s, fs in zip(state_el, final_state_el):
    #         assert s.shape == fs.shape


def assert_param_shapes_equal(params1: Dict, params2: Dict):
    """Assert that two parameter dictionaries are equal."""
    for k, v in params1.items():
        for k2, v2 in v.items():
            assert v2.shape == params2[k][k2].shape
