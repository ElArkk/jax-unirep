import numpy as np
from typing import Dict

from jax_unirep.evotuning import (
    evotuning_pairs,
    length_batch_input_outputs,
    input_output_pairs,
    predict,
    fit,
)
from jax_unirep.utils import (
    load_params_1900,
    load_dense_1900,
)
from jax.experimental.optimizers import adam


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
