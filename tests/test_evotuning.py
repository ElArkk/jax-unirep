import numpy as np

from jax_unirep.evotuning import (
    evotuning_pairs,
    length_batch_input_outputs,
    input_output_pairs,
    predict,
)
from jax_unirep.utils import (
    load_params_1900,
    load_dense_1900,
)


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
    """Unit test for ``predict``."""
    params = dict()
    params["mlstm1900"] = load_params_1900()
    params["dense"] = load_dense_1900()

    sequences = ["ASDFGHJKL", "ASDYGHTKW"]
    xs, ys = input_output_pairs(sequences)
    res = predict(params, xs)

    assert res.shape == ys.shape
