from jax_unirep.evotuning import length_batch_input_outputs, evotuning_pairs
import numpy as np


def test_length_batch_input_outputs():
    sequences = ["ASDF", "GHJKL", "PILKN"]
    xs, ys = length_batch_input_outputs(sequences)
    assert len(xs) == len(set([len(x) for x in sequences]))
    assert len(ys) == len(set([len(x) for x in sequences]))


def test_evotuning_pairs():
    sequence = "ACGHJKL"
    x, y = evotuning_pairs(sequence)
    assert x.shape == (len(sequence) + 1, 10)  # embeddings ("x") are width 10
    assert y.shape == (len(sequence) + 1, 25)  # output is one of 25 chars
