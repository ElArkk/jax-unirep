from contextlib import suppress as does_not_raise
from shutil import rmtree
from typing import Any, Callable

import numpy as np
import pytest

import pickle as pkl
from jax_unirep.evotuning_models import mlstm1900, mlstm64
from jax_unirep.utils import (
    aa_seq_to_int,
    batch_sequences,
    dump_params,
    evotuning_pairs,
    input_output_pairs,
    l2_normalize,
    length_batch_input_outputs,
    letter_seq,
    load_dense_params,
    load_embedding_1900,
    load_mlstm_params,
    load_params,
    one_hots,
    right_pad,
    validate_mLSTM_params,
)
from jax.random import normal, PRNGKey
from jax import vmap
from functools import partial
import warnings


@pytest.fixture
def model():
    """Dummy mLSTM64 model."""
    init_fun, apply_fun = mlstm64()
    return init_fun, apply_fun


def validate_params(model_func: Callable, params: Any):
    """
    Validate mLSTM parameters against a model.

    In here, we generate dummy embeddings
    and feed them through the model with the mLSTM parameters passed in.

    :param model_func: The model ``apply_func``.
        Should accept (params, input).
    :param params: Model parameters to validate.
    :raises: A generic exception if anything goes wrong,
        alongside a generic warning
        that parameter shape issues may be the problem.
    """
    dummy_embedding = normal(
        PRNGKey(42), shape=(2, 3, 10)  # n_samps, n_letters, n_embed_dims
    )
    try:
        vmap(partial(model_func, params))(dummy_embedding)
    except Exception as e:
        warnings.warn(
            "You may have shape issues! "
            "Check that your params are of the correct shapes "
            "for the specified model. "
            "Here's the original warning below:"
        )
        raise e


def test_l2_normalize():
    """Test for L2 normalization."""
    x = np.array([[3, -3, 5, 4], [4, 5, 3, -3]])

    expected = np.array(
        [
            [3 / 5, -3 / np.sqrt(34), 5 / np.sqrt(34), 4 / 5],
            [4 / 5, 5 / np.sqrt(34), 3 / np.sqrt(34), -3 / 5],
        ],
        dtype=np.float32,
    )

    assert np.allclose(l2_normalize(x, axis=0), expected)


@pytest.mark.parametrize(
    "seqs, expected",
    [
        (pytest.param([], [], marks=pytest.mark.xfail)),
        (["MTN"], [[0]]),
        (["MT", "MTN", "MD"], [[0, 2], [1]]),
        (["MD", "T", "D"], [[1, 2], [0]]),
    ],
)
def test_batch_sequences(seqs, expected):
    assert batch_sequences(seqs) == expected


def test_load_dense_params():
    """
    Make sure that parameters to be passed to
    the dense layer of the evotuning stax model have the right shapes.
    """
    dense = load_dense_params()
    assert dense[0].shape == (1900, 25)
    assert dense[1].shape == (25,)


def test_load_mlstm_params():
    """
    Make sure that parameters to be passed to
    the mLSTM have the right shapes.
    """
    params = load_mlstm_params()
    validate_mLSTM_params(params, n_outputs=1900)


def test_load_embedding_1900():
    """
    Make sure that the inital 10 dimensional aa embedding vectors
    have the right shapes.
    """
    emb = load_embedding_1900()
    assert emb.shape == (26, 10)


def test_load_params():
    """
    Make sure that all parameters needed for the evotuning stax model
    get loaded with the correct shapes.
    """
    _, apply_fun = mlstm1900()
    params = load_params()
    validate_params(model_func=apply_fun, params=params)


def test_dump_params(model):
    """
    Make sure that the parameter dumping function used in evotuning
    conserves all parameter shapes correctly.
    """
    init_fun, apply_fun = model
    _, params = init_fun(PRNGKey(42), input_shape=(-1, 10))
    dump_params(params, "tmp")
    with open("tmp/iter_0/model_weights.pkl", "rb") as f:
        dumped_params = pkl.load(f)
    rmtree("tmp")
    validate_params(model_func=apply_fun, params=dumped_params)


@pytest.mark.parametrize(
    "seqs, max_len, expected",
    [
        (["MT", "MTN", "M"], 4, ["MT--", "MTN-", "M---"]),
        (["MD", "T", "MDT", "MDT"], 2, ["MD", "T-", "MDT", "MDT"]),
    ],
)
def test_right_pad(seqs, max_len, expected):
    assert right_pad(seqs, max_len) == expected


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


def test_length_batch_input_outputs():
    """Example test for ``length_batch_input_outputs``."""
    sequences = ["ASDF", "GHJKL", "PILKN"]
    seqs_batches, seq_lens = length_batch_input_outputs(sequences)
    assert len(seqs_batches) == len(set([len(x) for x in sequences]))
    assert len(seq_lens) == len(set([len(x) for x in sequences]))


def test_evotuning_pairs():
    """Unit test for evotuning_pairs function."""
    sequence = "ACGHJKL"
    x, y = evotuning_pairs(sequence)
    assert x.shape == (len(sequence) + 1, 10)  # embeddings ("x") are width 10
    assert y.shape == (len(sequence) + 1, 25)  # output is one of 25 chars


def test_letter_seq():
    """Test letter_seq function."""
    seq = "ACDEF"
    ints = aa_seq_to_int(seq)
    one_hot = np.stack([one_hots[i] for i in ints])
    assert letter_seq(one_hot) == seq
