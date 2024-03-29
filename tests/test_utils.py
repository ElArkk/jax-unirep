import pickle as pkl
import warnings
from contextlib import suppress as does_not_raise
from functools import partial
from shutil import rmtree
from typing import Any, Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import vmap
from jax.random import PRNGKey, normal

from jax_unirep.evotuning_models import mlstm64, mlstm256, mlstm1900
from jax_unirep.utils import (
    aa_seq_to_int,
    batch_sequences,
    dump_params,
    evotuning_pairs,
    input_output_pairs,
    l2_normalize,
    length_batch_input_outputs,
    letter_seq,
    load_embedding,
    load_params,
    one_hots,
    right_pad,
    seq_to_oh,
)


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
        PRNGKey(42), shape=(2, 3, 26)  # n_samps, n_letters, n_embed_dims
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
    """Make sure sequences get batched together in the right way."""
    assert batch_sequences(seqs) == expected


def test_load_embedding():
    """
    Make sure that the inital 10 dimensional aa embedding vectors
    have the right shapes.
    """
    emb = load_embedding()
    assert emb.shape == (26, 10)


@pytest.mark.parametrize(
    "size, model",
    [
        (64, mlstm64),
        (256, mlstm256),
        (1900, mlstm1900),
    ],
)
def test_load_params(size, model):
    """
    Make sure that all parameters needed for the mlstm stax models
    get loaded with the correct shapes.
    """
    _, apply_fun = model()
    params = load_params(paper_weights=size)
    validate_params(model_func=apply_fun, params=params)


def test_dump_params(model):
    """
    Make sure that the parameter dumping function used in evotuning
    conserves all parameter shapes correctly.
    """
    init_fun, apply_fun = model
    _, params = init_fun(PRNGKey(42), input_shape=(-1, 26))
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
    """
    Make sure right padding sequences to same length
    works as expected.
    """
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
    """Test that the generation of input-output pairs works as expected."""
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
    assert x.shape == (len(sequence) + 1, 26)  # input is one of 26 chars
    assert y.shape == (
        len(sequence) + 1,
        25,
    )  # output is one of 25 chars (no "start")


def test_letter_seq():
    """Test letter_seq function."""
    seq = "ACDEF"
    ints = aa_seq_to_int(seq)
    one_hot = np.stack([one_hots[i] for i in ints])
    assert letter_seq(one_hot) == seq


@given(st.data())
@settings(deadline=None, max_examples=20)
def test_seq_to_oh(data):
    """Make sure the one-hot encoding returns properly shaped matrices."""
    length = data.draw(st.integers(min_value=1, max_value=10))
    sequence = data.draw(
        st.text(
            alphabet="MRHKDESTNQCUGPAVIFYWLOXZBJ",
            min_size=length,
            max_size=length,
        ),
    )

    oh_seq = seq_to_oh(sequence)
    assert oh_seq.shape == (length + 2, 26)
