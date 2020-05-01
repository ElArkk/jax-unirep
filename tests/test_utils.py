from shutil import rmtree

import numpy as np
import pytest

from jax_unirep.utils import (
    batch_sequences,
    dump_params,
    get_batch_len,
    l2_normalize,
    load_dense_1900,
    load_embedding_1900,
    load_params,
    load_params_1900,
    validate_mLSTM1900_params,
)


def test_l2_normalize():
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


def test_get_batch_len():
    batched_seqs = [["ABC", "ACD"], ["AABC", "EKQJ"], ["QWLRJK", "QJEFLK"]]
    mean_batch_length, batch_lengths = get_batch_len(batched_seqs)
    assert mean_batch_length == 2
    assert np.all(batch_lengths == np.array([2, 2, 2]))


def test_load_dense_1900():
    """
    Make sure that parameters to be passed to 
    the dense layer of the evotuning stax model have the right shapes.
    """
    dense = load_dense_1900()
    assert dense[0].shape == (1900, 25)
    assert dense[1].shape == (25,)


def test_load_params_1900():
    """
    Make sure that parameters to be passed to 
    the mlstm1900 have the right shapes.
    """
    params = load_params_1900()
    validate_mLSTM1900_params(params)


def test_load_embedding_1900():
    """
    Make sure that the inital 10 dimensional aa embedding vectors 
    have the right shapes.
    """
    emb = load_embedding_1900()
    assert emb.shape == (26, 10)


def validate_params(params):
    validate_mLSTM1900_params(params[0])
    assert params[1] == ()
    assert params[2][0].shape == (1900, 25)
    assert params[2][1].shape == (25,)
    assert params[3] == ()


def test_load_params():
    """
    Make sure that all parameters needed for the evotuning stax model
    get loaded with the correct shapes.
    """
    params = load_params()
    validate_params(params)


def test_dump_params():
    """
    Make sure that the parameter dumping function used in evotuning
    conserves all parameter shapes correctly.
    """
    params = load_params()
    dump_params(params, "tmp")
    dumped_params = load_params("tmp/iter_0")
    rmtree("tmp")
    validate_params(dumped_params)
