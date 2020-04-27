import numpy as np
import pytest

from jax_unirep.utils import batch_sequences, l2_normalize, get_batch_len


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
