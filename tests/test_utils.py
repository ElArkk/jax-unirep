import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

### TO REFACTOR INTO test_sampler.py ###
from jax_unirep.utils import (
    batch_sequences,
    l2_normalize,
    proposal_valid_letters,
    propose,
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


@given(seq=st.text(alphabet=proposal_valid_letters, min_size=1))
def test_propose_string(seq):
    new_seq = propose(seq)
    assert new_seq != seq

    # Check that only one position is different
    different_positions = []
    for i, (l1, l2) in enumerate(zip(seq, new_seq)):
        if l1 != l2:
            different_positions.append(i)
    assert len(different_positions) == 1
