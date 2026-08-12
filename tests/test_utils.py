from contextlib import suppress as does_not_raise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_unirep.utils import (
    aa_seq_to_int,
    batch_sequences,
    evotuning_pairs,
    input_output_pairs,
    l2_normalize,
    length_batch_input_outputs,
    letter_seq,
    one_hots,
    right_pad,
    seq_to_oh,
)


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
    # `does_not_raise` suppresses nothing, so the shape assertions below are
    # live for the passing case and skipped for the raising ones. Comparing
    # `expected` to a fresh `does_not_raise()` never matched, which left them
    # asserting a stale 10-dim embedding shape that nothing ever ran.
    with expected:
        xs, ys = input_output_pairs(seqs)
        assert xs.shape == (len(seqs), len(seqs[0]) + 1, 26)
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
