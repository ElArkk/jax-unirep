from contextlib import suppress as does_not_raise

import pytest

from jax_unirep.errors import SequenceLengthsError
from jax_unirep.featurize import rep_arbitrary_lengths, rep_same_lengths


@pytest.mark.parametrize(
    "seqs, expected",
    [
        ([], pytest.raises(SequenceLengthsError)),
        (["MT", "MTN", "MD"], pytest.raises(SequenceLengthsError)),
        (["MTN"], does_not_raise()),
        (["MD", "MT", "DF"], does_not_raise()),
    ],
)
def test_rep_same_lengths(seqs, expected):
    with expected:
        assert rep_same_lengths(seqs) is not None

    if expected == does_not_raise():
        h_final, c_final, h_avg = rep_same_lengths(seqs)
        assert h_final.shape == (len(seqs), 1900)
        assert c_final.shape == (len(seqs), 1900)
        assert h_avg.shape == (len(seqs), 1900)


@pytest.mark.parametrize(
    "seqs, expected",
    [
        ([], pytest.raises(SequenceLengthsError)),
        (["MT", "MTN", "MD"], does_not_raise()),
        (["MTN"], does_not_raise()),
        (["MD", "MT", "DF"], does_not_raise()),
    ],
)
def test_rep_arbitrary_lengths(seqs, expected):
    with expected:
        assert rep_arbitrary_lengths(seqs) is not None

    if expected == does_not_raise():
        h_final, c_final, h_avg = rep_arbitrary_lengths(seqs)
        assert h_final.shape == (len(seqs), 1900)
        assert c_final.shape == (len(seqs), 1900)
        assert h_avg.shape == (len(seqs), 1900)
