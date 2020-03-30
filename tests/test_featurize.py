from contextlib import suppress as does_not_raise

import numpy as np
import pytest

from jax_unirep import get_reps
from jax_unirep.errors import SequenceLengthsError
from jax_unirep.featurize import rep_arbitrary_lengths, rep_same_lengths
from jax_unirep.layers import mLSTM1900
from jax_unirep.utils import load_params_1900


@pytest.mark.parametrize(
    "seqs, expected",
    [
        ([], pytest.raises(SequenceLengthsError)),
        (["MT", "M1"], pytest.raises(ValueError)),
        (["MT", "MTN", "MD"], pytest.raises(SequenceLengthsError)),
        (["MTN"], does_not_raise()),
        (["MD", "MT", "DF"], does_not_raise()),
    ],
)
def test_rep_same_lengths(seqs, expected):
    params = load_params_1900()
    _, apply_fun = mLSTM1900()

    with expected:
        assert rep_same_lengths(seqs, params, apply_fun) is not None

    if expected == does_not_raise():
        h_final, c_final, h_avg = rep_same_lengths(seqs, params, apply_fun)
        assert h_final.shape == (len(seqs), 1900)
        assert c_final.shape == (len(seqs), 1900)
        assert h_avg.shape == (len(seqs), 1900)


@pytest.mark.parametrize(
    "seqs, expected",
    [
        ([], pytest.raises(SequenceLengthsError)),
        (["MT", "MTD", "M1"], pytest.raises(ValueError)),
        (["MT", "MTN", "MD"], does_not_raise()),
        (["MTN"], does_not_raise()),
        (["MD", "MT", "DF"], does_not_raise()),
    ],
)
def test_rep_arbitrary_lengths(seqs, expected):
    params = load_params_1900()
    _, apply_fun = mLSTM1900()

    with expected:
        assert rep_arbitrary_lengths(seqs, params, apply_fun) is not None

    if expected == does_not_raise():
        h_final, c_final, h_avg = rep_arbitrary_lengths(
            seqs, params, apply_fun
        )
        assert h_final.shape == (len(seqs), 1900)
        assert c_final.shape == (len(seqs), 1900)
        assert h_avg.shape == (len(seqs), 1900)


def test_get_reps():
    a, b, c = get_reps(["ABC"])
    d, e, f = get_reps("ABC")

    assert np.array_equal(a, d)
    assert np.array_equal(b, e)
    assert np.array_equal(c, f)
