from contextlib import suppress as does_not_raise

import numpy as np
import pytest

from jax_unirep import get_reps
from jax_unirep.errors import SequenceLengthsError
from jax_unirep.featurize import rep_arbitrary_lengths, rep_same_lengths
from jax_unirep.models import load_model

# The smallest published model: these tests check shapes and plumbing, not
# numbers, so there is no reason to load 73 MB of 1900-sized weights for them.
# The numerical checks live in test_regression.py.
MODEL = load_model(paper_weights=64)


@pytest.mark.parametrize(
    "seqs, expected",
    [
        (["MT", "M1"], pytest.raises(ValueError)),
        (["MTN"], does_not_raise()),
        (["MD", "MT", "DF"], does_not_raise()),
    ],
)
def test_rep_same_lengths(seqs, expected):
    with expected:
        h_avg, h_final, c_final = rep_same_lengths(seqs, MODEL)

        assert h_avg.shape == (len(seqs), MODEL.output_dim)
        assert h_final.shape == (len(seqs), MODEL.output_dim)
        assert c_final.shape == (len(seqs), MODEL.output_dim)


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
    with expected:
        h_avg, h_final, c_final = rep_arbitrary_lengths(seqs, MODEL)

        assert h_avg.shape == (len(seqs), MODEL.output_dim)
        assert h_final.shape == (len(seqs), MODEL.output_dim)
        assert c_final.shape == (len(seqs), MODEL.output_dim)


def test_rep_arbitrary_lengths_restores_order():
    """Sequences are repped grouped by length, but returned in input order."""
    seqs = ["MD", "MTN", "MT", "DFGH", "DF"]
    reps, _, _ = rep_arbitrary_lengths(seqs, MODEL)

    for i, seq in enumerate(seqs):
        np.testing.assert_array_equal(
            reps[i], rep_same_lengths([seq], MODEL)[0][0]
        )


def test_get_reps_accepts_a_bare_string():
    listed = get_reps(["ABC"], model=MODEL)
    bare = get_reps("ABC", model=MODEL)

    for from_list, from_string in zip(listed, bare):
        assert np.array_equal(from_list, from_string)


def test_get_reps_rejects_empty_input():
    with pytest.raises(SequenceLengthsError):
        get_reps([])


def test_get_reps_uses_the_model_it_is_given():
    """A passed-in model's own width wins; `mlstm_size` is only for loading."""
    h_avg, h_final, c_final = get_reps(
        ["ABC", "DEFGH", "DEF"], model=MODEL, mlstm_size=1900
    )

    assert h_avg.shape == (3, 64)
    assert h_final.shape == (3, 64)
    assert c_final.shape == (3, 64)


def test_get_reps():
    h_avg, h_final, c_final = get_reps(["ABC", "DEFGH", "DEF"])

    assert h_avg.shape == (3, 1900)
    assert h_final.shape == (3, 1900)
    assert c_final.shape == (3, 1900)
