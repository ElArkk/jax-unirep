import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_unirep import get_reps
from jax_unirep.sampler import (
    hamming_distance,
    is_accepted,
    propose,
    sample_one_chain,
)
from jax_unirep.utils import proposal_valid_letters


@given(st.data())
def test_is_accepted(data):
    best = data.draw(
        st.floats(
            min_value=0, max_value=1e2, allow_nan=False, allow_infinity=False
        )
    )
    candidate = data.draw(st.floats(allow_nan=False, allow_infinity=False))
    accept = is_accepted(best=best, candidate=candidate, temperature=1)
    assert isinstance(accept, bool)


@given(seq=st.text(alphabet=proposal_valid_letters, min_size=1))
def test_propose_string(seq):
    new_seq = propose(seq)
    assert new_seq != seq
    assert len(new_seq) == len(seq)

    # Check that only one position is different
    different_positions = []
    for i, (l1, l2) in enumerate(zip(seq, new_seq)):
        if l1 != l2:
            different_positions.append(i)
    assert len(different_positions) == 1


def test_propose_empty_string():
    """Check that ValueError is raised with an empty string."""
    with pytest.raises(ValueError):
        new_seq = propose("")


def test_propose_wrong_pos_prob_shape():
    """Check that ValueError is raised when pos_prob of wrong shape is added."""
    with pytest.raises(ValueError):
        new_seq = propose("ADSV", pos_prob=np.array([0.2] * 5))


def test_propose_wrong_pwm_shape():
    """Check that ValueError is raised when pwm of wrong shape is passed in."""
    with pytest.raises(ValueError):
        new_seq = propose("ASDV", pwm=np.array([0.2] * 5))


@pytest.mark.parametrize(
    "seq1,seq2,expected",
    [("AAA", "AAC", 1), ("AAE", "AAC", 1), ("ABC", "DEF", 3)],
)
def test_hamming_distance(seq1, seq2, expected):
    assert hamming_distance(seq1, seq2) == expected


def test_sample_one_chain():
    starter_sequence = "AAC"
    n_steps = 10
    scoring_func = lambda sequence: get_reps(sequence)[0].sum()

    chain_data = sample_one_chain(starter_sequence, n_steps, scoring_func)
    assert set(chain_data.keys()) == set(["sequences", "scores", "accept"])
    for k, v in chain_data.items():
        # +1 because the first step is included too.
        assert len(v) == n_steps + 1
