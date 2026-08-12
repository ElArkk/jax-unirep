"""Numerical regression tests against the original UniRep implementation.

The rest of the suite asserts shapes only, so embedding values could change
silently -- test_featurize.py checks `h_avg.shape == (len(seqs), 1900)` and
never a single value. These tests pin the actual numbers against per-position
hidden states captured from the original (churchlab/unirep) TensorFlow model,
generated with `mLSTMCellStackNPY`. Inputs include the start token and exclude
stop, matching both UniRep and jax-unirep inference.

Reference data lives in tests/data/. It is ground truth, not a snapshot of our
own output -- a failure here means we diverged from the original, so the fix
belongs in the code, never in the fixture.
"""

from pathlib import Path

import numpy as np
import pytest

from jax_unirep import get_reps
from jax_unirep.models import load_model
from jax_unirep.utils import aa_seq_to_int, one_hots

DATA = Path(__file__).parent / "data"

SIZES = [1900, 256, 64]
SEQUENCES = ["PROTEIN", "SEQWENCE"]

# Agreement with the original is ~1e-6 across all three sizes, i.e. float32
# accumulation noise through the recurrence. 1e-4 leaves headroom for op
# reordering during a framework port while still catching anything structural.
TOL = dict(rtol=1e-4, atol=1e-4)


def hidden_states(sequence: str, size: int) -> np.ndarray:
    """Per-position hidden states from the final mLSTM cell.

    The model carries its own embedding matrix and knows its own depth (one
    cell at 1900, four at 256 and 64), so the only thing left to get right is
    the input, and getting it wrong produces plausible-looking output rather
    than an error.

    This calls the model directly rather than `get_reps`, so the reference
    path stays independent of the code under test.
    """
    model = load_model(paper_weights=size)

    # Drop the stop token; the original includes start and excludes stop.
    indices = aa_seq_to_int(sequence)[:-1]
    one_hot = np.stack([one_hots[i] for i in indices])

    _, _, hidden = model(one_hot)
    return np.asarray(hidden)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("sequence", SEQUENCES)
def test_hidden_states_match_original_unirep(sequence, size):
    """Per-position hidden states must match the original model.

    Comparing every timestep rather than the pooled representation means a
    failure localises where the divergence starts.
    """
    reference = np.load(DATA / f"original_unirep_{size}_hidden_states.npz")
    expected = reference[sequence]
    ours = hidden_states(sequence, size)

    assert ours.shape == expected.shape
    np.testing.assert_allclose(ours, expected, **TOL)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("sequence", SEQUENCES)
def test_get_reps_matches_original_unirep(sequence, size):
    """get_reps must be correct for all three published sizes (issue #111).

    Previously it ran a single mLSTM layer with the 1900 embedding regardless
    of the requested size, so 256/64 returned plausible but wrong values.
    """
    reference = np.load(DATA / f"original_unirep_{size}_hidden_states.npz")
    expected = reference[sequence]

    h_avg, h_final, _ = get_reps([sequence], mlstm_size=size)

    np.testing.assert_allclose(
        np.asarray(h_avg)[0], expected.mean(axis=0), **TOL
    )
    np.testing.assert_allclose(np.asarray(h_final)[0], expected[-1], **TOL)


def test_get_reps_uses_the_model_it_is_given():
    """A model passed in must be used as-is, `mlstm_size` notwithstanding.

    Ignoring it and loading 1900 weights instead would still return plausible
    numbers, which is how the #111 failure stayed invisible.
    """
    reference = np.load(DATA / "original_unirep_64_hidden_states.npz")
    expected = reference["PROTEIN"]

    h_avg, h_final, _ = get_reps(
        ["PROTEIN"], model=load_model(paper_weights=64), mlstm_size=1900
    )

    np.testing.assert_allclose(
        np.asarray(h_avg)[0], expected.mean(axis=0), **TOL
    )
    np.testing.assert_allclose(np.asarray(h_final)[0], expected[-1], **TOL)


def test_reps_are_invocation_order_independent():
    """Guards the bug from issue #107.

    In v1, mLSTM params were mutated during inference, so a sequence's
    representation depended on which sequences had been embedded before it.
    Embedding a sequence alone must equal embedding it alongside others.
    """
    together, _, _ = get_reps(SEQUENCES)
    separately = np.vstack([np.asarray(get_reps([s])[0]) for s in SEQUENCES])

    np.testing.assert_allclose(np.asarray(together), separately, **TOL)


def test_reps_unchanged_by_prior_calls():
    """A sequence's representation must not depend on call history."""
    first = np.asarray(get_reps(["SEQWENCE"])[0])
    get_reps(["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQ"])
    get_reps(["PROTEIN", "MTN", "MD"])
    second = np.asarray(get_reps(["SEQWENCE"])[0])

    np.testing.assert_allclose(first, second, **TOL)
