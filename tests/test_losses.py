"""Tests for the evotuning loss."""

import jax.numpy as np
import numpy as onp

from jax_unirep.losses import cross_entropy_loss


def test_finite_when_the_true_class_underflows():
    """A confident, wrong model must not produce a nan loss.

    A logit gap of ~88 drives the true class's probability to exactly 0 in
    float32, so softmax-then-log returns nan. This is what the old tol=1e-10
    clamp patched; logsumexp removes the need for it. See the NaN loss bugs
    in CHANGELOG.md (April 2020, and issue #94).
    """
    loss = cross_entropy_loss(np.array([[1.0, 0.0]]), np.array([[0.0, 100.0]]))

    assert onp.isfinite(loss)
    onp.testing.assert_allclose(loss, 100.0, rtol=1e-5)


def test_masked_positions_cannot_affect_the_loss():
    """Padded positions must contribute nothing.

    right_pad fills with "-", which maps to class 0 -- a real class, not a
    sentinel. Without a mask the model is trained to predict gap characters,
    which is most of the signal for any sequence shorter than the longest in
    its batch.
    """
    # Positions 2 and 3 are padding: their target is class 0.
    targets = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    logits = np.array(
        [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0], [2.0, 0.0, 1.0]]
    )
    mask = 1.0 - targets[..., 0]

    before = cross_entropy_loss(targets, logits, mask)
    wrecked = logits.at[2].set(1e3).at[3].set(-1e3)
    after = cross_entropy_loss(targets, wrecked, mask)

    onp.testing.assert_allclose(before, after)
    # Without the mask, the same change moves the loss.
    assert not onp.allclose(
        cross_entropy_loss(targets, logits),
        cross_entropy_loss(targets, wrecked),
    )
