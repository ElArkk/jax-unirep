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
