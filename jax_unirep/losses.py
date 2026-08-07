"""Loss functions for evotuning."""

import jax.numpy as np
from jax.scipy.special import logsumexp


def cross_entropy_loss(targets, logits):
    """Categorical cross-entropy, computed from logits.

    Takes logits rather than probabilities so that logsumexp can fuse the
    softmax with the log, avoiding overflow and log(0).

    :param targets: One-hot targets, shape (..., n_classes).
    :param logits: Unnormalized model outputs, shape (..., n_classes).
    :returns: Mean negative log-likelihood over all positions.
    """
    log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
    # Sum over classes to select the true one; mean over positions and batch.
    return -np.sum(targets * log_probs, axis=-1).mean()
