"""Loss functions for evotuning."""

import jax.numpy as np
from jax.scipy.special import logsumexp


def cross_entropy_loss(targets, logits, mask=None):
    """Categorical cross-entropy, computed from logits.

    Takes logits rather than probabilities so that logsumexp can fuse the
    softmax with the log, avoiding overflow and log(0).

    :param targets: One-hot targets, shape (..., n_classes).
    :param logits: Unnormalized model outputs, shape (..., n_classes).
    :param mask: Optional weights per position, shape (...). Use 0.0 to drop a
        position from the loss and 1.0 to keep it. The result is averaged over
        the kept positions only.
    :returns: Mean negative log-likelihood over the kept positions.
    """
    log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
    # Sum over classes to select the true one.
    nll = -np.sum(targets * log_probs, axis=-1)

    if mask is None:
        return nll.mean()
    return np.sum(nll * mask) / np.sum(mask)
