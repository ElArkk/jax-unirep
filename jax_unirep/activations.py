import jax.numpy as np


def softmax(y_hat):
    """
    Returns a 3D array of the same dimensions but with softmax applied to the probabilities,
    to make them probabilities!
    :param y_hat: A 3D array of dimensions (# sequences, # AA's, 25),
        input sequences will be batched to have consistent # AA's,
        & 25 represents a probability of each possible AA.

    """
    return np.exp(y_hat - y_hat.max(axis=2, keepdims=True)) / np.sum(
        (np.exp(y_hat) - y_hat.max(axis=2, keepdims=True)),
        axis=2,
        keepdims=True,
    )


def sigmoid(x, version="tanh"):
    if version not in ["tanh", "exp"]:
        raise ValueError("version must be one of ['tanh' or 'exp']")

    sigmoids = {
        "tanh": lambda x: 0.5 * np.tanh(x) + 0.5,
        "exp": lambda x: 1 / (1 + np.exp(-x)),
    }
    return sigmoids[version](x)


def relu(x, alpha=0.1):
    return x * (x > 0) + alpha * x * (x <= 0)


def tanh(x):
    return np.tanh(x)


def identity(x):
    return x
