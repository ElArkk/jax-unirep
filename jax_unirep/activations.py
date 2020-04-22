import jax.numpy as np


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
