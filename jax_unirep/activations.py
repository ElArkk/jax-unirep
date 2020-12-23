import jax.numpy as np


def sigmoid(x, version="tanh"):
    """Sigmoid activation function.

    Two versions are provided: "tanh" and "exp".
    "exp" is used in the re-implementation.
    """
    sigmoids = {
        "tanh": lambda x: 0.5 * np.tanh(x) + 0.5,
        "exp": lambda x: safe_sigmoid_exp(x),
    }
    return sigmoids[version](x)


def safe_sigmoid_exp(x, clip_value=-88):
    """
    Safe version of exp version of sigmoid.

    Based on the test "test_sigmoid", we found that a value of -89.0
    gives us NaN values when calculating the gradient of sigmoid.
    As such, we clip -x to a minimum value of -88.0.
    """
    x = np.clip(x, a_min=-88)
    return 1 / (1 + np.exp(-x))


def relu(x, alpha=0.1):
    return x * (x > 0) + alpha * x * (x <= 0)


def tanh(x):
    return np.tanh(x)


def identity(x):
    return x
