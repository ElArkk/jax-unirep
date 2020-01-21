from functools import partial

import jax.numpy as np
from jax import lax, vmap

from activations import sigmoid, tanh
from utils import l2_normalize


def mlstm1900(params: dict, x: np.ndarray) -> np.ndarray:
    """
    mLSTM layer for UniRep, in which we pass in the entire dataset.
    :param params: A dictionary of parameters for the model.
        See ``mlstm1900_step`` for exact definitions
        of what parameter names are expected
    :param x: Input tensor,
        which should be of shape (n_samples, n_windows, n_features).
    """
    # Wrap mlstm1900_batch to only take one argument,
    # so that we can vmap it properly.
    # functools partial doesn't work very well
    # with our design that puts params in the first kwarg position,
    # because vmap will then try to pass x to the first positional argument.
    # Explicit wrapping might be the better way to approach this.
    def mlstm1900_vmappable(x):
        return mlstm1900_batch(params=params, batch=x)

    h_final, c_final, outputs = vmap(mlstm1900_vmappable)(x)
    return h_final, c_final, outputs


def mlstm1900_batch(params: dict, batch: np.ndarray) -> np.ndarray:
    """
    LSTM layer implemented according to UniRep,
    found here:
    https://github.com/churchlab/UniRep/blob/master/unirep.py#L43,
    for a batch of data.
    This layer processes one encoded sequence at a time,
    passed as a two dimensional array, with number of rows
    being number of sliding windows, and number of columns
    being the size of the sliding window (for the exact
    reimplementation, window size is fixed to length 10)
    :param params: All weights and biases for a single
        mlstm1900 rnn cell.
    :param batch: One sequence batch, sliced by window size,
        into an array of shape (:, n_windows, n_features).
    """
    h_t = np.zeros(params["wmh"].shape[0])
    c_t = np.zeros(params["wmh"].shape[0])

    step_func = partial(mlstm1900_step, params)
    (h_final, c_final), outputs = lax.scan(
        step_func, init=(h_t, c_t), xs=batch
    )
    return h_final, c_final, outputs


def mlstm1900_step(params: dict, carry: tuple, x_t: np.ndarray):
    """
    Implementation of mLSTMCell from UniRep paper, with weight normalization.
    Exact source code reference:
    https://github.com/churchlab/UniRep/blob/master/unirep.py#L75
    Shapes of parameters:
    - wmx: 10, 1900
    - wmh: 1900, 1900
    - wx: 10, 7600
    - wh: 1900, 7600
    - gmx: 1900
    - gmh: 1900
    - gx: 7600
    - gh: 7600
    - b: 7600
    Shapes of inputs:
    - x_t: (1, 10)
    - carry:
        - h_t: (1, 1900)
        - c_t: (1, 1900)
    """
    h_t, c_t = carry

    # Perform weight normalization first
    # (Corresponds to Line 113).
    # In the original implementation, this is toggled by a boolean flag,
    # but here we are enabling it by default.
    params["wx"] = l2_normalize(params["wx"], axis=0) * params["gx"]
    params["wh"] = l2_normalize(params["wh"], axis=0) * params["gh"]
    params["wmx"] = l2_normalize(params["wmx"], axis=0) * params["gmx"]
    params["wmh"] = l2_normalize(params["wmh"], axis=0) * params["gmh"]

    # Shape annotation
    # (:, 10) @ (10, 1900) * (:, 1900) @ (1900, 1900) => (:, 1900)
    m = np.matmul(x_t, params["wmx"]) * np.matmul(h_t, params["wmh"])

    # (:, 10) @ (10, 7600) * (:, 1900) @ (1900, 7600) + (7600, ) => (:, 7600)
    z = np.matmul(x_t, params["wx"]) + np.matmul(m, params["wh"]) + params["b"]

    # Splitting along axis 1, four-ways, gets us (:, 1900) as the shape
    # for each of i, f, o and u
    i, f, o, u = np.split(z, 4, axis=-1)  # input, forget, output, update

    # Elementwise transforms here.
    # Shapes are are (:, 1900) for each of the four.
    i = sigmoid(i, version="exp")
    f = sigmoid(f, version="exp")
    o = sigmoid(o, version="exp")
    u = tanh(u)

    # (:, 1900) * (:, 1900) + (:, 1900) * (:, 1900) => (:, 1900)
    c_t = f * c_t + i * u

    # (:, 1900) * (:, 1900) => (:, 1900)
    h_t = o * tanh(c_t)

    # h, c each have shape (:, 1900)
    return (h_t, c_t), h_t  # returned this way to match rest of fundl API.
