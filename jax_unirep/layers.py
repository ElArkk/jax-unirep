from functools import partial
from typing import Dict, Tuple

import jax.numpy as np
from jax import lax, random, vmap
from jax.nn.initializers import glorot_normal, normal

from .activations import identity, sigmoid, tanh
from .utils import l2_normalize


def dense(params: Dict[str, np.ndarray], x: np.ndarray, activation=identity):
    """
    "dense" layers are just affine shifts + activation functions.
    Affine shifts are represented
    by multiplication by weights and adding biases.
    Assumes that params is a dictionary with 'w' and 'b' as keys.
    Activation defaults to identity,
    but any elementwise numpy function can be applied.
    Shapes of inputs should be:
    :param params: A dictionary of weights.
        Should have "w" and "b" as keywords.
        "w" should be of shape (input_dim, output_dim),
        "b" should be of shape (output_dim,),
    :param x: Input data, which should be of shape (:, input_dim).
    :param activation: A callable
        that applies an elementwise activation function on the output array.
    """
    a = activation(np.dot(x, params["w"]) + params["b"])
    return a


## replace with mlstm1900 stax layer
# def mlstm1900(
#     params: Dict[str, np.ndarray], x: np.ndarray
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     mLSTM layer for UniRep, in which we pass in the entire dataset.
#     :param params: A dictionary of parameters for the model.
#         See ``mlstm1900_step`` for exact definitions
#         of what parameter names are expected
#     :param x: Input tensor,
#         which should be of shape (n_samples, n_windows, n_features).
#     :returns: Three tensors.
#         `h_final` is of shape (n_samples, n_features).
#         `c_final` is of shape (n_samples, n_features).
#         `outputs` is of shape (n_samples, n_windows, n_features).
#     """
#     # Wrap mlstm1900_batch to only take one argument,
#     # so that we can vmap it properly.
#     # functools partial doesn't work very well
#     # with our design that puts params in the first kwarg position,
#     # because vmap will then try to pass x to the first positional argument.
#     # Explicit wrapping might be the better way to approach this.
#     def mlstm1900_vmappable(x):
#         return mlstm1900_batch(params=params, batch=x)

#     h_final, c_final, outputs = vmap(mlstm1900_vmappable)(x)
#     return h_final, c_final, outputs

## write tests for init_fun , and stax layer with a sequence

## we will need a stax layer that does the averaging of hidden states
## and one that does concatenation for unirep fusion
## both will not return weight in their init_fun, but instead an empty tuple.
## see stax.elementwise


def mlstm1900(output_dim=1900, W_init=glorot_normal(), b_init=normal()):
    """
    mLSTM cell from the UniRep paper, stax compatible
    
    This function works on a per-sequence basis,
    meaning that mapping over batches of sequences
    needs to happen outside this function, like this:

    .. code-block:: python

        def apply_fun_vmapped(x):
            return apply_fun(params=params, inputs=x)
        h_final, c_final, outputs = vmap(apply_fun_vmapped)(emb_seqs)

    It returns the average hidden, final hidden and final cell states
    of the mlstm.

    """

    def init_fun(rng, input_shape):
        """
        Initialize parameters for mlstm1900
        
        output_dim: 
            mlstm cell size -> (1900,)
        input_shape:
            one embedded sequence -> (n_letters, 10)
        output_shape:
            one sequence in 1900 dims -> (n_letters, 1900)
            
        """
        input_dim = input_shape[1]

        k1, k2, k3, k4 = random.split(rng, num=4)
        wmx, wmh, wx, wh = (
            W_init(k1, (input_dim, output_dim)),
            W_init(k2, (output_dim, output_dim)),
            W_init(k3, (input_dim, output_dim * 4)),
            W_init(k4, (output_dim, output_dim * 4)),
        )

        k1, k2, k3, k4 = random.split(k1, num=4)
        gmx, gmh, gx, gh = (
            b_init(k1, (output_dim,)),
            b_init(k2, (output_dim,)),
            b_init(k3, (output_dim * 4,)),
            b_init(k4, (output_dim * 4,)),
        )

        k1, k2 = random.split(k1)
        b = b_init(k1, (output_dim * 4,))

        params = {
            "wmx": wmx,
            "wmh": wmh,
            "wx": wx,
            "wh": wh,
            "gmx": gmx,
            "gmh": gmh,
            "gx": gx,
            "gh": gh,
            "b": b,
        }
        output_shape = (input_shape[0], output_dim)
        return output_shape, params

    def apply_fun_scan(params, carry, x_t):
        return mlstm1900_step(params=params, carry=carry, x_t=x_t)

    def apply_fun(params, inputs, **kwargs):
        return mlstm1900_batch(params=params, batch=inputs)

    return init_fun, apply_fun


def mlstm1900_avghidden(output_dim=1900, **kwargs):
    """
    Returns the average hidden state of the mlstm.

    This is the canonical "UniRep" representation from the paper.
    """

    def init_fun(rng, input_shape):
        # Maybe include a assertion here that output_dim == output_shape[0]?
        # Not sure how to handle output_dim and output_shape here
        output_shape = (input_shape[1],)
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return inputs[2].mean(axis=0)

    return init_fun, apply_fun


def mlstm1900_fusion(output_dim=5700, **kwargs):
    """
    Returns the concatenation of all states of the mlstm.

    This means, it concatenates the 
    average hidden, final hidden and final cell states.

    This is the canonical "UniRep fusion" representation from the paper.
    """

    def init_fun(rng, input_shape):
        # Maybe include a assertion here that output_dim == output_shape[0]?
        # Not sure how to handle output_dim and output_shape here
        output_shape = (input_shape[1] * 3,)
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return np.concatenate((inputs[2].mean(axis=0), inputs[0], inputs[1]))

    return init_fun, apply_fun


def mlstm1900_batch(
    params: Dict[str, np.ndarray], batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    :returns:
    """
    h_t = np.zeros(params["wmh"].shape[0])
    c_t = np.zeros(params["wmh"].shape[0])

    step_func = partial(mlstm1900_step, params)
    (h_final, c_final), outputs = lax.scan(
        step_func, init=(h_t, c_t), xs=batch
    )
    return h_final, c_final, outputs


def mlstm1900_step(
    params: Dict[str, np.ndarray],
    carry: Tuple[np.ndarray, np.ndarray],
    x_t: np.ndarray,
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
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
