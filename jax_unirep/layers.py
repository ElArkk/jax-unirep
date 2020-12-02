from typing import Dict, Tuple

import jax.numpy as np
from jax import lax, random
from jax.nn.initializers import glorot_normal, normal

from .activations import sigmoid, tanh
from .utils import l2_normalize

"""mLSTM cell layers."""


def AAEmbedding(embedding_dims: int = 10, E_init=glorot_normal(), **kwargs):
    """
    Initial n-dimensional embedding of each amino-acid
    """

    def init_fun(rng, input_shape):
        """
        Generates the inital AA embedding matrix.

        `input_shape`:
            one-hot encoded AA sequence -> (n_aa, n_unique_aa)
        `output_dims`:
            embedded sequence -> (n_aa, embedding_dims)
        `emb_matrix`:
            embedding matrix -> (n_unique_aa, embedding_dims)
        """
        k1, _ = random.split(rng)
        emb_matrix = E_init(k1, (input_shape[1], embedding_dims))
        output_dims = (-1, embedding_dims)

        return output_dims, emb_matrix

    def apply_fun(params, inputs, **kwargs):
        """
        Embed a single AA sequence
        """
        emb_matrix = params
        # (n_aa, n_unique_aa) * (n_unique_aa, embedding_dims) => (n_aa, embedding_dims) # noqa: E501
        return np.matmul(inputs, emb_matrix)

    return init_fun, apply_fun


def mLSTM(output_dim=1900, W_init=glorot_normal(), b_init=normal()):
    """
    Return stax-compatible mLSTM layer from the UniRep paper.

    This function works on a per-sequence basis,
    meaning that mapping over batches of sequences
    needs to happen outside this function, like this:

    .. code-block:: python

        def apply_fun_vmapped(x):
            return apply_fun(params=params, inputs=x)
        h_final, c_final, outputs = vmap(apply_fun_vmapped)(emb_seqs)

    It returns the average hidden, final hidden and final cell states
    of the mLSTM.
    """

    def init_fun(rng, input_shape):
        """
        Initialize parameters for mLSTM.

        output_dim:
            mlstm cell size -> (1900,)
        input_shape:
            one embedded sequence -> (n_letters, 10)
        output_shape:
            one sequence in 1900 dims -> (n_letters, 1900)
        """
        input_dim = input_shape[-1]

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

    def apply_fun(params, inputs, **kwargs):
        return mLSTMBatch(params=params, batch=inputs)

    return init_fun, apply_fun


def mLSTMAvgHidden(**kwargs):
    """
    Returns the average hidden state of the mlstm.

    This is the canonical "UniRep" representation from the paper.
    """

    def init_fun(rng, input_shape):
        output_shape = (input_shape[-1],)
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return np.mean(inputs[2], axis=0)

    return init_fun, apply_fun


def mLSTMHiddenStates(**kwargs):
    """
    Returns the full hidden states (last element) of the mLSTM layer.
    """

    def init_fun(rng, input_shape):
        output_shape = (input_shape[1],)
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return inputs[2]

    return init_fun, apply_fun


def mLSTMFusion(**kwargs):
    """
    Return the concatenation of all states of the mLSTM.

    This means, it concatenates the average hidden,
    final hidden and final cell states.

    This is the canonical "UniRep fusion" representation from the paper.
    """

    def init_fun(rng, input_shape):
        output_shape = (input_shape[1] * 3,)
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return np.concatenate((inputs[2].mean(axis=0), inputs[0], inputs[1]))

    return init_fun, apply_fun


def mLSTMBatch(
    params: Dict[str, np.ndarray], batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scan mLSTMCell over one sequence batch.

    LSTM layer implemented according to UniRep,
    found here:
    https://github.com/churchlab/UniRep/blob/master/unirep.py#L43,
    for a batch of data.

    This function processes a single embedded sequence,
    passed in as a two dimensional array,
    with number of rows being number of sequence positions,
    and the number of columns being the embedding of each sequence letter.

    :param params: All weights and biases for a single
        mLSTM RNN cell.
    :param batch: One sequence embedded in a (n, 10) matrix,
        where `n` is the number of sequences
    :returns:
    """
    h_t = np.zeros(params["wmh"].shape[0])
    c_t = np.zeros(params["wmh"].shape[0])

    def mLSTMCell(
        carry: Tuple[np.ndarray, np.ndarray],
        x_t: np.ndarray,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Run one step of mLSTM cell.

        Implementation of mLSTMCell from UniRep paper,
        with weight normalization.

        Exact source code reference:
        https://github.com/churchlab/UniRep/blob/master/unirep.py#L75

        Shapes of parameters:

        - wmx: 10, n_outputs
        - wmh: n_outputs, n_outputs
        - wx: 10, n_outputs * 4
        - wh: n_outputs, n_outputs * 4
        - gmx: n_outputs
        - gmh: n_outputs
        - gx: n_outputs * 4
        - gh: n_outputs * 4
        - b: n_outputs * 4

        Shapes of inputs:

        - x_t: (1, 10)
        - carry:
            - h_t: (1, n_outputs)
            - c_t: (1, n_outputs)
        """
        h_t, c_t = carry

        # Perform weight normalization first
        # In the original implementation, this is toggled by a boolean flag,
        # but here we are enabling it by default.
        wx = l2_normalize(params["wx"], axis=0) * params["gx"]
        wh = l2_normalize(params["wh"], axis=0) * params["gh"]
        wmx = l2_normalize(params["wmx"], axis=0) * params["gmx"]
        wmh = l2_normalize(params["wmh"], axis=0) * params["gmh"]

        # Shape annotation
        # (:, 10) @ (10, n_outputs) * (:, n_outputs) @ (n_outputs, n_outputs) => (:, n_outputs)  # noqa: E501
        m = np.matmul(x_t, wmx) * np.matmul(h_t, wmh)

        # (:, 10) @ (10, n_outputs * 4) * (:, n_outputs) @ (n_outputs, n_outputs * 4) + (n_outputs * 4, ) => (:, n_outputs * 4)  # noqa: E501
        z = np.matmul(x_t, wx) + np.matmul(m, wh) + params["b"]

        # Splitting along axis 1, four-ways, gets us (:, n_outputs) as the shape  # noqa: E501
        # for each of i, f, o and u
        i, f, o, u = np.split(z, 4, axis=-1)  # input, forget, output, update

        # Elementwise transforms here.
        # Shapes are are (:, n_outputs) for each of the four.
        i = sigmoid(i, version="exp")
        f = sigmoid(f, version="exp")
        o = sigmoid(o, version="exp")
        u = tanh(u)

        # (:, n_outputs) * (:, n_outputs) + (:, n_outputs) * (:, n_outputs) => (:, n_outputs)  # noqa: E501
        c_t = f * c_t + i * u

        # (:, n_outputs) * (:, n_outputs) => (:, n_outputs)
        h_t = o * tanh(c_t)

        # h, c each have shape (:, n_outputs)
        return (h_t, c_t), h_t

    (h_final, c_final), outputs = lax.scan(
        mLSTMCell, init=(h_t, c_t), xs=batch
    )
    return h_final, c_final, outputs
