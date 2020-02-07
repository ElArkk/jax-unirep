from functools import partial
from typing import Dict, List, Tuple

from jax import grad, numpy as np, jit
from jax.experimental.optimizers import adam

from .activations import softmax
from .layers import dense, mlstm1900
from .losses import neg_cross_entropy_loss
from .params import add_dense_params
from .utils import (
    get_embeddings,
    batch_sequences,
    aa_seq_to_int,
    load_embeddings,
    one_hots,
)


# def evotune(
#     mlstm1900_params: Dict[str, np.array], seqs: List[str]
# ) -> Dict[str, np.array]:
#     """
#     Given a set of weights for the `mlstm1900` UniRep model,
#     as well as protein sequences of arbitrary length,
#     this function will perform weight updates on the mLSTM,
#     under the pretext learning task of predicting the next
#     amino acid in the protein sequences, given the output of the mLSTM.
#     The prediction itself is being done by a single, fully-connected
#     layer with 26 output nodes and using softmax activation
#     (Each node corresponding to one AA).

#     :param params: Either pre-trained or random weights to initalize
#         the mLSTM with, as `np.arrays`.
#     :param seqs: A list of protein sequences as strings
#     """

#     params = dict()
#     params["mlstm1900"] = mlstm1900_params
#     params["dense"] = add_dense_params()

#     def predict(params, batch):
#         batch = mlstm1900(params["mlstm1900"], batch)
#         batch = dense(params["dense"], batch, activation=softmax)
#         return batch

#     loss = partial(neg_cross_entropy_loss, model=predict)
#     dloss = grad(loss)

#     init, update, get_params = adam(step_size=0.005)

#     state = init(params)

#     for i in range(epochs):
#         g = dloss(params, x=x, y=y)

#         state = update(i, g, state)
#         params = get_params(state)

#         if i % 10 == 0:
#             l = loss(params, x=x, y=y)
#             print(f"iteration: {i}, loss: {l}")
#     # def update(params, x, y):

#     # prepare the sequences
#     # batch sequences according to length


def evotuning_pairs(s: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a sequence, return input-output pairs for evotuning.

    The goal of evotuning is to get the RNN to accurately predict
    the next character in a sequence.
    This convenience function exists to prep a single sequence
    into its corresponding input-output tensor pairs.

    Given a 1D sequence of length `k`,
    it gets represented as a 2D array of shape (k, 10),
    where 10 is the size of the embedding of each amino acid,
    and k-1 ranges from the zeroth a.a. to the nth a.a.
    This is the first element in the returned tuple.

    Given the same 1D sequence,
    the output is defined as a 2D array of shape (k-1, 25),
    where 25 is number of indices available to us
    in ``aa_to_int``,
    and k-1 corresponds to the first a.a. to the nth a.a.
    This is the second element in the returned tuple.

    :param s: The protein sequence to featurize.
    :returns: Two 2D NumPy arrays,
        the first corresponding to the input to evotuning with shape (n_letters, 10),
        and the second corresponding to the output amino acid to predict with shape (n_letters, 25).
    """
    seq_int = aa_seq_to_int(s[:-1])
    next_letters_int = aa_seq_to_int(s[1:])
    embeddings = load_embeddings()
    x = np.stack([embeddings[i] for i in seq_int])
    y = np.stack([one_hots[i] for i in next_letters_int])
    return x, y


def input_output_pairs(sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate input-output tensor pairs for evo-tuning.

    We check that lengths of sequences are identical,
    as this is necessary to ensure stacking of tensors happens correctly.

    :param sequences: A list of sequences
        to generate input-output tensor pairs.
    :returns: Two NumPy arrays,
        the first corresponding to the input to evotuning
        with shape (n_sequences, n_letters, 10),
        and the second corresponding to the output amino acids to predict
        with shape (n_sequences, n_letters, 25).
        Both will have an additional "sample" dimension as the first dim.
    """
    seqlengths = set(map(len, sequences))
    if not len(seqlengths) == 1:
        raise ValueError(
            """
Sequences should be of uniform length, but are not.
Please ensure that they are all of the same length before passing them in.
"""
        )

    xs = []
    ys = []
    for s in sequences:
        x, y = evotuning_pairs(s)
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.stack(ys)


def predict(params, x) -> np.ndarray:
    """
    Prediction model for evotuning.

    Architecture is a single softmax layer on top of the RNN.

    :param params: Dictionary of parameters.
        Should have keys ``mlstm1900`` and ``dense`` in there.
    :param x: Input tensor.
        Should be the result of calling ``input_output_pairs``,
        and be of shape (n_sequences, n_letters, 10).
    :returns: Prediction tensor, of shape (n_sequences, n_letters, 25).
    """
    # Defensive programming checks.
    if not len(x.shape) == 3:
        raise ValueError("Input tensor should be 3-dimensional.")
    if not x.shape[-1] == 10:
        raise ValueError(
            "Input tensor's 3rd dimension should be of length 10."
        )

    # Actual forward model happens here.
    _, _, x = mlstm1900(params["mlstm1900"], x)
    x = dense(params["dense"], x, activation=softmax)
    return x


from typing import Tuple, Callable


def evotune_step(
    i: int,
    state,
    optimizer_funcs: Tuple[Callable, Callable],
    loss_funcs: Tuple[Callable, Callable],
    x: np.ndarray,
    y: np.ndarray,
    # params: Dict[str, Dict[str, np.ndarray]],
    # x: np.ndarray,
    # y: np.ndarray,
    # n: int,
    # verbose=False,
):
    """
    ;param i: The current iteration of the training loop.
    :param state: Current state of parameters from jax.
    :param optimizer_funcs: The (update, get_params) functions
        from jax's optimizers.
    :param loss_funcs: The loss and dloss functions.
        dloss should be generated by calling grad(loss).
    :return state: Updated state of parameters from jax.
    """
    # Unpack optimizer funcs
    update, get_params = optimizer_funcs
    params = get_params(state)

    # Unpack loss funcs
    loss, dloss = loss_funcs

    l = loss(params, x=x, y=y)
    if np.isnan(l):
        raise Exception("NaN occurred in optimization.")
    print(f"Iteration: {i}, Loss: {l:.4f}")

    g = dloss(params, x=x, y=y)

    state = update(i, g, state)
    return state


def evotune(params: Dict, sequences: List[str], n: int) -> Dict:
    """
    Return evolutionarily-tuned weights.

    Evotuning is described in the original UniRep and eUniRep papers.
    This reimplementation of evotune provides a nicer API
    that automatically handles multiple sequences of variable lengths.

    The training loop is as follows.
    Per step in the training loop,
    we loop over each "length batch" of sequences and tune weights
    in order of the length of each sequence.
    For example, if we have sequences of length 302, 305, and 309,
    over K training epochs,
    we will perform 3xK updates,
    one step of updates for each length.

    To get batching of sequences by length done,
    we call on ``batch_sequences`` from our ``utils.py`` module,
    which returns a list of sub-lists,
    in which each sub-list contains the indices
    in the original list of sequences
    that are of a particular length.

    :param params: mLSTM1900 parameters.
    :param sequences: List of sequences to evotune on.
    :param n: The number of iterations to evotune on.
    """
    idxs_batched = batch_sequences(sequences)

    # Obtain x,y pairs
    xs = []
    ys = []
    for idxs in idxs_batched:
        seqs = [sequences[i] for i in idxs]
        x, y = input_output_pairs(seqs)
        xs.append(x)
        ys.append(y)

    init, update, get_params = adam(step_size=0.005)
    optimizer_funcs = update, get_params

    loss = partial(neg_cross_entropy_loss, model=predict)
    dloss = jit(grad(loss))
    loss_funcs = (loss, dloss)

    state = init(params)
    for i in range(n):  # TODO: Magic number
        for x, y in zip(xs, ys):
            state = evotune_step(i, state, optimizer_funcs, loss_funcs, x, y)

    return get_params(state)
    # """
    # Master function for tuning.

    # :param x: Input tensor.
    # :param y: Output tensor to train against.
    # :param n: Number of epochs (iterations) to train model for.
    # """
    # # `predict` must be defined in the same source file as this function.
    # loss = partial(neg_cross_entropy_loss, model=predict)
    # dloss = jit(grad(loss))

    # init, update, get_params = adam(step_size=0.005)

    # state = init(params)

    # for i in range(20):

    #     l = loss(params, x=x, y=y)
    #     if np.isnan(l):
    #         break
    #     if verbose:
    #         print(f"Iteration: {i}, Loss: {l:.4f}")

    #     g = dloss(params, x=x, y=y)

    #     state = update(i, g, state)
    #     params = get_params(state)
    # return params
