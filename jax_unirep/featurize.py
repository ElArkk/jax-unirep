from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from jax import vmap

from .errors import SequenceLengthsError
from .layers import mLSTM
from .utils import batch_sequences, get_embeddings, load_params


def unpack_params(params) -> Tuple[np.ndarray, List[Dict]]:
    """Split a packed parameter tree into embedding and mLSTM layers.

    Both `load_params()` and `fit()` return the full stax tree, which looks
    like `(embedding, mlstm, (), [mlstm, (), ...], dense, ())`. The embedding
    matrix is the first element and every mLSTM layer is a dict, so the tree
    is self-describing: its depth tells us how many recurrent layers to run.

    :param params: A packed parameter tree.
    :returns: `(embedding_matrix, [mlstm_params, ...])` in layer order.
    """
    if isinstance(params, dict):
        raise TypeError(
            "get_reps expects the full parameter tree returned by "
            "load_params() or fit(), not a bare mLSTM weight dict. "
            "A bare dict cannot describe the stacked 256/64 models, and "
            "carries no embedding matrix. Pass load_params(paper_weights=N)."
        )

    embedding = params[0]
    recurrent_params = [p for p in params if isinstance(p, dict)]
    if not recurrent_params:
        raise ValueError("No mLSTM layers found in the parameter tree.")

    return embedding, recurrent_params


def apply_mlstm_stack(
    embedded_seqs: np.ndarray, recurrent_params: List[Dict], mlstm_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a stack of mLSTM layers, each consuming the previous one's states.

    The 1900 model has a single layer; the 256 and 64 models stack four.

    :returns: `(h_final, c_final, hidden_states)` of the *final* layer.
    """
    h_final = c_final = None
    activations = embedded_seqs

    for layer_params in recurrent_params:
        _, apply_fun = mLSTM(output_dim=mlstm_size)
        h_final, c_final, activations = vmap(partial(apply_fun, layer_params))(
            activations
        )

    return h_final, c_final, activations


def rep_same_lengths(
    seqs: Iterable[str],
    embedding: np.ndarray,
    recurrent_params: List[Dict],
    mlstm_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function generates representations of protein sequences that have the same length,
    by passing them through the UniRep mLSTM.

    :param seqs: A list of same length sequences as strings.
        If passing only a single sequence, it also needs to be passed inside a list.
    :param embedding: The model's amino acid embedding matrix.
    :param recurrent_params: The model's mLSTM layer params, in layer order.
    :param mlstm_size: Number of nodes per mLSTM layer.
    :returns: A tuple of np.arrays containing the reps.
        Each `np.array` has shape (n_sequences, mlstm_size).
    """
    embedded_seqs = get_embeddings(seqs, embedding)

    h_final, c_final, h = apply_mlstm_stack(
        embedded_seqs, recurrent_params, mlstm_size
    )
    h_avg = h.mean(axis=1)

    return np.array(h_avg), np.array(h_final), np.array(c_final)


def rep_arbitrary_lengths(
    seqs: Iterable[str],
    embedding: np.ndarray,
    recurrent_params: List[Dict],
    mlstm_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function generates representations of protein sequences of arbitrary length,
    by batching together all sequences of the same length and passing them through
    the mLSTM. Original order of sequences is restored in the final output.

    This function exists to speed up the original published workflow
    of "repping one sequence at a time", through repping of all sequences
    of the same length at once.
    Repping one sequence length at a time avoids generating noise
    in the reps from adding padding characters.

    :param seqs: A list of sequences as strings.
        If passing only a single sequence, it also needs to be passed inside a list.
    :param embedding: The model's amino acid embedding matrix.
    :param recurrent_params: The model's mLSTM layer params, in layer order.
    :param mlstm_size: Integer specifying the number of nodes in the mLSTM layer.
        Though the model architecture space is practically infinite,
        we assume that you are using the same number of nodes per mLSTM layer.
        (This is a common simplification used in the design of neural networks.)
    :returns: A 3-tuple of `np.array`s containing the reps.
        Each `np.array` has shape (n_sequences, mlstm_size).
        Return order: (h_avg, h_final, c_final).
    """
    order = batch_sequences(seqs)
    # TODO: Find a better way to do this, without code triplication
    ha_list, hf_list, cf_list = [], [], []
    # Each list in `order` contains the indexes of all sequences of a
    # given length from the original list of sequences.
    for idxs in order:
        subset = [seqs[i] for i in idxs]

        h_avg, h_final, c_final = rep_same_lengths(
            subset, embedding, recurrent_params, mlstm_size
        )
        ha_list.append(h_avg)
        hf_list.append(h_final)
        cf_list.append(c_final)

    h_avg, h_final, c_final = (
        np.zeros((len(seqs), mlstm_size)),
        np.zeros((len(seqs), mlstm_size)),
        np.zeros((len(seqs), mlstm_size)),
    )
    # Re-order generated reps to match sequence order in the original list.
    for i, subset in enumerate(order):
        for j, rep in enumerate(subset):
            h_avg[rep] = ha_list[i][j]
            h_final[rep] = hf_list[i][j]
            c_final[rep] = cf_list[i][j]

    return h_avg, h_final, c_final


def get_reps(
    seqs: Union[str, Iterable[str]],
    params: Optional[Dict] = None,
    mlstm_size: Optional[str] = 1900,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get reps of proteins.

    This function generates representations of protein sequences
    using the mLSTM model from the
    [UniRep paper](https://github.com/churchlab/UniRep).

    Each element of the output 3-tuple is a `np.array`
    of shape (n_input_sequences, mlstm_size):

    - `h_avg`: Average hidden state of the mLSTM over the whole sequence.
    - `h_final`: Final hidden state of the mLSTM
    - `c_final`: Final cell state of the mLSTM

    You should not use this function
    if you want to do further JAX-based computations
    on the output vectors!
    In that case, the `DeviceArray` futures returned by `mLSTM`
    should be passed directly into the next step
    instead of converting them to `np.array`s.
    The conversion to `np.array`s is done
    in the dispatched `rep_x_lengths` functions
    to force python to wait with returning the values
    until the computation is completed.

    All three published model sizes are supported. The 1900 model has a single
    mLSTM layer; the 256 and 64 models stack four, and each model carries its
    own amino acid embedding matrix. Both are read from the parameter tree, so
    no architecture description is needed beyond the weights themselves.

    ### Parameters

    - `seqs`: A list of sequences as strings or a single string.
    - `params`: A full parameter tree as returned by `load_params()` or `fit()`,
        containing the embedding matrix and every mLSTM layer. Passing a bare
        mLSTM weight dict raises `TypeError`: it carries no embedding and
        cannot describe a stacked model. When given, the tree's own width
        takes precedence over `mlstm_size`.
    - `mlstm_size`: Which set of pre-trained weights to load when `params` is
        None. One of 1900, 256 or 64.

    ### Returns

    A 3-tuple of `np.array`s containing the reps,
    in the order `h_avg`, `h_final`, and `c_final`.
    Each `np.array` has shape (n_sequences, mlstm_size).
    """
    if params is None:
        params = load_params(paper_weights=mlstm_size)
    embedding, recurrent_params = unpack_params(params)

    # The tree knows its own width, so derive it rather than trusting the
    # caller's mlstm_size, which only selects which weights to load. This is
    # also why nothing is validated here: the caller no longer declares a size
    # that could disagree with the weights.
    mlstm_size = recurrent_params[0]["gmh"].shape[0]

    # If single string sequence is passed, package it into a list
    if isinstance(seqs, str):
        seqs = [seqs]
    # Make sure list is not empty
    if len(seqs) == 0:
        raise SequenceLengthsError("Cannot pass in empty list of sequences.")

    # Differentiate between two cases:
    # 1. All sequences in the list have the same length
    # 2. There are sequences of different lengths in the list
    if len(set([len(s) for s in seqs])) == 1:
        return rep_same_lengths(seqs, embedding, recurrent_params, mlstm_size)
    return rep_arbitrary_lengths(seqs, embedding, recurrent_params, mlstm_size)
