from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from jax import vmap

from .errors import SequenceLengthsError
from .layers import mLSTM1900
from .utils import (
    batch_sequences,
    get_embeddings,
    load_params_1900,
    validate_mLSTM1900_params,
)

# instantiate the mLSTM
_, apply_fun = mLSTM1900()


def rep_same_lengths(
    seqs: Iterable[str], params: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function generates representations of protein sequences that have the same length,
    by passing them through the UniRep mLSTM.

    :param seqs: A list of same length sequences as strings.
        If passing only a single sequence, it also needs to be passed inside a list.
    :returns: A tuple of np.arrays containing the reps.
        Each `np.array` has shape (n_sequences, 1900).
    """

    embedded_seqs = get_embeddings(seqs)

    h_final, c_final, h = vmap(partial(apply_fun, params))(embedded_seqs)
    h_avg = h.mean(axis=1)

    return np.array(h_avg), np.array(h_final), np.array(c_final)


def rep_arbitrary_lengths(
    seqs: Iterable[str], params: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function generates representations of protein sequences of arbitrary length,
    by batching together all sequences of the same length and passing them through
    the mLSTM. Original order of sequences is restored in the final output.

    :param seqs: A list of sequences as strings.
        If passing only a single sequence, it also needs to be passed inside a list.
    :returns: A 3-tuple of `np.array`s containing the reps.
        Each `np.array` has shape (n_sequences, 1900).
        Return order: (h_avg, h_final, c_final).
    """
    order = batch_sequences(seqs)
    # TODO: Find a better way to do this, without code triplication
    ha_list, hf_list, cf_list = [], [], []
    # Each list in `order` contains the indexes of all sequences of a
    # given length from the original list of sequences.
    for idxs in order:
        subset = [seqs[i] for i in idxs]

        h_avg, h_final, c_final = rep_same_lengths(subset, params)
        ha_list.append(h_avg)
        hf_list.append(h_final)
        cf_list.append(c_final)

    h_avg, h_final, c_final = (
        np.zeros((len(seqs), 1900)),
        np.zeros((len(seqs), 1900)),
        np.zeros((len(seqs), 1900)),
    )
    # Re-order generated reps to match sequence order in the original list.
    for i, subset in enumerate(order):
        for j, rep in enumerate(subset):
            h_avg[rep] = ha_list[i][j]
            h_final[rep] = hf_list[i][j]
            c_final[rep] = cf_list[i][j]

    return h_avg, h_final, c_final


def get_reps(
    seqs: Union[str, Iterable[str]], params: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get reps of proteins.

    This function generates representations of protein sequences
    using the 1900 hidden-unit mLSTM model with pre-trained weights
    from the [UniRep paper](https://github.com/churchlab/UniRep).

    Each element of the output 3-tuple is a `np.array`
    of shape (n_input_sequences, 1900):

    - `h_avg`: Average hidden state of the mLSTM over the whole sequence.
    - `h_final`: Final hidden state of the mLSTM
    - `c_final`: Final cell state of the mLSTM

    You should not use this function
    if you want to do further JAX-based computations
    on the output vectors!
    In that case, the `DeviceArray` futures returned by `mLSTM1900`
    should be passed directly into the next step
    instead of converting them to `np.array`s.
    The conversion to `np.array`s is done
    in the dispatched `rep_x_lengths` functions
    to force python to wait with returning the values
    until the computation is completed.

    The keys of the `params` dictionary must be:

        b, gh, gmh, gmx, gx, wh, wmh, wmx, wx

    ### Parameters

    - `seqs`: A list of sequences as strings or a single string.
    - `params`: A dictionary of mLSTM1900 weights.

    ### Returns

    A 3-tuple of `np.array`s containing the reps,
    in the order `h_avg`, `h_final`, and `c_final`.
    Each `np.array` has shape (n_sequences, 1900).
    """
    if params is None:
        params = load_params_1900()
    # Check that params have correct keys and shapes
    validate_mLSTM1900_params(params)
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
        h_avg, h_final, c_final = rep_same_lengths(seqs, params)
        return h_avg, h_final, c_final
    else:
        h_avg, h_final, c_final = rep_arbitrary_lengths(seqs, params)
        return h_avg, h_final, c_final
