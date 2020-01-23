from typing import List, Tuple

import numpy as np

from .layers import mlstm1900
from .utils import batch_sequences, get_embeddings, load_params_1900


def rep_same_lengths(
    seqs: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function generates representations of protein sequences that have the same length,
    by passing them through the UniRep mLSTM.

    :param seqs: A list of same length sequences as strings.
        If passing only a single sequence, it also needs to be passed inside a list.
    :returns: A tuple of np.arrays containing the reps.
        Each `np.array` has shape (n_sequences, 1900).
    """

    params = load_params_1900()
    embedded_seqs = get_embeddings(seqs)

    h_final, c_final, h = mlstm1900(params, embedded_seqs)
    h_avg = h.mean(axis=1)

    return np.array(h_final), np.array(c_final), np.array(h_avg)


def rep_arbitrary_lengths(
    seqs: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function generates representations of protein sequences of arbitrary length,
    by batching together all sequences of the same length and passing them through
    the mLSTM. Original order of sequences is restored in the final output.

    :param seqs: A list of sequences as strings.
        If passing only a single sequence, it also needs to be passed inside a list.
    :returns: A 3-tuple of `np.array`s containing the reps.
        Each `np.array` has shape (n_sequences, 1900).
    """
    order = batch_sequences(seqs)
    # TODO: Find a better way to do this, without code triplication
    hf_list, cf_list, ha_list = [], [], []
    # Each list in `order` contains the indexes of all sequences of a
    # given length from the original list of sequences.
    for idxs in order:
        subset = [seqs[i] for i in idxs]

        h_final, c_final, h_avg = get_reps(subset)
        hf_list.append(h_final)
        cf_list.append(c_final)
        ha_list.append(h_avg)

    h_final, c_final, h_avg = (
        np.zeros((len(seqs), 1900)),
        np.zeros((len(seqs), 1900)),
        np.zeros((len(seqs), 1900)),
    )
    # Re-order generated reps to match sequence order in the original list.
    for i, subset in enumerate(order):
        for j, rep in enumerate(subset):
            h_final[rep] = hf_list[i][j]
            c_final[rep] = cf_list[i][j]
            h_avg[rep] = ha_list[i][j]

    return h_final, c_final, h_avg


def get_reps(seqs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function generates representations of protein sequences using the 
    1900 hidden-unit mLSTM model with pre-trained weights from the UniRep
    paper (https://github.com/churchlab/UniRep).

    Each element of the output 3-tuple is a `np.array`
    of shape (n_input_sequences, 1900):
    - `h_final`: Final hidden state of the mLSTM
    - `c_final`: Final cell state of the mLSTM
    - `h_avg`: Average hidden state of the mLSTM over the whole sequence.

    You should not use this function if you want to do further JAX-based computations
    on the output vectors! In that case, the `DeviceArray` futures returned by `mlstm1900`
    should be passed directly into the next step instead of converting them to `np.array`s.
    The conversion to `np.array`s is done here to force python to wait with returning the values
    until the computation is actually completed.


    :param seqs: A list of sequences as strings.
        If passing only a single sequence, it also needs to be passed inside a list.
    :returns: A 3-tuple of `np.array`s containing the reps.
        Each `np.array` has shape (n_sequences, 1900).
    """
    # Make sure list is not empty
    if len(seqs) == 0:
        raise SequenceLengthsError("Cannot pass in empty list of sequences.")

    # Differentiate between two cases:
    # 1. All sequences in the list have the same length
    # 2. There are sequences of different lengths in the list
    if len(set([len(s) for s in seqs])) == 1:
        h_final, c_final, h_avg = rep_same_lengths(seqs)
        return h_final, c_final, h_avg
    else:
        h_final, c_final, h_avg = rep_arbitrary_lengths(seqs)
        return h_final, c_final, h_avg
