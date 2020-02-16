from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import jax.numpy as np
import numpy as onp
import pkg_resources

from .errors import SequenceLengthsError

aa_to_int = {
    "M": 1,
    "R": 2,
    "H": 3,
    "K": 4,
    "D": 5,
    "E": 6,
    "S": 7,
    "T": 8,
    "N": 9,
    "Q": 10,
    "C": 11,
    "U": 12,
    "G": 13,
    "P": 14,
    "A": 15,
    "V": 16,
    "I": 17,
    "F": 18,
    "Y": 19,
    "W": 20,
    "L": 21,
    "O": 22,  # Pyrrolysine
    "X": 23,  # Unknown
    "Z": 23,  # Glutamic acid or GLutamine
    "B": 23,  # Asparagine or aspartic acid
    "J": 23,  # Leucine or isoleucine
    "start": 24,
    "stop": 25,
}

weights_1900_dir = Path(
    pkg_resources.resource_filename("jax_unirep", "weights/1900_weights")
)


def aa_seq_to_int(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    # Make sure only valid aa's are passed
    if not set(s).issubset(set(aa_to_int.keys())):
        raise ValueError(
            f"Unsupported character(s) in sequence found:"
            f" {set(s).difference(set(aa_to_int.keys()))}"
        )
    return [24] + [aa_to_int[a] for a in s] + [25]


def load_embedding_1900(name: str = "UniRef50"):
    return np.load(weights_1900_dir / name / "embed_matrix:0.npy")


def get_embedding(sequence: str, embeddings: np.ndarray) -> np.ndarray:
    """Get embeddings for one sequence"""
    if len(sequence) < 1:
        raise SequenceLengthsError("Sequence must be at least of length one.")
    sequence = aa_seq_to_int(sequence)[:-1]
    x = onp.vstack([embeddings[i] for i in sequence])
    return x


def get_embeddings(sequences: List[str]) -> np.ndarray:
    """
    This function takes a list of protein sequences as strings,
    all sequences being of the same length,
    and returns the 10-dimensional embedding of those sequences.
    Input shapes should be (n_sequences, sequence_length),
    output shape is (n_sequences, sequence_length, 10).
    """

    # Defensive programming checks.
    # 1. Make sure list is not empty
    if len(sequences) == 0:
        raise SequenceLengthsError("Cannot pass in empty list of sequences.")
    # 2. Ensure that all sequences are of the same length
    seq_lengths = Counter([len(s) for s in sequences])
    if not len(seq_lengths) == 1:
        error = f"""
Sequences passed in are not all of the same length.
Sequence length: number of sequences information in the dictionary below.
{seq_lengths}
"""
        raise SequenceLengthsError(error)
    embeddings = load_embedding_1900()

    seq_embeddings = [get_embedding(s, embeddings) for s in sequences]
    return onp.stack(seq_embeddings, axis=0)


def load_params_1900(name: str = "UniRef50") -> dict:

    params = dict()
    params["gh"] = np.load(
        weights_1900_dir / name / "rnn_mlstm_mlstm_gh:0.npy"
    )
    params["gmh"] = np.load(
        weights_1900_dir / name / "rnn_mlstm_mlstm_gmh:0.npy"
    )
    params["gmx"] = np.load(
        weights_1900_dir / name / "rnn_mlstm_mlstm_gmx:0.npy"
    )
    params["gx"] = np.load(
        weights_1900_dir / name / "rnn_mlstm_mlstm_gx:0.npy"
    )

    params["wh"] = np.load(
        weights_1900_dir / name / "rnn_mlstm_mlstm_wh:0.npy"
    )
    params["wmh"] = np.load(
        weights_1900_dir / name / "rnn_mlstm_mlstm_wmh:0.npy"
    )
    params["wmx"] = np.load(
        weights_1900_dir / name / "rnn_mlstm_mlstm_wmx:0.npy"
    )
    params["wx"] = np.load(
        weights_1900_dir / name / "rnn_mlstm_mlstm_wx:0.npy"
    )

    params["b"] = np.load(weights_1900_dir / name / "rnn_mlstm_mlstm_b:0.npy")

    return params


def load_embeddings(name: str = "UniRef50"):
    return np.load(weights_1900_dir / name / "embed_matrix:0.npy")


def l2_normalize(arr, axis, epsilon=1e-12):
    """
    L2 normalize along a particular axis.

    Doc taken from tf.nn.l2_normalize:

    https://www.tensorflow.org/api_docs/python/tf/math/l2_normalize

        output = x / (
            sqrt(
                max(
                    sum(x**2),
                    epsilon
                )
            )
        )
    """
    sq_arr = np.power(arr, 2)
    square_sum = np.sum(sq_arr, axis=axis, keepdims=True)
    max_weights = np.maximum(square_sum, epsilon)
    return np.divide(arr, np.sqrt(max_weights))


def batch_sequences(seqs: List[str]) -> List[List]:
    """
    Given a list of strings, returns a list of lists,
    where each sub-list contains the positions of same-length sequences
    in the original list.

    Example:
    ['MTN', 'MT', 'MDN', 'M'] -> [[3], [1], [0, 2]]

    :param seqs: List of sequences as strings.
    :returns: List of lists, where each sub-list contains the positions of
        same-length sequences in the original list.
    """

    # Make sure list is not empty
    if len(seqs) == 0:
        raise SequenceLengthsError("Cannot pass in empty list of sequences.")

    order = []
    for l in set([len(s) for s in seqs]):
        order.append([i for i, s in enumerate(seqs) if len(s) == l])
    return order


# This block of code generates one-hot-encoded arrays.
oh_arrs = np.eye(max(aa_to_int.values()))

# one_hots maps from aa_to_int integers to an array
one_hots = {v: oh_arrs[v - 1] for k, v in aa_to_int.items()}

# oh_idx_to_aa maps from oh_arrs index to aa_to_int letter.
oh_idx_to_aa = {v - 1: k for k, v in aa_to_int.items()}
oh_idx_to_aa[22] = "[XZBJ]"


def boolean_true_idxs(mask: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """
    Return the index where the mask equals the array.

    We expect the ``mask`` to be a 1D array,
    while the ``arr`` to be a 2D array.

    np.where returns a tuple,
    and under the assumptions of this convenience function,
    we only need the first element.
    Hence, the magic number ``[0]`` in the return statement.

    The intended use of this function is to mkae arr_to_letter
    _really fast_.

    :param mask: The 1-D array mask.
    :param arr: The 2-D array on which to check mask equality.
    :returns: A 1-D array of indices where the mask
        equals the array.
    """
    return np.array(np.where(np.all(mask == arr, axis=-1)))[0]


def arr_to_letter(arr) -> str:
    """
    Convert a 1D one-hot array into a letter.

    This is intended to operate on a single array.
    """
    idx = int(boolean_true_idxs(mask=arr, arr=oh_arrs)[0])
    letter = oh_idx_to_aa[idx]
    return letter


def letter_seq(arr: np.array) -> str:
    """
    Convert a 2D one-hot array into a string representation.

    TODO: More docstrings needed.
    """
    sequence = ""
    for letter in arr:
        sequence += arr_to_letter(np.round(letter))
    return sequence.strip("start").strip("stop")
