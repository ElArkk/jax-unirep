from collections import Counter
from typing import List

import numpy as np
from pyprojroot import here

from errors import SequenceLengthsError

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


def aa_seq_to_int(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return [24] + [aa_to_int[a] for a in s] + [25]


def get_embedding(sequence: str, embeddings: np.ndarray) -> np.ndarray:
    """Get embeddings for one sequence"""
    if len(sequence) < 1:
        raise SequenceLengthsError("Sequence must be at least of length one.")
    sequence = aa_seq_to_int(sequence)[:-1]
    x = np.vstack([embeddings[i] for i in sequence])
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
    embeddings = np.load(here("./weights/1900_weights/embed_matrix:0.npy"))

    seq_embeddings = [get_embedding(s, embeddings) for s in sequences]
    return np.stack(seq_embeddings, axis=0)


def load_params_1900() -> dict:

    params = dict()
    params["gh"] = np.load(
        here("./weights/1900_weights/rnn_mlstm_mlstm_gh:0.npy")
    )
    params["gmh"] = np.load(
        here("./weights/1900_weights/rnn_mlstm_mlstm_gmh:0.npy")
    )
    params["gmx"] = np.load(
        here("./weights/1900_weights/rnn_mlstm_mlstm_gmx:0.npy")
    )
    params["gx"] = np.load(
        here("./weights/1900_weights/rnn_mlstm_mlstm_gx:0.npy")
    )

    params["wh"] = np.load(
        here("./weights/1900_weights/rnn_mlstm_mlstm_wh:0.npy")
    )
    params["wmh"] = np.load(
        here("./weights/1900_weights/rnn_mlstm_mlstm_wmh:0.npy")
    )
    params["wmx"] = np.load(
        here("./weights/1900_weights/rnn_mlstm_mlstm_wmx:0.npy")
    )
    params["wx"] = np.load(
        here("./weights/1900_weights/rnn_mlstm_mlstm_wx:0.npy")
    )

    params["b"] = np.load(
        here("./weights/1900_weights/rnn_mlstm_mlstm_b:0.npy")
    )

    return params


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
