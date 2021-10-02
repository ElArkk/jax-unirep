"""Utility functions for jax-unirep."""
import logging
import os
import pickle as pkl
from collections import Counter
from functools import lru_cache
from pathlib import Path
from random import sample
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import jax.numpy as np
import numpy as onp
import pkg_resources
from tqdm.autonotebook import tqdm

from .errors import SequenceLengthsError

aa_to_int = {
    "-": 0,
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
proposal_valid_letters = "ACDEFGHIKLMNPQRSTVWY"


def get_weights_dir(
    folderpath: Optional[str] = None, paper_weights: Optional[int] = 1900
):
    """
    Fetch model weights.

    If `folderpath` and `paper_weights` is None, retrieve the mLSTM1900 weights.

    :param folderpath: Path to the folder containing the model weights
    :param paper_weights: If paper weights should be loaded (folderpath set to None),
        specify from which model architecture. Possible values are 1900, 256 and 64.
        Defaults to 1900 weights.
    """
    if folderpath:
        return Path(folderpath)
    else:
        return Path(
            pkg_resources.resource_filename(
                "jax_unirep", f"weights/uniref50/{paper_weights}_weights"
            )
        )


def dump_params(
    params: Dict,
    dir_path: Path = Path("temp"),
    step: Optional[int] = 0,
):
    """
    Dump the current params of model being trained to a .pkl file.

    Note: We used to dump to a `.npy` file for each weight.
    This was tied to a previously strong assumption
    that the weights were from an mLSTM1900 model.

    With the change from a single model architecture assumption
    to one that allows for more flexibility,
    we can no longer assume that the weights match up.
    Hence, we now simply do a Python pickle dump instead.
    As with before,
    the embedding weights are not dumped.

    The directory is specified by dir_path,
    and will be created, if it does not exist yet.

    `dir_path`, by convention, should be relative to
    the current working directory
    in which you executed your Python script or Jupyter notebook.

    :param params: the parameters at the current state of training,
        input as a tuple of dicts.
    :param step: the number of training steps to get to this state.
    :param dir_name: path of directory params will save to.
    """
    # create directory if it doesn't already exist:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"created directory at {dir_path}")

    iteration_path = Path(dir_path) / f"iter_{step}"
    iteration_path.mkdir(exist_ok=True)

    with open(iteration_path / "model_weights.pkl", "wb") as f:
        pkl.dump(params, f)


def aa_seq_to_int(s: str) -> List[int]:
    """Return the int sequence as a list for a given string of amino acids."""
    # Make sure only valid aa's are passed
    if not set(s).issubset(set(aa_to_int.keys())):
        raise ValueError(
            f"Unsupported character(s) in sequence found:"
            f" {set(s).difference(set(aa_to_int.keys()))}"
        )
    return [24] + [aa_to_int[a] for a in s] + [25]


def load_embedding(
    folderpath: Optional[str] = None, paper_weights: Optional[int] = 1900
):
    """
    Load pre-trained embedding weights for UniRep paper models.

    :param folderpath: Path to the folder containing the model weights
    :param paper_weights: If paper weights should be loaded (folderpath set to None),
        specify from which model architecture. Possible values are `1900`, `256` and `64`.
        Defaults to 1900 weights.
    """
    weights_dir = get_weights_dir(
        folderpath=folderpath, paper_weights=paper_weights
    )
    with open(weights_dir / "model_weights.pkl", "rb") as f:
        params = pkl.load(f)
    return params[0]


def get_embedding(sequence: str, embeddings: np.ndarray) -> np.ndarray:
    """Get embeddings for one sequence."""
    if len(sequence) < 1:
        raise SequenceLengthsError("Sequence must be at least of length one.")
    sequence = aa_seq_to_int(sequence)[:-1]
    x = onp.vstack([embeddings[i] for i in sequence])
    return x


def get_embeddings(sequences: Iterable[str]) -> np.ndarray:
    """
    Return embedding of a list of sequences.

    This function takes a list of protein sequences as strings,
    all sequences being of the same length,
    and returns the 10-dimensional embedding of those sequences.
    Input shapes should be (n_sequences, sequence_length),
    output shape is (n_sequences, sequence_length, 10).

    :param sequences: A list of sequences to obtain embeddings for.
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
    embeddings = load_embedding()

    seq_embeddings = [get_embedding(s, embeddings) for s in sequences]
    return onp.stack(seq_embeddings, axis=0)


def validate_mLSTM_params(params: Dict, n_outputs):
    """
    Validate shapes of mLSTM parameter dictionary.

    Check that mLSTM params dictionary contains the correct set of keys
    and that the shapes of the params are correct.

    :param params: A dictionary of mLSTM weights.
    """
    expected = {
        "gh": (n_outputs * 4,),
        "gmh": (n_outputs,),
        "gmx": (n_outputs,),
        "gx": (n_outputs * 4,),
        "wh": (n_outputs, n_outputs * 4),
        "wmh": (n_outputs, n_outputs),
        "wmx": (10, n_outputs),
        "wx": (10, n_outputs * 4),
        "b": (n_outputs * 4,),
    }

    for key, value in params.items():
        if hasattr(value, "shape") and value.shape != expected[key]:
            raise ValueError(
                f"Param {key} does not have the right shape. Expected: {expected[key]}, got: {value.shape} instead."
            )


def load_params(
    folderpath: Optional[str] = None, paper_weights: Optional[int] = 1900
):
    """
    Load params for passing to evotuning stax model.

    The weights are saved as a single pickle file.
    We did this in version 1.1 to unify how weights are stored and dumped.
    When loaded into memory, the weights object `params`
    will be a nested tuple of arrays and dictionaries. In order, they are:

    - embedding params
    - mLSMT1900 params (with gating weights `g*`, matrix multiplication weights `w*`, and bias `b` as keys)
    - dense params to predict one-hot encoded next letter.

    Loading a Pickle file can pose a security issue,
    so if you wish to verify the MD5 of the pickles before loading them,
    you can do so using the following block of code:

    ```python
    from jax_unirep.utils import get_weights_dir

    weights_dir = get_weights_dir(folderpath=None)
    weights_path = weights_dir / "model_weights.pkl"
    # shell out to the system by calling on md5.
    os.system(f"md5 {str(weights_path)}")
    ```

    The return should be identical to the following:

        MD5 (model_weights.pkl) = 87c89ab62929485e43474c8b24cda5c8

    :param folderpath: Path to the folder containing the model weights
    :param paper_weights: If paper weights should be loaded (folderpath set to None),
        specify from which model architecture. Possible values are `1900`, `256` and `64`.
        Defaults to 1900 weights.
    """
    weights_dir = get_weights_dir(
        folderpath=folderpath, paper_weights=paper_weights
    )
    with open(weights_dir / "model_weights.pkl", "rb") as f:
        params = pkl.load(f)
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


def batch_sequences(seqs: Iterable[str]) -> List[List]:
    """
    Batch up sequences according to size.

    Given a list of strings, returns a list of lists,
    where each sub-list contains the positions of same-length sequences
    in the original list.

    For example:

    ```
    ['MTN', 'MT', 'MDN', 'M'] -> [[3], [1], [0, 2]]
    ```

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


def right_pad(seqs: Iterable[str], max_len: int):
    """Pad all seqs in a list to longest length on the right with "-"."""
    return [
        seq.ljust(max_len, "-")
        for seq in tqdm(seqs, desc="right-padding sequences")
    ]


def get_batching_func(seq_batch, batch_size: int = 25) -> Callable:
    """
    Create a function which returns batches of embedded sequences.

    :param xs: array of embedded same-length sequences
    :param ys: array of one-hot encoded groud truth next-AA labels
    """

    def batching_func():
        seqs = seq_batch
        if len(seqs) > batch_size:
            seqs = sample(seqs, batch_size)
        xs, ys = input_output_pairs(seqs)
        return xs, ys

    return batching_func


# This block of code generates one-hot-encoded arrays.
oh_arrs = np.eye(max(aa_to_int.values()) + 1)

# one_hots maps from aa_to_int integers to an array
one_hots = {v: oh_arrs[v] for k, v in aa_to_int.items()}

# oh_idx_to_aa maps from oh_arrs index to aa_to_int letter.
oh_idx_to_aa = {v: k for k, v in aa_to_int.items()}
oh_idx_to_aa[22] = "[XZBJ]"


def seq_to_oh(seq: str):
    """
    One-hot encode a single AA sequence
    """
    seq_int = aa_seq_to_int(seq)
    return onp.vstack([one_hots[i] for i in seq_int])


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
    in `aa_to_int`,
    and k-1 corresponds to the first a.a. to the nth a.a.
    This is the second element in the returned tuple.

    ### Parameters

    - `s`: The protein sequence to featurize.

    ### Returns

    Two 2D NumPy arrays,
    the first corresponding to
    the input to evotuning with shape (n_letters, 10),
    and the second corresponding to
    the output amino acid to predict with shape (n_letters, 25).
    """
    seq_int = aa_seq_to_int(s[:-1])
    next_letters_int = aa_seq_to_int(s[1:])

    x = onp.vstack([one_hots[i] for i in seq_int])
    # We delete the 24th one-hot position in the y vector,
    # since we never need to predict the "start" token.
    y = onp.vstack([onp.delete(one_hots[i], 24) for i in next_letters_int])
    return x, y


def input_output_pairs(
    sequences: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate input-output tensor pairs for evo-tuning.
    We check that lengths of sequences are identical,
    as this is necessary to ensure stacking of tensors happens correctly.
    :param sequences: A list of sequences
        to generate input-output tensor pairs.
    :returns: Two NumPy arrays,
        the first corresponding to the input to evotuning
        with shape (n_sequences, n_letters+1, 10),
        and the second corresponding to the output amino acids to predict
        with shape (n_sequences, n_letters+1, 25).
        Both will have an additional "sample" dimension as the first dim.
    """
    seqlengths = set(map(len, sequences))
    logging.debug(seqlengths)
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
    return onp.stack(xs), onp.stack(ys)


def length_batch_input_outputs(
    sequences: Iterable[str],
) -> Tuple[List[List[str]], List[int]]:
    """
    Return sequences, batched by their length, plus a list of unique lengths.

    This function exists because we need a way of
    batching sequences by size conveniently.

    :param sequences: A list of sequences to evotune on.
    :returns: Two lists, sequences and lengths.
    """
    idxs_batched = batch_sequences(sequences)

    seqs_batched = []
    lens = []
    for idxs in tqdm(idxs_batched):
        seqs = [sequences[i] for i in idxs]
        seqs_batched.append(seqs)
        lens.append(len(seqs[0]))
    return seqs_batched, lens
