import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from random import choice, sample
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import jax.numpy as np
import numpy as onp
import pkg_resources
from jax.nn.initializers import glorot_normal
from jax.random import PRNGKey
from jax.tree_util import tree_map
from tqdm.autonotebook import tqdm

from .errors import SequenceLengthsError

"""jax-unirep utils."""


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


def get_weights_dir(folderpath: Optional[str] = None):
    """
    Fetch the paper weights per default, or from a specified folderpath
    """
    if folderpath:
        return Path(folderpath)
    else:
        return Path(
            pkg_resources.resource_filename(
                "jax_unirep", "weights/1900_weights/uniref50"
            )
        )


def dump_params(
    params: Dict,
    dir_path: Optional[str] = "temp",
    step: Optional[int] = 0,
):
    """
    Dumps the current params of model being trained to a .npy file.

    The directory is specified by dir_path,
    and will be created, if it does not exist yet.

    The weights that will be dumped are the mLSTM weights as well
    as the dense layer weights. The embedding matrix weights are
    not dumped, as they never get modified.
    The weights will have the same naming convention as the original:
        dir_path/iter_x/fully_connected_biases:0.npy
        dir_path/iter_x/fully_connected_weights:0.npy
        dir_path/iter_x/rnn_mlstm_mlstm_b:0.npy
        dir_path/iter_x/rnn_mlstm_mlstm_gh:0.npy
        dir_path/iter_x/rnn_mlstm_mlstm_gmh:0.npy
        dir_path/iter_x/rnn_mlstm_mlstm_gmx:0.npy
        dir_path/iter_x/rnn_mlstm_mlstm_gx:0.npy
        dir_path/iter_x/rnn_mlstm_mlstm_wh:0.npy
        dir_path/iter_x/rnn_mlstm_mlstm_wmh:0.npy
        dir_path/iter_x/rnn_mlstm_mlstm_wmx:0.npy
        dir_path/iter_x/rnn_mlstm_mlstm_wx:0.npy

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

    # iterate through and save mlstm params as npy files.
    for name, val in params[0].items():
        # Construct filename
        fname = f"rnn_mlstm_mlstm_{name}:0.npy"

        # Construct directory for dumping.
        iteration_path = Path(dir_path) / f"iter_{step}"
        iteration_path.mkdir(exist_ok=True)

        # Save file
        fpath = iteration_path / fname
        onp.save(
            fpath,
            onp.array(val),
        )
    # iterate through and save dense params as npy files.
    dense_names = [
        "fully_connected_weights:0.npy",
        "fully_connected_biases:0.npy",
    ]
    for i, val in enumerate(params[2]):
        # Construct directory for dumping.
        iteration_path = Path(dir_path) / f"iter_{step}"
        iteration_path.mkdir(exist_ok=True)

        # Save file
        fpath = iteration_path / dense_names[i]
        onp.save(
            fpath,
            onp.array(val),
        )


def aa_seq_to_int(s: str) -> List[int]:
    """Return the int sequence as a list for a given string of amino acids."""
    # Make sure only valid aa's are passed
    if not set(s).issubset(set(aa_to_int.keys())):
        raise ValueError(
            f"Unsupported character(s) in sequence found:"
            f" {set(s).difference(set(aa_to_int.keys()))}"
        )
    return [24] + [aa_to_int[a] for a in s] + [25]


@lru_cache(maxsize=128)
def load_embedding_1900(folderpath: Optional[str] = None):
    """Load pre-trained embedding weights for uniref50 model."""
    weights_1900_dir = get_weights_dir(folderpath=folderpath)

    return np.load(weights_1900_dir / "embed_matrix:0.npy")


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
    embeddings = load_embedding_1900()

    seq_embeddings = [get_embedding(s, embeddings) for s in sequences]
    return onp.stack(seq_embeddings, axis=0)


def load_dense_1900(folderpath: Optional[str] = None) -> Tuple:
    """
    Load pre-trained dense layer weights from the UniRep paper.

    The dense layer weights are used to predict next character
    from the output of the mLSTM1900.
    """
    weights_1900_dir = get_weights_dir(folderpath=folderpath)

    w = np.load(weights_1900_dir / "fully_connected_weights:0.npy")
    b = np.load(weights_1900_dir / "fully_connected_biases:0.npy")
    return w, b


def load_params_1900(folderpath: Optional[str] = None) -> Dict:
    """Load pre-trained mLSTM1900 weights from the UniRep paper."""
    weights_1900_dir = get_weights_dir(folderpath=folderpath)

    params = dict()
    params["gh"] = np.load(weights_1900_dir / "rnn_mlstm_mlstm_gh:0.npy")
    params["gmh"] = np.load(weights_1900_dir / "rnn_mlstm_mlstm_gmh:0.npy")
    params["gmx"] = np.load(weights_1900_dir / "rnn_mlstm_mlstm_gmx:0.npy")
    params["gx"] = np.load(weights_1900_dir / "rnn_mlstm_mlstm_gx:0.npy")

    params["wh"] = np.load(weights_1900_dir / "rnn_mlstm_mlstm_wh:0.npy")
    params["wmh"] = np.load(weights_1900_dir / "rnn_mlstm_mlstm_wmh:0.npy")
    params["wmx"] = np.load(weights_1900_dir / "rnn_mlstm_mlstm_wmx:0.npy")
    params["wx"] = np.load(weights_1900_dir / "rnn_mlstm_mlstm_wx:0.npy")

    params["b"] = np.load(weights_1900_dir / "rnn_mlstm_mlstm_b:0.npy")

    return params


def validate_mLSTM1900_params(params: Dict):
    """
    Validate shapes of mLSTM1900 parameter dictionary.

    Check that mLSTM1900 params dictionary contains the correct set of keys
    and that the shapes of the params are correct.

    :param params: A dictionary of mLSTM1900 weights.
    """
    expected = {
        "gh": (7600,),
        "gmh": (1900,),
        "gmx": (1900,),
        "gx": (7600,),
        "wh": (1900, 7600),
        "wmh": (1900, 1900),
        "wmx": (10, 1900),
        "wx": (10, 7600),
        "b": (7600,),
    }

    for key, value in params.items():
        if value.shape != expected[key]:
            raise ValueError(
                f"Param {key} does not have the right shape. Expected: {expected[key]}, got: {value.shape} instead."
            )


def load_params(folderpath: Optional[str] = None):
    """load params for passing to evotuning stax model"""
    return (
        load_params_1900(folderpath=folderpath),
        (),
        load_dense_1900(folderpath=folderpath),
        (),
    )


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


def right_pad(seqs: Iterable[str], max_len: int):
    """Pad all seqs in a list to longest length on the right with "-"."""
    return [
        seq.ljust(max_len, "-")
        for seq in tqdm(seqs, desc="right-padding sequences")
    ]


def get_batching_func(
    xs: np.ndarray, ys: np.ndarray, batch_size: int = 25
) -> Callable:
    """
    Create a function which returns batches of sequences

    :param xs: array of embedded same-length sequences
    :param ys: array of one-hot encoded groud truth next-AA labels
    """

    def batching_func():
        pairs = list(zip(xs, ys))
        if len(pairs) > batch_size:
            pairs = sample(pairs, batch_size)
        x, y = zip(*pairs)
        return onp.stack(x), onp.stack(y)

    return batching_func


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


def random_like(param):
    key = PRNGKey(39)
    return glorot_normal(key, param.shape)


def load_random_evotuning_params():
    params_1900 = load_params_1900()
    random_params_1900 = tree_map(random_like, params_1900)
    params_dense = load_dense_1900()
    random_dense_1900 = tree_map(random_like, params_dense)
    return (params_1900, (), params_dense, ())
