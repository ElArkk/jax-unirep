"""Utility functions for jax-unirep."""
import logging
import os
import pickle as pkl
import warnings
from collections import Counter
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from random import sample
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import jax.numpy as np
import numpy as onp
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
            str(
                files("jax_unirep")
                / f"weights/uniref50/{paper_weights}_weights"
            )
        )


MLSTM_PARAM_KEYS = ("b", "gh", "gmh", "gmx", "gx", "wh", "wmh", "wmx", "wx")

WEIGHTS_NPZ = "model_weights.npz"
WEIGHTS_PKL = "model_weights.pkl"


def params_to_arrays(params) -> Dict[str, onp.ndarray]:
    """Flatten a parameter tree into semantically named arrays.

    Keys name what each array *is*, rather than where it sits in a stax
    tuple::

        embedding
        mlstm.0.wx   mlstm.0.wh   ...
        mlstm.1.wx   ...
        dense.w      dense.b

    Depth is the number of distinct `mlstm.N` prefixes and width is
    `mlstm.0.wmh.shape[0]`, so the file stays self-describing without
    encoding any particular framework's container layout.

    :param params: A parameter tree from `load_params` or `fit`.
    :returns: A flat dict of arrays, suitable for `np.savez`.
    """
    arrays = {"embedding": onp.asarray(params[0])}

    layer = 0
    for element in params[1:]:
        if isinstance(element, dict):
            for key in sorted(element):
                arrays[f"mlstm.{layer}.{key}"] = onp.asarray(element[key])
            layer += 1
        elif isinstance(element, (tuple, list)) and len(element) == 2:
            arrays["dense.w"] = onp.asarray(element[0])
            arrays["dense.b"] = onp.asarray(element[1])

    return arrays


def arrays_to_params(arrays: Dict[str, onp.ndarray]) -> Tuple:
    """Rebuild a stax parameter tree from named arrays.

    Inverse of `params_to_arrays`. The tree interleaves an empty tuple after
    every mLSTM layer, because stax gives the parameterless
    `mLSTMHiddenStates` and `Softmax` layers empty params.

    :param arrays: A flat dict of arrays, as read from a `.npz` file.
    :returns: A parameter tree matching the evotuning stax models.
    """
    n_layers = 1 + max(
        int(key.split(".")[1]) for key in arrays if key.startswith("mlstm.")
    )

    tree = [arrays["embedding"]]
    for i in range(n_layers):
        tree.append(
            {key: arrays[f"mlstm.{i}.{key}"] for key in MLSTM_PARAM_KEYS}
        )
        tree.append(())

    if "dense.w" in arrays:
        tree.append((arrays["dense.w"], arrays["dense.b"]))
        tree.append(())

    return tuple(tree)


def dump_params(
    params: Dict,
    dir_path: Path = Path("temp"),
    step: Optional[int] = 0,
):
    """
    Dump the current params of model being trained to a `.npz` file.

    Note: versions up to 2.x pickled the parameter tree. Pickle executes
    arbitrary code on load, which is why the old docs told you to verify an
    MD5 before loading. `.npz` with `allow_pickle=False` is pure data, so
    that caveat goes away. `load_params` still reads the old `.pkl` files,
    so weights dumped by earlier versions keep working.

    Arrays are stored under semantic names (`embedding`, `mlstm.0.wx`,
    `dense.w`, ...) rather than positional indices, so the file does not
    encode any particular framework's container layout.

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

    onp.savez(iteration_path / WEIGHTS_NPZ, **params_to_arrays(params))


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

    Note that each pre-trained model carries its own embedding matrix, so the
    1900 default is *only* correct for the 1900 model.

    :param folderpath: Path to the folder containing the model weights
    :param paper_weights: If paper weights should be loaded (folderpath set to None),
        specify from which model architecture. Possible values are `1900`, `256` and `64`.
        Defaults to 1900 weights.
    """
    weights_dir = get_weights_dir(
        folderpath=folderpath, paper_weights=paper_weights
    )
    npz_path = weights_dir / WEIGHTS_NPZ
    if npz_path.exists():
        # Reads one array rather than the whole 73 MB tree.
        with onp.load(npz_path, allow_pickle=False) as arrays:
            return arrays["embedding"]

    with open(weights_dir / WEIGHTS_PKL, "rb") as f:
        params = pkl.load(f)
    return params[0]


def get_embedding(sequence: str, embeddings: np.ndarray) -> np.ndarray:
    """Get embeddings for one sequence."""
    if len(sequence) < 1:
        raise SequenceLengthsError("Sequence must be at least of length one.")
    sequence = aa_seq_to_int(sequence)[:-1]
    x = onp.vstack([embeddings[i] for i in sequence])
    return x


def get_embeddings(
    sequences: Iterable[str], embedding: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Return embedding of a list of sequences.

    This function takes a list of protein sequences as strings,
    all sequences being of the same length,
    and returns the 10-dimensional embedding of those sequences.
    Input shapes should be (n_sequences, sequence_length),
    output shape is (n_sequences, sequence_length, 10).

    :param sequences: A list of sequences to obtain embeddings for.
    :param embedding: The amino acid embedding matrix to use, of shape
        (26, 10). Each pre-trained model carries its own, stored at
        `load_params(paper_weights=size)[0]`. Defaults to the 1900 model's
        embedding, which is *only* correct for the 1900 model.
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
    if embedding is None:
        embedding = load_embedding()

    seq_embeddings = [get_embedding(s, embedding) for s in sequences]
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

    The weights are saved as a single `.npz` file of named arrays, read with
    `allow_pickle=False`. When loaded into memory, the weights object `params`
    will be a nested tuple of arrays and dictionaries. In order, they are:

    - embedding params
    - mLSTM params, one dict per layer (with gating weights `g*`, matrix
      multiplication weights `w*`, and bias `b` as keys), each followed by an
      empty tuple for the parameterless hidden-state layer
    - dense params to predict one-hot encoded next letter.

    Weights dumped by version 2.x and earlier are Python pickles. Those are
    still read, so previously saved weights keep working, but note that
    unpickling executes arbitrary code and is only as trustworthy as the file
    itself. Re-dumping with `dump_params` converts to the safe format.

    :param folderpath: Path to the folder containing the model weights
    :param paper_weights: If paper weights should be loaded (folderpath set to None),
        specify from which model architecture. Possible values are `1900`, `256` and `64`.
        Defaults to 1900 weights.
    """
    weights_dir = get_weights_dir(
        folderpath=folderpath, paper_weights=paper_weights
    )

    npz_path = weights_dir / WEIGHTS_NPZ
    if npz_path.exists():
        with onp.load(npz_path, allow_pickle=False) as arrays:
            return arrays_to_params({k: arrays[k] for k in arrays.files})

    pkl_path = weights_dir / WEIGHTS_PKL
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"No model weights found in {weights_dir}. Expected "
            f"{WEIGHTS_NPZ}, or {WEIGHTS_PKL} if these were dumped by "
            f"jax-unirep 2.x or earlier."
        )

    warnings.warn(
        f"Loading legacy pickled weights from {pkl_path}. Unpickling "
        f"executes arbitrary code, so only load files you trust. Re-dump "
        f"them with dump_params to convert to {WEIGHTS_NPZ}.",
        FutureWarning,
        stacklevel=2,
    )
    with open(pkl_path, "rb") as f:
        return pkl.load(f)


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
