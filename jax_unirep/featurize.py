from typing import Iterable, Optional, Tuple, Union

import numpy as onp
from jax import vmap

from .errors import SequenceLengthsError
from .models import MLSTM, load_model
from .utils import batch_sequences, seq_to_oh


def rep_same_lengths(
    seqs: Iterable[str], model: MLSTM
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    Generate reps for protein sequences that all have the same length.

    :param seqs: A list of same length sequences as strings.
        If passing only a single sequence, it also needs to be passed
        inside a list.
    :param model: The `MLSTM` to featurize with.
    :returns: A 3-tuple of `np.array`s containing the reps,
        in the order `h_avg`, `h_final`, and `c_final`.
        Each `np.array` has shape (n_sequences, model.output_dim).
    """
    # The model consumes one-hots and does its own embedding lookup.
    # Sequences carry a start token but no stop token at inference time.
    one_hot = onp.stack([seq_to_oh(s)[:-1] for s in seqs])

    h_final, c_final, h = vmap(model)(one_hot)

    # Converting to `np.array` blocks until the computation has completed.
    return (
        onp.asarray(h.mean(axis=1)),
        onp.asarray(h_final),
        onp.asarray(c_final),
    )


def rep_arbitrary_lengths(
    seqs: Iterable[str], model: MLSTM
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    Generate reps for protein sequences of arbitrary length.

    All sequences of the same length are batched together and passed through
    the mLSTM in one go, which is much faster than the original published
    workflow of repping one sequence at a time. Batching by exact length
    rather than padding to a common length keeps padding characters from
    adding noise to the reps.

    :param seqs: A list of sequences as strings.
        If passing only a single sequence, it also needs to be passed
        inside a list.
    :param model: The `MLSTM` to featurize with.
    :returns: A 3-tuple of `np.array`s containing the reps,
        in the order `h_avg`, `h_final`, and `c_final`.
        Each `np.array` has shape (n_sequences, model.output_dim).
    """
    order = batch_sequences(seqs)
    reps = [rep_same_lengths([seqs[i] for i in idxs], model) for idxs in order]

    # `order` lists the original positions of the sequences in the order they
    # were repped, so argsorting it restores the caller's order.
    restore = onp.argsort(onp.concatenate(order))
    return tuple(onp.concatenate(rep)[restore] for rep in zip(*reps))


def get_reps(
    seqs: Union[str, Iterable[str]],
    model: Optional[MLSTM] = None,
    mlstm_size: int = 1900,
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
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
    In that case, call the `MLSTM` directly,
    so that the JAX arrays it returns
    can be passed into the next step
    instead of being converted to `np.array`s.

    All three published model sizes are supported. The model knows its own
    depth and width, so nothing needs to be declared about its architecture:
    the 1900 model has one mLSTM cell and the 256 and 64 models have four.

    :param seqs: A list of sequences as strings, or a single string.
    :param model: The `MLSTM` to featurize with, as returned by `load_model()`
        or `fit()`. When given, its own width is used and `mlstm_size` is
        ignored.
    :param mlstm_size: Which set of pre-trained weights to load when `model`
        is None. One of 1900, 256 or 64.
    :returns: A 3-tuple of `np.array`s containing the reps,
        in the order `h_avg`, `h_final`, and `c_final`.
        Each `np.array` has shape (n_sequences, mlstm_size).
    """
    # If single string sequence is passed, package it into a list
    if isinstance(seqs, str):
        seqs = [seqs]
    # Check before loading 73 MB of weights on the caller's behalf.
    if len(seqs) == 0:
        raise SequenceLengthsError("Cannot pass in empty list of sequences.")

    if model is None:
        model = load_model(paper_weights=mlstm_size)

    return rep_arbitrary_lengths(seqs, model)


def fusion_reps(
    seqs: Union[str, Iterable[str]],
    model: Optional[MLSTM] = None,
    mlstm_size: int = 1900,
) -> onp.ndarray:
    """
    Get "UniRep Fusion" representations of proteins.

    Alley et al. 2019 define UniRep Fusion as the concatenation of all three
    representations -- average hidden, final hidden and final cell state --
    into a single vector, and use it for the supervised stability and
    quantitative function prediction tasks. For the 1900 model that is 5700
    dimensions.

    This is a one-line composition of `get_reps`, and exists because the
    concatenation is a named quantity from the paper rather than an obvious
    thing to guess:

    ```python
    h_avg, h_final, c_final = get_reps(seqs)
    fusion = np.hstack([h_avg, h_final, c_final])
    ```

    If you are fine-tuning rather than featurizing, do not reach for this.
    Concatenate inside your own `equinox.Module` instead, so the gradient
    reaches the mLSTM -- see the "End-to-end differentiable models" section of
    the docs.

    :param seqs: A list of sequences as strings, or a single string.
    :param model: The `MLSTM` to featurize with, as returned by `load_model()`
        or `fit()`. When given, its own width is used and `mlstm_size` is
        ignored.
    :param mlstm_size: Which set of pre-trained weights to load when `model`
        is None. One of 1900, 256 or 64.
    :returns: An `np.array` of shape (n_sequences, 3 * mlstm_size), the
        components in the order `h_avg`, `h_final`, `c_final`.
    """
    return onp.hstack(get_reps(seqs, model=model, mlstm_size=mlstm_size))
