"""API for evolutionary tuning."""
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as onp
import optuna
from jax import grad, jit
from jax import numpy as np
from jax.experimental.optimizers import adam
from sklearn.model_selection import train_test_split

from .activations import softmax
from .layers import dense, mlstm1900
from .losses import neg_cross_entropy_loss
from .params import add_dense_params
from .utils import (
    aa_seq_to_int,
    batch_sequences,
    get_embeddings,
    load_dense_1900,
    load_embeddings,
    load_params_1900,
    one_hots,
    validate_mlstm1900_params,
)


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


def evotune_step(
    i: int,
    state,
    optimizer_funcs: Tuple[Callable, Callable],
    loss_funcs: Tuple[Callable, Callable],
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Perform one step of evolutionary updating.

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


evotune_loss = partial(neg_cross_entropy_loss, model=predict)
devotune_loss = jit(grad(evotune_loss))
evotune_loss_funcs = (evotune_loss, devotune_loss)


def length_batch_input_outputs(
    sequences: List[str],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Return lists of x and y tensors for evotuning, batched by their length.

    This function exists because we need a way of
    batching sequences by size conveniently.

    :param sequences: A list of sequences to evotune on.
    :returns: Two lists of NumPy arrays, one for xs and the other for ys.
    """
    idxs_batched = batch_sequences(sequences)

    xs = []
    ys = []
    for idxs in idxs_batched:
        seqs = [sequences[i] for i in idxs]
        x, y = input_output_pairs(seqs)
        xs.append(x)
        ys.append(y)
    return xs, ys


def fit(params: Dict, sequences: List[str], n: int) -> Dict:
    """
    Return weights fitted to predict the next letter in each sequence.

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
    xs, ys = length_batch_input_outputs(sequences)

    init, update, get_params = adam(step_size=0.005)
    optimizer_funcs = update, get_params

    state = init(params)
    for i in range(n):
        for x, y in zip(xs, ys):
            state = evotune_step(
                i, state, optimizer_funcs, evotune_loss_funcs, x, y
            )

    return get_params(state)


from sklearn.model_selection import KFold


def objective(
    trial,
    sequences: List[str],
    params: Optional[Dict] = None,
    n_epochs_config: Dict = None,
) -> float:
    """
    Objective function for an Optuna trial.

    The goal with the objective function is
    to automatically find the number of epochs to train
    that minimizes the average of 5-fold test loss.
    Doing so allows us to avoid babysitting the model.

    :param trial: An Optuna trial object.
    :param sequences: A list of strings corresponding to the sequences
        that we want to evotune against.
    :param params: A dictionary of parameters.
        Should have the keys ``mlstm1900`` and ``dense``,
        which correspond to the mLSTM weights and dense layer weights
        (output dimensions = 25)
        to predict the next character in the sequence.
    :returns: Average of 5-fold test loss.
    """
    n_epochs = trial.suggest_discrete_uniform(
        name="n_epochs",
        low=1,
        # high=len(sequences) * 3,
        high=2,
        q=1,
    )
    print(f"Trying out {n_epochs} epochs.")

    kf = KFold(shuffle=True)
    sequences = onp.array(sequences)

    avg_test_losses = []
    for train_index, test_index in kf.split(sequences):
        train_sequences, test_sequences = (
            sequences[train_index],
            sequences[test_index],
        )

        evotuned_params = fit(params, train_sequences, n=int(n_epochs))

        xs, ys = length_batch_input_outputs(test_sequences)

        sum_loss = 0
        for x, y in zip(xs, ys):
            sum_loss += evotune_loss(evotuned_params, x=x, y=y) * len(x)
        avg_test_losses.append(sum_loss / len(test_sequences))

    return sum(avg_test_losses) / len(avg_test_losses)


def evotune(
    sequences: List[str], n_trials: int, params: Optional[Dict] = None
) -> Dict:
    """
    Evolutionarily tune the model to a set of sequences.

    Evotuning is described in the original UniRep and eUniRep papers.
    This reimplementation of evotune provides a nicer API
    that automatically handles multiple sequences of variable lengths.

    Evotuning always needs a starter set of weights.
    By default, the pre-trained weights from the Nature Methods paper are used.
    However, other pre-trained weights are legitimate.

    We first use optuna to figure out how many epochs to fit
    before overfitting happens.
    To save on computation time, the number of trials run
    defaults to 20, but can be configured.
    """
    if params is None:
        params = dict()
        params["dense"] = load_dense_1900()
        params["mlstm1900"] = load_params_1900()

    # Check that params have correct keys and shapes
    validate_mlstm1900_params(params["mlstm1900"])

    study = optuna.create_study()

    objective_func = lambda x: objective(x, params=params, sequences=sequences)
    study.optimize(objective_func, n_trials=n_trials)
    num_epochs = int(study.best_params["n_epochs"])

    evotuned_params = fit(params, sequences=sequences, n=num_epochs)
    return evotuned_params
