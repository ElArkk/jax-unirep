"""API for evolutionary tuning."""
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as onp
import optuna
from jax import numpy as np
from jax import grad, jit, lax, random, vmap
from jax.experimental import stax
from jax.experimental.optimizers import adam
from jax.experimental.stax import Dense, Softmax
from jax.nn import softmax
from jax_unirep.losses import _neg_cross_entropy_loss
from sklearn.model_selection import KFold, train_test_split

from .layers import mLSTM1900, mLSTM1900_AvgHidden, mLSTM1900_HiddenStates
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
    validate_mLSTM1900_params,
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
        with shape (n_sequences, n_letters+1, 10),
        and the second corresponding to the output amino acids to predict
        with shape (n_sequences, n_letters+1, 25).
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


# HERE LIES THE DRAG.. MODEL!
init_fun, predict = stax.serial(
    mLSTM1900(), mLSTM1900_HiddenStates(), stax.Dense(25), stax.Softmax
)


@jit
def evotune_loss(params, inputs, targets):
    predictions = vmap(partial(predict, params))(inputs)
    return _neg_cross_entropy_loss(targets, predictions)


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


def fit(
    params: Dict, sequences: List[str], n: int, step_size: float = 0.001
) -> Dict:
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

    init, update, get_params = adam(step_size=step_size)
    # optimizer_funcs = jit(update), jit(get_params)

    @jit
    def step(i, state):
        """
        Perform one step of evolutionary updating.

        This function is closed inside `fit` because we need access
        to the variables in its scope,
        particularly the update and get_params functions.

        By structuring the function this way, we can JIT-compile it,
        and thus gain a massive speed-up!

        :param i: The current iteration of the training loop.
        :param state: Current state of parameters from jax.
        """
        params = get_params(state)
        g = grad(evotune_loss)(params, x, y)
        state = update(i, g, state)
        return state

    state = init(params)
    from time import time

    for i in range(n):
        for x, y in zip(xs, ys):
            state = step(i, state)
            params = get_params(state)
    return get_params(state)


def avg_loss(sequences, params):
    """
    Return average loss of a set of parameters,
    on a set of sequences.

    :param sequences: sequences (in standard AA format)
    :param params: parameters (i.e. from training)
    """
    xs, ys = length_batch_input_outputs(sequences)

    sum_loss = 0
    for x, y in zip(xs, ys):
        sum_loss += evotune_loss(params, inputs=x, targets=y) * len(x)

    return sum_loss / len(sequences)


def fit_manual(
    params: Dict,
    train_seqs: List[str],
    in_val_seqs: List[str],
    out_val_seqs: List[str],
    n: int,
    step_size: float = 0.001,
    steps_per_print: int = 200,
) -> Dict:
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
    :param train_seqs: The set of training sequences to tune to.
    :param in_val_seqs: In-domain validation set of sequences,
        to check for loss on to prevent overfitting.
    :param out_val_seqs: Out-domain validation set of sequences,
        to check for loss on to prevent overfitting.
    :param n: The number of iterations to evotune on.
    :param step_size: Optimizer learning rate. 
    :param steps_per_print: Iteration spacing between
        printing current params and loss. 

    """
    xs, ys = length_batch_input_outputs(train_seqs)

    init, update, get_params = adam(step_size=step_size)
    # optimizer_funcs = jit(update), jit(get_params)

    @jit
    def step(i, state):
        """
        Perform one step of evolutionary updating.

        This function is closed inside `fit` because we need access
        to the variables in its scope,
        particularly the update and get_params functions.

        By structuring the function this way, we can JIT-compile it,
        and thus gain a massive speed-up!

        :param i: The current iteration of the training loop.
        :param state: Current state of parameters from jax.
        """
        params = get_params(state)
        g = grad(evotune_loss)(params, x, y)
        state = update(i, g, state)
        return state

    state = init(params)
    from time import time

    for i in range(n):
        for x, y in zip(xs, ys):
            state = step(i, state)
            params = get_params(state)

        if (i + 1) % steps_per_print == 0:

            # calculate loss for in & out domain validation sets
            # then print it.
            # TODO: print to .log file??

            print(
                "Iteration {0}: in-val-loss={1}, out-val-loss:{2}".format(
                    i,
                    avg_loss(in_val_seqs, params),
                    avg_loss(out_val_seqs, params),
                )
            )

            # TODO: dump current params in case run crashes or loss increases.

    return get_params(state)


# def evotune_step(
#     i: int,
#     state,
#     optimizer_funcs: Tuple[Callable, Callable],
#     loss_func: Callable,
#     x: np.ndarray,
#     y: np.ndarray,
# ):
#     """
#     Perform one step of evolutionary updating.

#     ;param i: The current iteration of the training loop.
#     :param state: Current state of parameters from jax.
#     :param optimizer_funcs: The (update, get_params) functions
#         from jax's optimizers.
#     :param loss_func: The loss function.
#     :return state: Updated state of parameters from jax.
#     """
#     # Unpack optimizer funcs
#     update, get_params = optimizer_funcs
#     params = get_params(state)

#     # Unpack loss funcs
#     dloss = grad(loss_func)

#     l = loss_func(params, x, y)

#     # Conditional check
# #     pred = np.isnan(l)
# #     def true_fun(x):
# #         return optuna.exceptions.TrialPruned()
# #     true_operand = None
# #     def false_fun(x):
# #         pass
# #     false_operand = None

# #     lax.cond(pred, true_operand, true_fun, false_operand, false_fun)

# #     Rewrite the following using lax.cond
# #     if np.isnan(l):
# #         l = np.inf
# #         print("NaN occured in optimization. Skipping trial.")
# #         raise optuna.exceptions.TrialPruned()
# #     print(f"Iteration: {i}, Loss: {l:.4f}")

#     g = dloss(params, x, y)

#     state = update(i, g, state)
#     return state


def objective(
    trial,
    sequences: List[str],
    params: Optional[Dict] = None,
    n_epochs_config: Dict = None,
    learning_rate_config: Dict = None,
) -> float:
    """
    Objective function for an Optuna trial.

    The goal with the objective function is
    to automatically find the number of epochs to train
    that minimizes the average of 5-fold test loss.
    Doing so allows us to avoid babysitting the model manually.

    :param trial: An Optuna trial object.
    :param sequences: A list of strings corresponding to the sequences
        that we want to evotune against.
    :param params: A dictionary of parameters.
        Should have the keys ``mLSTM1900`` and ``dense``,
        which correspond to the mLSTM weights and dense layer weights
        (output dimensions = 25)
        to predict the next character in the sequence.
    :param n_epochs_config: A dictionary of kwargs
        to ``trial.suggest_discrete_uniform``,
        which are: ``name``, ``low``, ``high``, ``q``.
        This controls how many epochs to have Optuna test.
        See source code for default configuration,
        at the definition of ``n_epochs_kwargs``.
    :returns: Average of 5-fold test loss.
    """
    # Default settings for n_epochs_kwargs
    n_epochs_kwargs = {
        "name": "n_epochs",
        "low": 1,
        "high": len(sequences) * 3,
        "q": 1,
    }

    # Default settings for learning_rate_kwargs
    learning_rate_kwargs = {
        "name": "learning_rate",
        "low": 0.00001,
        "high": 0.01,
    }

    if n_epochs_config is not None:
        n_epochs_kwargs.update(n_epochs_config)
    if learning_rate_config is not None:
        learning_rate_kwargs.update(learning_rate_config)

    n_epochs = trial.suggest_discrete_uniform(**n_epochs_kwargs)
    learning_rate = trial.suggest_loguniform(**learning_rate_kwargs)
    print(f"Trying out {n_epochs} epochs with learning rate {learning_rate}.")

    kf = KFold(shuffle=True)
    sequences = onp.array(sequences)

    avg_test_losses = []
    for i, (train_index, test_index) in enumerate(kf.split(sequences)):
        print(f"Split #{i}")
        train_sequences, test_sequences = (
            sequences[train_index],
            sequences[test_index],
        )

        evotuned_params = fit(
            params, train_sequences, n=int(n_epochs), step_size=learning_rate
        )

        xs, ys = length_batch_input_outputs(test_sequences)

        sum_loss = 0
        for x, y in zip(xs, ys):
            sum_loss += evotune_loss(
                evotuned_params, inputs=x, targets=y
            ) * len(x)
        avg_test_losses.append(sum_loss / len(test_sequences))

    return sum(avg_test_losses) / len(avg_test_losses)


def evotune(
    sequences: List[str],
    n_trials: int = 20,
    params: Optional[Dict] = None,
    n_epochs_config: Dict = None,
    learning_rate_config: Dict = None,
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


    By default, mLSTM1900 weights from the paper are used by passing in `params=None`,
    but if you want to use randomly intialized weights,

        from jax_unirep.evotuning import init_fun
        from jax.random import PRNGKey
        _, params = init_fun(PRNGKey(0), input_shape=(-1, 10))

    :param sequences: Sequences to evotune against.
    :param n_trials: The number of trials Optuna should attempt.
    :param params: Parameters to be passed into `mLSTM1900`.
        Optional; if None, will default to mLSTM1900 from paper,
        or you can pass in your own set of parameters,
        as long as they are stax-compatible.
    :param n_epochs_config: A dictionary of kwargs
        to ``trial.suggest_discrete_uniform``,
        which are: ``name``, ``low``, ``high``, ``q``.
        This controls how many epochs to have Optuna test.
        See source code for default configuration,
        at the definition of ``n_epochs_kwargs``.
    :param learning_rate_config: A dictionary of kwargs
        to ``trial.suggest_loguniform``,
        which are: ``name``, ``low``, ``high``.
        This controls the learning rate of the model.
        See source code for default configuration,
        at the definition of ``learning_rate_kwargs``.
    :returns:
        - study - The optuna study object, containing information
        about all evotuning trials.
        - evotuned_params - A dictionary of optimized weights
    """
    if params is None:
        # _, params = init_fun(random.PRNGKey(0), input_shape=(-1, 10))
        params = (load_params_1900(), (), load_dense_1900())

        # params = dict()
        # params["dense"] = load_dense_1900()
        # params["mLSTM1900"] = load_params_1900()

    # Check that params have correct keys and shapes

    validate_mLSTM1900_params(params[0])

    study = optuna.create_study()

    objective_func = lambda x: objective(
        x,
        params=params,
        sequences=sequences,
        n_epochs_config=n_epochs_config,
        learning_rate_config=learning_rate_config,
    )
    study.optimize(objective_func, n_trials=n_trials)
    num_epochs = int(study.best_params["n_epochs"])
    learning_rate = float(study.best_params["learning_rate"])

    evotuned_params = fit(
        params, sequences=sequences, n=num_epochs, step_size=learning_rate
    )
    return study, evotuned_params


def evotune_manual(
    train_seqs: List[str],
    in_val_seqs: List[str],
    out_val_seqs: List[str],
    params: Optional[Dict] = None,
    n_epochs: int = 65,
    learning_rate: float = 0.00001,
    steps_per_print: int = 200,
):
    """
    Evolutionarily tune the model to a set of training sequences,
    Largely the same as evotune() function but manually selecting epochs and learning rate,
    instead of using Optuna.
    Also takes in validation sets to check for loss on.

    :param train_seqs: The set of training sequences to tune to.
    :param in_val_seqs: In-domain validation set of sequences,
        to check for loss on to prevent overfitting.
    :param out_val_seqs: Out-domain validation set of sequences,
        to check for loss on to prevent overfitting.
    :param params: Parameters to be passed into `mLSTM1900`.
        Optional; if None, will default to mLSTM1900 from paper,
        or you can pass in your own set of parameters,
        as long as they are stax-compatible.
    :param steps_per_print: the number of steps between each print,
        will print out current evotuned_params.
    :param n_epochs: The number of epochs to fit.
    :param learning_rate: Optimizer learning rate.
    """
    if params is None:
        params = (load_params_1900(), (), load_dense_1900())

    # Check that params have correct keys and shapes

    validate_mLSTM1900_params(params[0])

    evotuned_params = fit_manual(
        params,
        train_seqs=train_seqs,
        in_val_seqs=in_val_seqs,
        out_val_seqs=out_val_seqs,
        n=num_epochs,
        step_size=learning_rate,
        steps_per_print=steps_per_print,
    )

    return evotuned_params
