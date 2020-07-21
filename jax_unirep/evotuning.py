"""API for evolutionary tuning."""
import logging
from functools import partial
from random import choice
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as onp
import optuna
from jax import grad, jit
from jax import numpy as np
from jax import vmap
from jax.experimental.stax import Dense, Softmax, serial
from sklearn.model_selection import KFold
from tqdm.autonotebook import tqdm

from jax_unirep.losses import _neg_cross_entropy_loss

from .layers import mLSTM1900, mLSTM1900_HiddenStates
from .optimizers import adamW
from .utils import (
    aa_seq_to_int,
    batch_sequences,
    dump_params,
    get_batching_func,
    load_embedding_1900,
    load_params,
    one_hots,
    right_pad,
    validate_mLSTM1900_params,
)

# setup logger
logger = logging.getLogger("evotuning")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("evotuning.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

# setup model
model_layers = (mLSTM1900(), mLSTM1900_HiddenStates(), Dense(25), Softmax)
init_fun, predict = serial(*model_layers)


def evotune_loss(params, inputs, targets):
    logging.debug(f"Input shape: {inputs.shape}")
    logging.debug(f"Output shape: {targets.shape}")
    predictions = vmap(partial(predict, params))(inputs)

    return _neg_cross_entropy_loss(targets, predictions)


def avg_loss(
    xs: List[np.ndarray],
    ys: List[np.ndarray],
    params,
    backend: str = "cpu",
    batch_size: int = 50,
) -> float:
    """
    Return average loss of a set of parameters,
    on a set of sequences.

    :param xs: List of NumPy arrays
    :param ys: List of NumPy arrays
    :param params: parameters (i.e. from training)
    :param backend: Whether to use GPU ('gpu') or CPU ('cpu')
        to perform calculation.
        Defaults to 'cpu'.
    :param batch_size: Size of batch when calculating average loss
        over train or holdout set.
        Controlling this parameter helps with memory allocation issues -
        reduce this parameter's size to reduce the amount of RAM allocation
        needed to calculate loss.
        As a rule of thumb, batch size of 50 consumes about 5GB of GPU RAM.
    """
    logging.debug("Calculating average loss.")
    sum_loss = 0
    num_seqs = 0
    global evotune_loss  # this is necessary for JIT to reference evotune_loss
    evotune_loss_jit = jit(evotune_loss, backend=backend)

    def batch_iter(
        xs: np.ndarray, ys: np.ndarray, batch_size: int = batch_size
    ):
        i = 0
        for i in tqdm(
            range(0, len(xs), batch_size),
            desc=f"Avg loss on dataset length {len(xs)}",
        ):
            yield xs[i : i + batch_size], ys[i : i + batch_size]
            i += batch_size

    for xmat, ymat in zip(xs, ys):
        # Send x and y in small batches of 100 to control memory usage.
        for x, y in batch_iter(xmat, ymat, batch_size=batch_size):
            sum_loss += evotune_loss_jit(params, inputs=x, targets=y) * len(x)
            num_seqs += len(x)

    return sum_loss / num_seqs


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
    embeddings = load_embedding_1900()
    x = onp.stack([embeddings[i] for i in seq_int])
    y = onp.stack([one_hots[i] for i in next_letters_int])
    return x, y


def input_output_pairs(sequences: List[str],) -> Tuple[np.ndarray, np.ndarray]:
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
    for s in tqdm(sequences, desc="evotuning pairs"):
        x, y = evotuning_pairs(s)
        xs.append(x)
        ys.append(y)
    return onp.stack(xs), onp.stack(ys)


def length_batch_input_outputs(
    sequences: Iterable[str],
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
    lens = []
    for idxs in tqdm(idxs_batched):
        seqs = [sequences[i] for i in idxs]
        x, y = input_output_pairs(seqs)
        xs.append(x)
        ys.append(y)
        lens.append(len(seqs[0]))
    return xs, ys, lens


def fit(
    params: Dict,
    sequences: Set[str],
    n_epochs: int,
    batch_method: Optional[str] = "random",
    batch_size: Optional[int] = 25,
    step_size: float = 0.0001,
    holdout_seqs: Optional[Set[str]] = None,
    proj_name: Optional[str] = "temp",
    epochs_per_print: Optional[int] = 1,
    backend="cpu",
) -> Dict:
    """
    Return mLSTM weights fitted to predict the next letter in each AA sequence.

    The training loop is as follows, depending on the batching strategy:

    Length batching:

    - At each iteration,
    of all sequence lengths present in ``sequences``,
    one length gets chosen at random.
    - Next, ``batch_size`` number of sequences of the chosen length
    get selected at random.
    - If there are less sequences of a given length than `batch_size`,
    all sequences of that length get chosen.
    - Those sequences then get passed through the model.
    No padding of sequences occurs.

    To get batching of sequences by length done,
    we call on ``batch_sequences`` from our ``utils.py`` module,
    which returns a list of sub-lists,
    in which each sub-list contains the indices
    in the original list of sequences
    that are of a particular length.

    Random batching:

    - Before training, all sequences get padded
    to be the same length as the longest sequence
    in ``sequences``.
    - Then, at each iteration,
    we randomly sample ``batch_size`` sequences
    and pass them through the model.

    The training loop does not adhere
    to the common notion of `epochs`,
    where all sequences would be seen by the model
    exactly once per epoch.
    Instead sequences always get sampled at random,
    and one epoch approximately consists of
    ``round(len(sequences) / batch_size)`` weight updates.
    Asymptotically, this should be approximately equiavlent
    to doing epoch passes over the dataset.

    To learn more about the passing of ``params``,
    have a look at the ``evotune`` function docstring.

    You can optionally dump parameters
    and print weights every ``epochs_per_print`` epochs
    to monitor training progress.
    For ergonomics, training/holdout set losses are estimated
    on a batch size the same as ``batch_size``,
    rather than calculated exactly on the entire set.
    Set ``epochs_per_print`` to ``None`` to avoid parameter dumping.

    :param params: mLSTM1900 and Dense parameters.
    :param sequences: List of sequences to evotune on.
    :param n: The number of iterations to evotune on.
    :param batch_method: One of "length" or "random".
    :param batch_size: If random batching is used,
        number of sequences per batch.
        As a rule of thumb, batch size of 50 consumes
        about 5GB of GPU RAM.
    :param step_size: The learning rate.
    :param holdout_seqs: Holdout set, an optional input.
    :param proj_name: The directory path for weights to be output to.
    :param epochs_per_print: Number of epochs to progress before printing
        and dumping of weights.
        Must be greater than or equal to 1.
    :param backend: Whether or not to use the GPU. Defaults to "cpu",
        but can be set to "gpu" if desired.
    """

    @jit
    def step(i, x, y, state):
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

    # Load and check that params have correct keys and shapes
    if params is None:
        params = load_params()

    # Defensive programming checks
    if len(params) != len(model_layers):
        raise ValueError(
            "The number of parameters specified must "
            "match the number of stax.serial layers"
        )
    validate_mLSTM1900_params(params[0])

    # Defensive programming checks
    if batch_method not in ["length", "random"]:
        raise ValueError("batch_method must be one of 'length' or 'random'")
    if not isinstance(epochs_per_print, int):
        raise TypeError("epochs_per_print must be an integer.")
    if epochs_per_print < 1:
        raise ValueError(
            "epochs_per_print must be greater than or equal to 1."
        )

    if batch_method == "random":
        # First pad to the same length, effectively giving us one length batch.
        all_sequences = set(sequences)
        if holdout_seqs is not None:
            all_sequences = all_sequences.union(set(holdout_seqs))
        max_len = max([len(seq) for seq in all_sequences])
        sequences = right_pad(sequences, max_len)
        if holdout_seqs is not None:
            holdout_seqs = right_pad(holdout_seqs, max_len)

    # batch sequences by length
    xs, ys, seq_lens = length_batch_input_outputs(sequences)
    len_batching_funcs = {
        sl: get_batching_func(x, y, batch_size)
        for (sl, x, y) in zip(seq_lens, xs, ys)
    }

    if holdout_seqs is not None:
        holdout_xs, holdout_ys, holdout_seq_lens = length_batch_input_outputs(
            holdout_seqs
        )
        holdout_len_batching_funcs = {
            sl: get_batching_func(x, y, batch_size)
            for (sl, x, y) in zip(holdout_seq_lens, holdout_xs, holdout_ys)
        }

    batch_lens = [len(batch) for batch in xs]
    logger.info(
        f"Length-batching done: "
        f"{len(batch_lens)} unique lengths, "
        f"with average length {onp.mean(batch_lens)}, "
        f"max length {max(batch_lens)} and min length {min(batch_lens)}."
    )

    init, update, get_params = adamW(step_size=step_size)
    get_params = jit(get_params)
    state = init(params)

    # calculate how many iterations constitute one epoch approximately
    epoch_len = int(onp.ceil(len(sequences) / batch_size))

    n = n_epochs * epoch_len
    for i in tqdm(range(n), desc="Iteration"):
        logger.debug(f"Iteration {i}")
        current_epoch = (i // epoch_len) + 1
        is_starting_new_printing_epoch = (
            i % (epochs_per_print * epoch_len) == 0
        )
        # Choose a sequence length at random for this iteration
        l = choice(seq_lens)

        if is_starting_new_printing_epoch:
            logger.info(f"Starting epoch {current_epoch}")
            params = get_params(state)
            x, y = len_batching_funcs[l]()
            loss = avg_loss([x], [y], params, backend=backend)
            logger.info(
                f"Epoch {current_epoch - 1}: "
                f"Estimated average training-set loss: {loss}. "
                "Weights dumped."
            )

            if holdout_seqs is not None:
                # calculate and print loss for out-domain holdout set
                hl = choice(holdout_seq_lens)
                x, y = holdout_len_batching_funcs[hl]()
                loss = avg_loss([x], [y], params, backend=backend)
                logger.info(
                    f"Epoch {current_epoch - 1}: "
                    + f"Estimaged average holdout-set loss: {loss}"
                )

            # dump current params in case run crashes or loss increases
            dump_params(get_params(state), proj_name, current_epoch - 1)

        logger.debug("Getting batches")
        x, y = len_batching_funcs[l]()

        # actual forward & backwrd pass happens here
        logger.debug("Getting state")
        state = step(i, x, y, state)

    return get_params(state)


def objective(
    trial,
    sequences: Iterable[str],
    params: Optional[Dict] = None,
    n_epochs_config: Dict = None,
    learning_rate_config: Dict = None,
    n_splits: Optional[int] = 5,
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
    :param n_splits: The number of folds of cross-validation to do.

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
    logger.info(
        f"Trying out {n_epochs} epochs with learning rate {learning_rate}."
    )

    kf = KFold(n_splits=n_splits, shuffle=True)
    sequences = onp.array(sequences)

    avg_test_losses = []
    for i, (train_index, test_index) in enumerate(kf.split(sequences)):
        logger.info(f"Split #{i}")
        train_sequences, test_sequences = (
            sequences[train_index],
            sequences[test_index],
        )

        evotuned_params = fit(
            params,
            train_sequences,
            n_epochs=int(n_epochs),
            step_size=learning_rate,
        )

        xs, ys, _ = length_batch_input_outputs(sequences)

        avg_test_losses.append(avg_loss(xs, ys, evotuned_params))

    return sum(avg_test_losses) / len(avg_test_losses)


def evotune(
    sequences: Iterable[str],
    params: Optional[Dict] = None,
    proj_name: Optional[str] = "temp",
    out_dom_seqs: Optional[Iterable[str]] = None,
    n_trials: Optional[int] = 20,
    n_epochs_config: Dict = None,
    learning_rate_config: Dict = None,
    n_splits: Optional[int] = 5,
    epochs_per_print: Optional[int] = 200,
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

    By default, mLSTM1900 and Dense weights from the paper are used
    by passing in `params=None`,
    but if you want to use randomly intialized weights:

        from jax_unirep.evotuning import init_fun
        from jax.random import PRNGKey
        _, params = init_fun(PRNGKey(0), input_shape=(-1, 10))

    or dumped weights:

        from jax_unirep.utils import load_params
        params = load_params(folderpath="path/to/params/folder")

    This function is intended as an automagic way of identifying
    the best model and training routine hyperparameters.
    If you want more control over how fitting happens,
    please use the `fit()` function directly.
    There is an example in the `examples/` directory
    that shows how to use it.

    :param sequences: Sequences to evotune against.
    :param params: Parameters to be passed into `mLSTM1900` and `Dense`.
        Optional; if None, will default to weights from paper,
        or you can pass in your own set of parameters,
        as long as they are stax-compatible.
    :param proj_name: Name of the project,
        used to name created output directory.
    :param out_dom_seqs: Out-domain holdout set of sequences,
        to check for loss on to prevent overfitting.
    :param n_trials: The number of trials Optuna should attempt.
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
    :param n_splits: The number of folds of cross-validation to do.
    :param epochs_per_print: The number of steps between each
        printing and dumping of weights in the final
        evotuning step using the optimized hyperparameters.

    :returns:
        - study - The optuna study object, containing information
        about all evotuning trials.
        - evotuned_params - A dictionary of optimized weights
    """

    study = optuna.create_study()

    objective_func = lambda x: objective(
        x,
        params=params,
        sequences=sequences,
        n_epochs_config=n_epochs_config,
        learning_rate_config=learning_rate_config,
        n_splits=n_splits,
    )
    study.optimize(objective_func, n_trials=n_trials)
    num_epochs = int(study.best_params["n_epochs"])
    learning_rate = float(study.best_params["learning_rate"])

    logger.info(
        f"Optuna done, starting tuning with learning rate={learning_rate}, "
    )

    evotuned_params = fit(
        params=params,
        sequences=sequences,
        n_epochs=num_epochs,
        step_size=learning_rate,
        holdout_seqs=out_dom_seqs,
        proj_name=proj_name,
        epochs_per_print=epochs_per_print,
    )

    return study, evotuned_params
