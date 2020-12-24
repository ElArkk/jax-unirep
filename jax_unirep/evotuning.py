import logging
import os
from functools import partial
from random import choice
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as onp
import optuna
from jax import grad, jit
from jax import numpy as np
from jax import vmap
from sklearn.model_selection import KFold
from tqdm.autonotebook import tqdm

from jax_unirep.losses import _neg_cross_entropy_loss

from .evotuning_models import mlstm1900_apply_fun
from .optimizers import adamW
from .utils import (
    dump_params,
    get_batching_func,
    input_output_pairs,
    length_batch_input_outputs,
    load_params,
    right_pad,
    validate_mLSTM_params,
)

"""API for evolutionary tuning."""


logger = logging.getLogger("evotuning")


def setup_evotuning_log():
    logger.setLevel(logging.INFO)
    if os.path.exists("evotuning.log"):
        os.remove("evotuning.log")
    fh = logging.FileHandler("evotuning.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s :: %(levelname)s :: %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def evotune_loss(params, predict, inputs, targets):
    logging.debug(f"Input shape: {inputs.shape}")
    logging.debug(f"Output shape: {targets.shape}")
    predictions = vmap(partial(predict, params))(inputs)

    return _neg_cross_entropy_loss(targets, predictions)


def avg_loss(
    xs: List[np.ndarray],
    ys: List[np.ndarray],
    params: Tuple,
    model_func: Callable,
    backend: str = "cpu",
    batch_size: int = 50,
) -> float:
    """
    Return average loss of a set of parameters, on a set of sequences.

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
    evotune_loss_jit = jit(
        partial(evotune_loss, predict=model_func), backend=backend
    )

    def batch_iter(xs: np.ndarray, ys: np.ndarray, batch_size: int):
        for i in range(0, len(xs), batch_size):
            yield xs[i : i + batch_size], ys[i : i + batch_size]

    for xmat, ymat in zip(xs, ys):
        # Send x and y in small batches to control memory usage.
        for x, y in batch_iter(xmat, ymat, batch_size=batch_size):
            sum_loss += evotune_loss_jit(
                params=params, inputs=x, targets=y
            ) * len(x)
            num_seqs += len(x)

    return sum_loss / num_seqs


def generate_single_length_batch(
    sequences: Iterable[str], holdout_seqs: Optional[Iterable[str]] = None
) -> Tuple[int, Iterable[str], Optional[Iterable[str]]]:
    """
    Generates a single-length batch.

    This function is refactored out of the ``fit`` function
    to make it easier to read.

    :param sequences: Sequences to generate one length batch for.
    :param holdout_seqs: Holdout sequences.
    """
    # First pad to the same length, effectively giving us one length batch.
    all_sequences = set(sequences)
    if holdout_seqs is not None:
        all_sequences = all_sequences.union(set(holdout_seqs))
    max_len = max([len(seq) for seq in all_sequences])
    sequences = right_pad(sequences, max_len)
    if holdout_seqs is not None:
        holdout_seqs = right_pad(holdout_seqs, max_len)
    return max_len, sequences, holdout_seqs


def generate_batching_funcs(
    sequences: Iterable[str], batch_size: int
) -> Tuple[Dict[int, Callable], List[List[str]], List[int]]:
    """
    Generate a batching function for each sequence length

    Given a set of sequences and a batch size,
    this function generates a dictionary,
    where each key value pair consists of a
    unique sequence length and a batching function for that length
    respectively.
    It also returns the batched sequences,
    as well as the unique sequence lenghts.

    :param sequences: Sequences to generate batching functions for
    :param batch_size: batch size for all batching functions
    """
    seqs_batched, seq_lens = length_batch_input_outputs(sequences)
    len_batching_funcs = {
        sl: get_batching_func(seq_batch, batch_size)
        for (sl, seq_batch) in zip(seq_lens, seqs_batched)
    }

    return len_batching_funcs, seqs_batched, seq_lens


def fit(
    sequences: Iterable[str],
    n_epochs: int,
    model_func: Callable = mlstm1900_apply_fun,
    params: Any = None,
    batch_method: str = "random",
    batch_size: int = 25,
    step_size: float = 0.0001,
    holdout_seqs: Optional[Iterable[str]] = None,
    proj_name: str = "temp",
    epochs_per_print: int = 1,
    backend: str = "cpu",
) -> Dict:
    """
    Return mLSTM weights fitted to predict the next letter in each AA sequence.

    The training loop is as follows, depending on the batching strategy:

    Length batching:

    - At each iteration,
    of all sequence lengths present in `sequences`,
    one length gets chosen at random.
    - Next, `batch_size` number of sequences of the chosen length
    get selected at random.
    - If there are less sequences of a given length than `batch_size`,
    all sequences of that length get chosen.
    - Those sequences then get passed through the model.
    No padding of sequences occurs.

    To get batching of sequences by length done,
    we call on `batch_sequences` from our `utils.py` module,
    which returns a list of sub-lists,
    in which each sub-list contains the indices
    in the original list of sequences
    that are of a particular length.

    Random batching:

    - Before training, all sequences get padded
    to be the same length as the longest sequence
    in `sequences`.
    - Then, at each iteration,
    we randomly sample `batch_size` sequences
    and pass them through the model.

    The training loop does not adhere
    to the common notion of `epochs`,
    where all sequences would be seen by the model
    exactly once per epoch.
    Instead sequences always get sampled at random,
    and one epoch approximately consists of
    `round(len(sequences) / batch_size)` weight updates.
    Asymptotically, this should be approximately equivalent
    to doing epoch passes over the dataset.

    To learn more about the passing of `params`,
    have a look at the `evotune` function docstring.

    You can optionally dump parameters
    and print weights every `epochs_per_print` epochs
    to monitor training progress.
    For ergonomics, training/holdout set losses are estimated
    on a batch size the same as `batch_size`,
    rather than calculated exactly on the entire set.
    Set `epochs_per_print` to `None` to avoid parameter dumping.

    ### Parameters

    - `sequences`: List of sequences to evotune on.
    - `n_epochs`: The number of iterations to evotune on.
    - `model_func`: A function that accepts (params, x).
        Defaults to the mLSTM1900 model function.
    - `params`: Optionally pass in the params you want to use.
        These params must yield a correctly-sized mLSTM,
        otherwise you will get cryptic shape errors!
        If None, params will be randomly generated,
        except for mlstm_size of 1900,
        where the pre-trained weights from
        the original publication are used.
    - `batch_method`: One of "length" or "random".
    - `batch_size`: If random batching is used,
        number of sequences per batch.
        As a rule of thumb, batch size of 50 consumes
        about 5GB of GPU RAM.
    - `step_size`: The learning rate.
    - `holdout_seqs`: Holdout set, an optional input.
    - `proj_name`: The directory path for weights to be output to.
    - `epochs_per_print`: Number of epochs to progress before printing
        and dumping of weights.
        Must be greater than or equal to 1.
    - `backend`: Whether or not to use the GPU. Defaults to "cpu",
        but can be set to "gpu" if desired.
        If you set it to GPU, make sure you have
        a version of `jax` that is pre-compiled to work with GPUs.

    ### Returns

    Final optimized parameters.
    """

    setup_evotuning_log()
    model_func = jit(model_func)

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
        g = grad(partial(evotune_loss, predict=model_func))(
            params, inputs=x, targets=y
        )
        state = update(i, g, state)

        return state

    if params is None:
        params = load_params()
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
        _, sequences, holdout_seqs = generate_single_length_batch(
            sequences, holdout_seqs
        )

    # batch sequences by length
    (
        training_len_batching_funcs,
        training_seqs_batched,
        training_seq_lens,
    ) = generate_batching_funcs(sequences, batch_size)
    if holdout_seqs is not None:
        (
            holdout_len_batching_funcs,
            holdout_seqs_batched,
            holdout_seq_lens,
        ) = generate_batching_funcs(holdout_seqs, batch_size)

    batch_lens = [len(batch) for batch in training_seqs_batched]
    if batch_method == "length":
        logger.info(
            f"Length-batching done: "
            f"{len(batch_lens)} unique sequence lengths, "
            f"with average batch length {onp.mean(batch_lens)}, "
            f"max batch length {max(batch_lens)} "
            f"and min batch length {min(batch_lens)}."
        )
    elif batch_method == "random":
        # Both training_seq_lens
        logger.info(
            f"Random batching done: "
            f"All sequences padded to max sequence length of {max(training_seq_lens)}"
        )

    init, update, get_params = adamW(step_size=step_size)
    get_params = jit(get_params)
    state = init(params)

    # calculate how many iterations constitute one epoch approximately
    epoch_len = round(len(sequences) / batch_size)

    n = n_epochs * epoch_len
    for i in tqdm(range(n), desc="Iteration"):
        logger.debug(f"Iteration {i}")
        current_epoch = (i // epoch_len) + 1
        is_starting_new_printing_epoch = (
            i % (epochs_per_print * epoch_len) == 0
        )
        # Choose a sequence length at random for this iteration
        length = choice(training_seq_lens)
        avg_loss_func = partial(
            avg_loss, model_func=model_func, backend=backend
        )

        if is_starting_new_printing_epoch:
            log_epoch_func = partial(
                log_epoch,
                current_epoch=current_epoch,
                get_params_func=get_params,
                state=state,
                avg_loss_func=avg_loss_func,
            )

            log_epoch_func(
                length=length,
                len_batching_funcs=training_len_batching_funcs,
            )

            if holdout_seqs is not None:
                holdout_length = choice(holdout_seq_lens)
                log_epoch_func(
                    length=holdout_length,
                    len_batching_funcs=holdout_len_batching_funcs,
                    is_holdout_set=True,
                )
            dump_params(get_params(state), proj_name, current_epoch - 1)

        logger.debug("Getting batches")
        x, y = training_len_batching_funcs[length]()

        # actual forward & backwrd pass happens here
        logger.debug("Getting state")
        state = step(i, x, y, state)

    return get_params(state)


def log_epoch(
    current_epoch: int,
    state,
    length: int,
    len_batching_funcs: Dict[int, Callable],
    get_params_func: Callable,
    avg_loss_func: Callable,
    is_holdout_set: bool = False,
):
    """
    Log relevant information from one epoch.

    :param current_epoch: The current epoch that is being logged.
    :param state: Parameters wrapped in a jax optimizer's state.
    :param length: The length chosen.
    :param len_batching_funcs: A dictionary of length-batching functions,
        each of which accepts no arguments and returns an x, y matrix pair.
    :param get_params_func: A `get_params` function returned from
        the JAX optimizer triplet.
    :param avg_loss_func: A function that calculates the average loss
        over all of the data points x, y (returned from the len_batching_funcs)
        which accepts elements ([x], [y], and state_params)
    :param is_holdout_set: Whether or not we are using the holdout set.
        Affects the logging text only.
    """
    state_params = get_params_func(state)
    x, y = len_batching_funcs[length]()
    loss = avg_loss_func([x], [y], state_params)
    data_set = "holdout" if is_holdout_set else "training"
    logger.info(f"Calculations for {data_set} set:")
    logger.info(f"Epoch {current_epoch - 1}: Estimated average loss: {loss}. ")
    return None


def objective(
    trial,
    sequences: Iterable[str],
    model_func: Callable,
    params: Any,
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
    :param model_func: A model forward pass function that accepts (params, x).
    :param params: Model parameters that are compatible with the model_func.
    :param n_epochs_config: A dictionary of kwargs
        to `trial.suggest_discrete_uniform`,
        which are: `name`, `low`, `high`, `q`.
        This controls how many epochs to have Optuna test.
        See source code for default configuration,
        at the definition of `n_epochs_kwargs`.
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
            sequences=train_sequences,
            model_func=model_func,
            params=params,
            n_epochs=int(n_epochs),
            step_size=learning_rate,
        )

        seqs_batched, _ = length_batch_input_outputs(test_sequences)
        xs, ys = [], []
        for seq_batch in seqs_batched:
            x, y = input_output_pairs(seq_batch)
            xs.append(x)
            ys.append(y)

        avg_test_losses.append(
            avg_loss(xs, ys, evotuned_params, model_func=model_func)
        )

    return sum(avg_test_losses) / len(avg_test_losses)


def evotune(
    sequences: Iterable[str],
    model_func: Callable = mlstm1900_apply_fun,
    params: Any = None,
    n_trials: Optional[int] = 20,
    n_epochs_config: Dict = None,
    learning_rate_config: Dict = None,
    n_splits: Optional[int] = 5,
    out_dom_seqs: Optional[List[str]] = None,
) -> Tuple:
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

    By default, mLSTM and Dense weights from the paper are used
    by setting `mlstm_size=1900` and `params=None`
    in the partially-evaluated fit function (`fit_func`),
    but if you want to use randomly intialized weights:

    ```python
    from jax_unirep.evotuning import evotuning_funcs, fit
    from jax.random import PRNGKey
    from functools import partial

    init_func, _ = evotuning_funcs(mlstm_size=256) # works for any size
    _, params = init_func(PRNGKey(0), input_shape=(-1, 26))
    fit_func = partial(fit, mlstm_size=256, params=params)
    ```

    or dumped weights:

    ```python
    from jax_unirep.evotuning import fit
    from jax_unirep.utils import load_params

    params = load_params(folderpath="path/to/params/folder")
    fit_func = partial(fit, mlstm_size=256, params=params)
    ```

    The examples above use mLSTM sizes of 256, but any size works in theory!
    Just make sure that the mLSTM size of your randomly initialized or dumped
    `params` matches the one you set in the partially-evaluated fit function.

    This function is intended as an automagic way of identifying
    the best model and training routine hyperparameters.
    If you want more control over how fitting happens,
    please use the `fit()` function directly.
    There is an example in the `examples/` directory
    that shows how to use it.

    ### Parameters

    - `sequences`: Sequences to evotune against.
    - `model_func`: Model apply func.
        Defaults to the mLSTM1900 apply function.
    - `params`: Model params that are compatible with model apply func.
        Defaults to the mLSTM1900 params.
    - `n_trials: The number of trials Optuna should attempt.
    - `n_epochs_config`: A dictionary of kwargs
        to `trial.suggest_discrete_uniform`,
        which are: `name`, `low`, `high`, `q`.
        This controls how many epochs to have Optuna test.
        See source code for default configuration,
        at the definition of `n_epochs_kwargs`.
    - `learning_rate_config`: A dictionary of kwargs
        to `trial.suggest_loguniform`,
        which are: `name`, `low`, `high`.
        This controls the learning rate of the model.
        See source code for default configuration,
        at the definition of `learning_rate_kwargs`.
    - `n_splits`: The number of folds of cross-validation to do.
    - `out_dom_seqs`: Out-domain holdout set of sequences,
        to check for loss on to prevent overfitting.

    ### Returns

    - `study`: The optuna study object, containing information
        about all evotuning trials.
    - `evotuned_params`: A dictionary of the final, optimized weights.
    """
    study = optuna.create_study()
    if params is None:
        params = load_params()

    def objective_func(trial):
        return objective(
            trial,
            sequences=sequences,
            model_func=model_func,
            params=params,
            n_epochs_config=n_epochs_config,
            learning_rate_config=learning_rate_config,
            n_splits=n_splits,
        )

    study.optimize(objective_func, n_trials=n_trials)
    n_epochs = int(study.best_params["n_epochs"])
    learning_rate = float(study.best_params["learning_rate"])

    logger.info(
        f"Optuna done, starting tuning with learning rate={learning_rate}, "
    )

    evotuned_params = fit(
        sequences=sequences,
        model_func=model_func,
        params=params,
        n_epochs=n_epochs,
        step_size=learning_rate,
        holdout_seqs=out_dom_seqs,
    )

    return study, evotuned_params
