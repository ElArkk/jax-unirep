"""API for evolutionary tuning."""

import logging
import os
from random import choice
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import equinox as eqx
import numpy as onp
import optax
import optuna
from jax import numpy as np
from jax import vmap
from sklearn.model_selection import KFold
from tqdm.autonotebook import tqdm

from .losses import cross_entropy_loss
from .models import MLSTM, load_model, save_model
from .utils import (
    get_batching_func,
    input_output_pairs,
    length_batch_input_outputs,
    right_pad,
)

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


def evotune_loss(model: MLSTM, inputs, targets):
    """Masked next-amino-acid loss of a model on one batch of sequences.

    :param model: The `MLSTM` being tuned.
    :param inputs: One-hot sequences, shape (n_sequences, n_positions, 26).
    :param targets: One-hot next letters, shape (n_sequences, n_positions, 25).
    :returns: A scalar loss.
    """
    # `logits` handles one sequence, so vmap it over the batch. The softmax is
    # applied inside the loss, fused with the log.
    logits = vmap(model.logits)(inputs)

    # Class 0 is the "-" character that right_pad adds. It is a real class, so
    # an unmasked loss trains the model to predict gaps. With random batching
    # that is most of the signal for any sequence shorter than the longest one.
    mask = 1.0 - targets[..., 0]

    return cross_entropy_loss(targets, logits, mask)


evotune_loss_jit = eqx.filter_jit(evotune_loss)


def avg_loss(
    xs: List[np.ndarray],
    ys: List[np.ndarray],
    model: MLSTM,
    batch_size: int = 50,
) -> float:
    """
    Return average loss of a model on a set of sequences.

    :param xs: List of NumPy arrays
    :param ys: List of NumPy arrays
    :param model: The `MLSTM` to evaluate.
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

    def batch_iter(xs: np.ndarray, ys: np.ndarray, batch_size: int):
        for i in range(0, len(xs), batch_size):
            yield xs[i : i + batch_size], ys[i : i + batch_size]

    for xmat, ymat in zip(xs, ys):
        # Send x and y in small batches to control memory usage.
        for x, y in batch_iter(xmat, ymat, batch_size=batch_size):
            sum_loss += evotune_loss_jit(model, x, y) * len(x)
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
    model: Optional[MLSTM] = None,
    batch_method: str = "length",
    batch_size: int = 25,
    step_size: float = 0.0001,
    holdout_seqs: Optional[Iterable[str]] = None,
    proj_name: str = "temp",
    epochs_per_print: int = 1,
) -> MLSTM:
    """
    Return an mLSTM fitted to predict the next letter in each AA sequence.

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

    You can optionally dump weights
    and print losses every `epochs_per_print` epochs
    to monitor training progress.
    For ergonomics, training/holdout set losses are estimated
    on a batch size the same as `batch_size`,
    rather than calculated exactly on the entire set.
    Dumped weights are written in the same layout `load_model` reads,
    so training can be resumed from `load_model(folderpath=...)`.

    ### Parameters

    - `sequences`: List of sequences to evotune on.
    - `n_epochs`: The number of iterations to evotune on.
    - `model`: The `MLSTM` to tune.
        Defaults to the pre-trained mLSTM1900 from the paper.
        Pass `MLSTM(n_cells=..., output_dim=..., key=...)`
        to start from randomly initialized weights of any size,
        or `load_model(folderpath=...)` to resume from dumped weights.
    - `batch_method`: One of "length" or "random". Defaults to "length",
        which groups sequences of identical length and pads nothing.
        "random" pads every sequence to the longest in the *whole dataset*,
        which on a realistic length distribution wastes about half the
        compute and feeds gap characters through the recurrent state.
        Prefer "length" unless the sequences are already near-uniform.
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

    ### Returns

    The tuned `MLSTM`.
    """

    setup_evotuning_log()

    if model is None:
        model = load_model(paper_weights=1900)
    # Defensive programming checks
    if batch_method not in ["length", "random"]:
        raise ValueError("batch_method must be one of 'length' or 'random'")
    if not isinstance(epochs_per_print, int):
        raise TypeError("epochs_per_print must be an integer.")
    if epochs_per_print < 1:
        raise ValueError(
            "epochs_per_print must be greater than or equal to 1."
        )

    # weight_decay=0.01 reproduces the hand-rolled adamW this replaced, whose
    # update was `x - lr * (mhat / (sqrt(vhat) + eps) + w * x)` with w=0.01.
    # optax.adamw chains scale_by_adam -> add_decayed_weights -> scale_by_lr,
    # giving -lr * (adam + wd * params): the same expression. Its own default
    # is 1e-4, which would quietly change evotuning results.
    optim = optax.adamw(learning_rate=step_size, weight_decay=0.01)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, x, y):
        """
        Perform one step of evolutionary updating.

        The model *is* the parameters, so a step takes them in and hands the
        updated ones back rather than threading an opaque optimizer state that
        the parameters have to be dug out of.

        :param model: The model at the start of this step.
        :param opt_state: The optimizer's momentum estimates.
        :param x: One-hot input sequences.
        :param y: One-hot next letters to predict.
        :returns: The updated model, the updated optimizer state, and the loss.
        """
        loss, grads = eqx.filter_value_and_grad(evotune_loss)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        # optax's update already negates the gradient, so these are added.
        return eqx.apply_updates(model, updates), opt_state, loss

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
        logger.info(
            f"Random batching done: All sequences padded to max sequence "
            f"length of {max(training_seq_lens)}"
        )

    # calculate how many iterations constitute one epoch approximately
    epoch_len = round(len(sequences) / batch_size)

    n = n_epochs * epoch_len
    for i in tqdm(range(n), desc="Iteration"):
        logger.debug(f"Iteration {i}")
        current_epoch = (i // epoch_len) + 1
        # Choose a sequence length at random for this iteration
        length = choice(training_seq_lens)

        if i % (epochs_per_print * epoch_len) == 0:
            log_epoch(
                current_epoch=current_epoch,
                model=model,
                length=length,
                len_batching_funcs=training_len_batching_funcs,
            )

            if holdout_seqs is not None:
                log_epoch(
                    current_epoch=current_epoch,
                    model=model,
                    length=choice(holdout_seq_lens),
                    len_batching_funcs=holdout_len_batching_funcs,
                    is_holdout_set=True,
                )
            save_model(model, proj_name, current_epoch - 1)

        logger.debug("Getting batches")
        x, y = training_len_batching_funcs[length]()

        # actual forward & backward pass happens here
        model, opt_state, loss = step(model, opt_state, x, y)
        logger.debug(f"Iteration {i}: loss {loss}")

    return model


def log_epoch(
    current_epoch: int,
    model: MLSTM,
    length: int,
    len_batching_funcs: Dict[int, Callable],
    is_holdout_set: bool = False,
):
    """
    Log relevant information from one epoch.

    :param current_epoch: The current epoch that is being logged.
    :param model: The model at the start of this epoch.
    :param length: The length chosen.
    :param len_batching_funcs: A dictionary of length-batching functions,
        each of which accepts no arguments and returns an x, y matrix pair.
    :param is_holdout_set: Whether or not we are using the holdout set.
        Affects the logging text only.
    """
    x, y = len_batching_funcs[length]()
    loss = avg_loss([x], [y], model)
    data_set = "holdout" if is_holdout_set else "training"
    logger.info(f"Calculations for {data_set} set:")
    logger.info(f"Epoch {current_epoch - 1}: Estimated average loss: {loss}. ")
    return None


def objective(
    trial,
    sequences: Iterable[str],
    model: MLSTM,
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
    :param model: The `MLSTM` that each fold starts from.
    :param n_epochs_config: A dictionary of kwargs
        to `trial.suggest_float`,
        which are: `name`, `low`, `high`, `step`.
        This controls how many epochs to have Optuna test.
        See source code for default configuration,
        at the definition of `n_epochs_kwargs`.
    :param learning_rate_config: A dictionary of kwargs
        to `trial.suggest_float`,
        which are: `name`, `low`, `high`.
        This controls the learning rate of the model.
    :param n_splits: The number of folds of cross-validation to do.

    :returns: Average of 5-fold test loss.
    """
    # Default settings for n_epochs_kwargs
    n_epochs_kwargs = {
        "name": "n_epochs",
        "low": 1,
        "high": len(sequences) * 3,
        "step": 1,
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

    n_epochs = trial.suggest_float(**n_epochs_kwargs)
    learning_rate = trial.suggest_float(**learning_rate_kwargs, log=True)
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

        tuned_model = fit(
            sequences=train_sequences,
            model=model,
            n_epochs=int(n_epochs),
            step_size=learning_rate,
        )

        seqs_batched, _ = length_batch_input_outputs(test_sequences)
        xs, ys = [], []
        for seq_batch in seqs_batched:
            x, y = input_output_pairs(seq_batch)
            xs.append(x)
            ys.append(y)

        avg_test_losses.append(avg_loss(xs, ys, tuned_model))

    return sum(avg_test_losses) / len(avg_test_losses)


def evotune(
    sequences: Iterable[str],
    model: Optional[MLSTM] = None,
    n_trials: Optional[int] = 20,
    n_epochs_config: Dict = None,
    learning_rate_config: Dict = None,
    n_splits: Optional[int] = 5,
    out_dom_seqs: Optional[List[str]] = None,
) -> Tuple[optuna.Study, MLSTM]:
    """
    Evolutionarily tune the model to a set of sequences.

    Evotuning is described in the original UniRep and eUniRep papers.
    This reimplementation of evotune provides a nicer API
    that automatically handles multiple sequences of variable lengths.

    Evotuning always needs a starter model.
    By default, the pre-trained weights from the Nature Methods paper are used.
    However, other pre-trained weights are legitimate.

    We first use optuna to figure out how many epochs to fit
    before overfitting happens.
    To save on computation time, the number of trials run
    defaults to 20, but can be configured.

    If you want to start from randomly initialized weights of any size:

    ```python
    from jax.random import PRNGKey
    from jax_unirep.evotuning import evotune
    from jax_unirep.models import MLSTM

    model = MLSTM(n_cells=4, output_dim=256, key=PRNGKey(0))
    study, tuned_model = evotune(sequences, model=model)
    ```

    or from previously dumped weights:

    ```python
    from jax_unirep.models import load_model

    model = load_model(folderpath="path/to/weights/folder")
    ```

    The model states its own architecture, so nothing needs to be told
    what size it is.

    This function is intended as an automagic way of identifying
    the best model and training routine hyperparameters.
    If you want more control over how fitting happens,
    please use the `fit()` function directly.
    There is an example in the `examples/` directory
    that shows how to use it.

    ### Parameters

    - `sequences`: Sequences to evotune against.
    - `model`: The `MLSTM` to tune.
        Defaults to the pre-trained mLSTM1900 from the paper.
    - `n_trials`: The number of trials Optuna should attempt.
    - `n_epochs_config`: A dictionary of kwargs
        to `trial.suggest_float`,
        which are: `name`, `low`, `high`, `step`.
        This controls how many epochs to have Optuna test.
        See source code for default configuration,
        at the definition of `n_epochs_kwargs`.
    - `learning_rate_config`: A dictionary of kwargs
        to `trial.suggest_float`,
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
    - `tuned_model`: The final, optimized `MLSTM`.
    """
    study = optuna.create_study()
    if model is None:
        model = load_model(paper_weights=1900)

    def objective_func(trial):
        return objective(
            trial,
            sequences=sequences,
            model=model,
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

    tuned_model = fit(
        sequences=sequences,
        model=model,
        n_epochs=n_epochs,
        step_size=learning_rate,
        holdout_seqs=out_dom_seqs,
    )

    return study, tuned_model
