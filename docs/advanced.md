# Advanced Usage

APIs that support "advanced" tasks are available in `jax-unirep`.
Read on to learn how to use them.

## Evotuning

In the original paper the concept of 'evolutionary finetuning' is introduced,
where the pre-trained mLSTM weights get fine-tuned through weight-updates
using homolog protein sequences of a given protein of interest as input.
This feature is available as well in `jax-unirep`.
Given a set of starter weights for the mLSTM (defaults to
the weights from the paper) as well as a set of sequences,
the weights get fine-tuned in such a way that test set loss
in the 'next-aa prediction task' is minimized.
There are two functions with differing levels of control available.

The `evotune` function uses `optuna` under the hood
to automatically find:
1. the optimal number of epochs to train for, and
2. the optimal learning rate,

given a set of sequences.
The `study` object will contain all the information
about the training process of each trial.
`evotuned_params` will contain the fine-tuned mLSTM and dense weights
from the trial with the lowest test set loss.

!!! note "Speed freaks read this!"

    As a heads-up, using `evotune` is kind of slow,
    so read on if you're of the impatient kind -- use `fit`!

If you want to directly fine-tune the weights
for a fixed number of epochs
while using a fixed learning rate,
you should use the `fit` function instead.
The `fit` function has further customization options,
such as different batching strategies.
Please see the function docstring [here][fitdoc] for more information.

!!! note "GPU usage"

    The `fit` function will always default to using a
    GPU `backend` if available for the forward and backward passes
    during training of the LSTM.
    However, for the calulation of the average loss
    on the dataset after every epoch, you can decide
    if the CPU or GPU `backend` should be used (default is CPU).

You can find an example usages of both `evotune` and `fit` [here][examples],
but for convenience, here's a code block that you can copy/paste
to get kickstarted.

!!! warning "Read the docs!"

    Can't emphasize this enough:
    Be sure to read the [API docs for the `fit` function][fitdoc]
    to learn about what's going on underneath the hood!

```python
from jax_unirep.utils import load_random_evotuning_params
# Prepare your sequences as a list of strings,
# using whatever parsers you need.
# This is a pre-requisite step that will likely be project-specific.
seqs = [...]

# You can optionally split the dataset so that you have a holdout set.
shuffle(seqs)
break_point = int(len(seqs) * 0.7)
sequences = seqs[0:break_point]
holdout_sequences = seqs[break_point:]

# Set some evotuning parameters.
N_EPOCHS = 20  # probably want this to be quite high, like in the hundreds.
LEARN_RATE = 1e-5  # this is a very sane default to start with.
PROJECT_NAME = "temp"  # where the weights will be dumped

# Pre-load some evotuning params that are randomly initialized.
params = load_random_evotuning_params()

# Now to evotuning
evotuned_params = fit(
    params=params,
    sequences=sequences,
    n_epochs=N_EPOCHS,
    step_size=LEARN_RATE,
    holdout_seqs=holdout_sequences,
    batch_method="random",
    proj_name=PROJECT_NAME,
    epochs_per_print=1,  # also controls how often weights are dumped.
    backend="cpu",  # default is "cpu", can also set to "gpu" if you have it.
)

dump_params(evotuned_params, PROJECT_NAME)
print("Evotuning done! Find output weights in", PROJECT_NAME)

```

If you want to pass a set of mLSTM and dense weights
that were dumped in an earlier run,
create params as follows:

```python
from jax_unirep.utils import load_params

params = load_params(folderpath="path/to/params/folder")
```

If you want to start from randomly initialized mLSTM and dense weights instead:

```python
from jax_unirep.evotuning import init_fun
from jax.random import PRNGKey

_, params = init_fun(PRNGKey(0), input_shape=(-1, 10))
```

The weights used in the 10-dimensional embedding of the input sequences
always default to the weights from the paper,
since they do not get updated during evotuning.

[fitdoc]: https://elarkk.github.io/jax-unirep/api/#evotuning
[examples]: https://github.com/ElArkk/jax-unirep/blob/master/examples

## End-to-end differentiable models

As a user, you might want to write custom "top models",
such as a linear model on top of the reps,
but might want to jointly optimize the UniRep weights
with the top model reps.
You're in luck!

We implemented the mLSTM layers in such a way that
they are compatible with `jax.experimental.stax`.
This means that they can easily be plugged into
a `stax.serial` model, e.g. to train both the mLSTM
and a top-model at once:

```python
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu

from jax_unirep.layers import mLSTM1900, mLSTM1900_AvgHidden

init_fun, apply_fun = stax.serial(
    mLSTM1900(),
    mLSTM1900_AvgHidden(),
    # Add two layers, one dense layer that results in 512-dim activations
    Dense(512), Relu(),
    # And then a linear layer to produce a 1-dim activation
    Dense(1)
)
```

Have a look at the [documentation][stax] and [examples][staxex]
for more information about how to implement a model in `jax`.

[stax]: https://jax.readthedocs.io/en/latest/jax.experimental.stax.html
[staxex]: https://github.com/google/jax/tree/master/examples

## Sampling new protein sequences

When doing protein engineering,
one core task is proposing new sequences to order by gene synthesis.
`jax-unirep` provides a number of utility functions inside `jax_unirep.sampler`
that help with this task.

### Basic sampling

The key one to focus on is the `sample_one_chain` function.

This function takes in a starting sequence,
and uses Monte Carlo sampling alongside the Metropolis-Hastings criteria
to score and rank-order new sequences to try out.
The usage pattern is as follows.

Firstly, you _must_ have a scoring function defined
that takes in a string sequence, and outputs a number.
This can be, for example, in the form of a pre-trained machine learning model
that you have created.

```python
from jax_unirep import get_reps
model = SomeSKLearnModel()
model.fit(training_X, training_y)

def scoring_func(sequence: str):
    reps, _, _ = get_reps(sequence)
    return model.predict(reps)
```

Now, we can use MCMC sampling to propose new sequences.

```python
from jax_unirep import sample_one_chain
starter_seq = "MKLNEQLJLA"  # can be longer!
sampled_sequences = sample_one_chain(starter_seq, n_steps=10, scoring_func=scoring_func)
sampled_seqs_df = pd.DataFrame(sampled_sequences)
```

`sampled_sequences` is a dictionary
that can be converted directly into a `pandas.DataFrame`.
In there, every single sequence that was ever sampled is recorded,
as well as its score (given by the scoring function)
and whether it was accepted by the MCMC sampler or not.
(All generated sequences are recorded,
just in case there was something good that was rejected!)

### Parallel sampling

If you want to do parallel sampling,
you can use any library that does parallel processing.
We're going to show you one example using [Dask](https://dask.org/),
which happens to be out favourite library for scalable Python!

Assuming you have a Dask `client` object instantiated:

```python
client = Client(...)  # you'll have to configure this according to your own circumstances

starter_seq = "MKLNEQLJLA"  # can be longer!
chain_results_futures = []
for i in range(100):  # sample 100 independent chains
    chain_results_futures.append(
        # Submit tasks to workers
        client.submit(
            sample_one_chain,
            starter_seq,
            n_steps=10,
            scoring_func=scoring_func,
            pure=False  # this is important, esp. with random sampling methods
        )
    )
# Gather results from distributed workers
chain_results = client.gather(chain_results_futures)
# Convert everything into a single DataFrame
chain_data = pd.concat([pd.DataFrame(r) for r in chain_results])
```

## Your contribution here

Is there an "advanced" protocol that you've developed surrounding `jax-unirep`?
If so, please consider contributing it here!
