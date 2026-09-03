# Advanced Usage

APIs that support "advanced" tasks are available in `jax-unirep`.
Read on to learn how to use them.

## Evotuning

In the original UniRep paper,
the authors introduced the concept of 'evolutionary finetuning'.
Here the pre-trained mLSTM weights get fine-tuned through weight-updates
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

You can find an example usage of the evotuning function [here][evotuneex].

!!! warning "Read the docs!"

    Can't emphasize this enough:
    Be sure to read the [API docs for the `fit` function][fitdoc]
    to learn about what's going on underneath the hood!

If you want to pass a set of embedding, mLSTM and dense weights
that were dumped in an earlier run,
create params as follows:

```python
from jax_unirep.models import load_model

params = load_model(folderpath="path/to/params/folder")
```

Make sure that the params were created using the same
model architecture that you want to use them with!

If you want to start from randomly initialized embedding, mLSTM and dense weights instead:

```python
from jax_unirep.models import MLSTM
from jax.random import PRNGKey

# For the canonical single-stack 1900-model:
model = MLSTM(n_cells=1, output_dim=1900, key=PRNGKey(42))
```

[fitdoc]: https://elarkk.github.io/jax-unirep/api/#jax_unirepfit
[evotuneex]: https://github.com/ElArkk/jax-unirep/blob/master/examples/evotuning.py

## End-to-end differentiable models

As a user, you might want to write custom "top models",
such as a linear model on top of the reps,
but might want to jointly optimize the UniRep weights
with the top model reps.
You're in luck!

Because `MLSTM` is an [equinox][equinoxdoc] module, and equinox modules are
just PyTrees, you compose one by holding it as a field on your own module.
Its parameters then become part of your model's tree, and gradients reach
them with no extra plumbing.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from jax_unirep import MLSTM, load_model
from jax_unirep.utils import seq_to_oh


class TopModel(eqx.Module):
    """A UniRep trunk with a linear head, trained end to end."""

    trunk: MLSTM
    weight: jax.Array
    bias: jax.Array

    def __init__(self, trunk: MLSTM, key: jax.Array):
        self.trunk = trunk
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (trunk.output_dim,)) * 0.01
        self.bias = jnp.zeros(())

    def __call__(self, one_hot):
        # __call__ handles ONE sequence; vmap it for a batch.
        _, _, hidden_states = self.trunk(one_hot)
        h_avg = hidden_states.mean(axis=0)   # the canonical UniRep rep
        return jnp.dot(h_avg, self.weight) + self.bias
```

Three choices worth making deliberately:

- **Drop the next-amino-acid head.** The built-in `Dense(25)` head predicts the
  next residue and is only used for evotuning. Under your own head it is dead
  parameters.
- **Start from pre-trained weights.** A randomly initialised `MLSTM` returns
  nearly the same representation for every sequence, because the
  weight-normalisation gains start near zero and the gates sit at
  `sigmoid(0)`. A linear head on a constant representation can only learn the
  mean of your targets.

The shipped weights all include the amino-acid head, so drop it explicitly
after loading. The `is_leaf` argument is required, without it `tree_at`
cannot address a field you are replacing *with* `None`:

```python
trunk = load_model(paper_weights=64)
trunk = eqx.tree_at(
    lambda m: (m.dense_w, m.dense_b),
    trunk,
    replace=(None, None),
    is_leaf=lambda x: x is None,
)

model = TopModel(trunk, key=PRNGKey(1))
```

Then train it like any other equinox model. Note that `optax`'s `update`
returns already-negated updates, so `apply_updates` **adds** them:

```python
import optax

sequences = ["HASTA", "VISTA", "ALAVA", "LIMED"]
x = jnp.stack([seq_to_oh(s)[:-1] for s in sequences])
y = jnp.array([1.0, 0.5, -0.5, -1.0])


@eqx.filter_value_and_grad
def loss_fn(model, x, y):
    predictions = jax.vmap(model)(x)
    return jnp.mean((predictions - y) ** 2)


optim = optax.adam(1e-3)
opt_state = optim.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def step(model, opt_state, x, y):
    loss, grads = loss_fn(model, x, y)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    return eqx.apply_updates(model, updates), opt_state, loss


for _ in range(100):
    model, opt_state, loss = step(model, opt_state, x, y)
```

Gradients flow into both the head and every mLSTM cell, which is what makes
this end to end.

### Training only the head

To fit the head against frozen representations, partition the model with a
boolean tree marking what is trainable. **Initialise the optimizer from the
trainable part, not the whole model** -- otherwise `optax` tries to match a
full-model state against partial gradients, and the error surfaces deep inside
`tree_map`:

```python
trainable = jax.tree_util.tree_map(lambda _: False, model)
trainable = eqx.tree_at(
    lambda m: (m.weight, m.bias), trainable, replace=(True, True)
)

diff, static = eqx.partition(model, trainable)

optim = optax.adam(1e-2)
opt_state = optim.init(diff)          # from `diff`, not from `model`


@eqx.filter_value_and_grad
def head_loss(diff, static, x, y):
    model = eqx.combine(diff, static)
    return jnp.mean((jax.vmap(model)(x) - y) ** 2)


@eqx.filter_jit
def head_step(diff, static, opt_state, x, y):
    loss, grads = head_loss(diff, static, x, y)
    updates, opt_state = optim.update(grads, opt_state, diff)
    return eqx.apply_updates(diff, updates), opt_state, loss


for _ in range(20):
    diff, opt_state, loss = head_step(diff, static, opt_state, x, y)

model = eqx.combine(diff, static)
```

Have a look at the [equinox documentation][equinoxdoc]
for more on building and manipulating models this way.

[equinoxdoc]: https://docs.kidger.site/equinox/
[unirep]: https://www.nature.com/articles/s41592-019-0598-1

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
