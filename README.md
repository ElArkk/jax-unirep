[![Build Status](https://travis-ci.com/ElArkk/jax-unirep.svg?branch=master)](https://travis-ci.com/ElArkk/jax-unirep)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# jax-unirep

Reimplementation of the UniRep protein featurization model in JAX.

The UniRep model was developed in George Church's lab,
see the original publication
[here][arxiv] (bioRxiv) or [here][nature] (Nature Methods),
as well as the [repository][repo] containing the original model.

The idea to reimplement the TF-based model in the much lighter JAX framework
was coined by [Eric Ma][ericmjl], who also developed a first version of it
inside his functional deep-learning library [fundl][fundl].

This repo is a self-contained version of the UniRep model
(so far only the 1900 hidden-unit mLSTM),
adapted and extended from fundl.

## Installation

Ensure that your compute environment allows you to run JAX code.
(A modern Linux or macOS with a GLIBC>=2.23 is probably necessary.)

For now, `jax-unirep` is available by pip installing from source.

Installation from GitHub:

```bash
pip install git+https://github.com/ElArkk/jax-unirep.git
```

## Usage

### Getting UniReps

To generate representations of protein sequences,
pass a list of sequences as strings
or a single sequence to `jax_unirep.get_reps`.
It will return a tuple consisting of the
following representations for each sequence:

- `h_avg`: Average hidden state of the mLSTM over the whole sequence.
- `h_final`: Final hidden state of the mLSTM
- `c_final`: Final cell state of the mLSTM

From the original paper,
`h_avg` is considered the "representation" (or "rep") of the protein sequence.

Only valid amino acid sequence letters belonging to the set:

    MRHKDESTNQCUGPAVIFYWLOXZBJ

are allowed as inputs to `get_reps`.
They may be passed in as a single string or an iterable of strings,
and need _not_ necessarily be of the same length.

In Python code, for a single sequence:

```python
from jax_unirep import get_reps

sequence = "ASDFGHJKL"

# h_avg is the canonical "reps"
h_avg, h_final, c_final = get_reps(sequence)
```

And for multiple sequences:

```python
from jax_unirep import get_reps

sequences = ["ASDF", "YJKAL", "QQLAMEHALQP"]

# h_avg is the canonical "reps"
h_avg, h_final, c_final= get_reps(sequences)

# each of the arrays will be of shape (len(sequences), 1900),
# with the correct order of sequences preserved
```

### Evotuning

In the original paper the concept of 'evolutionary finetuning' is introduced,
where the pre-trained mLSTM weights get fine-tuned through weight-updates
using homolog protein sequences of a given protein of interest as input.
This feature is available as well in `jax-unirep`.
Given a set of starter weights for the mLSTM (defaults to
the weights from the paper) as well as a set of sequences,
the weights get fine-tuned in such a way that test set loss
in the 'next-aa prediction task' is minimized.
In order to achieve this, `optuna` is used to
find the optimal number of epochs to train for as well as the
optimal learning rate.
Example usage:

```python
from jax_unirep import evotune

sequences = ["ASDF", "YJKAL", "QQLAMEHALQP", ...]

# parameters for optuna need to be passed in form of a dictionary
# the range of epochs and learning rates to try can be adapted this way
n_epochs_config = {"low": 5, "high": 10}
lr_config = {"low": 0.001, "high": 0.1}

study, evotuned_params = evotune(
    sequences=sequences,
    n_trials=20, # how many trials optuna should do
    params=None, # defaults to weights from the paper
    n_epochs_config=n_epochs_config,
    learning_rate_config=lr_config
)
```

The `study` object will contain all the information about the
training process of each trial. `evotuned_params` will contain the
fine-tuned weights from the trial with the lowest test set loss.

In order to start from randomly initialized mLSTM weights instead,
create `params` as follows:

```python
from jax_unirep.evotuning import init_fun
from jax.random import PRNGKey

_, params = init_fun(PRNGKey(0), input_shape=(-1, 10))
```

In principle, any set of weights which is compatible
with the mLSTM `init_fun` in `jax_unirep.layers` can be passed.

### UniRep stax

We implemented the mLSTM layers in such a way that
they are compatible with `jax.experimental.stax`.
This means that they can easily be plugged into
a `stax.serial` model, e.g. to train both the mLSTM
and a top-model at once:

```python
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu

from jax_unirep.layers import mLSTM190, mLSTM1900_AvgHidden

init_fun, apply_fun = stax.serial(
    mLSTM1900(),
    mLSTM1900_AvgHidden(),
    Dense(512), Relu(),
    Dense(1)
)
```

Have a look at the [documentation][stax] and [examples][staxex]
for more information about how to implement a model in `jax`.

# More Details

To read more about how we reimplemented the model in JAX, we wrote it up.
Both the [HTML](https://elarkk.github.io/jax-unirep/)
and [PDF](https://elarkk.github.io/jax-unirep/paper.pdf)
are available.

# License

All the model weights are licensed under the terms of
Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Otherwise the code in this repository
is licensed under the terms of [GPL v3][gpl3].

[arxiv]: https://www.biorxiv.org/content/10.1101/589333v1
[nature]: https://www.nature.com/articles/s41592-019-0598-1
[repo]: https://github.com/churchlab/UniRep
[ericmjl]: https://github.com/ericmjl
[fundl]: https://github.com/ericmjl/fundl
[gpl3]: https://www.gnu.org/licenses/gpl-3.0.html
[stax]: https://jax.readthedocs.io/en/latest/jax.experimental.stax.html
[staxex]: https://github.com/google/jax/tree/master/examples
