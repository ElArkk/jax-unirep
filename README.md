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
There are two functions with differing levels of
control available for the task.

The [`evotune`][evotunefunc] function uses `optuna` under the hood, to automatically
find the optimal number of epochs to train for given a set of sequences,
as well as the optimal learning rate. 
The `study` object will contain all the information about the
training process of each trial. 
`evotuned_params` will contain the fine-tuned mLSTM and dense weights 
from the trial with the lowest test set loss.

If you want to directly fine-tune the weights for a fixed number
of epochs and with a fixed learning rate, then you should use
the [`fit`][fitfunc] function.

You can find an example usage of both functions [here][exampleevotune]

If you want to pass a set of mLSTM and dense weights that were
dumped in an earlier run, create params as follows:

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

The weights used in the 10-dimensional embedding
of the input sequences always default to the 
weights from the paper, since they do not
get updated during evotuning.

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
[evotunefunc]: https://github.com/ElArkk/jax-unirep/blob/master/jax_unirep/evotuning.py#L417
[fitfunc]: https://github.com/ElArkk/jax-unirep/blob/master/jax_unirep/evotuning.py#L164
[exampleevotune]: https://github.com/ElArkk/jax-unirep/blob/master/examples/evotuning.py
[stax]: https://jax.readthedocs.io/en/latest/jax.experimental.stax.html
[staxex]: https://github.com/google/jax/tree/master/examples
