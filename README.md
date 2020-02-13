[![Build Status](https://travis-ci.com/ElArkk/jax-unirep.svg?branch=master)](https://travis-ci.com/ElArkk/jax-unirep)

# jax-unirep

Reimplementation of the UniRep protein featurization model in JAX.

The UniRep model was developed in George Church's lab,
see the original publication [here][arxiv] (bioRxiv) or [here][nature] (Nature Methods),
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

To generate representations of protein sequences,
pass a list of sequences as strings or a single sequence to `jax_unirep.get_reps`.
It will return a tuple consisting of the following representations for each sequence:

- `h_avg`: Average hidden state of the mLSTM over the whole sequence.
- `h_final`: Final hidden state of the mLSTM
- `c_final`: Final cell state of the mLSTM

From the original paper, `h_avg` is considered the "representation" (or "rep") of the protein sequence.

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

# License

All the model weights are licensed under the terms of
Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Otherwise the code in this repository is licensed under the terms of [GPL v3][gpl3].

[arxiv]: https://www.biorxiv.org/content/10.1101/589333v1
[nature]: https://www.nature.com/articles/s41592-019-0598-1
[repo]: https://github.com/churchlab/UniRep
[ericmjl]: https://github.com/ericmjl
[fundl]: https://github.com/ericmjl/fundl
[gpl3]: https://www.gnu.org/licenses/gpl-3.0.html
