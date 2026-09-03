# Getting Started

## Installation

Ensure that your compute environment allows you to run JAX code.
(A modern Linux or macOS with a GLIBC>=2.23 is probably necessary.)

Then, install from PyPI:

```bash
pip install jax-unirep
```

If you want to run on a GPU, install a CUDA-enabled JAX alongside it:

```bash
pip install -U "jax[cuda12]"
```

## Basic Usage

The core activity with using UniRep
is to produce fixed-length representations of protein sequences.
This is done by using the `get_reps()` function.
You can "rep" a single sequence:

```python
from jax_unirep import get_reps

sequence = "ASDFGHJKL"

# h_avg is the canonical "reps"
h_avg, h_final, c_final = get_reps(sequence)
```

Or you can "rep" a bunch of sequences together:

```python
from jax_unirep import get_reps

sequences = ["ASDF", "YJKAL", "QQLAMEHALQP"]

# h_avg is the canonical "reps"
h_avg, h_final, c_final= get_reps(sequences)
# each of the arrays will be of shape (len(sequences), 1900),
# with the correct order of sequences preserved
```

Canonically, you would use `h_avg` as the "reps".

## UniRep Fusion

The [original paper][unirep] also defines "UniRep Fusion": the three
representations concatenated into one 5,700-dimensional vector, used for the
supervised stability and function prediction tasks. `get_reps` returns exactly
those three, in that order, so building it is one line:

```python
from jax_unirep import fusion_reps

reps = fusion_reps(["HASTA", "VISTA"])   # (n_sequences, 5700)
```

The components come back in the paper's order: average hidden, final hidden,
final cell. If you are fine-tuning rather than featurizing, do not use this --
concatenate inside your own `equinox.Module` so that gradients reach the
mLSTM. See [End-to-end differentiable models](advanced.md).

[unirep]: https://www.nature.com/articles/s41592-019-0598-1
