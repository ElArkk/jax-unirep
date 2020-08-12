# Getting Started

## Installation

Ensure that your compute environment allows you to run JAX code.
(A modern Linux or macOS with a GLIBC>=2.23 is probably necessary.)

For now, `jax-unirep` is available by pip installing from source.

Installation for the moment is done from GitHub:

```bash
pip install git+https://github.com/ElArkk/jax-unirep.git
```

On the roadmap is support for installation from PyPI and conda-forge.

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
