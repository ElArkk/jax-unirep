# Example workflow using `fit`

This page walks through using `fit` to evotune a UniRep model.

## Sequences

Prepare your sequences as a plain **list of strings**. In a real use case you
would load these from a FASTA file or a database.

```python
sequences = ["HASTA", "VISTA", "ALAVA", "LIMED", "HAST", "HAS", "HASVASTA"] * 5
holdout_sequences = ["HASTA", "VISTA", "ALAVA", "LIMED", "HAST", "HASVALTA"] * 5
```

## Example 1: the shipped mLSTM1900 weights

The fastest way to get going. Nothing is required except the sequences and a
number of epochs -- `fit` loads the pre-trained 1900 model for you.

```python
from jax_unirep import fit

tuned_model = fit(sequences, n_epochs=2)
```

`fit` returns an `MLSTM`. Save it with `save_model`, and read it back with
`load_model(folderpath=...)`.

## Example 2: a different published architecture

The paper's three architectures are all shipped: 1900, 256 and 64. The 1900
model is a single mLSTM cell; the 256 and 64 models stack four.

```python
from jax.random import PRNGKey

from jax_unirep import MLSTM, fit, load_model

# Either start from the paper's pre-trained weights...
model = load_model(paper_weights=64)

# ...or from reproducibly random ones.
model = MLSTM(n_cells=4, output_dim=64, key=PRNGKey(42))

tuned_model = fit(sequences, n_epochs=2, model=model)
```

The 256 model usually performs better than the 64 at the price of longer
training time.

!!! warning "Random initialisation is a cold start"

    A randomly initialised `MLSTM` returns nearly the same representation for
    every sequence. The weight-normalisation gains start near zero, so the
    gates sit at `sigmoid(0)` and the cell barely responds to its input. It
    trains out of this eventually, but starting from `load_model` converges
    far faster.

## Example 3: your own architecture

The model carries its own shape, so trying a different one is a constructor
call rather than an assembly of layers. Here are two stacked mLSTMs of 512
units each, on a 20-dimensional amino acid embedding instead of the default 10:

```python
from jax.random import PRNGKey

from jax_unirep import MLSTM, fit

model = MLSTM(
    n_cells=2,
    output_dim=512,
    key=PRNGKey(42),
    embedding_dim=20,
)

tuned_model = fit(sequences, n_epochs=2, model=model)
```

## Holdout sequences

Pass `holdout_seqs` to have `fit` report loss on a set it never trains on,
which is how you spot overfitting:

```python
tuned_model = fit(
    sequences,
    n_epochs=2,
    holdout_seqs=holdout_sequences,
)
```

Obviously, you would swap in your own sequences and train for considerably
longer. See the [API docs for `fit`][fitdoc] for batching strategies, learning
rate, and checkpointing options.

[fitdoc]: https://elarkk.github.io/jax-unirep/api/#jax_unirepfit
