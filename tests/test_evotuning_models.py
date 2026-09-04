import pytest
from jax.random import PRNGKey

from jax_unirep.models import MLSTM
from jax_unirep.utils import seq_to_oh

"""Tests of the evotuning readout of the paper model architectures.

`model.logits` is the next-amino-acid head used for evotuning, as opposed to
the pooled hidden states `get_reps` uses. Same weights, different readout.
"""


@pytest.mark.parametrize("n_cells, output_dim", [(1, 1900), (4, 256), (4, 64)])
def test_logits_shape(n_cells, output_dim):
    """Every paper architecture predicts one of 25 classes per position."""
    model = MLSTM(n_cells=n_cells, output_dim=output_dim, key=PRNGKey(42))

    logits = model.logits(seq_to_oh("HASTA"))

    assert logits.shape == (7, 25)


def test_logits_without_head():
    """A model built without a head says so, rather than failing cryptically."""
    model = MLSTM(n_cells=1, output_dim=64, key=PRNGKey(42), with_head=False)

    with pytest.raises(ValueError, match="no dense head"):
        model.logits(seq_to_oh("HASTA"))
