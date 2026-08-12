"""Tests for the model and its weight I/O."""

import pickle as pkl

import numpy as onp
import pytest
from jax.random import PRNGKey

from jax_unirep.models import (
    MLSTM,
    MLSTM_PARAM_KEYS,
    load_model,
    model_to_arrays,
    save_model,
)


@pytest.fixture
def model():
    """A small randomly initialized model, cheap enough to build per test."""
    return MLSTM(n_cells=4, output_dim=64, key=PRNGKey(0))


def assert_same_weights(actual: MLSTM, expected: MLSTM):
    """Every named array of two models is bit-for-bit identical."""
    actual_arrays = model_to_arrays(actual)
    expected_arrays = model_to_arrays(expected)

    assert actual_arrays.keys() == expected_arrays.keys()
    for key, value in expected_arrays.items():
        assert onp.array_equal(actual_arrays[key], value), key


def legacy_tree(model: MLSTM):
    """Build the parameter tree that versions up to 2.x pickled.

    The embedding, then one dict per mLSTM layer each followed by the empty
    tuple that the parameterless hidden-state layer contributed, then the
    dense pair.
    """
    tree = [model.embedding]
    for cell in model.cells:
        tree.append({key: getattr(cell, key) for key in MLSTM_PARAM_KEYS})
        tree.append(())
    tree.append((model.dense_w, model.dense_b))
    return tuple(tree)


def test_save_load_round_trip(model, tmp_path):
    """Saved weights come back off disk unchanged.

    This covers the flatten/rebuild pair, not just the writer: `save_model`
    writes the layout the shipped weights use, so a checkpoint has to be
    readable by `load_model(folderpath=...)`.
    """
    iteration_path = save_model(model, tmp_path, step=3)

    assert iteration_path == tmp_path / "iter_3"

    assert_same_weights(load_model(folderpath=str(iteration_path)), model)


def test_load_model_reads_legacy_pickle(model, tmp_path):
    """Weights dumped by v2.x are pickles and must keep loading.

    Nothing else exercises that branch, so without this it could rot silently
    and only break for users with previously saved weights.

    The warning is FutureWarning rather than DeprecationWarning because the
    latter is ignored by default outside __main__, so a library emitting one
    would be warning nobody.
    """
    (tmp_path / "model_weights.pkl").write_bytes(pkl.dumps(legacy_tree(model)))

    with pytest.warns(FutureWarning, match="legacy pickled weights"):
        loaded = load_model(folderpath=str(tmp_path))

    assert_same_weights(loaded, model)


def test_load_model_errors_when_no_weights_present(tmp_path):
    """The error must name the expected format, not just the legacy one.

    Falling through to `open(...pkl)` reports the pickle as missing, which
    misdirects anyone who pointed folderpath at the wrong directory.
    """
    with pytest.raises(FileNotFoundError, match="model_weights.npz"):
        load_model(folderpath=str(tmp_path))
