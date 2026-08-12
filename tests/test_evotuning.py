import pytest
from jax.random import PRNGKey

from jax_unirep.evotuning import evotune, fit
from jax_unirep.models import MLSTM, load_model

"""Evolutionary tuning function tests."""


@pytest.fixture
def model():
    """A small, randomly initialized mLSTM."""
    return MLSTM(n_cells=4, output_dim=64, key=PRNGKey(0))


@pytest.mark.parametrize("holdout_seqs", (["ASDV", None]))
@pytest.mark.parametrize("batch_method", (["length", "random"]))
def test_fit(model, holdout_seqs, batch_method, tmp_path):
    """Execution test for ``jax_unirep.evotuning.fit``."""
    sequences = ["ASDFGHJKL", "ASDYGHTKW", "HSKS", "HSGL", "ER"]

    tuned_model = fit(
        model=model,
        sequences=sequences,
        n_epochs=1,
        batch_method=batch_method,
        batch_size=2,
        holdout_seqs=holdout_seqs,
        proj_name=str(tmp_path),
    )

    # The architecture is preserved and the weights actually moved.
    assert tuned_model.output_dim == model.output_dim
    assert len(tuned_model.cells) == len(model.cells)
    assert not (tuned_model.cells[0].wx == model.cells[0].wx).all()

    # Weights get dumped in the layout `load_model` reads back.
    dumped = load_model(folderpath=tmp_path / "iter_0")
    assert (dumped.cells[0].wx == model.cells[0].wx).all()


@pytest.mark.slow
def test_fit_defaults(tmp_path):
    """``fit`` defaults to tuning the pre-trained mLSTM1900."""
    sequences = ["ASDFGHJKL", "ASDYGHTKW", "HSKS", "HSGL", "ER"]

    tuned_model = fit(
        sequences=sequences,
        n_epochs=1,
        batch_size=2,
        proj_name=str(tmp_path),
    )

    assert tuned_model.output_dim == 1900
    assert len(tuned_model.cells) == 1


@pytest.mark.slow
def test_evotune(model):
    """Simple execution test for evotune."""
    seqs = ["MTN", "BDD"] * 5
    n_epochs_config = {"high": 1}

    _, _ = evotune(
        sequences=seqs,
        model=model,
        n_trials=1,
        n_epochs_config=n_epochs_config,
    )
    # now test using all defaults
    _, _ = evotune(
        sequences=seqs,
        n_trials=1,
        n_epochs_config=n_epochs_config,
    )
