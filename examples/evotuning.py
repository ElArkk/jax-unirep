"""Evotuning with Optuna."""

from jax.random import PRNGKey

from jax_unirep import evotune
from jax_unirep.models import MLSTM, save_model

# Test sequences:
sequences = ["HASTA", "VISTA", "ALAVA", "LIMED", "HAST", "HAS", "HASVASTA"] * 5
holdout_sequences = [
    "HASTA",
    "VISTA",
    "ALAVA",
    "LIMED",
    "HAST",
    "HASVALTA",
] * 5
PROJECT_NAME = "evotuning_temp"

# Start from randomly initialized weights of any size. To start from the
# pre-trained paper weights instead, pass no `model` at all, or
# `load_model(paper_weights=64)` / `load_model(folderpath=...)` to resume.
model = MLSTM(n_cells=4, output_dim=64, key=PRNGKey(42))

n_epochs_config = {"low": 1, "high": 1}
lr_config = {"low": 1e-5, "high": 1e-3}
study, tuned_model = evotune(
    sequences=sequences,
    model=model,
    out_dom_seqs=holdout_sequences,
    n_trials=2,
    n_splits=2,
    n_epochs_config=n_epochs_config,
    learning_rate_config=lr_config,
)

save_model(tuned_model, PROJECT_NAME)
print("Evotuning done! Find output weights in", PROJECT_NAME)
print(study.trials_dataframe())
