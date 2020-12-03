"""Evotuning two ways!"""

from pathlib import Path

from jax.random import PRNGKey

from jax_unirep import evotune
from jax_unirep.evotuning_models import mlstm64
from jax_unirep.utils import dump_params

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

init_fun, apply_fun = mlstm64()

# The input_shape is always going to be (-1, 26),
# because that is the number of unique AA, one-hot encoded.
_, inital_params = init_fun(PRNGKey(42), input_shape=(-1, 26))

# 1. Evotuning with Optuna
n_epochs_config = {"low": 1, "high": 1}
lr_config = {"low": 1e-5, "high": 1e-3}
study, evotuned_params = evotune(
    sequences=sequences,
    model_func=apply_fun,
    params=inital_params,
    out_dom_seqs=holdout_sequences,
    n_trials=2,
    n_splits=2,
    n_epochs_config=n_epochs_config,
    learning_rate_config=lr_config,
)

dump_params(evotuned_params, Path(PROJECT_NAME))
print("Evotuning done! Find output weights in", PROJECT_NAME)
print(study.trials_dataframe())
