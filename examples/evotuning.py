"""Evotuning two ways!"""
from functools import partial

from jax.random import PRNGKey

from jax_unirep import evotune, fit
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
fit()
fit_func = partial(
    fit,
)

# 1. Evotuning with Optuna
n_epochs_config = {"low": 1, "high": 1}
lr_config = {"low": 1e-5, "high": 1e-3}
study, evotuned_params = evotune(
    sequences=sequences,
    fit_func=fit_func,
    out_dom_seqs=holdout_sequences,
    n_trials=2,
    n_splits=2,
    n_epochs_config=n_epochs_config,
    learning_rate_config=lr_config,
)

dump_params(evotuned_params, PROJECT_NAME)
print("Evotuning done! Find output weights in", PROJECT_NAME)
print(study.trials_dataframe())
