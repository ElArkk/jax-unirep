"""Functions for evolutionary tuning."""

import pandas as pd
from jax_unirep import evotune, fit
from jax_unirep.params import add_dense_params
from jax_unirep.utils import load_params_1900

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

# To start with Random params instead of mLSTM weights from UniRep:
"""
params = dict()
params = add_dense_params(params, "dense", 1900, 25)
params["mLSTM1900"] = load_params_1900()
params = fit(params=params, sequences=sequences, n=10)
"""

# Evotuning with Optuna

PROJECT_NAME = "temp"
n_epochs_config = {"low": 1, "high": 1}
lr_config = {"low": 1e-5, "high": 1e-3}
study, evotuned_params = evotune(
    sequences=sequences,
    params=None,
    proj_name=PROJECT_NAME,
    out_dom_seqs=holdout_sequences,
    n_trials=2,
    n_splits=2,
    n_epochs_config=n_epochs_config,
    learning_rate_config=lr_config,
    steps_per_print=1,
)

print("Evotuning done! Find output weights in", PROJECT_NAME)
print(study.trials_dataframe())


# Evotuning without Optuna
"""
N_EPOCHS = 3
LEARN_RATE = 1e-4
PROJECT_NAME = "temp"

evotuned_params = fit(
    params=None,
    sequences=sequences,
    n=N_EPOCHS,
    step_size=LEARN_RATE,
    holdout_seqs=holdout_sequences,
    proj_name=PROJECT_NAME,
    steps_per_print=1,
)

print("Evotuning done! Find output weights in", PROJECT_NAME)
"""
