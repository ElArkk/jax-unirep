"""Evotuning two ways."""
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

## 1. Evotuning with Optuna
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

dump_params(evotuned_params, PROJECT_NAME)
print("Evotuning done! Find output weights in", PROJECT_NAME)
print(study.trials_dataframe())


## 2. Evotuning without Optuna

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

dump_params(evotuned_params, PROJECT_NAME)
print("Evotuning done! Find output weights in", PROJECT_NAME)
