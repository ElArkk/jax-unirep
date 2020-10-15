"""Two ways to use the `fit` function."""
from jax_unirep import fit
from jax_unirep.evotuning_models import mlstm64
from jax.random import PRNGKey

sequences = ["HASTA", "VISTA", "ALAVA", "LIMED", "HAST", "HAS", "HASVASTA"] * 5
holdout_sequences = [
    "HASTA",
    "VISTA",
    "ALAVA",
    "LIMED",
    "HAST",
    "HASVALTA",
] * 5


# First way: Use the default mLSTM1900 weights with mLSTM1900 model.

tuned_params = fit(sequences, n_epochs=2)

# Second way: Use one of the pre-built evotuning models.

# In this example, we use the mLSTM64 model.
init_func, apply_func = mlstm64()
# The init_func always requires a PRNGKey,
# and input_shape should be set to (-1, 10)
_, params = init_func(PRNGKey(42), input_shape=(-1, 10))
tuned_params = fit(sequences, n_epochs=2, model_func=apply_func, params=params)
