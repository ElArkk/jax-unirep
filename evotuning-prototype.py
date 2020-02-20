"""Functions for evolutionary tuning."""

from jax_unirep import evotune
from jax_unirep.evotuning import fit
from jax_unirep.params import add_dense_params
from jax_unirep.utils import load_params_1900

sequences = ["HASTA", "VISTA", "ALAVA", "LIMED", "HAST"] * 3

# params = dict()
# params = add_dense_params(params, "dense", 1900, 25)
# params["mlstm1900"] = load_params_1900()
# params = fit(params=params, sequences=sequences, n=10)

n_epochs_config = {"high": 1}
evotuned_params = evotune(
    params=None,
    sequences=sequences,
    n_trials=1,
    n_epochs_config=n_epochs_config,
)

print(evotuned_params)
