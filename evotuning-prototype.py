from jax_unirep.evotuning import evotune
from jax_unirep.utils import load_params_1900
from jax_unirep.params import add_dense_params


sequences = ["HASTA", "VISTA", "ALAVA", "LIMED", "HAST"]

params = dict()
params["mlstm1900"] = load_params_1900()
params = add_dense_params(params, "dense", 1900, 25)

params = evotune(params, sequences, n=10)

# Diagnosis - from integers back to sequences.
