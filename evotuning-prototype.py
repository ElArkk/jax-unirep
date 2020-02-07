from jax_unirep import evotune
from jax_unirep.utils import load_params_1900
from jax_unirep.params import add_dense_params


sequences = ["HASTA", "VISTA", "ALAVA", "LIMED", "HAST"]

params = dict()
params["mlstm1900"] = load_params_1900()
params = add_dense_params(params, "dense", 1900, 25)

params = evotune(params, sequences, n=10)

# from jax_unirep.evotuning import evotuning_pairs
# from jax_unirep.utils import letter_seq

# x, y = evotuning_pairs("HASTA")
# seq = letter_seq(y)
# print(seq)
# Convert y back to string


# Diagnosis - from integers back to sequences.
