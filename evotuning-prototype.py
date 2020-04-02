"""Functions for evolutionary tuning."""

from jax_unirep import evotune, evotune_manual
from jax_unirep.evotuning import fit
from jax_unirep.params import add_dense_params
from jax_unirep.utils import load_params_1900

sequences = ["HASTA", "VISTA", "ALAVA", "LIMED", "HAST"] * 3
long_seqs = [
    "AJBGAJHLVSJHLVDJLGDJKGHDKJDGBFGFJKHFBKJHDBDHKJGDHJDGKLGLKJSGSHDGVHJDBDKLJFKJFHDJGDHKJLSGHJSG",
    "HDKJLFGHLKJFGFKLHJGFDHJDGHJDFGJKDFGDJKDGLHJKDGDJKLHDKJDHJKDGDLHJKGDLJHDKDLDJKDHJDJHGDJKGSDJG",
    "JDHJKDHJKDHJKDGDKJLGDJHDGDHDGKJDHGKJDLHDKJHFKJLHFJKDJHIDHDIOUDHODHUKDUHDHDUHDKUHDKUDHKDUKKKK",
] * 5

# params = dict()
# params = add_dense_params(params, "dense", 1900, 25)
# params["mLSTM1900"] = load_params_1900()
# params = fit(params=params, sequences=sequences, n=10)

n_epochs_config = {"high": 1}
lr_config = {"low": 0.1, "high": 0.5}
evotuned_params = evotune(
    params=None,
    sequences=long_seqs,
    n_trials=2,
    n_epochs_config=n_epochs_config,
    learning_rate_config=lr_config,
)

print(evotuned_params)
