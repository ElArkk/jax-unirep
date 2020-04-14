"""Functions for evolutionary tuning."""

from jax_unirep import evotune
from jax_unirep.evotuning import fit
from jax_unirep.params import add_dense_params
from jax_unirep.utils import load_params_1900

sequences = ["HASTA", "VISTA", "ALAVA", "LIMED", "HAST"] * 3
long_seqs = [
    "AJBGAJHLVSJHLVDJLGDJKGHDKJDGBFGFJKHFBKJHDBDHKJGDHJDGKLGLKJSGSHDGVHJDBDKLJFKJFHDJGDHKJLSGHJSG",
    "HDKJLFGHLKJFGFKLHJGFDHJDGHJDFGJKDFGDJKDGLHJKDGDJKLHDKJDHJKDGDLHJKGDLJHDKDLDJKDHJDJHGDJKGSDJG",
    "JDHJKDHJKDHJKDGDKJLGDJHDGDHDGKJDHGKJDLHDKJHFKJLHFJKDJHIDHDIOUDHODHUKDUHDHDUHDKUHDKUDHKDUKKKK",
] * 5

holdout_seqs = [
    "AJBGAJHLVSJHLVDJLGDAKGHDKJDGBFGFJKHFBKJHDBDHKJGDHJDGKLGLKJSGSHDGVHJDBDKLAAKJFHDJGDHKJLSGHJSG",
    "HDKJLFGHLKJFGFKLHJGFDHJDGHJDFGJKDFGDJKDGLHJKDGDAAAHDKJDHJKDGDLHJKGDLJHDKDLDJKDHJDJHGDJKGSDJG",
] * 2

# params = dict()
# params = add_dense_params(params, "dense", 1900, 25)
# params["mLSTM1900"] = load_params_1900()
# params = fit(params=params, sequences=sequences, n=10)

n_epochs_config = {"low": 1, "high": 20}
lr_config = {"low": 0.00001, "high": 0.01}
evotuned_params = evotune(
    sequences=long_seqs,
    params=None,
    proj_name="temp",
    out_dom_seqs=holdout_seqs,
    n_trials=20,
    n_epochs_config=n_epochs_config,
    learning_rate_config=lr_config,
    steps_per_print=1,
)

print("Evotuning done! Find output weights in" + PROJECT_NAME)
