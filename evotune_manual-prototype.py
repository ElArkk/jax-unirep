"""Functions for evolutionary tuning."""

from jax_unirep import evotune_manual

PROJECT_NAME = "temp"

train_sequences = ["HASTA", "VISTA", "ALAVA", "LIMED", "HAST"] * 3
in_val_sequences = ["HASTE", "VISTE", "ALAVE", "LIMAD", "HEST"] * 3
out_val_sequences = ["HASTO", "VISTO", "ALAVO", "LIMOD", "HOST"] * 3


# CREATE OUTPUT FOLDER BEFORE RUNNING THIS SCRIPT.
# currently hardcoded to output to folder jax_unirep/weights/temp

evotuned_params = evotune_manual(
    train_seqs=train_sequences,
    in_val_seqs=in_val_sequences,
    out_val_seqs=out_val_sequences,
    proj_name=PROJECT_NAME,
    params=None,
    n_epochs=1,
    learning_rate=0.00001,
    steps_per_print=1,
)

print(
    "Evotuning done! Find output weights in 'jax_unirep/weights/"
    + PROJECT_NAME
)
