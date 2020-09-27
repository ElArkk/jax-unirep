from pathlib import Path
import pandas as pd
import pickle as pkl
from random import shuffle
from time import time
import os

print("Opening pickle...")
start = time()
with open(Path.home() / "uniref/uniref50_preprocessed.pkl", "rb") as f:
    sequences = pkl.load(f)
end = time()
print(f"Time taken: {end - start:.2f} seconds")

print("Splitting sequences...")
start = time()
# shuffle(sequences)
break_point = int(len(sequences) * 0.99)
training_sequences = sequences[0:break_point]
holdout_sequences = sequences[break_point:]
end = time()
print(f"Time taken: {end - start:.2f} seconds")

# Set some evotuning parameters.
N_EPOCHS = 1  # probably want this to be quite high, like in the hundreds.
LEARN_RATE = 1e-5  # this is a very sane default to start with.
PROJECT_NAME = "uniref_retrain"  # where the weights will be dumped
BATCH_SIZE = 20

from jax_unirep.utils import load_random_evotuning_params
from jax_unirep import fit
from jax_unirep.utils import right_pad

# Pre-load some evotuning params that are randomly initialized.
params = load_random_evotuning_params()

print("Evotuning...")

# Now to evotuning
evotuned_params = fit(
    params=params,  # you can also set this to None if you want to use the published weights as the starting point.
    sequences=training_sequences,
    n_epochs=N_EPOCHS,
    step_size=LEARN_RATE,
    batch_size=BATCH_SIZE,
    holdout_seqs=holdout_sequences,
    seq_max_length=None,
    proj_name=PROJECT_NAME,
    epochs_per_print=1,  # also controls how often weights are dumped.
    backend="gpu",  # default is "cpu", can also set to "gpu" if you have it.
)

dump_params(evotuned_params, PROJECT_NAME)
print("Evotuning done! Find output weights in", PROJECT_NAME)
