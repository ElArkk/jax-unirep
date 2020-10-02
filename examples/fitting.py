import logging
from random import shuffle

from Bio import SeqIO
from pyprojroot import here

from jax_unirep import fit
from jax_unirep.utils import dump_params
from jax.random import PRNGKey

"""Fitting mLSTM weights based on randomly initialized weights."""


seqs = []
with open(here() / "examples/enzymes.fa", "r+") as f:
    for record in SeqIO.parse(f, "fasta"):
        seqs.append(str(record.seq))


seqs = seqs[:50]
shuffle(seqs)
break_point = int(len(seqs) * 0.7)
sequences = seqs[0:break_point]
holdout_sequences = seqs[break_point:]


logger = logging.getLogger("fitting.py")
logger.setLevel(logging.DEBUG)
logger.info(f"There are {len(sequences)} sequences.")

N_EPOCHS = 20
LEARN_RATE = 1e-5
PROJECT_NAME = "temp"

evotuned_params = fit(
    mlstm_size=256,
    rng=PRNGKey(42),
    params=None,
    sequences=sequences,
    n_epochs=N_EPOCHS,
    step_size=LEARN_RATE,
    holdout_seqs=holdout_sequences,
    batch_size=2000,
    batch_method="random",
    proj_name=PROJECT_NAME,
    epochs_per_print=1,
    backend="cpu",  # default is "cpu", can be "gpu" if you have JAX-GPU installed.
)
dump_params(evotuned_params, PROJECT_NAME)
print("Evotuning done! Find output weights in", PROJECT_NAME)
