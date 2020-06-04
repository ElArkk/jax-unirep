"""Fitting on a bunch of sequences."""
import logging
from random import shuffle

from Bio import SeqIO
from pyprojroot import here

from jax_unirep import fit
from jax_unirep.utils import dump_params


seqs = []
with open(here() / "examples/ired.fa", "r+") as f:
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
    params=None,
    sequences=sequences,
    n_epochs=N_EPOCHS,
    step_size=LEARN_RATE,
    holdout_seqs=holdout_sequences,
    batch_method="random",
    proj_name=PROJECT_NAME,
    steps_per_print=None,
)

dump_params(evotuned_params, PROJECT_NAME)
print("Evotuning done! Find output weights in", PROJECT_NAME)
