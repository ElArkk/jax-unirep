import logging
from enum import Enum
from random import choice, shuffle
from typing import Optional

from Bio import SeqIO
from pyprojroot import here

import typer
from jax.random import PRNGKey
from jax_unirep import fit
from jax_unirep.utils import dump_params

app = typer.Typer()

"""Fitting mLSTM weights based on randomly initialized weights."""


class DataType(str, Enum):
    enzymes = "enzymes"
    random = "random"


class BatchMethod(str, Enum):
    random = "random"
    length = "length"


def make_string(length: int):
    return "".join(choice("ACDEFGHIKLMNPQRSTVWY") for i in range(length))


def load_enzymes():
    seqs = []
    with open(here() / "examples/enzymes.fa", "r+") as f:
        for record in SeqIO.parse(f, "fasta"):
            seqs.append(str(record.seq))
    return seqs


def load_random(num_sequences, length):
    seqs = [make_string(length) for _ in range(num_sequences)]
    return seqs


@app.command()
def main(
    kind: DataType,
    num_sequences: Optional[int] = None,
    length: Optional[int] = None,
    validation_fraction: float = None,
    backend: str = "gpu",
    n_epochs: int = 20,
    learning_rate_power: int = -3,
    mlstm_size: int = 256,
    batch_size: int = 100,
    batch_method: BatchMethod = "random",
):
    if kind == "enzymes":
        seqs = load_enzymes()
    if kind == "random":
        seqs = load_random(num_sequences, length)
    shuffle(seqs)
    break_point = int(len(seqs) * validation_fraction)
    sequences = seqs[0:break_point]
    holdout_sequences = seqs[break_point:]

    logger = logging.getLogger("fitting.py")
    logger.setLevel(logging.DEBUG)
    logger.info(f"There are {len(sequences)} sequences.")

    LEARN_RATE = 10 ** learning_rate_power
    PROJECT_NAME = "temp"

    evotuned_params = fit(
        mlstm_size=mlstm_size,
        rng=PRNGKey(42),
        params=None,
        sequences=sequences,
        n_epochs=n_epochs,
        step_size=LEARN_RATE,
        holdout_seqs=holdout_sequences,
        batch_size=batch_size,
        batch_method="random",
        proj_name=PROJECT_NAME,
        epochs_per_print=1,
        backend=backend,  # default is "cpu", can be "gpu" if you have JAX-GPU installed.
    )
    dump_params(evotuned_params, PROJECT_NAME)
    print("Evotuning done! Find output weights in", PROJECT_NAME)


if __name__ == "__main__":
    app()
