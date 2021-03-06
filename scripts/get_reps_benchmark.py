from pprint import pprint
from random import choice
from time import time

import numpy as np

from jax_unirep.featurize import get_reps
from jax_unirep.utils import aa_to_int


def generate_sequence(length: int):
    alphabet = list(set(aa_to_int.keys()).difference(["start", "stop"]))

    return "".join(choice(alphabet) for i in range(length))


sequences = dict()
for n in [10, 100, 1000, 10000]:  # number of sequences
    sequences[n] = [generate_sequence(50) for i in range(n)]


timings = dict()
reps = dict()
for n, seqs in sequences.items():
    print(f"Processing {n} sequences...")
    start = time()
    _, _, out = get_reps(seqs)
    reps[n] = np.array(out)
    timings[n] = time() - start


pprint(timings)
