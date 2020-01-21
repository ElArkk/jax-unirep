from typing import List, Tuple

import numpy as np

from .layers import mlstm1900
from .utils import get_embeddings, load_params_1900


def get_reps(seqs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = load_params_1900()
    embedded_seqs = get_embeddings(seqs)

    h_final, c_final, h = mlstm1900(params, embedded_seqs)
    h_avg = h.mean(axis=1)

    return h_final, c_final, h_avg
