from typing import List, Tuple

import numpy as np

from .layers import mlstm1900
from .utils import get_embeddings, load_params_1900


def get_reps(seqs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function generates representations of protein sequences using the 
    1900 hidden-unit mLSTM model with pre-trained weights from the UniRep
    paper (https://github.com/churchlab/UniRep).

    All input sequences need to be of the same length!

    Each element of the output tuple is a np.array
    of shape (n_sequences, 1900).


    :param seqs: A list of same length sequences as strings.
        If passing only a single sequence, it also needs to be passed inside a list.
    :returns: A tuple of np.arrays containing the reps.
        Each np.array has shape (n_sequences, 1900).
    """

    params = load_params_1900()
    embedded_seqs = get_embeddings(seqs)

    h_final, c_final, h = mlstm1900(params, embedded_seqs)
    h_avg = h.mean(axis=1)

    return h_final, c_final, h_avg
