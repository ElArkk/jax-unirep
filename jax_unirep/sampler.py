from random import choice

import numpy as np
from multipledispatch import dispatch

from jax_unirep.utils import proposal_valid_letters


def is_accepted(best: float, candidate: float) -> bool:
    """
    Return boolean decision on whether the candidate mutant is accepted or not.

    This function checks whether we want to 
    accept a new mutant proposed by our MMC sampler,
    by comparing its predicted activity 
    to the current best mutants activity.

    :param best: Predicted activity of current best mutant
    :param candidate: Predicted activity of new candidate mutant
    :returns bool: Whether or not candidate mutant was accepted
    """
    if candidate < 1e-5:
        return False
    if candidate > 1e2:
        return True

    c = np.exp(candidate) / np.exp(best)

    if c >= 1:
        return True
    else:
        p = np.random.uniform(0, 1)
        if c >= p:
            return True
        else:
            return False


@dispatch(str)
def propose(sequence: str) -> str:
    """
    Given a string, return a proposed mutated string.

    The proposed mutant is generated as follows:

    - Uniformly pick a position
    - Given that position,
    uniformly pick a letter that is not the existing letter.

    :param sequence: The sequence to propose a new mutation on.
    :returns: A string.
    """
    if len(sequence) == 0:
        raise ValueError(
            "sequence passed into `propose` must be at least length 1."
        )
    position = choice(list(range(len(sequence))))
    new_sequence = ""
    for i, letter in enumerate(sequence):
        if position != i:
            new_sequence += letter
        else:
            new_letter = choice(
                list(set(proposal_valid_letters).difference(letter))
            )
            new_sequence += new_letter
    return new_sequence


@dispatch(np.ndarray)
def propose(sequence: np.ndarray) -> np.ndarray:
    """
    Given a bit-matrix, return a mutated bit-matrix.

    This function proposes a new protein sequence.
    The protein sequence in this particular case
    is represented as a bit matrix.

    The input bit-matrix should be of shape (N, 25),
    where N = number of positions, and 25 is the number of possible letters
    that UniRep's embedder hand handle.

    TODO when we have some bandwidth.
    """
    pass
    # pos = choice(list(range(len(sequence))))
    # existing_vector = sequence[pos, :]

    # indexing_vectors = np.eye(26)
    # valid_indices = [
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    #     6,
    #     7,
    #     8,
    #     9,
    #     10,
    #     11,
    #     13,
    #     14,
    #     15,
    #     16,
    #     17,
    #     18,
    #     19,
    #     20,
    #     21,
    # ]
    # index = choice(valid_indices)
    # # Get the index of the existing_vector so that we can choose another index
    # # to sample out.
    # np.where(np.equal(indexing_vectors, existing_vector).all(axis=1))
