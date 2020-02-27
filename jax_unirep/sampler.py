from random import choice
from typing import Dict

import numpy as np
import numpy.random as npr
from multipledispatch import dispatch

from jax_unirep.utils import proposal_valid_letters

letters_sorted = sorted(proposal_valid_letters)


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
def propose(
    sequence: str, pos_prob: np.ndarray = None, pwm: np.ndarray = None
) -> str:
    """
    Given a string, return a proposed mutated string.

    The proposed mutant is generated as follows:

    - Pick a position given the pick probabilities for each position
    - Given that position,
    pick a letter given the pick probabilities for each letter at that position
    that is not the existing letter.

    If no probabilites are specified for the positions or the pwm (position weight matrix),
    the sampler defaults to picking from an uniform distribution in both cases.

    :param sequence: The sequence to propose a new mutation on.
    :param pos_prob: Pick probability for each position in the sequence.
        Needs to be of shape (len(sequence), ),
        and probabilities need to sum to 1.
    :param pwm: Pick probability for each AA at each position.
        Needs to be of shape (len(sequence), 20),
        and probabilities need to sum to 1.
    :returns: A string.
    """
    if len(sequence) == 0:
        raise ValueError(
            "sequence passed into `propose` must be at least length 1."
        )

    if pos_prob is None:
        pos_prob = np.array([1.0 / len(sequence)] * len(sequence))
    if pwm is None:
        pwm = np.tile(np.array([[0.05] * 20]), (len(sequence), 1))

    if pos_prob.shape != (len(sequence),):
        raise ValueError(
            f"Position probability array needs to be of shape (len(sequence), ). Got shape {pos_prob.shape} instead."
        )
    if np.sum(pos_prob) != 1.0:
        raise ValueError(
            f"Position probabilities need to sum to 1. Sum is {np.sum(pos_prob)} instead."
        )
    if pwm.shape != (len(sequence), 20):
        raise ValueError(
            f"PWM needs to be of shape (len(sequence), 20). Got shape {pwm.shape} instead."
        )

    if not np.all(np.equal(np.round(np.sum(pwm, axis=1), 3), 1.0)):
        raise ValueError(
            f"PWM probabilities for each position need to sum to 1. "
            f"The following positions did not sum to 1: "
            f"{np.where(np.equal(np.round(np.sum(pwm, axis=1), 3), 1.0)==0)[0]}"
        )

    # position = choice(list(range(len(sequence))))
    position = np.argmax(npr.multinomial(1, pos_prob))
    new_sequence = ""
    for i, letter in enumerate(sequence):
        if position != i:
            new_sequence += letter
        else:
            # new_letter = choice(
            #     list(set(proposal_valid_letters).difference(letter))
            # )
            new_letter_idx = np.argmax(npr.multinomial(1, pwm[i, :]))
            new_letter = letters_sorted[new_letter_idx]
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
