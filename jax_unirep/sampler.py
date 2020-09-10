from collections import defaultdict
from random import choice
from typing import Callable, Dict

import numpy as np
import numpy.random as npr
from multipledispatch import dispatch
from tqdm.autonotebook import tqdm

from jax_unirep.utils import proposal_valid_letters

letters_sorted = sorted(proposal_valid_letters)

aa_dict = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
}


def is_accepted(best: float, candidate: float, temperature: float) -> bool:
    """
    Return boolean decision on whether the candidate mutant is accepted or not.

    This function checks whether we want to
    accept a new mutant proposed by our MMC sampler,
    by comparing its predicted activity
    to the current best mutants activity.

    :param best: Predicted activity of current best mutant
    :param candidate: Predicted activity of new candidate mutant
    :param temperature: Boltzmann distribution temperature.
        Controls acceptance probability.
        Low T decreases acceptance probability.
        High T increases acceptance probability.
    :returns bool: Whether or not candidate mutant was accepted
    """

    c = np.exp((candidate - best) / temperature)

    if c > 1:
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
        and probabilities at each position need to sum to 1.
        AA's need to be sorted alphabetically.
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
    # 28 February 2020: Temporarily delegating probability vector checks
    # to NumPy's random multinomial instead.
    # if np.round(np.sum(pos_prob), 3) != 1.0:
    #     raise ValueError(
    #         f"Position probabilities need to sum to 1. Sum is {np.sum(pos_prob)} instead."
    #     )
    if pwm.shape != (len(sequence), 20):
        raise ValueError(
            f"PWM needs to be of shape (len(sequence), 20). Got shape {pwm.shape} instead."
        )

    # 28 February 2020: Temporarily delegating probability vector checks
    # to NumPy's random multinomial instead.
    # if not np.all(np.equal(np.round(np.sum(pwm, axis=1), 3), 1.0)):
    #     raise ValueError(
    #         f"PWM probabilities for each position need to sum to 1. "
    #        f"The following positions did not sum to 1: "
    #          f"{np.where(np.equal(np.round(np.sum(pwm, axis=1), 3), 1.0)==0)[0]}"
    #     )

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
            letter_idx = aa_dict[letter] - 1
            pwm[i, letter_idx] = 0
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
    # pass
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


def hamming_distance(s1: str, s2: str):
    """Return hamming distance between two strings of the same length."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def sample_one_chain(
    starter_sequence: str,
    n_steps: int,
    scoring_func: Callable,
    is_accepted_kwargs: Dict = {},
    trust_radius: int = 7,
    propose_kwargs: Dict = {},
) -> Dict:
    """
    Return one chain of MCMC samples of new sequences.

    Given a `starter_sequence`,
    this function will sample one chain of protein sequences,
    scored using a user-provided `scoring_func`.

    Design choices made here include the following.

    Firstly, we record all sequences that were sampled,
    and not just the accepted ones.
    This behaviour differs from other MCMC samplers
    that record only the accepted values.
    We do this just in case sequences that are still "good"
    (but not better than current) are rejected.
    The effect here is that we get a cluster of sequences
    that are one-apart from newly accepted sequences.

    Secondly, we check the Hamming distance
    between the newly proposed sequences and the original.
    This corresponds to the "trust radius"
    specified in the [jax-unirep paper](https://doi.org/10.1101/2020.01.23.917682).
    If the hamming distance > trust radius,
    we reject the sequence outright.

    A dictionary containing the following key-value pairs are returned:

    - "sequences": All proposed sequences.
    - "scores": All scores from the scoring function.
    - "accept": Whether the sequence was accepted as the new 'current sequence'
        on which new sequences are proposed.

    This can be turned into a pandas DataFrame.

    ### Parameters

    - `starter_sequence`: The starting sequence.
    - `n_steps`: Number of steps for the MC chain to walk.
    - `scoring_func`: Scoring function for a new sequence.
        It should only accept a string `sequence`.
    - `is_accepted_kwargs`: Dictionary of kwargs to pass into
        `is_accepted` function.
        See `is_accepted` docstring for more details.
    - `trust_radius`: Maximum allowed number of mutations away from
        starter sequence.
    - `propose_kwargs`: Dictionary of kwargs to pass into
        `propose` function.
        See `propose` docstring for more details.
    - `verbose`: Whether or not to print iteration number
        and associated sequence + score. Defaults to False

    ### Returns

    A dictionary with `sequences`, `accept` and `score` as keys.
    """
    current_sequence = starter_sequence
    current_score = scoring_func(sequence=starter_sequence)

    chain_data = defaultdict(list)
    chain_data["sequences"].append(current_sequence)
    chain_data["scores"].append(current_score)
    chain_data["accept"].append(True)

    for i in tqdm(range(n_steps)):
        new_sequence = propose(current_sequence, **propose_kwargs)
        new_score = scoring_func(sequence=new_sequence)

        default_is_accepted_kwargs = {"temperature": 0.1}
        default_is_accepted_kwargs.update(is_accepted_kwargs)
        accept = is_accepted(
            best=current_score,
            candidate=new_score,
            **default_is_accepted_kwargs,
        )

        # Check hamming distance
        if hamming_distance(starter_sequence, new_sequence) > trust_radius:
            accept = False

        # Determine acceptance
        if accept:
            current_sequence = new_sequence
            current_score = new_score

        # Record data.
        chain_data["sequences"].append(new_sequence)
        chain_data["scores"].append(new_score)
        chain_data["accept"].append(accept)
    chain_data["scores"] = np.hstack(chain_data["scores"])
    return chain_data
