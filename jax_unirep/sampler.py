import numpy as np


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
