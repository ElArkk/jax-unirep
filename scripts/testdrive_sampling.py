"""
Sequence sampler test.

In this Python script, we start with a sequence,
and try to optimize for a better version of it,
as measured by the sum over reps.
It's a silly task,
but I think it gives us ability to sanity-check that we have
the right thing going.
"""

from jax_unirep import get_reps
from jax_unirep.sampler import is_accepted, propose

starting_sequence = "ASDFGHJKL"

current_sequence = starting_sequence
current_score = get_reps(current_sequence)[0].sum()
sequences = [current_sequence]
scores = [current_score]


for i in range(100):
    new_sequence = propose(current_sequence)
    new_score = get_reps(new_sequence)[0].sum()

    if is_accepted(best=current_score, candidate=new_score, temperature=1):
        current_sequence = new_sequence

    sequences.append(current_sequence)
    print(i, new_sequence, new_score)
