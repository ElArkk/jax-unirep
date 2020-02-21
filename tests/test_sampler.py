import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_unirep.sampler import check_acceptance


@given(st.data())
@settings(deadline=None, max_examples=100)
def test_check_acceptance(data):
    best = data.draw(
        st.floats(
            min_value=0, max_value=1e2, allow_nan=False, allow_infinity=False
        )
    )
    candidate = data.draw(st.floats(allow_nan=False, allow_infinity=False))
    accept = check_acceptance(best=best, candidate=candidate)
    assert isinstance(accept, bool)
