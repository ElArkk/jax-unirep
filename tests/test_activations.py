"""Tests for activation functions.

In particular, we are looking for tests that cause NaN errors in grads.
"""
import jax.numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import grad

from jax_unirep.activations import sigmoid


@pytest.mark.parametrize("version", ["tanh", "exp"])
@given(x=st.floats(allow_nan=False, allow_infinity=False))
@settings(deadline=None)
def test_sigmoid(x, version):
    """Check for null gradient issues."""
    result = sigmoid(x, version=version)
    assert not np.isnan(result)

    dsigmoid = grad(sigmoid)
    dresult = dsigmoid(x, version=version)
    assert not np.isnan(dresult)
