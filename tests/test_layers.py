import numpy as np
import numpy.random as npr
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import random
from jax.experimental import stax

from jax_unirep.layers import (  # mLSTM1900_step,
    mLSTM1900,
    mLSTM1900_AvgHidden,
    mLSTM1900_batch,
    mLSTM1900_Fusion,
)
from jax_unirep.utils import (
    get_embedding,
    get_embeddings,
    load_embedding_1900,
    load_params_1900,
)

rng = random.PRNGKey(0)


def test_mLSTM1900_batch():
    """
    Given one fake embedded sequence,
    ensure that we get out _an_ output from mLSTM1900.
    """
    emb = load_embedding_1900()
    x = get_embedding("TEST", emb)

    params = load_params_1900()
    original_params = params.copy()

    h_final, c_final, h = mLSTM1900_batch(params, x)
    assert h.shape == (x.shape[0], 1900)
    np.testing.assert_array_equal(original_params, params)


def validate_mLSTM1900_params(params):
    assert params["wmx"].shape == (10, 1900)
    assert params["wmh"].shape == (1900, 1900)
    assert params["wx"].shape == (10, 7600)
    assert params["wh"].shape == (1900, 7600)
    assert params["gmx"].shape == (1900,)
    assert params["gmh"].shape == (1900,)
    assert params["gx"].shape == (7600,)
    assert params["gh"].shape == (7600,)
    assert params["b"].shape == (7600,)
    return None


@given(st.data())
@settings(deadline=None, max_examples=20)
def test_mLSTM1900(data):
    params = load_params_1900()
    length = data.draw(st.integers(min_value=1, max_value=10))
    sequence = data.draw(
        st.text(
            alphabet="MRHKDESTNQCUGPAVIFYWLOXZBJ",
            min_size=length,
            max_size=length,
        ),
    )
    embedding = load_embedding_1900()
    x = get_embedding(sequence, embedding)
    init_fun, apply_fun = mLSTM1900(output_dim=1900)
    output_shape, params = init_fun(rng, (length, 10))
    h_final, c_final, outputs = apply_fun(params=params, inputs=x)
    assert output_shape == (length, 1900)
    validate_mLSTM1900_params(params)
    assert outputs.shape == (length + 1, 1900)


@given(st.data())
@settings(deadline=None, max_examples=20)
def test_mLSTM1900_AvgHidden(data):
    params = load_params_1900()
    length = data.draw(st.integers(min_value=1, max_value=10))
    sequence = data.draw(
        st.text(
            alphabet="MRHKDESTNQCUGPAVIFYWLOXZBJ",
            min_size=length,
            max_size=length,
        ),
    )
    embedding = load_embedding_1900()
    x = get_embedding(sequence, embedding)
    init_fun, apply_fun = stax.serial(
        mLSTM1900(output_dim=1900),
        mLSTM1900_AvgHidden(output_dim=1900),
    )
    output_shape, params = init_fun(rng, (length, 10))
    h_avg = apply_fun(params=params, inputs=x)
    assert output_shape == (1900,)
    validate_mLSTM1900_params(params[0])
    assert params[1] == ()
    assert h_avg.shape == (1900,)


@given(st.data())
@settings(deadline=None, max_examples=20)
def test_mLSTM1900_Fusion(data):
    params = load_params_1900()
    length = data.draw(st.integers(min_value=1, max_value=10))
    sequence = data.draw(
        st.text(
            alphabet="MRHKDESTNQCUGPAVIFYWLOXZBJ",
            min_size=length,
            max_size=length,
        ),
    )
    embedding = load_embedding_1900()
    x = get_embedding(sequence, embedding)
    init_fun, apply_fun = stax.serial(
        mLSTM1900(output_dim=1900),
        mLSTM1900_Fusion(output_dim=5700),
    )
    output_shape, params = init_fun(rng, (length, 10))
    h_avg = apply_fun(params=params, inputs=x)
    assert output_shape == (5700,)
    validate_mLSTM1900_params(params[0])
    assert params[1] == ()
    assert h_avg.shape == (5700,)
