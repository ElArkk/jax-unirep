"""Tests for neural network layers."""

from functools import partial

from hypothesis import given, settings
from hypothesis import strategies as st

from jax import random, vmap
from jax.experimental import stax
from jax_unirep.layers import mLSTM, mLSTMAvgHidden, mLSTMFusion
from jax_unirep.utils import (
    get_embedding,
    load_embedding_1900,
    load_params_1900,
    validate_mLSTM_params,
)

rng = random.PRNGKey(0)


# def test_mLSTMBatch():
#     """
#     Given one fake embedded sequence,
#     ensure that we get out _an_ output from mLSTM.
#     """
#     emb = load_embedding_1900()
#     x = get_embedding("TEST", emb)

#     params = load_params_1900()

#     h_final, c_final, h = mLSTMBatch(params, x)
#     assert h.shape == (x.shape[0], 1900)


@given(st.data())
@settings(deadline=None, max_examples=20)
def test_mLSTM(data):
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
    output_dim = 256
    init_fun, apply_fun = mLSTM(output_dim=output_dim)
    output_shape, params = init_fun(rng, (-1, 10))
    _, _, outputs = vmap(partial(apply_fun, params=params))(inputs=x)
    assert output_shape == (-1, output_dim)
    assert outputs.shape == (length + 1, output_dim)


@given(st.data())
@settings(deadline=None, max_examples=20)
def test_mLSTMAvgHidden(data):
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
    output_dim = 256
    init_fun, apply_fun = stax.serial(
        mLSTM(output_dim=output_dim),
        mLSTMAvgHidden(),
    )
    output_shape, params = init_fun(rng, (length, 10))
    h_avg = apply_fun(params=params, inputs=x)
    assert output_shape == (output_dim,)
    validate_mLSTM_params(params[0], output_dim)
    assert params[1] == ()
    assert h_avg.shape == (output_dim,)


@given(st.data())
@settings(deadline=None, max_examples=20)
def test_mLSTMFusion(data):
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
    output_dim = 256
    init_fun, apply_fun = stax.serial(
        mLSTM(output_dim=output_dim),
        mLSTMFusion(),
    )
    output_shape, params = init_fun(rng, (length, 10))
    h_avg = apply_fun(params=params, inputs=x)
    assert output_shape == (output_dim * 3,)
    validate_mLSTM_params(params[0], output_dim)
    assert params[1] == ()
    assert h_avg.shape == (output_dim * 3,)
