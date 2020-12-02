from hypothesis import given, settings
from hypothesis import strategies as st
from jax import random
from jax.experimental import stax

from jax_unirep.layers import AAEmbedding, mLSTM, mLSTMAvgHidden, mLSTMFusion
from jax_unirep.utils import (
    get_embedding,
    load_embedding,
    seq_to_oh,
    validate_mLSTM_params,
)

"""Tests for neural network layers."""


rng = random.PRNGKey(0)


@given(st.data())
@settings(deadline=None, max_examples=20)
def test_AAEmbedding(data):
    length = data.draw(st.integers(min_value=1, max_value=10))
    sequence = data.draw(
        st.text(
            alphabet="MRHKDESTNQCUGPAVIFYWLOXZBJ",
            min_size=length,
            max_size=length,
        ),
    )
    oh_seq = seq_to_oh(sequence)
    init_fun, apply_fun = AAEmbedding(embedding_dims=33)
    output_shape, emb_matrix = init_fun(rng, (-1, 26))
    emb_seq = apply_fun(params=emb_matrix, inputs=oh_seq)
    assert output_shape == (-1, 33)
    assert emb_matrix.shape == (26, 33)
    assert emb_seq.shape == (length + 2, 33)


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
    embedding = load_embedding()
    x = get_embedding(sequence, embedding)
    output_dim = 256
    init_fun, apply_fun = mLSTM(output_dim=output_dim)
    output_shape, params = init_fun(rng, (-1, 10))
    _, _, outputs = apply_fun(params=params, inputs=x)
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
    embedding = load_embedding()
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
    length = data.draw(st.integers(min_value=1, max_value=10))
    sequence = data.draw(
        st.text(
            alphabet="MRHKDESTNQCUGPAVIFYWLOXZBJ",
            min_size=length,
            max_size=length,
        ),
    )
    embedding = load_embedding()
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
