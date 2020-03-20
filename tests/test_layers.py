import numpy as np
import numpy.random as npr
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import random

from jax_unirep.layers import mlstm1900, mlstm1900_batch, mlstm1900_step
from jax_unirep.utils import (
    get_embedding,
    get_embeddings,
    load_embedding_1900,
    load_params_1900,
)

rng = random.PRNGKey(0)


def test_mlstm1900_step():
    """
    Given fake data of the correct input shapes,
    make sure that the output shapes are also correct.
    """
    params = load_params_1900()

    x_t = npr.normal(size=(1, 10))
    h_t = np.zeros(shape=(1, 1900))
    c_t = np.zeros(shape=(1, 1900))

    carry = (h_t, c_t)

    (h_t, c_t), _ = mlstm1900_step(params, carry, x_t)
    assert h_t.shape == (1, 1900)
    assert c_t.shape == (1, 1900)


def test_mlstm1900_batch():
    """
    Given one fake embedded sequence,
    ensure that we get out _an_ output from mLSTM1900.
    """
    emb = load_embedding_1900()
    x = get_embedding("TEST", emb)

    params = load_params_1900()

    h_final, c_final, h = mlstm1900_batch(params, x)
    assert h.shape == (x.shape[0], 1900)


# @given(st.data())
# @settings(deadline=None, max_examples=20)
# def test_mlstm1900(data):
#     params = load_params_1900()
#     length = data.draw(st.integers(min_value=1, max_value=10))
#     sequences = data.draw(
#         st.lists(
#             elements=st.text(
#                 alphabet="MRHKDESTNQCUGPAVIFYWLOXZBJ",
#                 min_size=length,
#                 max_size=length,
#             ),
#             min_size=1,
#         )
#     )

#     x = get_embeddings(sequences)
#     h_final, c_final, h = mlstm1900(params, x)
#     assert h.shape == (len(sequences), length + 1, 1900)


@given(st.data())
@settings(deadline=None, max_examples=20)
def test_mlstm1900(data):
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
    init_fun, apply_fun = mlstm1900(output_dim=1900)
    output_shape, params = init_fun(rng, (length, 10))

    h_final, c_final, outputs = apply_fun(params=params, inputs=x)

    assert output_shape == (length, 1900)

    assert params["wmx"].shape == (10, 1900)
    assert params["wmh"].shape == (1900, 1900)
    assert params["wx"].shape == (10, 7600)
    assert params["wh"].shape == (1900, 7600)
    assert params["gmx"].shape == (1900,)
    assert params["gmh"].shape == (1900,)
    assert params["gx"].shape == (7600,)
    assert params["gh"].shape == (7600,)
    assert params["b"].shape == (7600,)

    assert outputs.shape == (length + 1, 1900)
