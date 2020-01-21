import numpy as np
import numpy.random as npr
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_unirep.layers import mlstm1900, mlstm1900_batch, mlstm1900_step
from jax_unirep.utils import (
    get_embedding,
    get_embeddings,
    load_embedding_1900,
    load_params_1900,
)


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


@given(st.data())
@settings(deadline=None, max_examples=20)
def test_mlstm1900(data):
    params = load_params_1900()
    length = data.draw(st.integers(min_value=1, max_value=10))
    sequences = data.draw(
        st.lists(
            elements=st.text(
                alphabet="MRHKDESTNQCUGPAVIFYWLOXZBJ",
                min_size=length,
                max_size=length,
            ),
            min_size=1,
        )
    )

    x = get_embeddings(sequences)
    h_final, c_final, h = mlstm1900(params, x)
    assert h.shape == (len(sequences), length + 1, 1900)
