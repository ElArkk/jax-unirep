from jax.random import PRNGKey

from jax_unirep.evotuning_models import mlstm64, mlstm256, mlstm1900
from jax_unirep.utils import seq_to_oh


def test_mlstm1900():
    """Test forward pass of pre-built mlstm1900 model"""
    init_fun, model_fun = mlstm1900()
    _, params = init_fun(PRNGKey(42), input_shape=(-1, 26))

    oh = seq_to_oh("HASTA")
    out = model_fun(params, oh)

    assert out.shape == (7, 25)


def test_mlstm256():
    """Test forward pass of pre-built mlstm256 model"""
    init_fun, model_fun = mlstm256()
    _, params = init_fun(PRNGKey(42), input_shape=(-1, 26))

    oh = seq_to_oh("HASTA")
    out = model_fun(params, oh)

    assert out.shape == (7, 25)


def test_mlstm64():
    """Test forward pass of pre-built mlstm64 model"""
    init_fun, model_fun = mlstm64()
    _, params = init_fun(PRNGKey(42), input_shape=(-1, 26))

    oh = seq_to_oh("HASTA")
    out = model_fun(params, oh)

    assert out.shape == (7, 25)
