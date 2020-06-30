from jax_unirep.params import add_dense_params, add_mLSTM1900_params
from jax_unirep.utils import validate_mLSTM1900_params


def test_add_dense_params():
    """Unit test for add_dense_params."""
    params = dict()
    params = add_dense_params(
        params, name="param1", input_dim=40, output_dim=20
    )
    assert params["param1"]["w"].shape == (40, 20)
    assert params["param1"]["b"].shape == (20,)


def test_add_mLSTM1900_params():
    """Execution test for add_mLSTM1900_params.

    Forgive me, I was being lazy when doing this test,
    I thus defaulted to just using an execution test.
    """
    params = dict()
    params = add_mLSTM1900_params(params, name="mLSTM1900",)
    validate_mLSTM1900_params(params["mLSTM1900"])
