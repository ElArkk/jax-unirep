from jax_unirep.params import add_dense_params, add_mLSTM_params
from jax_unirep.utils import validate_mLSTM_params


def test_add_dense_params():
    """Unit test for add_dense_params."""
    params = dict()
    params = add_dense_params(
        params, name="param1", input_dim=40, output_dim=20
    )
    assert params["param1"]["w"].shape == (40, 20)
    assert params["param1"]["b"].shape == (20,)


def test_add_mLSTM_params():
    """Execution test for add_mLSTM_params.

    Forgive me, I was being lazy when doing this test,
    I thus defaulted to just using an execution test.
    """
    params = dict()
    params = add_mLSTM_params(
        params,
        name="mLSTM",
    )
    validate_mLSTM_params(params["mLSTM"])
