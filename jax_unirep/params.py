from jax.nn.initializers import glorot_normal
from jax.random import PRNGKey, normal, split

key = PRNGKey(42)


def add_dense_params(params, name, input_dim, output_dim, init=glorot_normal):
    params[name] = dict()
    params[name]["w"] = init(split(key)[0], (input_dim, output_dim)) * 0.01
    params[name]["b"] = init(split(key)[0], (output_dim,)) * 0.01
    return params


def add_mlstm1900_params(
    params, name, input_dim, output_dim, init=glorot_normal
):
    params[name] = dict()
    params[name]["wmx"] = init(split(key)[0], (input_dim, output_dim))
    params[name]["wmh"] = init(split(key)[0], (output_dim, output_dim))
    params[name]["wx"] = init(split(key)[0], (input_dim, output_dim * 4))
    params[name]["wh"] = init(split(key)[0], (output_dim, output_dim * 4))

    params[name]["gmx"] = init(split(key)[0], (output_dim,))
    params[name]["gmh"] = init(split(key)[0], (output_dim,))
    params[name]["gx"] = init(split(key)[0], (output_dim * 4,))
    params[name]["gh"] = init(split(key)[0], (output_dim * 4,))

    params[name]["b"] = init(split(key)[0], (output_dim * 4,))
    return params
