from jax.random import PRNGKey, normal, split

key = PRNGKey(42)


def add_dense_params(params, name, input_dim, output_dim):
    params[name] = dict()
    params[name]["w"] = normal(split(key)[0], (input_dim, output_dim)) * 0.01
    params[name]["b"] = normal(split(key)[0], (output_dim,)) * 0.01
    return params


def add_mlstm1900_params(params, name, input_dim, output_dim):
    params[name] = dict()
    params[name]["wmx"] = normal(split(key)[0], (input_dim, output_dim))
    params[name]["wmh"] = normal(split(key)[0], (output_dim, output_dim))
    params[name]["wx"] = normal(split(key)[0], (input_dim, output_dim * 4))
    params[name]["wh"] = normal(split(key)[0], (output_dim, output_dim * 4))

    params[name]["gmx"] = normal(split(key)[0], (output_dim,))
    params[name]["gmh"] = normal(split(key)[0], (output_dim,))
    params[name]["gx"] = normal(split(key)[0], (output_dim * 4,))
    params[name]["gh"] = normal(split(key)[0], (output_dim * 4,))

    params[name]["b"] = normal(split(key)[0], (output_dim * 4,))
    return params