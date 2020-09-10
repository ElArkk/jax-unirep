import jax.numpy as np

# def _cross_entropy_loss(y, y_hat):
#     """
#     Also corresponds to the log likelihood of the Bernoulli
#     distribution.
#     Intended to be used inside of another function that differentiates w.r.t.
#     parameters.
#     """
#     xent = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
#     return np.mean(xent)


def _neg_cross_entropy_loss(y, y_hat, tol=1e-10):
    """
    Also corresponds to the log likelihood of the Bernoulli
    distribution.
    Intended to be used inside of another function that differentiates w.r.t.
    parameters.
    """
    xent = -(
        y * np.log(np.maximum(tol, y_hat))
        + (1 - y) * np.log(np.maximum(tol, 1 - y_hat))
    )
    return np.mean(xent)


# def _mse_loss(y, y_hat):
#     """
#     Intended to be used inside of another function that differentiates w.r.t.
#     parameters.
#     """
#     return np.mean(np.power(y - y_hat, 2))


# def _mae_loss(y, y_hat):
#     """
#     Intended to be used inside of another function that differentiates w.r.t.
#     parameters.
#     """
#     return np.mean(np.abs(y - y_hat))


# def cross_entropy_loss(params, model, x, y):
#     y_hat = model(params, x)
#     return _cross_entropy_loss(y, y_hat)


# def neg_cross_entropy_loss(params, model, x, y):
#     y_hat = model(params, x)
#     return _neg_cross_entropy_loss(y, y_hat)


# def mseloss(params, model, x, y):
#     y_hat = model(params, x)
#     return _mse_loss(y, y_hat)


# def maeloss(params, model, x, y):
#     y_hat = model(params, x)
#     return _mae_loss(y, y_hat)
