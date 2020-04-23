import jax.numpy as np
from jax.experimental.optimizers import make_schedule, optimizer


@optimizer
def adamW(step_size, b1=0.9, b2=0.999, eps=1e-8, w=0.01):
    """Construct optimizer triple for Adam.

    This docstring is different from the rest because we want to submit this
    to the jax library, so DON'T CHANGE IT TO SPHINX-STYLE!

    Args:
        step_size: positive scalar, or a callable representing a step size schedule
            that maps the iteration index to positive scalar.
        b1: optional, a positive scalar value for beta_1, the exponential decay rate
            for the first moment estimates (default 0.9).
        b2: optional, a positive scalar value for beta_2, the exponential decay rate
            for the second moment estimates (default 0.999).
        eps: optional, a positive scalar value for epsilon, a small constant for
            numerical stability (default 1e-8).
        w: optional, weight decay term (default 0.01)

    Returns:
        An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)

    def init(x0):
        m0 = np.zeros_like(x0)
        v0 = np.zeros_like(x0)
        return x0, m0, v0

    def update(i, g, state):
        x, m, v = state
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))
        x = x - step_size(i) * (mhat / (np.sqrt(vhat) + eps) + w * x)
        return x, m, v

    def get_params(state):
        x, m, v = state
        return x

    return init, update, get_params
