from jax_unirep import fit
from jax_unirep.utils import load_params_1900
from jax_unirep import get_reps
import jax.numpy as np
from jax.lax import lax_control_flow
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


print(lax_control_flow.__file__)

sequences = ["MKLVIPJ", "MMLVIKJP", "MKLVIJ"]

params = fit(params=None, sequences=sequences, n_epochs=2)

mut_seq = sequences

for i in range(0, 6):
    print(f"------Iteration {i}------")
    # print(
    #     "Default parameters-sum of embeddings: {}".format(
    #         np.sum(get_reps(mut_seq)[0])
    #     )
    # )
    print(
        "Custom parameters-sum of embeddings: {}".format(
            np.sum(get_reps(mut_seq, params=params[0])[0])
        )
    )
