from functools import partial
from typing import Dict, List

from jax import grad
from jax.experimental.optimizers import adam

from .activations import softmax
from .layers import dense, mlstm1900
from .losses import neg_cross_entropy_loss
from .params import add_dense_params
from .utils import get_embeddings


def evotune(
    mlstm1900_params: Dict[str, np.array], seqs: List[str]
) -> Dict[str, np.array]:
    """
    Given a set of weights for the `mlstm1900` UniRep model,
    as well as protein sequences of arbitrary length,
    this function will perform weight updates on the mLSTM,
    under the pretext learning task of predicting the next
    amino acid in the protein sequences, given the output of the mLSTM.
    The prediction itself is being done by a single, fully-connected
    layer with 26 output nodes and using softmax activation 
    (Each node corresponding to one AA).

    :param params: Either pre-trained or random weights to initalize
        the mLSTM with, as `np.arrays`.
    :param seqs: A list of protein sequences as strings
    """

    params = dict()
    params["mlstm1900"] = mlstm1900_params
    params["dense"] = add_dense_params()

    def predict(params, batch):
        batch = mlstm1900(params["mlstm1900"], batch)
        batch = dense(params["dense"], batch, activation=softmax)
        return batch

    loss = partial(neg_cross_entropy_loss, model=predict)
    dloss = grad(loss)

    init, update, get_params = adam(step_size=0.005)

    state = init(params)

    for i in range(epochs):
        g = dloss(params, x=x, y=y)

        state = update(i, g, state)
        params = get_params(state)

        if i % 10 == 0:
            l = loss(params, x=x, y=y)
            print(f"iteration: {i}, loss: {l}")
    # def update(params, x, y):

    # prepare the sequences
    # batch sequences according to length
