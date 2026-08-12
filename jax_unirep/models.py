"""The UniRep mLSTM, as Equinox modules.

Parameters are fields on the module rather than entries in an anonymous tuple,
so the model describes its own architecture: `len(model.cells)` is the depth
and `model.cells[0].wmh.shape[0]` is the width. Nothing needs to be told what
shape the model is.
"""

import pickle as pkl
import warnings
from pathlib import Path
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax import lax, random
from jax.nn.initializers import glorot_normal, normal

from .activations import sigmoid, tanh
from .utils import (
    WEIGHTS_NPZ,
    WEIGHTS_PKL,
    get_weights_dir,
    l2_normalize,
)

EMBEDDING_DIM = 10
N_LETTERS = 26
N_TARGET_CLASSES = 25

MLSTM_PARAM_KEYS = ("b", "gh", "gmh", "gmx", "gx", "wh", "wmh", "wmx", "wx")


class MLSTMCell(eqx.Module):
    """One multiplicative LSTM layer, with weight normalization.

    Reference implementation:
    https://github.com/churchlab/UniRep/blob/master/unirep.py#L75
    """

    wmx: jax.Array
    wmh: jax.Array
    wx: jax.Array
    wh: jax.Array
    gmx: jax.Array
    gmh: jax.Array
    gx: jax.Array
    gh: jax.Array
    b: jax.Array

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        key: jax.Array,
        W_init=glorot_normal(),
        b_init=normal(),
    ):
        k1, k2, k3, k4 = random.split(key, num=4)
        self.wmx = W_init(k1, (input_dim, output_dim))
        self.wmh = W_init(k2, (output_dim, output_dim))
        self.wx = W_init(k3, (input_dim, output_dim * 4))
        self.wh = W_init(k4, (output_dim, output_dim * 4))

        k1, k2, k3, k4 = random.split(k1, num=4)
        self.gmx = b_init(k1, (output_dim,))
        self.gmh = b_init(k2, (output_dim,))
        self.gx = b_init(k3, (output_dim * 4,))
        self.gh = b_init(k4, (output_dim * 4,))

        k1, _ = random.split(k1)
        self.b = b_init(k1, (output_dim * 4,))

    @property
    def output_dim(self) -> int:
        return self.wmh.shape[0]

    def __call__(
        self, sequence: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Scan the cell over one sequence.

        :param sequence: shape (n_positions, input_dim).
        :returns: `(h_final, c_final, hidden_states)`, where `hidden_states`
            has shape (n_positions, output_dim).
        """
        h_t = np.zeros(self.output_dim)
        c_t = np.zeros(self.output_dim)

        # Weight normalization does not depend on the timestep, so it is
        # hoisted out of the scan rather than recomputed at every position as
        # the stax version did. These are locals: `self` is frozen, so the
        # in-place mutation behind issue #107 cannot be written here at all.
        wx = l2_normalize(self.wx, axis=0) * self.gx
        wh = l2_normalize(self.wh, axis=0) * self.gh
        wmx = l2_normalize(self.wmx, axis=0) * self.gmx
        wmh = l2_normalize(self.wmh, axis=0) * self.gmh

        def step(carry, x_t):
            h_t, c_t = carry

            m = np.matmul(x_t, wmx) * np.matmul(h_t, wmh)
            z = np.matmul(x_t, wx) + np.matmul(m, wh) + self.b

            # input, forget, output, update
            i, f, o, u = np.split(z, 4, axis=-1)
            i = sigmoid(i, version="exp")
            f = sigmoid(f, version="exp")
            o = sigmoid(o, version="exp")
            u = tanh(u)

            c_t = f * c_t + i * u
            h_t = o * tanh(c_t)

            return (h_t, c_t), h_t

        (h_final, c_final), hidden_states = lax.scan(
            step, (h_t, c_t), sequence
        )
        return h_final, c_final, hidden_states


class MLSTM(eqx.Module):
    """Amino acid embedding, a stack of mLSTM cells, and an optional head.

    The 1900 model has one cell; the 256 and 64 models have four. The dense
    head predicts the next amino acid and is only used for evotuning, so it is
    optional.
    """

    embedding: jax.Array
    cells: Tuple[MLSTMCell, ...]
    dense_w: Optional[jax.Array]
    dense_b: Optional[jax.Array]

    def __init__(
        self,
        n_cells: int,
        output_dim: int,
        key: jax.Array,
        with_head: bool = True,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        keys = random.split(key, num=n_cells + 2)
        self.embedding = glorot_normal()(keys[0], (N_LETTERS, embedding_dim))
        self.cells = tuple(
            MLSTMCell(
                input_dim=embedding_dim if i == 0 else output_dim,
                output_dim=output_dim,
                key=keys[i + 1],
            )
            for i in range(n_cells)
        )
        if with_head:
            w_key, b_key = random.split(keys[-1])
            self.dense_w = glorot_normal()(
                w_key, (output_dim, N_TARGET_CLASSES)
            )
            self.dense_b = normal()(b_key, (N_TARGET_CLASSES,))
        else:
            self.dense_w = None
            self.dense_b = None

    @property
    def output_dim(self) -> int:
        return self.cells[0].output_dim

    def embed(self, one_hot: jax.Array) -> jax.Array:
        """Project one-hot amino acids into the embedding space."""
        return np.matmul(one_hot, self.embedding)

    def __call__(
        self, one_hot: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Run the embedding and every cell over one sequence.

        :param one_hot: shape (n_positions, 26).
        :returns: `(h_final, c_final, hidden_states)` of the final cell.
        """
        activations = self.embed(one_hot)
        h_final = c_final = None
        for cell in self.cells:
            h_final, c_final, activations = cell(activations)
        return h_final, c_final, activations

    def logits(self, one_hot: jax.Array) -> jax.Array:
        """Next-amino-acid logits for one sequence, shape (n_positions, 25).

        Deliberately not softmaxed: the softmax is folded into the loss, where
        logsumexp can fuse it with the log.
        """
        if self.dense_w is None:
            raise ValueError(
                "This model has no dense head, so it cannot produce logits. "
                "Build it with with_head=True, or load weights that include "
                "dense.w and dense.b."
            )
        _, _, hidden_states = self(one_hot)
        return np.matmul(hidden_states, self.dense_w) + self.dense_b


def model_from_arrays(arrays) -> MLSTM:
    """Build an `MLSTM` from the named arrays stored in a weights `.npz`.

    Keys are `embedding`, `mlstm.<i>.<param>` and optionally `dense.w` /
    `dense.b`, so the file states its own depth and width.
    """
    n_cells = 1 + max(
        int(key.split(".")[1]) for key in arrays if key.startswith("mlstm.")
    )
    output_dim = arrays["mlstm.0.wmh"].shape[0]

    # Build a correctly-shaped skeleton, then replace every leaf. eqx.tree_at
    # is the supported way to write into a frozen module.
    model = MLSTM(
        n_cells=n_cells,
        output_dim=output_dim,
        key=random.PRNGKey(0),
        with_head="dense.w" in arrays,
    )

    replacements = {"embedding": np.asarray(arrays["embedding"])}
    if "dense.w" in arrays:
        replacements["dense_w"] = np.asarray(arrays["dense.w"])
        replacements["dense_b"] = np.asarray(arrays["dense.b"])

    model = eqx.tree_at(
        lambda m: [getattr(m, name) for name in replacements],
        model,
        [replacements[name] for name in replacements],
    )

    cells = tuple(
        eqx.tree_at(
            lambda c: [
                c.wmx,
                c.wmh,
                c.wx,
                c.wh,
                c.gmx,
                c.gmh,
                c.gx,
                c.gh,
                c.b,
            ],
            cell,
            [
                np.asarray(arrays[f"mlstm.{i}.{name}"])
                for name in (
                    "wmx",
                    "wmh",
                    "wx",
                    "wh",
                    "gmx",
                    "gmh",
                    "gx",
                    "gh",
                    "b",
                )
            ],
        )
        for i, cell in enumerate(model.cells)
    )
    return eqx.tree_at(lambda m: m.cells, model, cells)


def model_to_arrays(model: MLSTM) -> dict:
    """Flatten an `MLSTM` into named arrays, ready for `np.savez`.

    Inverse of `model_from_arrays`.
    """
    arrays = {"embedding": onp.asarray(model.embedding)}
    for i, cell in enumerate(model.cells):
        for key in MLSTM_PARAM_KEYS:
            arrays[f"mlstm.{i}.{key}"] = onp.asarray(getattr(cell, key))
    if model.dense_w is not None:
        arrays["dense.w"] = onp.asarray(model.dense_w)
        arrays["dense.b"] = onp.asarray(model.dense_b)
    return arrays


def _arrays_from_legacy_tree(params) -> dict:
    """Read the flat parameter tree that versions up to 2.x pickled.

    That tree is the embedding, then one dict per mLSTM layer interleaved with
    empty tuples for the parameterless stax layers, then the dense pair.
    """
    arrays = {"embedding": onp.asarray(params[0])}
    cell = 0
    for element in params[1:]:
        if isinstance(element, dict):
            for key in MLSTM_PARAM_KEYS:
                arrays[f"mlstm.{cell}.{key}"] = onp.asarray(element[key])
            cell += 1
        elif isinstance(element, (tuple, list)) and len(element) == 2:
            arrays["dense.w"] = onp.asarray(element[0])
            arrays["dense.b"] = onp.asarray(element[1])
    return arrays


def save_model(
    model: MLSTM,
    dir_path: Path = Path("temp"),
    step: Optional[int] = 0,
) -> Path:
    """Write a model's weights to `<dir_path>/iter_<step>/model_weights.npz`.

    The layout matches the shipped weights, so a checkpoint can be read back
    with `load_model(folderpath=...)`.

    :param model: The model to save.
    :param dir_path: Directory to write into. Created if it does not exist.
    :param step: Training step, used to name the subdirectory.
    :returns: The directory the weights were written to.
    """
    iteration_path = Path(dir_path) / f"iter_{step}"
    iteration_path.mkdir(parents=True, exist_ok=True)
    onp.savez(iteration_path / WEIGHTS_NPZ, **model_to_arrays(model))
    return iteration_path


def load_model(
    folderpath: Optional[str] = None, paper_weights: Optional[int] = 1900
) -> MLSTM:
    """Load a pre-trained `MLSTM`.

    Weights saved by versions up to 2.x are Python pickles. Those are still
    read, so previously saved weights keep working, but unpickling executes
    arbitrary code and is only as trustworthy as the file. Re-save with
    `save_model` to convert.

    :param folderpath: Directory holding a `model_weights.npz`.
    :param paper_weights: Which published model to load: 1900, 256 or 64.
    """
    weights_dir = Path(
        get_weights_dir(folderpath=folderpath, paper_weights=paper_weights)
    )

    npz_path = weights_dir / WEIGHTS_NPZ
    if npz_path.exists():
        with onp.load(npz_path, allow_pickle=False) as arrays:
            return model_from_arrays({k: arrays[k] for k in arrays.files})

    pkl_path = weights_dir / WEIGHTS_PKL
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"No model weights found in {weights_dir}. Expected "
            f"{WEIGHTS_NPZ}, or {WEIGHTS_PKL} if these were saved by "
            f"jax-unirep 2.x or earlier."
        )

    warnings.warn(
        f"Loading legacy pickled weights from {pkl_path}. Unpickling "
        f"executes arbitrary code, so only load files you trust. Re-save "
        f"them with save_model to convert to {WEIGHTS_NPZ}.",
        FutureWarning,
        stacklevel=2,
    )
    with open(pkl_path, "rb") as f:
        return model_from_arrays(_arrays_from_legacy_tree(pkl.load(f)))
