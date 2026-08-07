"""One-time conversion of the shipped weights from pickle to .npz.

Reads each `model_weights.pkl` and writes a `model_weights.npz` of
semantically named arrays alongside it, then verifies that reloading the npz
reconstructs the original tree exactly.

Committed rather than run ad hoc so the mapping from the old positional stax
tuple to the new names is reviewable and reproducible.

    .venv/bin/python scripts/convert_weights_to_npz.py
"""

import pickle
import sys

import numpy as np

from jax_unirep.utils import (
    WEIGHTS_NPZ,
    WEIGHTS_PKL,
    arrays_to_params,
    get_weights_dir,
    params_to_arrays,
)

SIZES = (1900, 256, 64)


def trees_match(a, b) -> bool:
    """Structural and exact numerical equality of two parameter trees."""
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(trees_match(a[k], b[k]) for k in a)
        )
    if isinstance(a, (tuple, list)):
        return (
            isinstance(b, (tuple, list))
            and len(a) == len(b)
            and all(trees_match(x, y) for x, y in zip(a, b))
        )
    return np.array_equal(np.asarray(a), np.asarray(b))


def main() -> int:
    failures = 0

    for size in SIZES:
        weights_dir = get_weights_dir(paper_weights=size)
        pkl_path = weights_dir / WEIGHTS_PKL
        npz_path = weights_dir / WEIGHTS_NPZ

        with open(pkl_path, "rb") as f:
            original = pickle.load(f)

        arrays = params_to_arrays(original)
        np.savez(npz_path, **arrays)

        with np.load(npz_path, allow_pickle=False) as loaded:
            roundtripped = arrays_to_params(
                {k: loaded[k] for k in loaded.files}
            )

        ok = trees_match(original, roundtripped)
        failures += not ok

        print(
            f"{size:>5}: {len(arrays):>2} arrays  "
            f"{pkl_path.stat().st_size / 1e6:6.1f} MB pkl -> "
            f"{npz_path.stat().st_size / 1e6:6.1f} MB npz  "
            f"exact={ok}"
        )
        if size == SIZES[0]:
            print(f"        keys: {', '.join(sorted(arrays)[:4])}, ...")

    return failures


if __name__ == "__main__":
    sys.exit(main())
