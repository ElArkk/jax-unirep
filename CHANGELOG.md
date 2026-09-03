# Changelog

In the changelog, @ElArkk and @ericmjl would like to acknowledge contributors who have helped us with anything on the project, big or small.

<!-- Please add your contribution to the top -->

## 3.0.0 (3 September 2026)

A breaking release. The model is now an [equinox](https://docs.kidger.site/equinox/)
module rather than a `jax.example_libraries.stax` layer stack, and parameters
live on the model instead of in an anonymous tuple beside it.

Representations are unchanged. The 1900, 256 and 64 models still reproduce the
original TensorFlow UniRep hidden states to ~1e-6, and the test suite now pins
those numbers against captured ground truth so a change that alters an
embedding fails the build.

### Migration

| v2 | v3 |
| --- | --- |
| `get_reps(seqs, params=...)` | `get_reps(seqs, model=...)` |
| `load_params(...)` | `load_model(...)` |
| `dump_params(params, dir)` | `save_model(model, dir)` |
| `fit(..., model_func=..., params=...)` | `fit(..., model=...)` |
| `evotuning_models.mlstm1900()` | `MLSTM(n_cells=1, output_dim=1900, key=...)` |
| `evotuning_models.mlstm256()` | `MLSTM(n_cells=4, output_dim=256, key=...)` |
| `jax_unirep.layers.*` | `jax_unirep.models.MLSTM` |

`fit` and `evotune` now return an `MLSTM`. Weights are stored as `.npz` rather
than pickle; `load_model` still reads 2.x pickles, with a warning.

### Fixed

- `get_reps` returned plausible but wrong values for the 256 and 64 models. It
  ignored `mlstm_size` when loading, always used the 1900 embedding matrix, and
  ran one mLSTM cell where those models stack four. With explicit params it
  returned a `(1, 256)` array with no error, off by 1.44 from ground truth
  (issue #111).
- `pkg_resources` was removed from setuptools, making the package unimportable
  in modern environments (issue #122).
- A bare `"jax"` requirement let resolvers install a version predating
  `jax.example_libraries`, which is the real cause of issue #118. Now
  `jax>=0.4`.
- Evotuning used *binary* cross-entropy elementwise across 25 softmax outputs,
  so roughly half the gradient went to classes softmax already handles. Now
  categorical cross-entropy computed from logits, which also removes the
  `log(0)` clamp that guarded the NaN losses recorded in April 2020 and issue
  #94.
- Padding was not masked out of the evotuning loss. Since `"-"` maps to class 0,
  a real class, the model was trained to predict gap characters -- 41% of the
  batch loss in a representative case.
- `fit` now defaults to `batch_method="length"`. `"random"` pads every sequence
  to the longest in the *whole dataset*, wasting about half the compute on a
  realistic length distribution.
- `sampler.is_accepted` overflowed `exp` on every run.

### Added

- `fusion_reps`, the "UniRep Fusion" representation from the paper: average
  hidden, final hidden and final cell concatenated.
- `MLSTM`, `load_model` and `save_model` are exported at the top level.

### Internal

- `optax.adamw` replaces the hand-rolled optimizer, with `weight_decay=0.01`
  preserved so results are unchanged.
- Packaging moved to `pyproject.toml`; conda and `environment.yml` replaced by
  uv. `layers.py`, `evotuning_models.py` and `optimizers.py` are deleted.


- 12 August 2022: Fixed jax dependency imports, by @aaroncsolomon
- 23 December 2020: Snuck in a fix for incorrect logger info, by @ericmjl.
- 23 December 2020: Fixed bug with NaN values in grad (issue #94), by @ericmjl
    1. h/t @r-karimi for discovering the bug.
- 14 December 2020: Evotune log format bugfix by @ericmjl and @ElArkk
    1. Reported by @jmahenriques in issue #93
- 2 December 2020: Implementation of embedding layer by @ElArkk
    1. Initial AA embedding layer is now trainable and of flexible size
    2. Pre-existing and dumped weights get stored in pkl format
    3. get_reps accepts variable size mLSTMs
- 20 November 2020: Major rework of fitting API, plus bugfixes by @ericmjl
    1. Custom model architectures can now be passed to `fit`
    2. Refactored lots of utility functions in the evotuing process for better readability
    3. Oscillating output bug of `get_reps` fixed (thank you @hhefzi, @tanggis and @hypostulate !)
    4. Confusing logging statements regarding length and random batching updated
- 29 August 2020: Fixed setup.py so that PEP 517 calls such as `pip install .` work, by @konstin.
- 29 August 2020: Require python 3.6 instead 3.7, by @konstin.
- 20 April 2020: Code fixes for major bug with negative and NaN losses due to Softmax issue by @ivanjayapurna,
- 20 April 2020: (Also by @ivanjayapurna) Overhauled evotuning.py with major changes including
    1. option to supply an out-domain holdout set and print params as training progresses,
    2. evotuning without Optuna by directly calling fit function,
    3. added avg_loss() function for calculation outputting of training and holdout set loss to a log file (number and length of batches are also calculated and printed to log file),
    4. introduction of "epochs_per_print" to periodically calculate losses and dump parameters
    5. Implemented adamW in JAX and switched optimizer to adamW,
    6. added option to change the number of folds in optuna KFolds,
    7. update evotuning-prototype.py example script
- 30 March 2020: Code fixes for correctness and readability, and a parameter dumping function by @ivanjayapurna,
- 28 June 2020: Improvements to evotuning ergonomics, by @ericmjl
    1. Adds a pre-commit configuration.
    2. Adds an installation script that makes easy the installation of jax on GPU.
    3. Provided backend specification of device (GPU/CPU).
    4. Switched preparation of sequences as input-output pairs exclusively on CPU, for speed.
    5. Added ergonomic UI features - progressbars! - that improve user experience.
    6. Added docs on recommended batch size and its relationship to GPU RAM consumption.
    7. Switched from exact calculation of train/holdout loss to estimated calculation.
- 9 July 2020: Add progress bar to sequence sampler, by @ericmjl
