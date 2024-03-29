# Changelog

In the changelog, @ElArkk and @ericmjl would like to acknowledge contributors who have helped us with anything on the project, big or small.

<!-- Please add your contribution to the top -->

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
