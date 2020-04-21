# Changelog

In the changelog, @ElArkk and @ericmjl would like to acknowledge contributors who have helped us with anything on the project, big or small.

<Please add your contribution to the top>

- 30 March 2020: Code fixes for correctness and readability, and a parameter dumping function by @ivanjayapurna,
- 20 April 20202: Modified evotuning functtion to add (1) option to supply an out-domain holdout set and print params as training progresses, (2) evotuning without Optuna by directly calling fit function (3) Fixed a major bug in evotuning (negative and NaN cross-entropy losses) by removing non-functional jax.stax Softmax with manually implemented softmax layer. Additionally added load_params() to utils to load params outputted by JAX evotuning, and overhauled evotuning-prototype.py example script, by @ivanjayapurna,
