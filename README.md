[![Build Status](https://travis-ci.com/ElArkk/jax-unirep.svg?branch=master)](https://travis-ci.com/ElArkk/jax-unirep)
[![codecov](https://codecov.io/gh/ElArkk/jax-unirep/branch/master/graph/badge.svg)](https://codecov.io/gh/ElArkk/jax-unirep)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# jax-unirep

A performant reimplementation of the UniRep protein featurization model in JAX.

The UniRep model was developed in George Church's lab,
see the original publication
[here][arxiv] (bioRxiv) or [here][nature] (Nature Methods),
as well as the [repository][repo] containing the original model.

This repo is a self-contained and easily customizable version of the UniRep model,
with additional utility APIs that support protein engineering workflows.

## Installation

Ensure that your compute environment allows you to run JAX code.
(A modern Linux or macOS with a GLIBC>=2.23 is probably necessary.)

Then, you can install from PyPI:

```bash
pip install jax-unirep
```

If you want the latest and greatest, you can install directly from source:

```bash
pip install git+https://github.com/ElArkk/jax-unirep.git
```

## More Details

Documentation on how to use `jax-unirep` is available [here][docs].

We highly encourage community contributions to this project!
For more information, see the section in the docs on our [contributing][cont] guidelines.

To read more about how we reimplemented the model in JAX,
check out our preprint
on [bioarxiv].

## License

All the model weights are licensed under the terms of
Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license,
visit [here][cc])
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Otherwise the code in this repository
is licensed under the terms of [GPL v3][gpl3].

[arxiv]: https://www.biorxiv.org/content/10.1101/589333v1
[nature]: https://www.nature.com/articles/s41592-019-0598-1
[repo]: https://github.com/churchlab/UniRep
[ericmjl]: https://github.com/ericmjl
[fundl]: https://github.com/ericmjl/fundl
[gpl3]: https://www.gnu.org/licenses/gpl-3.0.html
[evotunefunc]: https://github.com/ElArkk/jax-unirep/blob/master/jax_unirep/evotuning.py#L421
[fitfunc]: https://github.com/ElArkk/jax-unirep/blob/master/jax_unirep/evotuning.py#L163
[examples]: https://github.com/ElArkk/jax-unirep/blob/master/examples
[stax]: https://jax.readthedocs.io/en/latest/jax.experimental.stax.html
[staxex]: https://github.com/google/jax/tree/master/examples
[docs]: https://elarkk.github.io/jax-unirep/
[cont]: https://elarkk.github.io/jax-unirep/contributing/
[it]: https://github.com/ElArkk/jax-unirep/issues
[bioarxiv]: https://www.biorxiv.org/content/10.1101/2020.05.11.088344v1
[cc]: http://creativecommons.org/licenses/by-nc/4.0/
