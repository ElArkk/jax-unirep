[![Build Status](https://travis-ci.com/ElArkk/jax-unirep.svg?branch=master)](https://travis-ci.com/ElArkk/jax-unirep)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# jax-unirep

A performant reimplementation of the UniRep protein featurization model in JAX.

The UniRep model was developed in George Church's lab,
see the original publication
[here][arxiv] (bioRxiv) or [here][nature] (Nature Methods),
as well as the [repository][repo] containing the original model.

This repo is a self-contained version of the UniRep model
(so far only the 1900 hidden-unit mLSTM),
with additional utility APIs that support protein engineering workflows.

## Installation

Ensure that your compute environment allows you to run JAX code.
(A modern Linux or macOS with a GLIBC>=2.23 is probably necessary.)

For now, `jax-unirep` is available by pip installing from source.

Installation from GitHub:

```bash
pip install git+https://github.com/ElArkk/jax-unirep.git
```

## More Details

To read more about how we reimplemented the model in JAX,
check out our preprint
on [bioarxiv](https://www.biorxiv.org/content/10.1101/2020.05.11.088344v1).

Docs are in the making, keep your eyes peeled!

## License

All the model weights are licensed under the terms of
Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license,
visit [here](http://creativecommons.org/licenses/by-nc/4.0/)
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
[it]: https://github.com/ElArkk/jax-unirep/issues
