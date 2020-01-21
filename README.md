# jax-unirep
Reimplementation of the UniRep protein featurization model in JAX.

The UniRep model was developed in George Church's lab, see the original publication [here][arxiv] (bioRxiv) or [here][nature] (Nature Methods), as well as the [repository][repo] containing the original model.

The idea to reimplement the TF-based model in the much lighter JAX framework was coined by [Eric Ma][ericmjl], who also developed a first version of it inside his functional deep-learning library [fundl][fundl].

This repo is a self-contained version of the UniRep model (so far only the 1900 hidden-unit mLSTM), adapted and extended from fundl.


[arxiv]: https://www.biorxiv.org/content/10.1101/589333v1
[nature]: https://www.nature.com/articles/s41592-019-0598-1
[repo]: https://github.com/churchlab/UniRep
[ericmjl]: https://github.com/ericmjl
[fundl]: https://github.com/ericmjl/fundl