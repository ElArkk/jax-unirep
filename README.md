# jax-unirep

Reimplementation of the UniRep protein featurization model in JAX.

The UniRep model was developed in George Church's lab, see the original publication [here][arxiv] (bioRxiv) or [here][nature] (Nature Methods), as well as the [repository][repo] containing the original model.

The idea to reimplement the TF-based model in the much lighter JAX framework was coined by [Eric Ma][ericmjl], who also developed a first version of it inside his functional deep-learning library [fundl][fundl].

This repo is a self-contained version of the UniRep model (so far only the 1900 hidden-unit mLSTM), adapted and extended from fundl.

## Usage

To generate representations of same-length sequences, use `jax_unirep.featurize.get_reps`. It will return a tuple consisting of the following representations for each sequence:

- h_final: Final hidden state of the mLSTM
- c_final: Final cell state of the mLSTM
- h_avg: Average hidden state of the mLSTM over the whole sequence.

# License

All the model weights are licensed under the terms of Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Otherwise the code in this repository is licensed under the terms of [GPL v3][gpl3].

[arxiv]: https://www.biorxiv.org/content/10.1101/589333v1
[nature]: https://www.nature.com/articles/s41592-019-0598-1
[repo]: https://github.com/churchlab/UniRep
[ericmjl]: https://github.com/ericmjl
[fundl]: https://github.com/ericmjl/fundl
[gpl3]: https://www.gnu.org/licenses/gpl-3.0.html
