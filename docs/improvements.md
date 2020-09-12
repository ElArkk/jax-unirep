# Improvements

What have we done with `jax-unirep` that improve upon [the original][theog]?

Primarily, we have made improvements in three areas.

Firstly, speed through reimplementation:
By leveraging modern automatic differentiation packages, specifically JAX,
we were able to accelerate the UniRep model over 100X over the original.
Our preprint [on bioarxiv][preprint] provides additional detail on our claim.

[preprint]: https://www.biorxiv.org/content/10.1101/2020.05.11.088344v1

Secondly, robustness:
we have extensively unit tested the model code,
so that others may use it with confidence.
The code has 96% test coverage,
and we leverage Travis CI for continuous integration,
such that every change is tested extensively.

Thirdly, user-friendly APIs:
Most of our expected user base should be not-so-technical protein engineers.
("Technical" being extensively comfortable with Python;
protein engineering is a very technical discipline!)
As such, developed APIs that prevent many sources of user error
that may interfere with safe handling of UniRep.
One example is that our most commonly-used functions,
for calculating representations of proteins and evotuning,
always accept Python strings rather than hand-prepared tensors.

Finally, docs:
The fact that you're here means you're benefiting from the hosted API docs!
We hope you find it to be easy to use and navigate.

[theog]: https://github.com/churchlab/unirep
