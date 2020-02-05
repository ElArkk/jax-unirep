# Reimplementing Unirep in JAX

<!-- Things to check are prefixed with %% -->

## Abstract

UniRep is a recurrent neural network model
trained on 24 million protein sequences,
and has shown utility in protein engineering.
The original model, however, has rough spots in its implementation,
and a convenient API is not available for certain tasks.
To rectify this, we reimplemented the model in JAX/NumPy,
achieving X-fold speedups in forward pass performance,
and implemented a convenient API for specialized tasks.
In this article, we wish to document our model reimplementation process
with the goal of educating others interested in learning
how to dissect a deep learning model,
and engineer it for robustness and ease of use.

## Introduction

UniRep is a recurrent neural network,
trained using self-supervision
on 24 million protein sequences
to predict the next amino acid in a sequence [@alley2019unified].
Its most powerful model allows for embedding
arbitrary length sequences in a 1900-long feature vector
that can be used as the input to a "top model"
for unsupervised clustering or supervised prediction of protein properties.
<!-- Vectors can be used for clustering
or as inputs for the top model
to predict properties of proteins. -->

The original model was implemented in TensorFlow 1.13 [@abadi2016tensorflow],
and its original API only allowed
for one sequence to be transformed at once.
Thus, while the model itself holds great potential
for the protein engineering field,
the API prevents us from using it conveniently and productively.
We thus sought to reimplement and package the model
in a way that brings a robust yet easy-to-use experience
to protein modellers and engineers.

In particular, our engineering goals were to provide:

- A function that can process multiple sequences of arbitrary lengths,
- Vectorizing the inputs to make it fast.
- A single function call to "evotune" the global weights.

## Reimplementation Main Points

### Choice of JAX

JAX was our library choice to reimplement it in,
because it provides automatic differentiation machinery [@jax2018github]
on top of the highly idiomatic and widely-used NumPy API [@oliphant2006guide].
JAX uses a number of components shared with TensorFlow,
in particular the use of the
XLA (Accelerated Linear Algebra) library
to provide automatic compilation from the NumPy API to GPU and TPU.

Part of the exercise was also pedagogical:
by reimplementing the model in a pure NumPy API,
we are forced to become familiar with the mechanics of the model,
and learn the translation between NumPy and TensorFlow operations.

Because JAX provides automatic differentiation
and a number of optimization routines as utility functions,
we are thus not prohibited from fine-tuning UniRep weights
through gradient descent.

During the reimplementation,
we also discovered that JAX provided convenient utilities (`jit` and `vmap`)
to convert loops into vectorized operations on tensors.
This had a pleasant effect of helping us write more performant code,
while also forcing us to reason clearly about the semantic meaning
of our tensor dimensions.

### Tensor Ops Reimplementation

The process of tensor ops reimplementation were as follows.

Firstly, we started from the RNN cell (`mlstm1900_step`),
which sequentially walks down the protein sequence
and generates the single step embedding.
Secondly, we wrapped the RNN cell using `lax.scan`
to scan over a single sequence, generating `mlstm1900_batch`.
Thirdly, we then used `jax.vmap`
to vectorize the operation over multiple sequences,
thus generating `mlstm1900`.
These are available in the source `jax_unirep/layers.py`.

Besides reimplementation,
we also took care to document the semantic meaning of tensor dimensions.
This had the pleasant side effect of forcing us
to order our tensor dimensions in a sane fashion,
such that the "batch" or "sample" dimension was always the first one,
with explicit documentation written to guide a new user on this convention.

While reimplementing the model, we also generated a test suite for it.
Most of our tests check that the shapes of returned tensors were correct.
For the unit RNN cell, we provided an example-based test with random matrices.
The same applied to the batch function.
However, for the full forward model, we provided a property-based test,
which checked that tensor dimensions were correct
given different numbers of samples.
These are available in the source `tests/` directory.

During the reimplementation,
we found that certain operations that we took for granted
to be implemented one way
could actually be implemented in multiple ways.
One example was the `sigmoid()` function.
The TensorFlow implementation is:

$$\frac{1}{1 + e^{-x}}$$

However, an alternative implementation includes:

$$\frac{1}{2} \text{tanh}(x) + \frac{1}{2}$$

Yet both are called "sigmoid" functions.
Because their slopes differ,
their outputs also differ drastically.
We originally used the $\text{tanh}$ version,
but found a large discrepancy between the reimplemented model's output
and the original's output.
It took digging deep through the TensorFlow source code
to realize that the exponentiated sigmoid was the one being used.

A similar lesson was learned while reimplementing the L2 norm of our weights.

### Utility Reimplementation

More tricky than the tensors are
getting strings to tensor conversion done correct.
For the `get_reps()` functionality,
we copied quite a bit of source code from the original,
including the original authors' implementation of
embedding a sequence into an $l$-by-10 embedding matrix first.
However, we added tests to guarantee that they were robust,
as well as technical documentation to clarify how it works.

We did this because one way that deep learning models can be fragile
is that the input tensors can be generated incorrectly
but still have the expected shapes.
Thus, though the structure of input tensors might be correct,
their semantic meaning would be completely wrong.
(Adversarial examples can be generated this way.)
Thus, the input to the model has to be carefully controlled.
Moreover, input tensors are _not_ the raw-est form of data;
for a protein engineer, the protein sequence is.
Thus, having robustly tested functions that generate the input tensors
with correct semantic meaning
is crucial to having confidence
that the model works correctly end-to-end.

### APIs

Because we expect the model to be used as a Python library,
the model source and weights are packaged together.
This makes it much more convenient for end-users,
as the cognitive load of downloading starter weights is eliminated.

The `get_reps()` function is designed
such that it is flexible enough to accept a single sequence
or an iterable of sequences.
This also reduces cognitive load for end-users,
some of whom might want to process only a single sequence,
while others might be operating in batch mode.
`get_reps()` also correctly handles sequences of multiple lengths,
further simplifying usage for end-users.
As usual, tests are provided,
bringing the same degree of confidence as we would expect
from tested software.

Evotuning is an important task when using UniRep [@alley2019unified],
and we provide a convenient API through the `evotune()` function.
Here, we use of Optuna to automatically find the right hyperparameters
for finetuning weights, using the protocol that the original authors describe.
This enables end-users to "set and forget" the model fitting protocol
instead of needing to babysit a deep learning optimization routine.
Like `get_reps()`, `evotune()` and its associated utility functions
have at least an example-based test, if not a property-based test
associated with them.

## Reimplementation Performance

Anecdotally, on our benchmarks using internal data,
it takes about 30 seconds to process one sequence.
On a "lifted-and-shifted" version of the model running on cloud GPUs,
it took overnight to featurize 6,000 sequences.
By contrast, with our reimplementation, processing 10,000 dummy sequences
takes only 40 seconds on a single CPU core.

A formal speed comparison using the same CPU is available below.

<!-- %%figure -->
![
    Figure 1: 
    Speed comparison between the original implementation (UniRep)
    and our re-implementation (Jax-UniRep). Both one and ten random sequences of length ten
    were transformed by both implementations. 
    Our re-implementation could make use of vectorization
    in the multi-sequence case, 
    whereas in the original implementation each sequence was transformed
    one at a time.
](./figures/speed_barplot.png)

We also needed to check that our reimplementation correctly embeds sequences.
To do so, we ran a dummy sequence
through the original and through our reimplementation,
and compared the computed representations.
Because it is 1900-long, a visual check for correctness
is a trace of 1900-long embedding.

<!-- %%figure -->
![
    Figure 2:
    Comparison of the average hidden state between the implementations 
    when transforming the same sequence. 
    Because the two traces of the hidden state dimensions overlapped
    almost perfectly, a small constant was added to the UniRep values,
    such that both traces become visible. The inset shows
    50 out of the total 1900 dimensions.
](./figures/rep_trace.png)

We also verified that the embeddings of our reimplementation
were informative for top models,
and trained a model to predict avGFP brightness (as the authors did).
5-fold cross-validated performance is shown below.

<!-- %%figure -->
![](./figures/top_model.png)

<!--
Process of reimplementing the model parts in JAX (RNN cell, weight normalization).
Speed comparison (maybe on CPU and GPU), on one sequence and eg 10 sequences.
-> Figure of speed comparison (bar or scatter plot)
Rep comparison between TF and JAX implementation
-> Figure, x-axis Unirep positions (1-1900), y-axis embedding vals of TF and JAX on same sequence
    Do this for all three reps.
-->

## Lessons Learned

We found the reimplementation exercise to be highly educational,
and would not characterize it as a waste of time at all.
In particular, we gained a mechanical understanding of the model,
and through documenting the model functions thoroughly
with the semantic meaning of tensor dimensions,
we were able to greatly reduce our own confusion when
debugging why the model would fail.

Writing automated tests for the model functions,
in basically the same way as we would test software,
gave us the confidence that our code changes would not
inadvertently break something.
We also could more easily narrow down where failures were happening
when developing new code that interacted with the model
(such as providing input tensors).

Competing tensor libraries that do not interoperate seamlessly
means data scientists are forced to learn one
(and be mentally locked in).
To break free of framework lock-in,
being able to translate between frameworks is highly valuable.
Model reimplementation was highly beneficial for this.

If "models are software 2.0" [@kaparthy2017software2],
then data science teams might do well
to treat model weights as software artefacts
that are shipped to end-users,
and take care to design sane APIs
that enable other developers to use it in ways
that minimize cognitive load.

## Future Work

As we have, at this point, only implemented the 1900-cell model.
Going forth, we aim to work on implementing the 256- and 64-cell model.

## Acknowledgments

We thank the UniRep authors for open sourcing their model.
It is our hope that our reimplementation helps with adoption of the model
in a variety of settings, and increases its impact.

## References
