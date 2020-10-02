# Supplementary information - Jax-unirep

## Speed profiling of Unirep

The functions used for profiling the original UniRep implementation
and our `jax-unirep` reimplementation respectively:

```python
# assume babbler is imported from tf-unirep
def profile_tf_unirep(seqs):
    with tf.variable_scope("embed_matrix", reuse=tf.AUTO_REUSE):
        b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)
        for seq in seqs:
            avg, final, cell = b.get_rep(seq)

# assume get_reps is imported from jax-unirep
def profile_jax_unirep(seqs):
    get_reps(seqs)
```

![
    Flame graph of the original UniRep's implementation,
    down to 10 levels deep from the profiling function that was called.
](./figures/unirep-profile.png)

![
    Flame graph of the jax-unirep reimplementation,
    down to 10 levels deep from the profiling function that was called.
](./figures/jax-unirep-profile.png)

As is visible from the code execution flamegraph,
the unreasonably long time that it takes to process ten sequences
was probably due to the time spent in TensorFlow's session.
Because of TensorFlow's compiled nature,
we thus deduced that the majority of the execution time
was most likely in the graph compilation phase.
Unfortunately, cProfile could not give us any further detail
beyond the `_pywrap_tensorflow_internal.TF_SessionRun_wrapper`
in the call graph,
meaning we were unable to conveniently peer into the internals of TF execution
without digging further.


## Reimplementation Main Points

### Tensor Ops Reimplementation

The process of tensor ops reimplementation were as follows.

Firstly, we started from the RNN cell (`mLSTMCell`),
which sequentially walks down the protein sequence
and generates the single step embedding.
We thus end up with a "unit cell" function:

```python
def mLSTMCell(params, carry, x_t):
    h_t, c_t = carry
    # Unit cell implementation goes here.
    return (h_t, c_t), h_t
```

Secondly, we wrapped the RNN cell using `lax.scan`
to scan over a single sequence.
This is the `mLSTMBatch` function:

```python
def mLSTMBatch(params, batch):
    # code setup goes here.
    step_func = partial(mLSTMCell, params)

    # use of lax.scan below:
    (h_final, c_final), outputs = lax.scan(
        step_func, init=(h_t, c_t), xs=batch
    )
    return h_final, c_final, outputs
```

Thirdly, we then used `jax.vmap`
to vectorize the operation over multiple sequences,
thus generating `mLSTM`:

```python
def mLSTM(params, x):
    def mLSTM_vmappable(x):
        return mLSTMBatch(params=params, batch=x)

    h_final, c_final, outputs = vmap(mLSTM_vmappable)(x)
    return h_final, c_final, outputs
```

Effectively, `jax.vmap` and `lax.scan`
replace for-loops that we would otherwise write,
which would incur Python type-checking overhead that would accumulate.
`lax.scan` being effectively a pre-compiled for-loop
enables pre-allocation of the necessary memory needed for backpropagation,
which also contributes to a speed-up.
As the for-loop type checking penalty is well-known in Python,
a detailed comparison between `jax.vmap`, `lax.scan`, and a vanilla `for` loop
is out of scope for this paper.
The full source code is available in `jax_unirep/layers.py`.

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
As a known benefit with software testing,
our tests allowed us to rebuild the full model piece by piece,
while always making sure that each new piece did not break
the existing pieces.

### Utility Reimplementation

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
In particular, we spent time ensuring that
`get_reps()` correctly batches sequences of the same size together
before calculating their reps,
while returning the reps in the same order as the sequences passed in.
As usual, tests are provided,
bringing the same degree of confidence as we would expect
from tested software.


## Top model

To test our reimplementation in a practical workflow,
we recreated the GFP brightness prediction model
that was introduced in the original paper.

We trained the model on the same set of GFP mutants
as the orignial authors did, but used the `get_reps` function
to embed the protein sequences that we then passed to the
classification model.
The model performance shown below
is based on 5-fold crossvalidation of the whole GFP dataset.
The full analysis is available as a jupyter notebook in the
`jax-unirep` repository.

![
    GFP brightness classification using
    a logistic regression top model taking in the
    1900-long average hidden state representations
    of the GFP protein sequences.
    Left: Distribution of GFP brightness values in the dataset.
    Red dotted line indicates classification breakpoint.
    Points to the left get labeled as "Dark",
    while points to the right get labeled "Bright".
    Right: Confusion matrix showing the
    classification accuracy of the model.
](./figures/top_model.png)
