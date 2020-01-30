# Reimplementing Unirep in JAX

## Introduction

Unirep allows for embedding arbitrary length sequences in a 1900 x 1 feature vector. 
Vectors can be used for clustering or input for top model to predict properties of proteins.
Original model was implemented in TF1, and only allowed for one sequence to be transformed at once.
Explain JAX, difference to TF.
Now we can transform any number of same length sequences at once in a vectorised fashion. 
Also build a convenience function to transform arbitray length sequences together by batching same length seqs internally.

## Main

Process of reimplementing the model parts in JAX (RNN cell, weight normalization).
Speed comparison (maybe on CPU and GPU), on one sequence and eg 10 sequences. 
-> Figure of speed comparison (bar or scatter plot)
Rep comparison between TF and JAX implementation 
-> Figure, x-axis Unirep positions (1-1900), y-axis embedding vals of TF and JAX on same sequence
    Do this for all three reps.

## Outlook

Implementation of 256 and 64 cell model.
So far no training possible, only forward pass. Implement weight updates using new seqs (Evotuning)
