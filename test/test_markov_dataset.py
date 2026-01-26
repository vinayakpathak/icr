"""
Tests for the MarkovDataset data loader.
"""

import torch

from lpe.markov_transformer import MarkovDataset, count_markov_transitions


def test_markov_dataset_shapes_and_binary():
    dataset = MarkovDataset(seq_len=25, num_samples=5, fixed_p=0.6, fixed_q=0.4)
    seq = dataset[0]
    assert seq.shape == (25,)
    assert seq.dtype == torch.long
    assert torch.all((seq == 0) | (seq == 1))


def test_markov_dataset_transition_probs_fixed():
    torch.manual_seed(0)
    p = 0.8
    q = 0.3
    dataset = MarkovDataset(seq_len=100, num_samples=200, fixed_p=p, fixed_q=q, initial_state_prob=0.5)

    n00 = n01 = n10 = n11 = 0
    for i in range(len(dataset)):
        seq = dataset[i]
        c00, c01, c10, c11 = count_markov_transitions(seq)
        n00 += c00
        n01 += c01
        n10 += c10
        n11 += c11

    denom0 = n00 + n01
    denom1 = n10 + n11
    assert denom0 > 0
    assert denom1 > 0

    p_hat = n01 / denom0
    q_hat = n10 / denom1

    assert abs(p_hat - p) < 0.05
    assert abs(q_hat - q) < 0.05
