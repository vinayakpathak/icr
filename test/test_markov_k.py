import math

import torch

from lpe.markov_k_transformer import (
    bayes_target_logprob,
    compute_posterior_params,
    compute_transition_counts,
    encode_state_windows,
    markov_sequence_logprob_given_theta,
    rollout_with_cache,
    rollout_with_cache_batch,
)
from lpe.markov_transformer import MarkovTransformer


def test_encode_state_windows_k2():
    windows = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
    encoded = encode_state_windows(windows)
    assert encoded.tolist() == [0, 1, 2, 3]


def test_transition_counts_k2():
    seq = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
    ones, zeros = compute_transition_counts(seq, k=2)
    assert ones.tolist() == [0.0, 1.0, 1.0, 0.0]
    assert zeros.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_posterior_params_k2_beta11():
    context = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
    alpha_post, beta_post = compute_posterior_params(context, k=2, alpha=1.0, beta=1.0)
    assert alpha_post.tolist() == [1.0, 2.0, 2.0, 1.0]
    assert beta_post.tolist() == [1.0, 1.0, 1.0, 2.0]


def test_logprob_given_theta_matches_manual_k1():
    # k=1, context has last bit 0, theta[state=0]=0.8, theta[state=1]=0.3
    context = torch.tensor([0], dtype=torch.long)
    target = torch.tensor([1, 1], dtype=torch.long)
    theta = torch.tensor([0.8, 0.3], dtype=torch.float64)

    # P(1 then 1 | context=0) = 0.8 * 0.3
    expected = math.log(0.8) + math.log(0.3)
    got = markov_sequence_logprob_given_theta(context, target, theta, k=1)
    assert abs(got - expected) < 1e-10


def test_bayes_target_logprob_reasonable_range():
    context = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.long)
    target = torch.tensor([1, 0, 1], dtype=torch.long)
    lp = bayes_target_logprob(context=context, target=target, k=2, alpha=1.0, beta=1.0)
    assert lp < 0.0
    assert lp > -100.0


def test_rollout_with_cache_length_and_range():
    torch.manual_seed(7)
    model = MarkovTransformer(max_seq_len=32, d_model=16, n_layers=1, n_heads=1, d_mlp=16)
    context = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    out = rollout_with_cache(model, prefix=context, length=4)
    assert out.shape == (4,)
    assert out.dtype == torch.long
    assert set(out.tolist()).issubset({0, 1})


def test_rollout_with_cache_batch_shape_and_range():
    torch.manual_seed(11)
    model = MarkovTransformer(max_seq_len=40, d_model=16, n_layers=1, n_heads=1, d_mlp=16)
    context = torch.tensor([1, 0, 1, 1], dtype=torch.long)
    out = rollout_with_cache_batch(model, prefix=context, length=5, batch_size=3)
    assert out.shape == (3, 5)
    assert out.dtype == torch.long
    flat = set(out.reshape(-1).tolist())
    assert flat.issubset({0, 1})
