#!/usr/bin/env python3
"""
Markov Transformer for LPE Demonstration.

Implements a transformer that learns to do in-context Bayesian inference for
binary Markov chains with parameters p = P(x_n=1|x_{n-1}=0) and
q = P(x_n=0|x_{n-1}=1).
"""

import csv
import math
import os
import warnings
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# =====================
#  Transformer Architecture
# =====================

class TransformerBlock(nn.Module):
    """Single transformer block matching ic_regression.py architecture."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int, use_prenorm: bool = True):
        super().__init__()
        self.use_prenorm = use_prenorm
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        if self.use_prenorm:
            # Pre-layer-norm: norm before attention/MLP
            h = self.ln1(x)
            attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
            x = x + attn_out
            h = self.ln2(x)
            h = self.mlp(h)
            x = x + h
        else:
            # Post-layer-norm (GPT2 style): norm after residual connection
            attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
            x = x + attn_out
            x = self.ln1(x)
            h = self.mlp(x)
            x = x + h
            x = self.ln2(x)
        return x


class MarkovTransformer(nn.Module):
    """
    Decoder-only transformer for binary sequences, matching ic_regression.py architecture.

    Input: binary sequence of shape (B, T) where each element is 0 or 1
    Output: logits of shape (B, T, 2) for predicting next token
    """

    def __init__(
        self,
        max_seq_len: Optional[int] = None,
        d_model: int = 16,
        n_layers: int = 1,
        n_heads: int = 1,
        d_mlp: int = 16,
        use_prenorm: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.use_prenorm = use_prenorm

        self.token_emb = nn.Embedding(2, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp, use_prenorm=use_prenorm)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 2)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        for block in self.blocks:
            for m in block.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape

        if self.max_seq_len is not None and T > self.max_seq_len:
            warnings.warn(
                f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}. "
                "Model will process it, but performance may degrade or memory issues may occur.",
                UserWarning,
            )

        x = self.token_emb(x)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.output_proj(x)
        return logits

    def predict_next_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return logits[:, -1, :]

    def sample(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.predict_next_logits(x) / temperature
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        if sampled.numel() == 1:
            return sampled.item()
        return sampled

    def rollout(self, prefix: torch.Tensor, length: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            prefix_len = len(prefix)
            if self.max_seq_len is not None:
                max_allowed_length = self.max_seq_len - prefix_len
                if max_allowed_length <= 0:
                    warnings.warn(
                        f"Prefix length {prefix_len} exceeds or equals max_seq_len={self.max_seq_len}. "
                        "Cannot generate any tokens.",
                        UserWarning,
                    )
                    return torch.tensor([], dtype=prefix.dtype, device=prefix.device)
                actual_length = min(length, max_allowed_length)
                if actual_length < length:
                    warnings.warn(
                        f"Requested rollout length {length} exceeds maximum allowed {max_allowed_length} "
                        f"(prefix length: {prefix_len}, max_seq_len: {self.max_seq_len}). "
                        f"Reducing to {actual_length}.",
                        UserWarning,
                    )
            else:
                actual_length = length

            current = prefix.clone()
            generated = []

            for _ in range(actual_length):
                next_token = self.sample(current, temperature=temperature)
                next_token_val = int(next_token) if isinstance(next_token, (int, float)) else next_token.item()
                generated.append(next_token_val)
                current = torch.cat(
                    [current, torch.tensor([next_token_val], dtype=current.dtype, device=current.device)]
                )

            return torch.tensor(generated, dtype=current.dtype, device=current.device)


# =====================
#  Markov Data Utilities
# =====================

def sample_markov_sequence(
    p: float,
    q: float,
    seq_len: int,
    initial_state_prob: float = 0.5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample a binary sequence from a two-state Markov chain."""
    if seq_len <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    seq = torch.empty(seq_len, dtype=torch.long, device=device)
    seq[0] = 1 if torch.rand(1, device=device).item() < initial_state_prob else 0
    for t in range(1, seq_len):
        prev = int(seq[t - 1].item())
        if prev == 0:
            seq[t] = 1 if torch.rand(1, device=device).item() < p else 0
        else:
            seq[t] = 0 if torch.rand(1, device=device).item() < q else 1
    return seq


def count_markov_transitions(sequence: torch.Tensor) -> Tuple[int, int, int, int]:
    """Return counts (n00, n01, n10, n11) for transitions in the sequence."""
    if sequence.numel() < 2:
        return 0, 0, 0, 0
    prev = sequence[:-1]
    nxt = sequence[1:]
    n00 = int(((prev == 0) & (nxt == 0)).sum().item())
    n01 = int(((prev == 0) & (nxt == 1)).sum().item())
    n10 = int(((prev == 1) & (nxt == 0)).sum().item())
    n11 = int(((prev == 1) & (nxt == 1)).sum().item())
    return n00, n01, n10, n11


def estimate_transition_probs(sequence: torch.Tensor) -> Tuple[float, float]:
    """Estimate (p, q) from a sequence via empirical transitions."""
    n00, n01, n10, n11 = count_markov_transitions(sequence)
    denom0 = n00 + n01
    denom1 = n10 + n11
    p_hat = n01 / denom0 if denom0 > 0 else 0.5
    q_hat = n10 / denom1 if denom1 > 0 else 0.5
    return float(p_hat), float(q_hat)


# =====================
#  Training
# =====================

class MarkovDataset(Dataset):
    """
    Dataset for training:
    - sample p ~ Beta(alpha_p, beta_p)
    - sample q ~ Beta(alpha_q, beta_q)
    - sample sequence from Markov chain with (p, q)
    """

    def __init__(
        self,
        seq_len: int,
        num_samples: int,
        alpha_p: float = 1.0,
        beta_p: float = 1.0,
        alpha_q: float = 1.0,
        beta_q: float = 1.0,
        initial_state_prob: float = 0.5,
        fixed_p: Optional[float] = None,
        fixed_q: Optional[float] = None,
    ):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.alpha_q = alpha_q
        self.beta_q = beta_q
        self.initial_state_prob = initial_state_prob
        self.fixed_p = fixed_p
        self.fixed_q = fixed_q

        self._beta_p = dist.Beta(torch.tensor(alpha_p), torch.tensor(beta_p))
        self._beta_q = dist.Beta(torch.tensor(alpha_q), torch.tensor(beta_q))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.fixed_p is None:
            p = float(self._beta_p.sample().item())
        else:
            p = float(self.fixed_p)
        if self.fixed_q is None:
            q = float(self._beta_q.sample().item())
        else:
            q = float(self.fixed_q)
        sequence = sample_markov_sequence(
            p,
            q,
            self.seq_len,
            initial_state_prob=self.initial_state_prob,
            device=None,
        )
        return sequence


def train_markov_transformer(
    model: MarkovTransformer,
    seq_len: int = 256,
    batch_size: int = 64,
    num_steps: int = 50000,
    learning_rate: float = 3e-4,
    warmup_steps: int = 5000,
    grad_clip: Optional[float] = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    print_every: int = 1000,
    checkpoint_dir: Optional[str] = None,
    alpha_p: float = 1.0,
    beta_p: float = 1.0,
    alpha_q: float = 1.0,
    beta_q: float = 1.0,
) -> MarkovTransformer:
    """
    Train the transformer using the procedure:
    - Sample p ~ Beta(alpha_p, beta_p)
    - Sample q ~ Beta(alpha_q, beta_q)
    - Sample sequence from Markov chain
    - Train with autoregressive NLL loss
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    dataset = MarkovDataset(
        seq_len=seq_len,
        num_samples=100000,
        alpha_p=alpha_p,
        beta_p=beta_p,
        alpha_q=alpha_q,
        beta_q=beta_q,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    step = 0
    running_loss = 0.0
    data_iter = iter(dataloader)

    print(f"Training on device: {device}")
    print(f"Sequence length: {seq_len}, Batch size: {batch_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Learning rate: {learning_rate}, Warmup steps: {warmup_steps}, Gradient clip: {grad_clip}"
    )

    def get_lr(current_step: int) -> float:
        if current_step < warmup_steps:
            return learning_rate * (current_step / warmup_steps)
        return learning_rate

    while step < num_steps:
        try:
            sequences = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            sequences = next(data_iter)

        sequences = sequences.to(device)
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]

        logits = model.forward(inputs)
        loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % print_every == 0:
            avg_loss = running_loss / print_every
            current_lr = get_lr(step)
            print(f"[step {step}] loss={avg_loss:.4f}, lr={current_lr:.6f}")
            running_loss = 0.0

        if checkpoint_dir is not None and step % print_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"markov_step_{step}.pt")
            torch.save(model.state_dict(), checkpoint_path)

    print("Training completed!")
    return model


# =====================
#  Analytical Computations
# =====================

def compute_markov_posterior(
    context: torch.Tensor,
    alpha_p: float = 1.0,
    beta_p: float = 1.0,
    alpha_q: float = 1.0,
    beta_q: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Compute posterior parameters for p and q given a context.

    p ~ Beta(alpha_p, beta_p) and q ~ Beta(alpha_q, beta_q).
    Observed transition counts update each Beta independently.
    """
    n00, n01, n10, n11 = count_markov_transitions(context)
    alpha_p_post = alpha_p + n01
    beta_p_post = beta_p + n00
    alpha_q_post = alpha_q + n10
    beta_q_post = beta_q + n11
    return alpha_p_post, beta_p_post, alpha_q_post, beta_q_post


def compute_markov_posterior_means(
    context: torch.Tensor,
    alpha_p: float = 1.0,
    beta_p: float = 1.0,
    alpha_q: float = 1.0,
    beta_q: float = 1.0,
) -> Tuple[float, float]:
    """Return posterior mean of (p, q) given context."""
    alpha_p_post, beta_p_post, alpha_q_post, beta_q_post = compute_markov_posterior(
        context,
        alpha_p=alpha_p,
        beta_p=beta_p,
        alpha_q=alpha_q,
        beta_q=beta_q,
    )
    p_mean = alpha_p_post / (alpha_p_post + beta_p_post)
    q_mean = alpha_q_post / (alpha_q_post + beta_q_post)
    return float(p_mean), float(q_mean)


def compute_string_probability_given_pq(
    target_string: torch.Tensor,
    last_bit: int,
    p: float,
    q: float,
) -> float:
    """Compute P(target_string | last_bit, p, q) for a Markov chain."""
    prob = 1.0
    prev = int(last_bit)
    for bit in target_string:
        bit_val = int(bit.item())
        if prev == 0:
            prob *= p if bit_val == 1 else (1 - p)
        else:
            prob *= q if bit_val == 0 else (1 - q)
        prev = bit_val
    return prob


def compute_string_probability_analytical(
    target_string: torch.Tensor,
    context: torch.Tensor,
    alpha_p: float = 1.0,
    beta_p: float = 1.0,
    alpha_q: float = 1.0,
    beta_q: float = 1.0,
) -> float:
    """
    Compute P(target_string | context) by integrating out p and q.

    Uses sequential predictive updates based on Beta-Bernoulli conjugacy
    for transitions out of state 0 (p) and state 1 (q).
    """
    if context.numel() == 0:
        raise ValueError("Context must be non-empty for Markov predictive.")

    n00, n01, n10, n11 = count_markov_transitions(context)
    prev = int(context[-1].item())
    prob = 1.0

    for bit in target_string:
        bit_val = int(bit.item())
        if prev == 0:
            denom = alpha_p + beta_p + n00 + n01
            if bit_val == 1:
                prob *= (alpha_p + n01) / denom
                n01 += 1
            else:
                prob *= (beta_p + n00) / denom
                n00 += 1
        else:
            denom = alpha_q + beta_q + n10 + n11
            if bit_val == 0:
                prob *= (alpha_q + n10) / denom
                n10 += 1
            else:
                prob *= (beta_q + n11) / denom
                n11 += 1
        prev = bit_val

    return prob


# =====================
#  Model vs Bayes Comparison
# =====================

def get_model_nextbit_probability(
    model: MarkovTransformer,
    context: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """Return model probability of next bit being 1 given context."""
    model = model.to(device)
    model.eval()
    if context.dim() == 1:
        context = context.unsqueeze(0)
    context = context.to(device)
    with torch.no_grad():
        logits = model.predict_next_logits(context)
        probs = torch.softmax(logits, dim=-1)
        return float(probs[:, 1].detach().cpu().numpy()[0])


def get_bayes_optimal_prediction(
    context: torch.Tensor,
    alpha_p: float = 1.0,
    beta_p: float = 1.0,
    alpha_q: float = 1.0,
    beta_q: float = 1.0,
) -> float:
    """Return Bayes predictive probability for next bit being 1."""
    if context.numel() == 0:
        raise ValueError("Context must be non-empty for Markov predictive.")
    last_bit = int(context[-1].item())
    p_mean, q_mean = compute_markov_posterior_means(
        context,
        alpha_p=alpha_p,
        beta_p=beta_p,
        alpha_q=alpha_q,
        beta_q=beta_q,
    )
    if last_bit == 0:
        return p_mean
    return 1.0 - q_mean


def compare_model_vs_bayes(
    model: MarkovTransformer,
    num_contexts: int = 100,
    context_length: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    plot_dir: str = "plots",
    p_min: float = 0.05,
    p_max: float = 0.95,
    q_min: float = 0.05,
    q_max: float = 0.95,
    seed: int = 123,
) -> None:
    """Compare model predictions against ideal Bayes predictor."""
    print(f"Generating {num_contexts} contexts and comparing predictions...")

    rng = np.random.default_rng(seed)
    model = model.to(device)
    model.eval()

    true_ps = rng.uniform(p_min, p_max, size=num_contexts)
    true_qs = rng.uniform(q_min, q_max, size=num_contexts)

    model_predictions = []
    bayes_predictions = []
    true_next_probs = []
    last_bits = []

    with torch.no_grad():
        for i, (true_p, true_q) in enumerate(zip(true_ps, true_qs)):
            context = sample_markov_sequence(
                float(true_p),
                float(true_q),
                context_length,
                initial_state_prob=0.5,
                device=torch.device(device),
            )
            last_bit = int(context[-1].item())
            true_next = float(true_p if last_bit == 0 else (1.0 - true_q))

            model_p = get_model_nextbit_probability(model, context, device=device)
            bayes_p = get_bayes_optimal_prediction(context.cpu(), alpha_p=1.0, beta_p=1.0, alpha_q=1.0, beta_q=1.0)

            model_predictions.append(model_p)
            bayes_predictions.append(bayes_p)
            true_next_probs.append(true_next)
            last_bits.append(last_bit)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{num_contexts} contexts...")

    model_predictions = np.array(model_predictions)
    bayes_predictions = np.array(bayes_predictions)
    true_next_probs = np.array(true_next_probs)

    model_errors = model_predictions - true_next_probs
    bayes_errors = bayes_predictions - true_next_probs

    model_mse = np.mean(model_errors ** 2)
    bayes_mse = np.mean(bayes_errors ** 2)

    print(f"\nModel MSE: {model_mse:.6f}")
    print(f"Bayes MSE: {bayes_mse:.6f}")
    print(f"Model/Bayes MSE ratio: {model_mse / bayes_mse:.4f}")

    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.scatter(bayes_predictions, model_predictions, alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect match")
    ax.set_xlabel("Bayes Optimal Prediction")
    ax.set_ylabel("Model Prediction")
    ax.set_title("Model vs Bayes Optimal Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[0, 1]
    ax.scatter(true_next_probs, bayes_predictions, alpha=0.6, s=30, label="Bayes Optimal", color="green")
    ax.scatter(true_next_probs, model_predictions, alpha=0.6, s=30, label="Model", color="blue")
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect", alpha=0.5)
    ax.set_xlabel("True next-bit probability")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Predictions vs True Next-bit Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(model_errors, bins=30, alpha=0.7, label=f"Model (MSE={model_mse:.4f})", color="blue", density=True)
    ax.hist(bayes_errors, bins=30, alpha=0.7, label=f"Bayes (MSE={bayes_mse:.4f})", color="green", density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax.set_xlabel("Prediction Error (predicted - true)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(true_next_probs, model_errors, alpha=0.6, s=30, label="Model", color="blue")
    ax.scatter(true_next_probs, bayes_errors, alpha=0.6, s=30, label="Bayes", color="green")
    ax.axhline(0, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax.set_xlabel("True next-bit probability")
    ax.set_ylabel("Prediction Error")
    ax.set_title("Error vs True Next-bit Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_plot_path = os.path.join(plot_dir, "markov_model_vs_bayes_comparison.png")
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Comparison plot saved to {comparison_plot_path}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(true_next_probs, np.abs(model_errors), alpha=0.6, s=30, label="Model", color="blue")
    ax.scatter(true_next_probs, np.abs(bayes_errors), alpha=0.6, s=30, label="Bayes", color="green")
    ax.set_xlabel("True next-bit probability")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Absolute Error vs True Next-bit Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(true_next_probs, model_errors ** 2, alpha=0.6, s=30, label="Model", color="blue")
    ax.scatter(true_next_probs, bayes_errors ** 2, alpha=0.6, s=30, label="Bayes", color="green")
    ax.set_xlabel("True next-bit probability")
    ax.set_ylabel("Squared Error")
    ax.set_title("Squared Error vs True Next-bit Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    correlation = np.corrcoef(bayes_predictions, model_predictions)[0, 1]
    ax.scatter(bayes_predictions, model_predictions, alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect match")
    ax.set_xlabel("Bayes Optimal Prediction")
    ax.set_ylabel("Model Prediction")
    ax.set_title(f"Correlation: {correlation:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    detailed_plot_path = os.path.join(plot_dir, "markov_model_vs_bayes_detailed.png")
    plt.savefig(detailed_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Detailed analysis plot saved to {detailed_plot_path}")


# =====================
#  Estimation Methods
# =====================

def estimate_rollout_method(
    model: MarkovTransformer,
    context: torch.Tensor,
    target_string: torch.Tensor,
    num_rollouts: int = 10000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    print_every: int = 5000,
) -> Tuple[float, float]:
    """Direct rollout method for estimating P(target_string | context)."""
    model = model.to(device)
    model.eval()

    context = context.to(device)
    target_string = target_string.to(device)
    m = len(target_string)

    hits = 0
    print(f"Running {num_rollouts} rollouts...")
    with torch.no_grad():
        for i in range(num_rollouts):
            rollout = model.rollout(context, length=m, temperature=1.0)
            if torch.equal(rollout.to(device), target_string):
                hits += 1

            if (i + 1) % print_every == 0 or (i + 1) == num_rollouts:
                current_estimate = hits / (i + 1)
                print(
                    f"  Progress: {i+1}/{num_rollouts} rollouts, hits: {hits}, "
                    f"current estimate: {current_estimate:.8e}"
                )

    estimate = hits / num_rollouts
    std_error = math.sqrt(estimate * (1 - estimate) / num_rollouts) if num_rollouts > 0 else 0.0
    return estimate, std_error


def estimate_posterior_sampling_method(
    model: MarkovTransformer,
    context: torch.Tensor,
    target_string: torch.Tensor,
    num_samples: int = 1000,
    rollout_length: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    posterior_plot_path: Optional[str] = None,
    rollout_transition_plot_path: Optional[str] = None,
    alpha_p: float = 1.0,
    beta_p: float = 1.0,
    alpha_q: float = 1.0,
    beta_q: float = 1.0,
    print_every: int = 5,
    max_transition_curves: int = 100,
) -> Tuple[float, float]:
    """
    Posterior sampling method (Rao-Blackwellized).

    For each sample:
    1. Roll out y_{n+1:M} for large M
    2. Estimate (p, q) from the rollout transitions
    3. Compute P(target_string | p, q) analytically
    """
    model = model.to(device)
    model.eval()

    context = context.to(device)
    target_string = target_string.to(device)
    last_bit = int(context[-1].item())

    print(f"Running {num_samples} posterior samples, each with {rollout_length} rollout steps...")
    f_values = []
    p_samples = []
    q_samples = []
    transition_curves = []

    with torch.no_grad():
        for i in range(num_samples):
            rollout = model.rollout(context, length=rollout_length, temperature=1.0)
            p_hat, q_hat = estimate_transition_probs(rollout)
            p_samples.append(p_hat)
            q_samples.append(q_hat)

            f_pq = compute_string_probability_given_pq(target_string, last_bit, p_hat, q_hat)
            f_values.append(f_pq)

            if rollout.numel() > 1 and len(transition_curves) < max_transition_curves:
                prev = rollout[:-1]
                nxt = rollout[1:]
                trans0 = (prev == 0).float()
                trans1 = (prev == 1).float()
                trans01 = ((prev == 0) & (nxt == 1)).float()
                trans10 = ((prev == 1) & (nxt == 0)).float()

                c0 = trans0.cumsum(0)
                c1 = trans1.cumsum(0)
                c01 = trans01.cumsum(0)
                c10 = trans10.cumsum(0)

                p_curve = torch.where(c0 > 0, c01 / c0, torch.full_like(c0, 0.5))
                q_curve = torch.where(c1 > 0, c10 / c1, torch.full_like(c1, 0.5))
                transition_curves.append((p_curve.cpu().numpy(), q_curve.cpu().numpy()))

            if (i + 1) % print_every == 0 or (i + 1) == num_samples:
                current_estimate = np.mean(f_values)
                current_std = np.std(f_values) / math.sqrt(len(f_values)) if len(f_values) > 1 else 0.0
                print(
                    f"  Progress: {i+1}/{num_samples} samples, current estimate: "
                    f"{current_estimate:.8e} ± {current_std:.8e}"
                )

    f_values = np.array(f_values, dtype=np.float64)
    p_samples = np.array(p_samples, dtype=np.float64)
    q_samples = np.array(q_samples, dtype=np.float64)

    estimate = float(f_values.mean()) if f_values.size else 0.0
    std_error = float(f_values.std(ddof=1) / math.sqrt(num_samples)) if num_samples > 1 else 0.0

    alpha_p_post, beta_p_post, alpha_q_post, beta_q_post = compute_markov_posterior(
        context.cpu(),
        alpha_p=alpha_p,
        beta_p=beta_p,
        alpha_q=alpha_q,
        beta_q=beta_q,
    )

    if posterior_plot_path is None:
        os.makedirs("plots", exist_ok=True)
        posterior_plot_path = "plots/markov_posterior_sampling_comparison.png"
    else:
        plot_dir = os.path.dirname(posterior_plot_path) or "plots"
        os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    p_range = torch.linspace(0.001, 0.999, 1000)
    p_beta = dist.Beta(torch.tensor(alpha_p_post), torch.tensor(beta_p_post))
    p_pdf = p_beta.log_prob(p_range).exp().numpy()

    ax = axes[0]
    ax.hist(p_samples, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black")
    ax.plot(p_range.numpy(), p_pdf, "r-", linewidth=2)
    ax.set_xlabel("p = P(1|0)")
    ax.set_ylabel("Density")
    ax.set_title(f"Posterior for p: Beta({alpha_p_post:.2f}, {beta_p_post:.2f})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    q_range = torch.linspace(0.001, 0.999, 1000)
    q_beta = dist.Beta(torch.tensor(alpha_q_post), torch.tensor(beta_q_post))
    q_pdf = q_beta.log_prob(q_range).exp().numpy()

    ax = axes[1]
    ax.hist(q_samples, bins=50, density=True, alpha=0.7, color="seagreen", edgecolor="black")
    ax.plot(q_range.numpy(), q_pdf, "r-", linewidth=2)
    ax.set_xlabel("q = P(0|1)")
    ax.set_ylabel("Density")
    ax.set_title(f"Posterior for q: Beta({alpha_q_post:.2f}, {beta_q_post:.2f})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(posterior_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Posterior comparison plot saved to {posterior_plot_path}")

    if rollout_transition_plot_path is None:
        os.makedirs("plots", exist_ok=True)
        rollout_transition_plot_path = "plots/markov_rollout_transition_curves.png"
    else:
        plot_dir = os.path.dirname(rollout_transition_plot_path) or "plots"
        os.makedirs(plot_dir, exist_ok=True)

    if transition_curves:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x_axis = np.arange(1, len(transition_curves[0][0]) + 1)

        ax = axes[0]
        for p_curve, _ in transition_curves:
            ax.plot(x_axis, p_curve, alpha=0.25, linewidth=1.0)
        ax.set_xlabel("Rollout transitions seen")
        ax.set_ylabel("Estimated p")
        ax.set_title("Cumulative p estimates across rollouts")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for _, q_curve in transition_curves:
            ax.plot(x_axis, q_curve, alpha=0.25, linewidth=1.0)
        ax.set_xlabel("Rollout transitions seen")
        ax.set_ylabel("Estimated q")
        ax.set_title("Cumulative q estimates across rollouts")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(rollout_transition_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Rollout transition curves plot saved to {rollout_transition_plot_path}")

    return estimate, std_error


# =====================
#  Diagnostics Helpers
# =====================

def estimate_posterior_sampling_method_fast(
    model: MarkovTransformer,
    context: torch.Tensor,
    target_string: torch.Tensor,
    num_samples: int = 100,
    rollout_length: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[float, float, float, float, float, float, float]:
    """Fast posterior sampling without plotting or verbose output."""
    model = model.to(device)
    model.eval()
    context = context.to(device)
    target_string = target_string.to(device)
    last_bit = int(context[-1].item())

    f_values = []
    p_samples = []
    q_samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            rollout = model.rollout(context, length=rollout_length, temperature=1.0)
            p_hat, q_hat = estimate_transition_probs(rollout)
            p_samples.append(p_hat)
            q_samples.append(q_hat)
            f_values.append(compute_string_probability_given_pq(target_string, last_bit, p_hat, q_hat))

    f_values = np.array(f_values, dtype=np.float64)
    p_samples = np.array(p_samples, dtype=np.float64)
    q_samples = np.array(q_samples, dtype=np.float64)

    estimate = float(f_values.mean()) if f_values.size else 0.0
    f_std = float(f_values.std(ddof=1)) if f_values.size > 1 else 0.0
    std_error = f_std / math.sqrt(num_samples) if num_samples > 1 else 0.0
    p_mean = float(p_samples.mean()) if p_samples.size else 0.0
    p_std = float(p_samples.std(ddof=1)) if p_samples.size > 1 else 0.0
    q_mean = float(q_samples.mean()) if q_samples.size else 0.0
    q_std = float(q_samples.std(ddof=1)) if q_samples.size > 1 else 0.0
    coef_var = f_std / estimate if estimate > 0 else float("inf")

    return estimate, std_error, p_mean, p_std, q_mean, q_std, coef_var


def estimate_posterior_means_from_rollouts(
    model: MarkovTransformer,
    context: torch.Tensor,
    num_rollouts: int = 200,
    rollout_length: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[float, float, float, float]:
    """Estimate posterior mean of (p, q) using rollouts."""
    model = model.to(device)
    model.eval()
    context = context.to(device)

    p_hats = []
    q_hats = []
    with torch.no_grad():
        for _ in range(num_rollouts):
            rollout = model.rollout(context, length=rollout_length, temperature=1.0)
            p_hat, q_hat = estimate_transition_probs(rollout)
            p_hats.append(p_hat)
            q_hats.append(q_hat)

    p_hats = np.array(p_hats, dtype=np.float64)
    q_hats = np.array(q_hats, dtype=np.float64)

    p_mean = float(p_hats.mean()) if p_hats.size else 0.0
    p_std = float(p_hats.std(ddof=1)) if p_hats.size > 1 else 0.0
    q_mean = float(q_hats.mean()) if q_hats.size else 0.0
    q_std = float(q_hats.std(ddof=1)) if q_hats.size > 1 else 0.0
    return p_mean, p_std, q_mean, q_std


# =====================
#  Diagnostics Runner
# =====================

def _make_target_string(
    rng: np.random.Generator,
    target_length: int,
    target_mode: str,
    true_p: float,
    device: str,
) -> torch.Tensor:
    if target_mode == "alternating":
        seq = torch.tensor([1, 0] * (target_length // 2), device=device)
        if target_length % 2 == 1:
            seq = torch.cat([seq, torch.tensor([1], device=device)])
        return seq.long()
    if target_mode == "balanced":
        half = target_length // 2
        seq = torch.tensor([1] * half + [0] * (target_length - half), device=device)
        perm = torch.randperm(target_length, device=device)
        return seq[perm].long()
    if target_mode == "uniform":
        seq = (torch.rand(target_length, device=device) < 0.5).long()
        return seq
    seq = (torch.rand(target_length, device=device) < true_p).long()
    return seq


def run_diagnostics(
    model_path: Optional[str] = None,
    train_model: bool = True,
    train_seq_len: int = 256,
    d_model: int = 16,
    n_layers: int = 1,
    n_heads: int = 1,
    d_mlp: int = 16,
    num_trials: int = 50,
    context_length: int = 50,
    target_length: int = 50,
    num_posterior_samples: int = 100,
    rollout_length_for_posterior: int = 1000,
    num_p_rollouts: int = 200,
    p_rollout_length: int = 100,
    target_mode: str = "bernoulli",
    p_min: float = 0.05,
    p_max: float = 0.95,
    q_min: float = 0.05,
    q_max: float = 0.95,
    seed: int = 123,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    plot_dir: str = "plots",
    print_every: int = 10,
) -> None:
    """Run repeated trials to diagnose posterior sampling accuracy."""
    default_model_path = "checkpoints/markov/markov_transformer.pt"
    if model_path is None:
        model_path = default_model_path

    if train_model and os.path.exists(model_path):
        print(f"Found existing checkpoint at {model_path}")
        print("Set --no-train to skip training, or delete checkpoint to retrain")
        train_model = False

    if train_model:
        print("=" * 80)
        print("Step 1: Training Transformer")
        print("=" * 80)
        model = MarkovTransformer(
            max_seq_len=None,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_mlp=d_mlp,
            use_prenorm=True,
        )
        model = train_markov_transformer(
            model,
            seq_len=train_seq_len,
            batch_size=64,
            num_steps=4000,
            learning_rate=3e-4,
            warmup_steps=400,
            grad_clip=1.0,
            device=device,
            print_every=1000,
        )
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print("=" * 80)
        print(f"Step 1: Loading Model from {model_path}")
        print("=" * 80)
        model = MarkovTransformer(
            max_seq_len=None,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_mlp=d_mlp,
            use_prenorm=True,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        print(f"Model loaded successfully from {model_path}")

    print("\n" + "=" * 80)
    print("Step 2: Running Diagnostics")
    print("=" * 80)
    print(
        f"Trials={num_trials} | context_len={context_length} | target_len={target_length} | "
        f"posterior_samples={num_posterior_samples} | rollout_len={rollout_length_for_posterior}"
    )
    print(
        f"p_rollouts={num_p_rollouts} (len={p_rollout_length}) | "
        f"target_mode={target_mode} | p_range=[{p_min}, {p_max}] | q_range=[{q_min}, {q_max}]"
    )

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    results = []
    eps = 1e-300

    for trial in range(num_trials):
        true_p = float(rng.uniform(p_min, p_max))
        true_q = float(rng.uniform(q_min, q_max))
        context = sample_markov_sequence(true_p, true_q, context_length, device=torch.device(device))
        target = _make_target_string(rng, target_length, target_mode, true_p, device)

        context_cpu = context.cpu()
        target_cpu = target.cpu()
        true_prob = compute_string_probability_analytical(target_cpu, context_cpu)

        estimate, std_error, p_mean, p_std, q_mean, q_std, coef_var = estimate_posterior_sampling_method_fast(
            model=model,
            context=context,
            target_string=target,
            num_samples=num_posterior_samples,
            rollout_length=rollout_length_for_posterior,
            device=device,
        )

        p_rollout_mean, p_rollout_std, q_rollout_mean, q_rollout_std = estimate_posterior_means_from_rollouts(
            model=model,
            context=context,
            num_rollouts=num_p_rollouts,
            rollout_length=p_rollout_length,
            device=device,
        )

        bayes_p_mean, bayes_q_mean = compute_markov_posterior_means(context_cpu)

        denom = true_prob if true_prob > 0 else eps
        rel_error = (estimate - true_prob) / denom
        log10_ratio = math.log10(max(estimate, eps)) - math.log10(max(true_prob, eps))

        results.append(
            {
                "trial": trial,
                "true_p": true_p,
                "true_q": true_q,
                "context_ones": int(context_cpu.sum().item()),
                "context_len": int(context_length),
                "target_ones": int(target_cpu.sum().item()),
                "target_len": int(target_length),
                "true_prob": true_prob,
                "estimate": estimate,
                "std_error": std_error,
                "coef_var": coef_var,
                "rel_error": rel_error,
                "log10_ratio": log10_ratio,
                "p_rollout_mean": p_rollout_mean,
                "p_rollout_std": p_rollout_std,
                "q_rollout_mean": q_rollout_mean,
                "q_rollout_std": q_rollout_std,
                "bayes_p_mean": bayes_p_mean,
                "bayes_q_mean": bayes_q_mean,
                "target_mode": target_mode,
            }
        )

        if print_every and (trial + 1) % print_every == 0:
            print(f"  Completed {trial + 1}/{num_trials} trials")

    os.makedirs(plot_dir, exist_ok=True)

    csv_path = os.path.join(plot_dir, "markov_posterior_sampling_diagnostics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    true_probs = np.array([r["true_prob"] for r in results], dtype=np.float64)
    estimates = np.array([r["estimate"] for r in results], dtype=np.float64)
    log10_ratio = np.array([r["log10_ratio"] for r in results], dtype=np.float64)
    abs_log10_ratio = np.abs(log10_ratio)
    rel_error = np.array([r["rel_error"] for r in results], dtype=np.float64)
    abs_rel_error = np.abs(rel_error)
    p_rollout_mean = np.array([r["p_rollout_mean"] for r in results], dtype=np.float64)
    q_rollout_mean = np.array([r["q_rollout_mean"] for r in results], dtype=np.float64)
    bayes_p_mean = np.array([r["bayes_p_mean"] for r in results], dtype=np.float64)
    bayes_q_mean = np.array([r["bayes_q_mean"] for r in results], dtype=np.float64)
    coef_var = np.array([r["coef_var"] for r in results], dtype=np.float64)

    def pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q))

    p_mean_error = p_rollout_mean - bayes_p_mean
    q_mean_error = q_rollout_mean - bayes_q_mean
    p_mse = float(np.mean(p_mean_error ** 2))
    q_mse = float(np.mean(q_mean_error ** 2))
    p_corr = float(np.corrcoef(p_rollout_mean, bayes_p_mean)[0, 1]) if len(results) > 1 else 0.0
    q_corr = float(np.corrcoef(q_rollout_mean, bayes_q_mean)[0, 1]) if len(results) > 1 else 0.0
    log_corr = float(np.corrcoef(np.log10(true_probs + eps), np.log10(estimates + eps))[0, 1]) if len(results) > 1 else 0.0

    print("\n" + "=" * 80)
    print("Diagnostics Summary")
    print("=" * 80)
    print(f"Results saved to: {csv_path}")
    print(f"Posterior sampling log10 ratio mean={log10_ratio.mean():.3f}, std={log10_ratio.std():.3f}")
    print(
        "Abs log10 ratio percentiles: "
        f"p50={pct(abs_log10_ratio, 50):.3f}, "
        f"p90={pct(abs_log10_ratio, 90):.3f}, "
        f"p95={pct(abs_log10_ratio, 95):.3f}"
    )
    print(
        "Abs relative error percentiles: "
        f"p50={pct(abs_rel_error, 50):.3f}, "
        f"p90={pct(abs_rel_error, 90):.3f}, "
        f"p95={pct(abs_rel_error, 95):.3f}"
    )
    print(f"Posterior CV mean={float(np.mean(coef_var)):.2f}, median={float(np.median(coef_var)):.2f}")
    print(f"Posterior mean p error: MSE={p_mse:.6f}, corr={p_corr:.4f}")
    print(f"Posterior mean q error: MSE={q_mse:.6f}, corr={q_corr:.4f}")
    print(f"log10(true_prob) vs log10(estimate) corr={log_corr:.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.log10(true_probs + eps)
    y = np.log10(estimates + eps)
    ax.scatter(x, y, alpha=0.6, s=25)
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    ax.set_xlabel("log10(true_prob)")
    ax.set_ylabel("log10(estimate)")
    ax.set_title("Posterior Sampling: True vs Estimated Probability")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "markov_diag_true_vs_est.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(log10_ratio, bins=30, color="steelblue", alpha=0.8, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("log10(estimate / true_prob)")
    ax.set_ylabel("Count")
    ax.set_title("Posterior Sampling Error Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "markov_diag_log10_ratio_hist.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(bayes_p_mean, p_rollout_mean, alpha=0.6, s=25)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2)
    ax.set_xlabel("Bayes posterior mean p")
    ax.set_ylabel("Model rollout mean p")
    ax.set_title("Posterior Mean p: Model vs Bayes")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "markov_diag_posterior_mean_p.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(bayes_q_mean, q_rollout_mean, alpha=0.6, s=25)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2)
    ax.set_xlabel("Bayes posterior mean q")
    ax.set_ylabel("Model rollout mean q")
    ax.set_title("Posterior Mean q: Model vs Bayes")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "markov_diag_posterior_mean_q.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


# =====================
#  Main Demo
# =====================

def run_demo(
    model_path: Optional[str] = None,
    train_model: bool = True,
    train_seq_len: int = 256,
    d_model: int = 16,
    n_layers: int = 1,
    n_heads: int = 1,
    d_mlp: int = 16,
    context_length: int = 50,
    target_string_length: int = 50,
    num_rollouts: int = 10000,
    num_posterior_samples: int = 100,
    rollout_length_for_posterior: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    skip_rollout_method: bool = True,
) -> dict:
    """Run the full Markov demonstration."""
    default_model_path = "checkpoints/markov/markov_transformer.pt"
    if model_path is None:
        model_path = default_model_path

    if train_model and os.path.exists(model_path):
        print(f"Found existing checkpoint at {model_path}")
        print("Set --no-train to skip training, or delete checkpoint to retrain")
        train_model = False

    if train_model:
        print("=" * 80)
        print("Step 1: Training Transformer")
        print("=" * 80)
        model = MarkovTransformer(
            max_seq_len=None,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_mlp=d_mlp,
            use_prenorm=True,
        )
        model = train_markov_transformer(
            model,
            seq_len=train_seq_len,
            batch_size=64,
            num_steps=4000,
            learning_rate=3e-4,
            warmup_steps=400,
            grad_clip=1.0,
            device=device,
            print_every=1000,
        )
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print("=" * 80)
        print(f"Step 1: Loading Model from {model_path}")
        print("=" * 80)
        model = MarkovTransformer(
            max_seq_len=None,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_mlp=d_mlp,
            use_prenorm=True,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        print(f"Model loaded successfully from {model_path}")

    print("\n" + "=" * 80)
    print("Step 2: Generating Context")
    print("=" * 80)

    true_p = 0.7
    true_q = 0.3
    print(f"True p: {true_p}, true q: {true_q}")

    context = sample_markov_sequence(true_p, true_q, context_length, device=torch.device(device))
    n_ones = int(context.sum().item())
    n_zeros = len(context) - n_ones
    print(f"Context: {n_ones} ones, {n_zeros} zeros (length={len(context)})")
    print(f"Context preview: {context[:20].tolist()}...")

    print("\n" + "=" * 80)
    print("Step 3: Target String")
    print("=" * 80)

    target_string = torch.tensor([1, 0] * (target_string_length // 2), device=context.device)
    if target_string_length % 2 == 1:
        target_string = torch.cat([target_string, torch.tensor([1], device=context.device)])

    k_target = int(target_string.sum().item())
    print(f"Target string length: {len(target_string)}")
    print(f"Target string: {k_target} ones, {len(target_string) - k_target} zeros")
    print(f"Target string: {target_string.tolist()}")

    print("\n" + "=" * 80)
    print("Step 4: Model vs Ideal Bayes Predictor Analysis")
    print("=" * 80)
    compare_model_vs_bayes(
        model,
        num_contexts=100,
        context_length=context_length,
        device=device,
        plot_dir="plots",
    )

    print("\n" + "=" * 80)
    print("Step 5: True Probability (Analytical)")
    print("=" * 80)

    true_prob = compute_string_probability_analytical(target_string.cpu(), context.cpu())
    print(f"True P(target_string | context) = {true_prob:.8e}")

    if not skip_rollout_method:
        print("\n" + "=" * 80)
        print("Step 6: Estimation Method 1 (Direct Rollout)")
        print("=" * 80)
        print(f"Using {num_rollouts} rollouts...")

        rollout_estimate, rollout_std = estimate_rollout_method(
            model, context, target_string, num_rollouts=num_rollouts, device=device, print_every=5000
        )

        print(f"Estimate: {rollout_estimate:.8e}")
        print(f"Std Error: {rollout_std:.8e}")
        if true_prob > 0:
            print(f"Relative Error: {abs(rollout_estimate - true_prob) / true_prob * 100:.2f}%")
    else:
        print("\n" + "=" * 80)
        print("Step 6: Estimation Method 1 (Direct Rollout) - SKIPPED")
        print("=" * 80)
        rollout_estimate = None
        rollout_std = None

    print("\n" + "=" * 80)
    print("Step 7: Estimation Method 2 (Posterior Sampling)")
    print("=" * 80)
    print(
        f"Using {num_posterior_samples} samples, each with {rollout_length_for_posterior} rollout steps..."
    )

    posterior_estimate, posterior_std = estimate_posterior_sampling_method(
        model,
        context,
        target_string,
        num_samples=num_posterior_samples,
        rollout_length=rollout_length_for_posterior,
        device=device,
        posterior_plot_path="plots/markov_posterior_sampling_comparison.png",
        rollout_transition_plot_path="plots/markov_rollout_transition_curves.png",
        print_every=5,
    )

    print(f"Estimate: {posterior_estimate:.8e}")
    print(f"Std Error: {posterior_std:.8e}")
    if true_prob > 0:
        print(f"Relative Error: {abs(posterior_estimate - true_prob) / true_prob * 100:.2f}%")

    print("\n" + "=" * 80)
    print("Step 8: Summary")
    print("=" * 80)
    print(f"True Probability:     {true_prob:.8e}")
    if not skip_rollout_method:
        print(f"Rollout Method:       {rollout_estimate:.8e} ± {rollout_std:.8e}")
    print(f"Posterior Method:     {posterior_estimate:.8e} ± {posterior_std:.8e}")

    return {
        "true_prob": true_prob,
        "rollout_estimate": rollout_estimate,
        "rollout_std": rollout_std,
        "posterior_estimate": posterior_estimate,
        "posterior_std": posterior_std,
    }


def run_posterior_compare_only(
    model_path: Optional[str] = None,
    train_model: bool = True,
    train_seq_len: int = 256,
    d_model: int = 16,
    n_layers: int = 1,
    n_heads: int = 1,
    d_mlp: int = 16,
    context_length: int = 50,
    target_string_length: int = 50,
    num_posterior_samples: int = 100,
    rollout_length_for_posterior: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Run only the posterior comparison plot for a single context/target."""
    default_model_path = "checkpoints/markov/markov_transformer.pt"
    if model_path is None:
        model_path = default_model_path

    if train_model and os.path.exists(model_path):
        print(f"Found existing checkpoint at {model_path}")
        print("Set --no-train to skip training, or delete checkpoint to retrain")
        train_model = False

    if train_model:
        print("=" * 80)
        print("Step 1: Training Transformer")
        print("=" * 80)
        model = MarkovTransformer(
            max_seq_len=None,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_mlp=d_mlp,
            use_prenorm=True,
        )
        model = train_markov_transformer(
            model,
            seq_len=train_seq_len,
            batch_size=64,
            num_steps=4000,
            learning_rate=3e-4,
            warmup_steps=400,
            grad_clip=1.0,
            device=device,
            print_every=1000,
        )
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print("=" * 80)
        print(f"Step 1: Loading Model from {model_path}")
        print("=" * 80)
        model = MarkovTransformer(
            max_seq_len=None,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_mlp=d_mlp,
            use_prenorm=True,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        print(f"Model loaded successfully from {model_path}")

    print("\n" + "=" * 80)
    print("Step 2: Generating Context/Target")
    print("=" * 80)

    true_p = 0.7
    true_q = 0.3
    context = sample_markov_sequence(true_p, true_q, context_length, device=torch.device(device))
    n_ones = int(context.sum().item())
    n_zeros = len(context) - n_ones
    print(f"True p: {true_p}, true q: {true_q}")
    print(f"Context: {n_ones} ones, {n_zeros} zeros (length={len(context)})")

    target_string = torch.tensor([1, 0] * (target_string_length // 2), device=context.device)
    if target_string_length % 2 == 1:
        target_string = torch.cat([target_string, torch.tensor([1], device=context.device)])
    k_target = int(target_string.sum().item())
    print(
        f"Target string: {k_target} ones, {len(target_string) - k_target} zeros "
        f"(length={len(target_string)})"
    )

    print("\n" + "=" * 80)
    print("Step 3: Posterior Sampling Comparison")
    print("=" * 80)
    estimate, std_error = estimate_posterior_sampling_method(
        model,
        context,
        target_string,
        num_samples=num_posterior_samples,
        rollout_length=rollout_length_for_posterior,
        device=device,
        posterior_plot_path="plots/markov_posterior_sampling_comparison.png",
        rollout_transition_plot_path="plots/markov_rollout_transition_curves.png",
        print_every=5,
    )
    print(f"Estimate: {estimate:.8e}")
    print(f"Std Error: {std_error:.8e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Markov Transformer LPE Demo")
    parser.add_argument("--model-path", type=str, default=None, help="Path to save/load model")
    parser.add_argument("--no-train", action="store_true", help="Skip training (requires model-path)")
    parser.add_argument("--context-length", type=int, default=50, help="Length of context sequence")
    parser.add_argument("--target-length", type=int, default=50, help="Length of target string")
    parser.add_argument("--num-rollouts", type=int, default=10000, help="Number of rollouts for method 1")
    parser.add_argument("--num-posterior-samples", type=int, default=100, help="Number of posterior samples")
    parser.add_argument("--rollout-length", type=int, default=1000, help="Length of rollout for posterior sampling")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--train-seq-len", type=int, default=256, help="Sequence length used during training")
    parser.add_argument("--d-model", type=int, default=16, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--d-mlp", type=int, default=16, help="MLP hidden dimension")
    parser.add_argument("--posterior-compare-only", action="store_true", help="Only run posterior sampling comparison")
    parser.add_argument("--diagnostics", action="store_true", help="Run multi-trial diagnostics")
    parser.add_argument("--num-trials", type=int, default=50, help="Number of diagnostic trials")
    parser.add_argument(
        "--target-mode",
        type=str,
        default="bernoulli",
        choices=["bernoulli", "balanced", "alternating", "uniform"],
        help="Target string mode",
    )
    parser.add_argument("--p-min", type=float, default=0.05, help="Minimum true p")
    parser.add_argument("--p-max", type=float, default=0.95, help="Maximum true p")
    parser.add_argument("--q-min", type=float, default=0.05, help="Minimum true q")
    parser.add_argument("--q-max", type=float, default=0.95, help="Maximum true q")
    parser.add_argument("--num-p-rollouts", type=int, default=200, help="Number of rollouts for posterior mean p/q")
    parser.add_argument("--p-rollout-length", type=int, default=100, help="Rollout length for posterior mean p/q")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for diagnostics")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory for diagnostic plots")
    parser.add_argument("--print-every", type=int, default=10, help="Print progress every N trials")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.posterior_compare_only:
        run_posterior_compare_only(
            model_path=args.model_path,
            train_model=not args.no_train,
            train_seq_len=args.train_seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_mlp=args.d_mlp,
            context_length=args.context_length,
            target_string_length=args.target_length,
            num_posterior_samples=args.num_posterior_samples,
            rollout_length_for_posterior=args.rollout_length,
            device=device,
        )
    elif args.diagnostics:
        run_diagnostics(
            model_path=args.model_path,
            train_model=not args.no_train,
            train_seq_len=args.train_seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_mlp=args.d_mlp,
            num_trials=args.num_trials,
            context_length=args.context_length,
            target_length=args.target_length,
            num_posterior_samples=args.num_posterior_samples,
            rollout_length_for_posterior=args.rollout_length,
            num_p_rollouts=args.num_p_rollouts,
            p_rollout_length=args.p_rollout_length,
            target_mode=args.target_mode,
            p_min=args.p_min,
            p_max=args.p_max,
            q_min=args.q_min,
            q_max=args.q_max,
            seed=args.seed,
            device=device,
            plot_dir=args.plot_dir,
            print_every=args.print_every,
        )
    else:
        run_demo(
            model_path=args.model_path,
            train_model=not args.no_train,
            train_seq_len=args.train_seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_mlp=args.d_mlp,
            context_length=args.context_length,
            target_string_length=args.target_length,
            num_rollouts=args.num_rollouts,
            num_posterior_samples=args.num_posterior_samples,
            rollout_length_for_posterior=args.rollout_length,
            device=device,
            skip_rollout_method=True,
        )
