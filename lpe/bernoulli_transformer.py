#!/usr/bin/env python3
"""
Bernoulli Transformer for LPE Demonstration

Implements a transformer that learns to do in-context Bayesian inference for Bernoulli data.
Demonstrates the posterior sampling method from the LPE paper.
"""

import csv
import math
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import numpy as np
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
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
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


class BernoulliTransformer(nn.Module):
    """
    Decoder-only transformer for binary sequences, matching ic_regression.py architecture.
    
    Input: binary sequence of shape (B, T) where each element is 0 or 1
    Output: logits of shape (B, T, 2) for predicting next token
    
    Note: Without positional encoding, there's no architectural limit on sequence length.
    max_seq_len is optional and only used for warnings/practical limits (memory, etc.).

    attention_mode:
      - "causal": standard autoregressive causal attention
      - "set": full attention + mean pooling (permutation-invariant)
    """
    
    def __init__(
        self,
        max_seq_len: Optional[int] = None,
        d_model: int = 16,  # Minimal config for Bernoulli task
        n_layers: int = 1,  # Minimal config for Bernoulli task
        n_heads: int = 1,  # Minimal config for Bernoulli task
        d_mlp: int = 16,  # Minimal config for Bernoulli task
        use_prenorm: bool = True,
        attention_mode: str = "causal",
    ):
        super().__init__()
        self.max_seq_len = max_seq_len  # Optional: used for warnings/practical limits
        self.d_model = d_model
        self.use_prenorm = use_prenorm
        self.attention_mode = attention_mode
        
        # Embedding for binary tokens (0 and 1)
        # Use linear projection similar to ic_regression.py style
        self.token_emb = nn.Embedding(2, d_model)
        
        # Transformer blocks (matching ic_regression.py structure)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp, use_prenorm=use_prenorm)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection (matching ic_regression.py)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 2)  # 2 logits for 0 and 1
        
        # Note: Attention masks are generated dynamically in forward() to support variable-length sequences
        
        self._init_parameters()

        if self.attention_mode not in {"causal", "set"}:
            raise ValueError(f"Unknown attention_mode '{self.attention_mode}'. Expected 'causal' or 'set'.")
    
    def _init_parameters(self):
        """Initialize parameters matching ic_regression.py style."""
        # Initialization matching GPT2/nanoGPT: normal(0, 0.02) for embeddings and linear layers
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
        # Initialize MLP layers with normal(0, 0.02) (matching ic_regression.py)
        for block in self.blocks:
            for m in block.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(m.bias)
    
    def _build_attention_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.attention_mode == "causal":
            # For PyTorch's MultiheadAttention: attn_mask[i, j] = True means position i cannot attend to position j
            return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        if self.attention_mode == "set":
            # Full attention (permutation-equivariant without positional embeddings)
            return None
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) binary sequence
        returns:
          - causal: (B, T, 2) logits for next token
          - set: (B, 2) logits for next token (permutation-invariant pooling)
        """
        B, T = x.shape
        
        # Optional warning if sequence exceeds max_seq_len (if set)
        if self.max_seq_len is not None and T > self.max_seq_len:
            warnings.warn(
                f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}. "
                f"Model will process it, but performance may degrade or memory issues may occur.",
                UserWarning
            )
        
        # Embed tokens (no positional encoding)
        x = self.token_emb(x)  # (B, T, d_model)
        
        mask = self._build_attention_mask(T, x.device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        if self.attention_mode == "set":
            if T == 0:
                raise ValueError("Empty sequence is not supported in set attention mode.")
            pooled = x.mean(dim=1)  # (B, d_model), permutation-invariant pooling
            logits = self.output_proj(pooled)  # (B, 2)
        else:
            logits = self.output_proj(x)  # (B, T, 2)
        
        return logits
    
    def predict_next_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get logits for predicting the next token after sequence x.
        
        x: (B, T) binary sequence
        returns: (B, 2) logits for next token
        """
        logits = self.forward(x)
        if logits.dim() == 2:
            return logits
        return logits[:, -1, :]  # (B, 2)
    
    def sample(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample next token from the model's distribution.
        
        x: (T,) or (B, T) binary sequence
        returns: scalar or (B,) sampled next tokens
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, T)
        logits = self.predict_next_logits(x) / temperature  # (B, 2)
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
        if sampled.numel() == 1:
            return sampled.item()
        return sampled
    
    def rollout(self, prefix: torch.Tensor, length: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Roll out a sequence from the model given a prefix.
        
        prefix: (T_prefix,) binary sequence
        length: number of tokens to generate
        returns: (length,) generated sequence
        
        Note: If max_seq_len is set, the rollout length may be limited to prevent
        sequences that exceed max_seq_len. Otherwise, generation continues until
        the requested length is reached (subject to memory constraints).
        """
        self.eval()

        # Keep legacy behavior for set-attention mode (no autoregressive cache path).
        if self.attention_mode != "causal":
            with torch.no_grad():
                prefix_len = len(prefix)
                if self.max_seq_len is not None:
                    max_allowed_length = self.max_seq_len - prefix_len
                    if max_allowed_length <= 0:
                        warnings.warn(
                            f"Prefix length {prefix_len} exceeds or equals max_seq_len={self.max_seq_len}. "
                            f"Cannot generate any tokens.",
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
                generated: List[int] = []
                for _ in range(actual_length):
                    next_token = self.sample(current, temperature=temperature)
                    if isinstance(next_token, (int, float)):
                        next_token_val = int(next_token)
                    else:
                        next_token_val = int(next_token.item())
                    generated.append(next_token_val)
                    current = torch.cat(
                        [current, torch.tensor([next_token_val], dtype=current.dtype, device=current.device)]
                    )
                return torch.tensor(generated, dtype=current.dtype, device=current.device)

        # Causal mode: use incremental KV-cached decode by default.
        try:
            from lpe.markov_k_transformer import rollout_with_cache  # type: ignore
        except ImportError:
            from markov_k_transformer import rollout_with_cache  # type: ignore

        with torch.no_grad():
            return rollout_with_cache(
                self,
                prefix=prefix,
                length=length,
                temperature=temperature,
            )


# =====================
#  Training
# =====================

class BernoulliDataset(Dataset):
    """Dataset for training: sample p ~ Beta(1,1), then sample sequence from Bernoulli(p)."""
    
    def __init__(self, seq_len: int, num_samples: int):
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Sample p ~ Beta(1,1) (uniform on [0,1])
        p = torch.rand(1).item()
        
        # Sample sequence from Bernoulli(p)
        sequence = (torch.rand(self.seq_len) < p).long()
        
        return sequence


class BernoulliContextTargetDataset(Dataset):
    """Dataset for permutation-invariant training: context + next token target."""

    def __init__(self, context_len: int, num_samples: int):
        self.context_len = context_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sample p ~ Beta(1,1) (uniform on [0,1])
        p = torch.rand(1).item()

        # Sample context from Bernoulli(p)
        context = (torch.rand(self.context_len) < p).long()

        # Sample next token from Bernoulli(p)
        target = (torch.rand(1) < p).long().squeeze(0)

        return context, target


def train_bernoulli_transformer(
    model: BernoulliTransformer,
    seq_len: int = 256,
    batch_size: int = 64,
    num_steps: int = 50000,
    learning_rate: float = 3e-4,
    warmup_steps: int = 5000,
    grad_clip: Optional[float] = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    print_every: int = 1000,
    checkpoint_dir: Optional[str] = None,
) -> BernoulliTransformer:
    """
    Train the transformer using the procedure:
    - Sample p ~ Beta(1,1)
    - Sample sequence from Bernoulli(p)
    - Train with autoregressive NLL loss
    
    Uses learning rate warmup and optional gradient clipping.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    attention_mode = getattr(model, "attention_mode", "causal")

    # Create dataset and dataloader
    if attention_mode == "set":
        dataset = BernoulliContextTargetDataset(context_len=seq_len, num_samples=100000)
    else:
        dataset = BernoulliDataset(seq_len=seq_len, num_samples=100000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model.train()
    step = 0
    running_loss = 0.0
    data_iter = iter(dataloader)
    
    print(f"Training on device: {device}")
    if attention_mode == "set":
        print(f"Context length: {seq_len}, Batch size: {batch_size}")
    else:
        print(f"Sequence length: {seq_len}, Batch size: {batch_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Attention mode: {attention_mode}")
    print(f"Learning rate: {learning_rate}, Warmup steps: {warmup_steps}, Gradient clip: {grad_clip}")
    
    def get_lr(step):
        """Linear warmup followed by constant learning rate."""
        if step < warmup_steps:
            return learning_rate * (step / warmup_steps)
        return learning_rate
    
    while step < num_steps:
        try:
            sequences = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            sequences = next(data_iter)
        
        if attention_mode == "set":
            contexts, targets = sequences
            contexts = contexts.to(device)  # (B, context_len)
            targets = targets.to(device)  # (B,)

            logits = model.forward(contexts)  # (B, 2)
            loss = F.cross_entropy(logits, targets)
        else:
            sequences = sequences.to(device)  # (B, seq_len)

            # For autoregressive training, input is sequence[:-1], target is sequence[1:]
            inputs = sequences[:, :-1]  # (B, seq_len-1)
            targets = sequences[:, 1:]  # (B, seq_len-1)

            # Forward pass
            logits = model.forward(inputs)  # (B, seq_len-1, 2)

            # Compute loss (cross entropy)
            loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Learning rate warmup
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        optimizer.step()
        
        running_loss += loss.item()
        step += 1
        
        if step % print_every == 0:
            avg_loss = running_loss / print_every
            current_lr = get_lr(step)
            print(f"[step {step}] loss={avg_loss:.4f}, lr={current_lr:.6f}")
            running_loss = 0.0
    
    print("Training completed!")
    return model


# =====================
#  Analytical Computations
# =====================

def compute_bernoulli_posterior(n_ones: int, n_zeros: int, alpha: float = 1.0, beta: float = 1.0) -> Tuple[float, float]:
    """
    Compute posterior parameters for Beta-Bernoulli.
    
    With Beta(alpha, beta) prior and observing n_ones ones and n_zeros zeros,
    the posterior is Beta(alpha + n_ones, beta + n_zeros).
    
    Returns: (posterior_alpha, posterior_beta)
    """
    return (alpha + n_ones, beta + n_zeros)


def compute_string_probability_given_p(string: torch.Tensor, p: float) -> float:
    """
    Compute P(string | p) = p^k * (1-p)^(m-k) where k is number of ones.
    
    string: (m,) binary sequence
    p: Bernoulli parameter
    returns: probability
    """
    k = string.sum().item()
    m = len(string)
    return (p ** k) * ((1 - p) ** (m - k))


def compute_string_probability_analytical(
    string: torch.Tensor,
    context: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Compute P(string | context) analytically using Beta-Bernoulli.
    
    With Beta(alpha, beta) prior:
    P(y_{n+1:n+m} = string | y_{1:n} = context) can be computed analytically.
    
    For Beta(alpha, beta) prior and Beta(alpha + k, beta + (n-k)) posterior,
    the predictive for a string with k' ones and (m-k') zeros is:
    product over positions of (alpha + k + i) / (alpha + beta + n + i) for ones,
    and (beta + n - k + j) / (alpha + beta + n + j) for zeros,
    where we accumulate counts as we go.
    
    Actually, simpler: we can use the formula for Beta-Binomial.
    For a string with k' ones, the probability is:
    Beta(alpha + k + k', beta + n - k + m - k') / Beta(alpha + k, beta + n - k)
    
    string: (m,) binary sequence
    context: (n,) binary sequence
    returns: probability
    """
    n_ones_context = context.sum().item()
    n_zeros_context = len(context) - n_ones_context
    n = len(context)
    
    k_string = string.sum().item()
    m = len(string)
    
    # Posterior after context: Beta(alpha + n_ones_context, beta + n_zeros_context)
    alpha_post = alpha + n_ones_context
    beta_post = beta + n_zeros_context
    
    # Predictive probability for the string
    # Using Beta-Binomial: P(string | context) = Beta(alpha_post + k_string, beta_post + m - k_string) / Beta(alpha_post, beta_post)
    # But this is for the probability of k_string ones out of m, not the exact sequence.
    
    # For the exact sequence, we need to compute it position by position:
    # P(y_{n+1} = s_1 | context) * P(y_{n+2} = s_2 | context, s_1) * ...
    
    prob = 1.0
    ones_so_far = n_ones_context
    total_so_far = n
    
    for bit in string:
        if bit == 1:
            prob *= (alpha + ones_so_far) / (alpha + beta + total_so_far)
            ones_so_far += 1
        else:
            prob *= (beta + total_so_far - ones_so_far) / (alpha + beta + total_so_far)
        total_so_far += 1
    
    return prob


# =====================
#  Model vs Bayes Comparison
# =====================

def get_model_prediction_probability(
    model: BernoulliTransformer,
    context: torch.Tensor,
    num_rollouts: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    Get model's estimate of p (probability of next token being 1) by averaging rollouts.
    
    Args:
        model: Trained transformer model
        context: Context sequence
        num_rollouts: Number of rollouts to average
        device: Device to use
    
    Returns:
        Estimated probability p (fraction of 1's in rollouts)
    """
    model = model.to(device)
    model.eval()
    context = context.to(device)
    
    with torch.no_grad():
        # Roll out multiple sequences and average the fraction of 1's
        rollout_length = 100  # Short rollouts for efficiency
        p_estimates = []
        for _ in range(num_rollouts):
            rollout = model.rollout(context, length=rollout_length, temperature=1.0)
            p_hat = rollout.float().mean().item()
            p_estimates.append(p_hat)
        
        return np.mean(p_estimates)


def get_bayes_optimal_prediction(
    context: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Get ideal Bayes predictor's estimate of p (posterior mean).
    
    With Beta(alpha, beta) prior and context with n_ones ones and n_zeros zeros,
    the posterior is Beta(alpha + n_ones, beta + n_zeros).
    The posterior mean (optimal prediction) is (alpha + n_ones) / (alpha + beta + n_ones + n_zeros).
    
    Args:
        context: Context sequence
        alpha: Beta prior parameter alpha
        beta: Beta prior parameter beta
    
    Returns:
        Posterior mean (optimal Bayes prediction)
    """
    n_ones = context.sum().item()
    n_zeros = len(context) - n_ones
    return (alpha + n_ones) / (alpha + beta + n_ones + n_zeros)


def compare_model_vs_bayes(
    model: BernoulliTransformer,
    num_contexts: int = 100,
    context_length: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    plot_dir: str = "plots",
    num_rollouts: int = 100,
) -> None:
    """
    Compare model predictions against ideal Bayes predictor on multiple contexts.
    
    Generates multiple contexts with different true p values, then compares:
    - Model's estimate of p (from rollouts)
    - Ideal Bayes predictor's estimate (posterior mean)
    
    Creates visualizations showing the comparison.
    """
    print(f"Generating {num_contexts} contexts and comparing predictions...")
    
    model = model.to(device)
    model.eval()
    
    # Generate contexts with different true p values
    true_p_values = np.linspace(0.1, 0.9, num_contexts)
    model_predictions = []
    bayes_predictions = []
    true_p_used = []
    
    with torch.no_grad():
        for i, true_p in enumerate(true_p_values):
            # Sample context from Bernoulli(true_p)
            context = (torch.rand(context_length) < true_p).long()
            context = context.to(device)
            
            # Get model's prediction
            model_p = get_model_prediction_probability(model, context, num_rollouts=num_rollouts, device=device)
            
            # Get Bayes optimal prediction
            bayes_p = get_bayes_optimal_prediction(context.cpu(), alpha=1.0, beta=1.0)
            
            model_predictions.append(model_p)
            bayes_predictions.append(bayes_p)
            true_p_used.append(true_p)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{num_contexts} contexts...")
    
    model_predictions = np.array(model_predictions)
    bayes_predictions = np.array(bayes_predictions)
    true_p_used = np.array(true_p_used)
    
    # Compute errors
    model_errors = model_predictions - true_p_used
    bayes_errors = bayes_predictions - true_p_used
    
    model_mse = np.mean(model_errors ** 2)
    bayes_mse = np.mean(bayes_errors ** 2)
    
    print(f"\nModel MSE: {model_mse:.6f}")
    print(f"Bayes MSE: {bayes_mse:.6f}")
    print(f"Model/Bayes MSE ratio: {model_mse / bayes_mse:.4f}")
    
    # Create visualizations
    os.makedirs(plot_dir, exist_ok=True)
    
    # Figure 1: Scatter plot of predictions
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Model vs Bayes predictions
    ax = axes[0, 0]
    ax.scatter(bayes_predictions, model_predictions, alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect match')
    ax.set_xlabel('Bayes Optimal Prediction', fontsize=12)
    ax.set_ylabel('Model Prediction', fontsize=12)
    ax.set_title('Model vs Bayes Optimal Predictions', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot 2: Predictions vs True p
    ax = axes[0, 1]
    ax.scatter(true_p_used, bayes_predictions, alpha=0.6, s=30, label='Bayes Optimal', color='green')
    ax.scatter(true_p_used, model_predictions, alpha=0.6, s=30, label='Model', color='blue')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect', alpha=0.5)
    ax.set_xlabel('True p', fontsize=12)
    ax.set_ylabel('Predicted p', fontsize=12)
    ax.set_title('Predictions vs True p', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    ax = axes[1, 0]
    ax.hist(model_errors, bins=30, alpha=0.7, label=f'Model (MSE={model_mse:.4f})', color='blue', density=True)
    ax.hist(bayes_errors, bins=30, alpha=0.7, label=f'Bayes (MSE={bayes_mse:.4f})', color='green', density=True)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Prediction Error (predicted - true)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error vs True p
    ax = axes[1, 1]
    ax.scatter(true_p_used, model_errors, alpha=0.6, s=30, label='Model', color='blue')
    ax.scatter(true_p_used, bayes_errors, alpha=0.6, s=30, label='Bayes', color='green')
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('True p', fontsize=12)
    ax.set_ylabel('Prediction Error', fontsize=12)
    ax.set_title('Error vs True p', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(plot_dir, "model_vs_bayes_comparison.png")
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Comparison plot saved to {comparison_plot_path}")
    
    # Figure 2: Detailed analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Absolute errors
    ax = axes[0]
    ax.scatter(true_p_used, np.abs(model_errors), alpha=0.6, s=30, label='Model', color='blue')
    ax.scatter(true_p_used, np.abs(bayes_errors), alpha=0.6, s=30, label='Bayes', color='green')
    ax.set_xlabel('True p', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Absolute Error vs True p', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Squared errors
    ax = axes[1]
    ax.scatter(true_p_used, model_errors ** 2, alpha=0.6, s=30, label='Model', color='blue')
    ax.scatter(true_p_used, bayes_errors ** 2, alpha=0.6, s=30, label='Bayes', color='green')
    ax.set_xlabel('True p', fontsize=12)
    ax.set_ylabel('Squared Error', fontsize=12)
    ax.set_title('Squared Error vs True p', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Correlation
    ax = axes[2]
    correlation = np.corrcoef(bayes_predictions, model_predictions)[0, 1]
    ax.scatter(bayes_predictions, model_predictions, alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect match')
    ax.set_xlabel('Bayes Optimal Prediction', fontsize=12)
    ax.set_ylabel('Model Prediction', fontsize=12)
    ax.set_title(f'Correlation: {correlation:.4f}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    detailed_plot_path = os.path.join(plot_dir, "model_vs_bayes_detailed.png")
    plt.savefig(detailed_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Detailed analysis plot saved to {detailed_plot_path}")


# =====================
#  Estimation Methods
# =====================

def estimate_rollout_method(
    model: BernoulliTransformer,
    context: torch.Tensor,
    target_string: torch.Tensor,
    num_rollouts: int = 10000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    print_every: int = 5000,
) -> Tuple[float, float]:
    """
    Estimator 1: Direct rollout method.
    
    Sample R rollouts Y^{(r)}_{n+1:n+m} and count hits.
    
    Returns: (estimate, std_error)
    """
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
            
            # Print progress
            if (i + 1) % print_every == 0 or (i + 1) == num_rollouts:
                current_estimate = hits / (i + 1)
                print(f"  Progress: {i+1}/{num_rollouts} rollouts, hits: {hits}, current estimate: {current_estimate:.8e}")
    
    estimate = hits / num_rollouts
    
    # Standard error (for Bernoulli)
    std_error = math.sqrt(estimate * (1 - estimate) / num_rollouts)
    
    return estimate, std_error


def estimate_posterior_sampling_method(
    model: BernoulliTransformer,
    context: torch.Tensor,
    target_string: torch.Tensor,
    num_samples: int = 1000,
    rollout_length: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    plot_path: Optional[str] = None,
    rollout_fraction_plot_path: Optional[str] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    print_every: int = 5,
) -> Tuple[float, float]:
    """
    Estimator 2: Posterior sampling method (Rao-Blackwellized).
    
    For each sample:
    1. Roll out y_{n+1:M} for large M
    2. Compute fraction of 1's in rollout → this is θ (sample of p)
    3. Compute f(θ) = P(target_string | θ) analytically
    
    Then average: (1/M) * sum f(θ_i)
    
    Args:
        model: Trained transformer model
        context: Context sequence
        target_string: Target string to estimate probability for
        num_samples: Number of posterior samples
        rollout_length: Length of rollout for each sample
        device: Device to use
        plot_path: Path to save the posterior comparison plot (if None, auto-generates)
        alpha: Beta prior parameter alpha (default: 1.0)
        beta: Beta prior parameter beta (default: 1.0)
        print_every: Print progress every N samples
    
    Returns: (estimate, std_error)
    """
    model = model.to(device)
    model.eval()
    
    context = context.to(device)
    target_string = target_string.to(device)
    
    print(f"Running {num_samples} posterior samples, each with {rollout_length} rollout steps...")
    f_values = []
    p_samples = []  # Store p_hat samples for plotting
    rollout_fraction_curves = []  # Store cumulative fraction curves for each rollout
    
    with torch.no_grad():
        for i in range(num_samples):
            # Roll out a long sequence
            rollout = model.rollout(context, length=rollout_length, temperature=1.0)
            
            # Compute fraction of 1's → this is our sample of p
            p_hat = rollout.float().mean().item()
            p_samples.append(p_hat)

            # Store cumulative fraction of 1s across rollout length
            if rollout.numel() > 0:
                cum_frac = rollout.float().cumsum(0) / torch.arange(
                    1, rollout.numel() + 1, device=rollout.device
                )
                rollout_fraction_curves.append(cum_frac.cpu().numpy())
            
            # Compute f(p_hat) = P(target_string | p_hat) analytically
            f_p = compute_string_probability_given_p(target_string, p_hat)
            f_values.append(f_p)
            
            # Print progress
            if print_every and ((i + 1) % print_every == 0 or (i + 1) == num_samples):
                current_estimate = np.mean(f_values)
                current_std = np.std(f_values) / math.sqrt(len(f_values)) if len(f_values) > 1 else 0.0
                print(f"  Progress: {i+1}/{num_samples} samples, current estimate: {current_estimate:.8e} ± {current_std:.8e}")
    
    f_values = np.array(f_values)
    p_samples = np.array(p_samples)
    estimate = f_values.mean()
    std_error = f_values.std() / math.sqrt(num_samples)
    
    # Compute analytical posterior parameters from context
    n_ones_context = context.sum().item()
    n_zeros_context = len(context) - n_ones_context
    alpha_post = alpha + n_ones_context
    beta_post = beta + n_zeros_context
    
    # Create plot comparing analytical posterior vs sampled histogram
    if plot_path is None:
        os.makedirs("plots", exist_ok=True)
        plot_path = "plots/posterior_sampling_comparison.png"
    else:
        plot_dir = os.path.dirname(plot_path) if os.path.dirname(plot_path) else "plots"
        os.makedirs(plot_dir, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram of samples
    ax.hist(
        p_samples,
        bins=50,
        density=True,
        alpha=0.7,
        label=f"Sampled Posterior (n={num_samples})",
        color="steelblue",
        edgecolor="black",
    )
    
    # Plot analytical Beta posterior PDF using torch.distributions
    p_range = torch.linspace(0.001, 0.999, 1000)  # Avoid 0 and 1 for numerical stability
    beta_dist = dist.Beta(torch.tensor(alpha_post), torch.tensor(beta_post))
    beta_pdf = beta_dist.log_prob(p_range).exp().numpy()
    p_range_np = p_range.numpy()
    
    ax.plot(
        p_range_np,
        beta_pdf,
        "r-",
        linewidth=2,
        label=f"Analytical Beta({alpha_post:.2f}, {beta_post:.2f})",
    )
    
    # Formatting
    ax.set_xlabel("p (Bernoulli parameter)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Posterior Distribution Comparison\n"
        f"Context: {n_ones_context} ones, {n_zeros_context} zeros | "
        f"Prior: Beta({alpha}, {beta})",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Posterior comparison plot saved to {plot_path}")

    # Plot rollout cumulative fraction curves
    if rollout_fraction_plot_path is None:
        os.makedirs("plots", exist_ok=True)
        rollout_fraction_plot_path = "plots/rollout_fraction_curves.png"
    else:
        plot_dir = os.path.dirname(rollout_fraction_plot_path) if os.path.dirname(rollout_fraction_plot_path) else "plots"
        os.makedirs(plot_dir, exist_ok=True)

    if len(rollout_fraction_curves) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_axis = np.arange(1, len(rollout_fraction_curves[0]) + 1)
        for curve in rollout_fraction_curves:
            ax.plot(x_axis, curve, alpha=0.25, linewidth=1.0)
        ax.set_xlabel("Rollout length so far", fontsize=12)
        ax.set_ylabel("Fraction of y = 1", fontsize=12)
        ax.set_title(
            f"Cumulative Fraction of 1s Across Rollouts\n"
            f"{len(rollout_fraction_curves)} rollouts, length={len(rollout_fraction_curves[0])}",
            fontsize=13,
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(rollout_fraction_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Rollout fraction curves plot saved to {rollout_fraction_plot_path}")
    
    return estimate, std_error


# =====================
#  Diagnostics Helpers
# =====================

def estimate_posterior_sampling_method_fast(
    model: BernoulliTransformer,
    context: torch.Tensor,
    target_string: torch.Tensor,
    num_samples: int = 100,
    rollout_length: int = 1000,
    rollout_batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[float, float, float, float, float]:
    """
    Fast posterior sampling without plotting or verbose output.
    
    Returns: (estimate, std_error, sample_mean, sample_std, coef_var)
    """
    model = model.to(device)
    model.eval()
    context = context.to(device)
    target_string = target_string.to(device)

    rollout_batch_size = max(1, int(rollout_batch_size))

    # Prefer cached batched rollouts for diagnostics speed; fall back to legacy rollout if unavailable.
    try:
        from lpe.markov_k_transformer import rollout_with_cache_batch  # type: ignore
    except ImportError:
        from markov_k_transformer import rollout_with_cache_batch  # type: ignore

    p_samples_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, num_samples, rollout_batch_size):
            bsz = min(rollout_batch_size, num_samples - offset)
            rollout_batch = rollout_with_cache_batch(
                model,
                prefix=context.long(),
                length=rollout_length,
                batch_size=bsz,
                temperature=1.0,
            )
            if rollout_batch.numel() == 0:
                p_chunk = np.full((bsz,), 0.5, dtype=np.float64)
            else:
                p_chunk = rollout_batch.float().mean(dim=1).detach().cpu().numpy().astype(np.float64)
            p_samples_chunks.append(p_chunk)

    if p_samples_chunks:
        p_samples = np.concatenate(p_samples_chunks, axis=0)
    else:
        p_samples = np.zeros((0,), dtype=np.float64)

    k_ones = int(target_string.sum().item())
    target_len = int(target_string.numel())
    f_values = np.power(p_samples, k_ones) * np.power(1.0 - p_samples, target_len - k_ones)
    
    estimate = float(f_values.mean()) if f_values.size else 0.0
    f_std = float(f_values.std(ddof=1)) if f_values.size > 1 else 0.0
    std_error = f_std / math.sqrt(num_samples) if num_samples > 1 else 0.0
    p_mean = float(p_samples.mean()) if p_samples.size else 0.0
    p_std = float(p_samples.std(ddof=1)) if p_samples.size > 1 else 0.0
    coef_var = f_std / estimate if estimate > 0 else float("inf")
    
    return estimate, std_error, p_mean, p_std, coef_var


def estimate_posterior_mean_from_rollouts(
    model: BernoulliTransformer,
    context: torch.Tensor,
    num_rollouts: int = 200,
    rollout_length: int = 100,
    rollout_batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[float, float]:
    """Estimate posterior mean p using rollouts; returns (mean, std)."""
    model = model.to(device)
    model.eval()
    context = context.to(device)

    rollout_batch_size = max(1, int(rollout_batch_size))

    try:
        from lpe.markov_k_transformer import rollout_with_cache_batch  # type: ignore
    except ImportError:
        from markov_k_transformer import rollout_with_cache_batch  # type: ignore

    p_hats_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, num_rollouts, rollout_batch_size):
            bsz = min(rollout_batch_size, num_rollouts - offset)
            rollout_batch = rollout_with_cache_batch(
                model,
                prefix=context.long(),
                length=rollout_length,
                batch_size=bsz,
                temperature=1.0,
            )
            if rollout_batch.numel() == 0:
                p_chunk = np.full((bsz,), 0.5, dtype=np.float64)
            else:
                p_chunk = rollout_batch.float().mean(dim=1).detach().cpu().numpy().astype(np.float64)
            p_hats_chunks.append(p_chunk)

    if p_hats_chunks:
        p_hats = np.concatenate(p_hats_chunks, axis=0)
    else:
        p_hats = np.zeros((0,), dtype=np.float64)

    p_mean = float(p_hats.mean()) if p_hats.size else 0.0
    p_std = float(p_hats.std(ddof=1)) if p_hats.size > 1 else 0.0
    return p_mean, p_std


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
    # Default: Bernoulli(true_p)
    seq = (torch.rand(target_length, device=device) < true_p).long()
    return seq


def run_diagnostics(
    model_path: Optional[str] = None,
    train_model: bool = True,
    train_seq_len: int = 1000,
    num_trials: int = 50,
    context_length: int = 50,
    target_length: int = 50,
    num_posterior_samples: int = 100,
    rollout_length_for_posterior: int = 1000,
    posterior_rollout_batch_size: int = 32,
    num_p_rollouts: int = 200,
    p_rollout_length: int = 100,
    p_rollout_batch_size: int = 32,
    target_mode: str = "bernoulli",
    p_source: str = "uniform_range",
    p_min: float = 0.05,
    p_max: float = 0.95,
    seed: int = 123,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    plot_dir: str = "plots",
    print_every: int = 10,
    attention_mode: str = "causal",
    posterior_plot_path: Optional[str] = None,
) -> None:
    """Run repeated trials to diagnose posterior sampling accuracy."""
    # Step 1: Train or load model
    default_model_path = "checkpoints/bernoulli_transformer.pt"
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
        model = BernoulliTransformer(
            max_seq_len=None,
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_mlp=16,
            use_prenorm=True,
            attention_mode=attention_mode,
        )
        model = train_bernoulli_transformer(
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
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print("=" * 80)
        print(f"Step 1: Loading Model from {model_path}")
        print("=" * 80)
        model = BernoulliTransformer(
            max_seq_len=None,
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_mlp=16,
            use_prenorm=True,
            attention_mode=attention_mode,
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
    if p_source == "beta11":
        p_desc = "Beta(1,1)"
    else:
        p_desc = f"Uniform({p_min}, {p_max})"
    print(
        f"p_rollouts={num_p_rollouts} (len={p_rollout_length}) | "
        f"target_mode={target_mode} | p_source={p_desc} | "
        f"posterior_rollout_batch_size={posterior_rollout_batch_size} | "
        f"p_rollout_batch_size={p_rollout_batch_size}"
    )
    
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    results = []
    eps = 1e-300
    
    for trial in range(num_trials):
        if p_source == "beta11":
            true_p = float(rng.beta(1.0, 1.0))
        else:
            true_p = float(rng.uniform(p_min, p_max))
        context = (torch.rand(context_length, device=device) < true_p).long()
        target = _make_target_string(rng, target_length, target_mode, true_p, device)
        
        context_cpu = context.cpu()
        target_cpu = target.cpu()
        true_prob = compute_string_probability_analytical(target_cpu, context_cpu)
        
        estimate, std_error, p_mean, p_std, coef_var = estimate_posterior_sampling_method_fast(
            model=model,
            context=context,
            target_string=target,
            num_samples=num_posterior_samples,
            rollout_length=rollout_length_for_posterior,
            rollout_batch_size=posterior_rollout_batch_size,
            device=device,
        )

        if trial == 0 and posterior_plot_path is not None:
            estimate_posterior_sampling_method(
                model=model,
                context=context,
                target_string=target,
                num_samples=num_posterior_samples,
                rollout_length=rollout_length_for_posterior,
                device=device,
                plot_path=posterior_plot_path,
                rollout_fraction_plot_path=None,
                print_every=0,
            )
        
        p_rollout_mean, p_rollout_std = estimate_posterior_mean_from_rollouts(
            model=model,
            context=context,
            num_rollouts=num_p_rollouts,
            rollout_length=p_rollout_length,
            rollout_batch_size=p_rollout_batch_size,
            device=device,
        )
        
        bayes_mean = get_bayes_optimal_prediction(context_cpu, alpha=1.0, beta=1.0)
        
        denom = true_prob if true_prob > 0 else eps
        rel_error = (estimate - true_prob) / denom
        log10_ratio = math.log10(max(estimate, eps)) - math.log10(max(true_prob, eps))
        
        results.append({
            "trial": trial,
            "true_p": true_p,
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
            "bayes_mean": bayes_mean,
            "target_mode": target_mode,
        })
        
        if print_every and (trial + 1) % print_every == 0:
            print(f"  Completed {trial + 1}/{num_trials} trials")
    
    os.makedirs(plot_dir, exist_ok=True)
    
    csv_path = os.path.join(plot_dir, "bernoulli_posterior_sampling_diagnostics.csv")
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
    bayes_mean = np.array([r["bayes_mean"] for r in results], dtype=np.float64)
    coef_var = np.array([r["coef_var"] for r in results], dtype=np.float64)
    
    def pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q))
    
    p_mean_error = p_rollout_mean - bayes_mean
    p_mse = float(np.mean(p_mean_error ** 2))
    p_corr = float(np.corrcoef(p_rollout_mean, bayes_mean)[0, 1]) if len(results) > 1 else 0.0
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
    print(f"log10(true_prob) vs log10(estimate) corr={log_corr:.4f}")
    
    # Plot 1: log10(true) vs log10(estimate)
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
    plot_path = os.path.join(plot_dir, "bernoulli_diag_true_vs_est.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 2: log10 ratio histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(log10_ratio, bins=30, color="steelblue", alpha=0.8, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("log10(estimate / true_prob)")
    ax.set_ylabel("Count")
    ax.set_title("Posterior Sampling Error Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "bernoulli_diag_log10_ratio_hist.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 3: posterior mean vs Bayes mean
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(bayes_mean, p_rollout_mean, alpha=0.6, s=25)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2)
    ax.set_xlabel("Bayes posterior mean")
    ax.set_ylabel("Model rollout mean")
    ax.set_title("Posterior Mean: Model vs Bayes")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "bernoulli_diag_posterior_mean.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 4: absolute log10 error vs true p
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([r["true_p"] for r in results], abs_log10_ratio, alpha=0.6, s=25)
    ax.set_xlabel("True p")
    ax.set_ylabel("abs(log10 ratio)")
    ax.set_title("Error vs True p")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "bernoulli_diag_error_vs_true_p.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


# =====================
#  Main Demo
# =====================

def run_demo(
    model_path: Optional[str] = None,
    train_model: bool = True,
    train_seq_len: int = 1000,
    context_length: int = 50,
    target_string_length: int = 50,
    num_rollouts: int = 10000,
    num_posterior_samples: int = 100,
    rollout_length_for_posterior: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    skip_rollout_method: bool = True,
    attention_mode: str = "causal",
):
    """
    Run the full demonstration.
    
    1. Train (or load) the transformer
    2. Pick a p and sample context y_{1:n} from it
    3. Compute true P(target_string | context) analytically
    4. Estimate using both methods
    5. Compare results
    """
    
    # Step 1: Train or load model
    # Default checkpoint path
    default_model_path = "checkpoints/bernoulli_transformer.pt"
    
    # Check if we should load from checkpoint
    if model_path is None:
        model_path = default_model_path
    
    if train_model and os.path.exists(model_path):
        print(f"Found existing checkpoint at {model_path}")
        print("Set --no-train to skip training, or delete checkpoint to retrain")
        train_model = False  # Load existing model instead
    
    if train_model:
        print("=" * 80)
        print("Step 1: Training Transformer")
        print("=" * 80)
        model = BernoulliTransformer(
            max_seq_len=None,  # No architectural limit (masks generated dynamically)
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_mlp=16,
            use_prenorm=True,
            attention_mode=attention_mode,
        )
        model = train_bernoulli_transformer(
            model,
            seq_len=train_seq_len,
            batch_size=64,
            num_steps=4000,  # Reduced to 4k steps for fast execution
            learning_rate=3e-4,
            warmup_steps=400,  # 400 warmup steps (10% of total)
            grad_clip=1.0,
            device=device,
            print_every=1000,  # Print every 1k steps
        )
        # Save model checkpoint
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print("=" * 80)
        print(f"Step 1: Loading Model from {model_path}")
        print("=" * 80)
        model = BernoulliTransformer(
            max_seq_len=None,  # No architectural limit (masks generated dynamically)
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_mlp=16,
            use_prenorm=True,
            attention_mode=attention_mode,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        print(f"Model loaded successfully from {model_path}")
    
    # Step 2: Generate context
    print("\n" + "=" * 80)
    print("Step 2: Generating Context")
    print("=" * 80)
    
    # Pick a p (use 0.6 for demonstration)
    true_p = 0.6
    print(f"True p: {true_p}")
    
    # Sample context
    context = (torch.rand(context_length) < true_p).long()
    n_ones = context.sum().item()
    n_zeros = len(context) - n_ones
    print(f"Context: {n_ones} ones, {n_zeros} zeros (length={len(context)})")
    print(f"Context preview: {context[:20].tolist()}...")
    
    # Step 3: Choose target string (balanced string: 50% ones)
    print("\n" + "=" * 80)
    print("Step 3: Target String")
    print("=" * 80)
    
    # Create a balanced string (alternating pattern)
    target_string = torch.tensor([1, 0] * (target_string_length // 2))
    if target_string_length % 2 == 1:
        target_string = torch.cat([target_string, torch.tensor([1])])
    
    k_target = target_string.sum().item()
    print(f"Target string length: {len(target_string)}")
    print(f"Target string: {k_target} ones, {len(target_string) - k_target} zeros")
    print(f"Target string: {target_string.tolist()}")
    
    # Step 4: Compare model predictions vs ideal Bayes predictor
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
    
    # Step 5: Compute true probability analytically
    print("\n" + "=" * 80)
    print("Step 5: True Probability (Analytical)")
    print("=" * 80)
    
    true_prob = compute_string_probability_analytical(target_string, context)
    print(f"True P(target_string | context) = {true_prob:.8e}")
    
    # Step 6: Estimate using rollout method (optional)
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
        print(f"Relative Error: {abs(rollout_estimate - true_prob) / true_prob * 100:.2f}%")
    else:
        print("\n" + "=" * 80)
        print("Step 6: Estimation Method 1 (Direct Rollout) - SKIPPED")
        print("=" * 80)
        rollout_estimate = None
        rollout_std = None
    
    # Step 7: Estimate using posterior sampling method
    print("\n" + "=" * 80)
    print("Step 7: Estimation Method 2 (Posterior Sampling)")
    print("=" * 80)
    print(f"Using {num_posterior_samples} samples, each with {rollout_length_for_posterior} rollout steps...")
    
    posterior_estimate, posterior_std = estimate_posterior_sampling_method(
        model,
        context,
        target_string,
        num_samples=num_posterior_samples,
        rollout_length=rollout_length_for_posterior,
        device=device,
        plot_path="plots/posterior_sampling_comparison.png",
        print_every=5,
    )
    
    print(f"Estimate: {posterior_estimate:.8e}")
    print(f"Std Error: {posterior_std:.8e}")
    print(f"Relative Error: {abs(posterior_estimate - true_prob) / true_prob * 100:.2f}%")
    
    # Step 8: Summary
    print("\n" + "=" * 80)
    print("Step 8: Summary")
    print("=" * 80)
    print(f"True Probability:     {true_prob:.8e}")
    if not skip_rollout_method:
        print(f"Rollout Method:       {rollout_estimate:.8e} ± {rollout_std:.8e}")
        print(f"Rollout relative error:   {abs(rollout_estimate - true_prob) / true_prob * 100:.2f}%")
    print(f"Posterior Method:     {posterior_estimate:.8e} ± {posterior_std:.8e}")
    print(f"Posterior relative error: {abs(posterior_estimate - true_prob) / true_prob * 100:.2f}%")
    
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
    train_seq_len: int = 1000,
    context_length: int = 50,
    target_string_length: int = 50,
    num_posterior_samples: int = 100,
    rollout_length_for_posterior: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    attention_mode: str = "causal",
):
    """Run only the posterior comparison plot for a single context/target."""
    default_model_path = "checkpoints/bernoulli_transformer.pt"
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
        model = BernoulliTransformer(
            max_seq_len=None,
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_mlp=16,
            use_prenorm=True,
            attention_mode=attention_mode,
        )
        model = train_bernoulli_transformer(
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
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print("=" * 80)
        print(f"Step 1: Loading Model from {model_path}")
        print("=" * 80)
        model = BernoulliTransformer(
            max_seq_len=None,
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_mlp=16,
            use_prenorm=True,
            attention_mode=attention_mode,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        print(f"Model loaded successfully from {model_path}")
    
    print("\n" + "=" * 80)
    print("Step 2: Generating Context/Target")
    print("=" * 80)
    
    true_p = 0.6
    context = (torch.rand(context_length) < true_p).long()
    n_ones = context.sum().item()
    n_zeros = len(context) - n_ones
    print(f"True p: {true_p}")
    print(f"Context: {n_ones} ones, {n_zeros} zeros (length={len(context)})")
    
    target_string = torch.tensor([1, 0] * (target_string_length // 2))
    if target_string_length % 2 == 1:
        target_string = torch.cat([target_string, torch.tensor([1])])
    k_target = target_string.sum().item()
    print(f"Target string: {k_target} ones, {len(target_string) - k_target} zeros (length={len(target_string)})")
    
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
        plot_path="plots/posterior_sampling_comparison.png",
        print_every=5,
    )
    print(f"Estimate: {estimate:.8e}")
    print(f"Std Error: {std_error:.8e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bernoulli Transformer LPE Demo")
    parser.add_argument("--model-path", type=str, default=None, help="Path to save/load model")
    parser.add_argument("--no-train", action="store_true", help="Skip training (requires model-path)")
    parser.add_argument("--context-length", type=int, default=50, help="Length of context sequence")
    parser.add_argument("--target-length", type=int, default=50, help="Length of target string")
    parser.add_argument("--num-rollouts", type=int, default=10000, help="Number of rollouts for method 1")
    parser.add_argument("--num-posterior-samples", type=int, default=100, help="Number of posterior samples")
    parser.add_argument("--rollout-length", type=int, default=1000, help="Length of rollout for posterior sampling")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--train-seq-len", type=int, default=1000, help="Sequence length used during training")
    parser.add_argument("--posterior-compare-only", action="store_true", help="Only run posterior sampling comparison")
    parser.add_argument("--diagnostics", action="store_true", help="Run multi-trial diagnostics")
    parser.add_argument("--num-trials", type=int, default=50, help="Number of diagnostic trials")
    parser.add_argument("--target-mode", type=str, default="bernoulli", choices=["bernoulli", "balanced", "alternating", "uniform"], help="Target string mode")
    parser.add_argument(
        "--p-source",
        type=str,
        default="uniform_range",
        choices=["uniform_range", "beta11"],
        help="How to sample latent Bernoulli p for each diagnostic context",
    )
    parser.add_argument("--p-min", type=float, default=0.05, help="Minimum true p")
    parser.add_argument("--p-max", type=float, default=0.95, help="Maximum true p")
    parser.add_argument(
        "--posterior-rollout-batch-size",
        type=int,
        default=32,
        help="Batch size for cached rollouts in posterior-sampling estimator",
    )
    parser.add_argument("--num-p-rollouts", type=int, default=200, help="Number of rollouts for posterior mean p")
    parser.add_argument("--p-rollout-length", type=int, default=100, help="Rollout length for posterior mean p")
    parser.add_argument(
        "--p-rollout-batch-size",
        type=int,
        default=32,
        help="Batch size for cached rollouts in posterior-mean estimator",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for diagnostics")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory for diagnostic plots")
    parser.add_argument(
        "--posterior-plot-path",
        type=str,
        default=None,
        help="Optional path for posterior vs analytical comparison plot (diagnostics only)",
    )
    parser.add_argument("--print-every", type=int, default=10, help="Print progress every N trials")
    parser.add_argument(
        "--attention-mode",
        type=str,
        default="causal",
        choices=["causal", "set"],
        help="Attention mode: causal (default) or permutation-invariant set",
    )
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.posterior_compare_only:
        run_posterior_compare_only(
            model_path=args.model_path,
            train_model=not args.no_train,
            train_seq_len=args.train_seq_len,
            context_length=args.context_length,
            target_string_length=args.target_length,
            num_posterior_samples=args.num_posterior_samples,
            rollout_length_for_posterior=args.rollout_length,
            device=device,
            attention_mode=args.attention_mode,
        )
    elif args.diagnostics:
        run_diagnostics(
            model_path=args.model_path,
            train_model=not args.no_train,
            train_seq_len=args.train_seq_len,
            num_trials=args.num_trials,
            context_length=args.context_length,
            target_length=args.target_length,
            num_posterior_samples=args.num_posterior_samples,
            rollout_length_for_posterior=args.rollout_length,
            posterior_rollout_batch_size=args.posterior_rollout_batch_size,
            num_p_rollouts=args.num_p_rollouts,
            p_rollout_length=args.p_rollout_length,
            p_rollout_batch_size=args.p_rollout_batch_size,
            target_mode=args.target_mode,
            p_source=args.p_source,
            p_min=args.p_min,
            p_max=args.p_max,
            seed=args.seed,
            device=device,
            plot_dir=args.plot_dir,
            print_every=args.print_every,
            attention_mode=args.attention_mode,
            posterior_plot_path=args.posterior_plot_path,
        )
    else:
        run_demo(
            model_path=args.model_path,
            train_model=not args.no_train,
            train_seq_len=args.train_seq_len,
            context_length=args.context_length,
            target_string_length=args.target_length,
            num_rollouts=args.num_rollouts,
            num_posterior_samples=args.num_posterior_samples,
            rollout_length_for_posterior=args.rollout_length,
            device=device,
            skip_rollout_method=True,  # Skip direct rollout method
            attention_mode=args.attention_mode,
        )
