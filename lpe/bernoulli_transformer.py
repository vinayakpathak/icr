#!/usr/bin/env python3
"""
Bernoulli Transformer for LPE Demonstration

Implements a transformer that learns to do in-context Bayesian inference for Bernoulli data.
Demonstrates the posterior sampling method from the LPE paper.
"""

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


class BernoulliTransformer(nn.Module):
    """
    Decoder-only transformer for binary sequences, matching ic_regression.py architecture.
    
    Input: binary sequence of shape (B, T) where each element is 0 or 1
    Output: logits of shape (B, T, 2) for predicting next token
    
    Note: Without positional encoding, there's no architectural limit on sequence length.
    max_seq_len is optional and only used for warnings/practical limits (memory, etc.).
    """
    
    def __init__(
        self,
        max_seq_len: Optional[int] = None,
        d_model: int = 16,  # Minimal config for Bernoulli task
        n_layers: int = 1,  # Minimal config for Bernoulli task
        n_heads: int = 1,  # Minimal config for Bernoulli task
        d_mlp: int = 16,  # Minimal config for Bernoulli task
        use_prenorm: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len  # Optional: used for warnings/practical limits
        self.d_model = d_model
        self.use_prenorm = use_prenorm
        
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
        
        # Note: Causal mask is now generated dynamically in forward() to support variable-length sequences
        
        self._init_parameters()
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) binary sequence
        returns: (B, T, 2) logits for next token
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
        
        # Generate causal mask dynamically
        # For PyTorch's MultiheadAttention: attn_mask[i, j] = True means position i cannot attend to position j
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.output_proj(x)  # (B, T, 2)
        
        return logits
    
    def predict_next_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get logits for predicting the next token after sequence x.
        
        x: (B, T) binary sequence
        returns: (B, 2) logits for next token
        """
        logits = self.forward(x)  # (B, T, 2)
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
        with torch.no_grad():
            prefix_len = len(prefix)
            
            # If max_seq_len is set, limit the rollout length
            if self.max_seq_len is not None:
                max_allowed_length = self.max_seq_len - prefix_len
                
                if max_allowed_length <= 0:
                    # Prefix is already at or exceeds max_seq_len, cannot generate any tokens
                    warnings.warn(
                        f"Prefix length {prefix_len} exceeds or equals max_seq_len={self.max_seq_len}. "
                        f"Cannot generate any tokens.",
                        UserWarning
                    )
                    return torch.tensor([], dtype=prefix.dtype, device=prefix.device)
                
                actual_length = min(length, max_allowed_length)
                if actual_length < length:
                    warnings.warn(
                        f"Requested rollout length {length} exceeds maximum allowed {max_allowed_length} "
                        f"(prefix length: {prefix_len}, max_seq_len: {self.max_seq_len}). "
                        f"Reducing to {actual_length}.",
                        UserWarning
                    )
            else:
                # No limit set, use requested length
                actual_length = length
            
            current = prefix.clone()
            generated = []
            
            for _ in range(actual_length):
                next_token = self.sample(current, temperature=temperature)
                if isinstance(next_token, (int, float)):
                    next_token_val = int(next_token)
                else:
                    next_token_val = next_token.item()
                generated.append(next_token_val)
                current = torch.cat([current, torch.tensor([next_token_val], dtype=current.dtype, device=current.device)])
            
            return torch.tensor(generated, dtype=current.dtype, device=current.device)


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
    
    # Create dataset and dataloader
    dataset = BernoulliDataset(seq_len=seq_len, num_samples=100000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model.train()
    step = 0
    running_loss = 0.0
    data_iter = iter(dataloader)
    
    print(f"Training on device: {device}")
    print(f"Sequence length: {seq_len}, Batch size: {batch_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
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
            if (i + 1) % print_every == 0 or (i + 1) == num_samples:
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
#  Main Demo
# =====================

def run_demo(
    model_path: Optional[str] = None,
    train_model: bool = True,
    context_length: int = 50,
    target_string_length: int = 50,
    num_rollouts: int = 10000,
    num_posterior_samples: int = 100,
    rollout_length_for_posterior: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    skip_rollout_method: bool = True,
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
        )
        model = train_bernoulli_transformer(
            model,
            seq_len=256,
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
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    run_demo(
        model_path=args.model_path,
        train_model=not args.no_train,
        context_length=args.context_length,
        target_string_length=args.target_length,
        num_rollouts=args.num_rollouts,
        num_posterior_samples=args.num_posterior_samples,
        rollout_length_for_posterior=args.rollout_length,
        device=device,
        skip_rollout_method=True,  # Skip direct rollout method
    )
