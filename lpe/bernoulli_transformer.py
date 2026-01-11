#!/usr/bin/env python3
"""
Bernoulli Transformer for LPE Demonstration

Implements a transformer that learns to do in-context Bayesian inference for Bernoulli data.
Demonstrates the posterior sampling method from the LPE paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import numpy as np


# =====================
#  Transformer Architecture
# =====================

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
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
        # Pre-layer norm
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out
        h = self.ln2(x)
        h = self.mlp(h)
        x = x + h
        return x


class BernoulliTransformer(nn.Module):
    """
    Simple decoder-only transformer for binary sequences.
    
    Input: binary sequence of shape (B, T) where each element is 0 or 1
    Output: logits of shape (B, T, 2) for predicting next token
    """
    
    def __init__(
        self,
        max_seq_len: int = 512,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_mlp: int = 512,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Embedding for binary tokens (0 and 1)
        self.token_emb = nn.Embedding(2, d_model)
        
        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 2)  # 2 logits for 0 and 1
        
        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("attn_mask", mask)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
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
        assert T <= self.max_seq_len
        
        # Embed tokens
        token_emb = self.token_emb(x)  # (B, T, d_model)
        pos_emb = self.pos_emb[:, :T, :]  # (1, T, d_model)
        x = token_emb + pos_emb
        
        # Apply transformer blocks
        mask = self.attn_mask[:T, :T]
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
        """
        self.eval()
        with torch.no_grad():
            current = prefix.clone()
            generated = []
            
            for _ in range(length):
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    print_every: int = 1000,
    checkpoint_dir: Optional[str] = None,
) -> BernoulliTransformer:
    """
    Train the transformer using the procedure:
    - Sample p ~ Beta(1,1)
    - Sample sequence from Bernoulli(p)
    - Train with autoregressive NLL loss
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
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
        optimizer.step()
        
        running_loss += loss.item()
        step += 1
        
        if step % print_every == 0:
            avg_loss = running_loss / print_every
            print(f"[step {step}] loss={avg_loss:.4f}")
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
#  Estimation Methods
# =====================

def estimate_rollout_method(
    model: BernoulliTransformer,
    context: torch.Tensor,
    target_string: torch.Tensor,
    num_rollouts: int = 10000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
    
    with torch.no_grad():
        for _ in range(num_rollouts):
            rollout = model.rollout(context, length=m, temperature=1.0)
            if torch.equal(rollout.to(device), target_string):
                hits += 1
    
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
) -> Tuple[float, float]:
    """
    Estimator 2: Posterior sampling method (Rao-Blackwellized).
    
    For each sample:
    1. Roll out y_{n+1:M} for large M
    2. Compute fraction of 1's in rollout → this is θ (sample of p)
    3. Compute f(θ) = P(target_string | θ) analytically
    
    Then average: (1/M) * sum f(θ_i)
    
    Returns: (estimate, std_error)
    """
    model = model.to(device)
    model.eval()
    
    context = context.to(device)
    target_string = target_string.to(device)
    
    f_values = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Roll out a long sequence
            rollout = model.rollout(context, length=rollout_length, temperature=1.0)
            
            # Compute fraction of 1's → this is our sample of p
            p_hat = rollout.float().mean().item()
            
            # Compute f(p_hat) = P(target_string | p_hat) analytically
            f_p = compute_string_probability_given_p(target_string, p_hat)
            f_values.append(f_p)
    
    f_values = np.array(f_values)
    estimate = f_values.mean()
    std_error = f_values.std() / math.sqrt(num_samples)
    
    return estimate, std_error


# =====================
#  Main Demo
# =====================

def run_demo(
    model_path: Optional[str] = None,
    train_model: bool = True,
    context_length: int = 50,
    target_string_length: int = 10,
    num_rollouts: int = 50000,
    num_posterior_samples: int = 1000,
    rollout_length_for_posterior: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
    if train_model or model_path is None:
        print("=" * 80)
        print("Step 1: Training Transformer")
        print("=" * 80)
        model = BernoulliTransformer(
            max_seq_len=512,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_mlp=512,
        )
        model = train_bernoulli_transformer(
            model,
            seq_len=256,
            batch_size=64,
            num_steps=10000,  # Reduced for faster execution
            device=device,
        )
        if model_path is not None:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    else:
        print(f"Loading model from {model_path}")
        model = BernoulliTransformer(
            max_seq_len=512,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_mlp=512,
        )
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
    
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
    
    # Step 4: Compute true probability analytically
    print("\n" + "=" * 80)
    print("Step 4: True Probability (Analytical)")
    print("=" * 80)
    
    true_prob = compute_string_probability_analytical(target_string, context)
    print(f"True P(target_string | context) = {true_prob:.8e}")
    
    # Step 5: Estimate using rollout method
    print("\n" + "=" * 80)
    print("Step 5: Estimation Method 1 (Direct Rollout)")
    print("=" * 80)
    print(f"Using {num_rollouts} rollouts...")
    
    rollout_estimate, rollout_std = estimate_rollout_method(
        model, context, target_string, num_rollouts=num_rollouts, device=device
    )
    
    print(f"Estimate: {rollout_estimate:.8e}")
    print(f"Std Error: {rollout_std:.8e}")
    print(f"Relative Error: {abs(rollout_estimate - true_prob) / true_prob * 100:.2f}%")
    
    # Step 6: Estimate using posterior sampling method
    print("\n" + "=" * 80)
    print("Step 6: Estimation Method 2 (Posterior Sampling)")
    print("=" * 80)
    print(f"Using {num_posterior_samples} samples, each with {rollout_length_for_posterior} rollout steps...")
    
    posterior_estimate, posterior_std = estimate_posterior_sampling_method(
        model,
        context,
        target_string,
        num_samples=num_posterior_samples,
        rollout_length=rollout_length_for_posterior,
        device=device,
    )
    
    print(f"Estimate: {posterior_estimate:.8e}")
    print(f"Std Error: {posterior_std:.8e}")
    print(f"Relative Error: {abs(posterior_estimate - true_prob) / true_prob * 100:.2f}%")
    
    # Step 7: Summary
    print("\n" + "=" * 80)
    print("Step 7: Summary")
    print("=" * 80)
    print(f"True Probability:     {true_prob:.8e}")
    print(f"Rollout Method:       {rollout_estimate:.8e} ± {rollout_std:.8e}")
    print(f"Posterior Method:     {posterior_estimate:.8e} ± {posterior_std:.8e}")
    print()
    print(f"Rollout relative error:   {abs(rollout_estimate - true_prob) / true_prob * 100:.2f}%")
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
    parser.add_argument("--target-length", type=int, default=10, help="Length of target string")
    parser.add_argument("--num-rollouts", type=int, default=50000, help="Number of rollouts for method 1")
    parser.add_argument("--num-posterior-samples", type=int, default=1000, help="Number of posterior samples")
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
    )
