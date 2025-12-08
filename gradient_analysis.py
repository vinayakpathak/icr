"""
Gradient computation for distribution parameters using score function estimator.
"""

import math
from typing import Optional, Tuple

import torch
import torch.distributions as dist

from ic_regression import ICLinearRegressionTransformer, encode_sequence_tokens


def generate_gaussian_prompt(
    mu_x: torch.Tensor,
    sigma_x: torch.Tensor,
    mu_theta: torch.Tensor,
    sigma_theta: torch.Tensor,
    mu_noise: torch.Tensor,
    sigma_noise: torch.Tensor,
    n: int,
    D: int,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a prompt from multivariate Gaussian distributions.
    
    Process:
    1. Sample x from N(μ_x, σ_x² I_D) for each of n examples
    2. Sample θ from N(μ_θ, σ_θ² I_D) for each of n examples
    3. Sample y from θ^T x + N(μ_noise, σ_noise²) for each of n examples
    
    Args:
        mu_x: Mean for x distribution (scalar, requires_grad=True)
        sigma_x: Std for x distribution (scalar, requires_grad=True, must be > 0)
        mu_theta: Mean for θ distribution (scalar, requires_grad=True)
        sigma_theta: Std for θ distribution (scalar, requires_grad=True, must be > 0)
        mu_noise: Mean for noise distribution (scalar, requires_grad=True)
        sigma_noise: Std for noise distribution (scalar, requires_grad=True, must be > 0)
        n: Number of examples to generate
        D: Dimension of x and θ
        seed: Random seed (optional, for reproducibility)
    
    Returns:
        Tuple of (x_context, y_context, x_query, y_query, theta_query) where:
        - x_context: (n-1, D) first n-1 x values
        - y_context: (n-1,) first n-1 y values
        - x_query: (1, D) nth x value
        - y_query: (1,) nth y value
        - theta_query: (D,) theta used for nth example
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Sample n thetas: each component ~ N(μ_θ, σ_θ²)
    theta_dist = dist.Normal(mu_theta, sigma_theta)
    thetas = theta_dist.sample((n, D))  # (n, D)
    
    # Sample n x values: each component ~ N(μ_x, σ_x²)
    x_dist = dist.Normal(mu_x, sigma_x)
    x_samples = x_dist.sample((n, D))  # (n, D)
    
    # Compute y = θ^T x + noise for each example
    # y = sum over D of (theta * x) + noise
    y_noiseless = (x_samples * thetas).sum(dim=1)  # (n,)
    
    # Sample noise: N(μ_noise, σ_noise²)
    noise_dist = dist.Normal(mu_noise, sigma_noise)
    noise = noise_dist.sample((n,))  # (n,)
    
    y_samples = y_noiseless + noise  # (n,)
    
    # Split into context (first n-1) and query (nth)
    x_context = x_samples[:n-1]  # (n-1, D)
    y_context = y_samples[:n-1]  # (n-1,)
    x_query = x_samples[n-1:n]  # (1, D)
    y_query = y_samples[n-1:n]  # (1,)
    theta_query = thetas[n-1]  # (D,)
    
    return x_context, y_context, x_query, y_query, theta_query


def compute_log_prob(
    x: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    mu_x: torch.Tensor,
    sigma_x: torch.Tensor,
    mu_theta: torch.Tensor,
    sigma_theta: torch.Tensor,
    mu_noise: torch.Tensor,
    sigma_noise: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probability log p(x, θ, y | parameters).
    
    Since x, θ, and noise are independent:
    log p(x, θ, y | params) = log p(x | μ_x, σ_x) + log p(θ | μ_θ, σ_θ) + log p(y | x, θ, μ_noise, σ_noise)
    
    Args:
        x: (1, D) or (D,) x value
        theta: (D,) theta value
        y: (1,) or scalar y value
        mu_x, sigma_x, mu_theta, sigma_theta, mu_noise, sigma_noise: Distribution parameters
    
    Returns:
        Scalar log probability (with gradients)
    """
    # Flatten if needed
    if x.dim() > 1:
        x = x.squeeze(0)  # (D,)
    if y.dim() > 0:
        y = y.squeeze()  # scalar
    
    # log p(x | μ_x, σ_x): sum over D components
    x_dist = dist.Normal(mu_x, sigma_x)
    log_p_x = x_dist.log_prob(x).sum()  # sum over D dimensions
    
    # log p(θ | μ_θ, σ_θ): sum over D components
    theta_dist = dist.Normal(mu_theta, sigma_theta)
    log_p_theta = theta_dist.log_prob(theta).sum()  # sum over D dimensions
    
    # log p(y | x, θ, μ_noise, σ_noise)
    # y = θ^T x + noise, so noise = y - θ^T x
    y_expected = (x * theta).sum()  # θ^T x
    noise_value = y - y_expected
    noise_dist = dist.Normal(mu_noise, sigma_noise)
    log_p_noise = noise_dist.log_prob(noise_value)
    
    # Total log probability
    log_p = log_p_x + log_p_theta + log_p_noise
    
    return log_p


def compute_expected_loss_gradient(
    model: ICLinearRegressionTransformer,
    mu_x: torch.Tensor,
    sigma_x: torch.Tensor,
    mu_theta: torch.Tensor,
    sigma_theta: torch.Tensor,
    mu_noise: torch.Tensor,
    sigma_noise: torch.Tensor,
    n_prompts: int = 1000,
    D: int = 8,
    K: int = 16,
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """
    Compute gradient of expected loss w.r.t. distribution parameters using score function estimator.
    
    Score function estimator: ∇_φ E[L] ≈ (1/n_prompts) Σ L_i * ∇_φ log p(x_i, θ_i, y_i | φ)
    
    For in-distribution gradients, use parameters matching the training distribution:
    - mu_x=0.0, sigma_x=1.0: matches x ~ N(0, 1) used in training
    - mu_theta=0.0, sigma_theta=1.0: approximates Uniform({t_1,...,t_M}) where t_i ~ N(0, I_D)
      (uses Gaussian approximation for the discrete Uniform distribution)
    - mu_noise=0.0, sigma_noise=sqrt(0.125) ≈ 0.354: matches y = theta^T x + N(0, 0.125)
    
    Args:
        model: Trained transformer model
        mu_x, sigma_x, mu_theta, sigma_theta, mu_noise, sigma_noise: Distribution parameters (with requires_grad=True)
        n_prompts: Number of prompts for Monte Carlo estimation
        D: Dimension of x and θ
        K: Number of examples per prompt (n = K)
        device: Device to run on
        verbose: Whether to print progress
    
    Returns:
        Dictionary with:
        - 'gradients': dict with gradient for each parameter
        - 'expected_loss': average loss over prompts
        - 'parameter_values': dict with current parameter values
    """
    model.eval()
    
    # Ensure parameters are on correct device
    mu_x = mu_x.to(device)
    sigma_x = sigma_x.to(device)
    mu_theta = mu_theta.to(device)
    sigma_theta = sigma_theta.to(device)
    mu_noise = mu_noise.to(device)
    sigma_noise = sigma_noise.to(device)
    
    # Accumulators for score function estimator
    total_loss = 0.0
    
    # We'll accumulate gradients manually
    # Initialize gradient accumulators
    grad_mu_x = torch.zeros_like(mu_x)
    grad_sigma_x = torch.zeros_like(sigma_x)
    grad_mu_theta = torch.zeros_like(mu_theta)
    grad_sigma_theta = torch.zeros_like(sigma_theta)
    grad_mu_noise = torch.zeros_like(mu_noise)
    grad_sigma_noise = torch.zeros_like(sigma_noise)
    
    for i in range(n_prompts):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processing prompt {i+1}/{n_prompts}")
        
        # Generate prompt
        x_context, y_context, x_query, y_query, theta_query = generate_gaussian_prompt(
            mu_x, sigma_x, mu_theta, sigma_theta, mu_noise, sigma_noise,
            n=K, D=D, seed=None  # Don't fix seed for Monte Carlo
        )
        
        # Move to device
        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_query = x_query.to(device)
        y_query = y_query.to(device)
        theta_query = theta_query.to(device)
        
        # Get model prediction (no gradients needed for model)
        with torch.no_grad():
            # Encode sequence
            x_full = torch.cat([x_context, x_query], dim=0)  # (K, D)
            y_full = torch.cat([y_context, torch.zeros(1, dtype=y_context.dtype, device=device)], dim=0)
            tokens = encode_sequence_tokens(x_full, y_full)  # (2K, D+1)
            tokens = tokens.unsqueeze(0).to(device)  # (1, 2K, D+1)
            
            # Get prediction for query point (last x-token position)
            y_pred_all = model.predict_y_from_x_tokens(tokens)  # (1, K)
            y_pred = y_pred_all[0, K-1]  # Prediction for last (query) position
        
        # Compute loss (detached scalar - we don't need gradients through model)
        loss_value = (y_pred.detach() - y_query[0].detach()) ** 2
        total_loss += loss_value.item()
        
        # Compute log probability (with gradients w.r.t. distribution parameters)
        # Note: x_query, theta_query, y_query[0] should have gradients from the sampling process
        log_prob = compute_log_prob(
            x_query, theta_query, y_query[0],
            mu_x, sigma_x, mu_theta, sigma_theta, mu_noise, sigma_noise
        )
        
        # Score function estimator: ∇E[L] = E[L * ∇log_p]
        # We multiply loss (detached scalar) with log_prob (has gradients)
        # Then backpropagate to get: loss * ∇log_p
        loss_times_log_prob = loss_value * log_prob
        
        # Compute gradients
        grad_mu_x_i, grad_sigma_x_i, grad_mu_theta_i, grad_sigma_theta_i, grad_mu_noise_i, grad_sigma_noise_i = torch.autograd.grad(
            loss_times_log_prob,
            [mu_x, sigma_x, mu_theta, sigma_theta, mu_noise, sigma_noise],
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )
        
        grad_mu_x += grad_mu_x_i
        grad_sigma_x += grad_sigma_x_i
        grad_mu_theta += grad_mu_theta_i
        grad_sigma_theta += grad_sigma_theta_i
        grad_mu_noise += grad_mu_noise_i
        grad_sigma_noise += grad_sigma_noise_i
    
    # Average over prompts
    n_prompts_tensor = torch.tensor(n_prompts, dtype=torch.float32, device=device)
    expected_loss = total_loss / n_prompts
    
    gradients = {
        'mu_x': (grad_mu_x / n_prompts_tensor).item(),
        'sigma_x': (grad_sigma_x / n_prompts_tensor).item(),
        'mu_theta': (grad_mu_theta / n_prompts_tensor).item(),
        'sigma_theta': (grad_sigma_theta / n_prompts_tensor).item(),
        'mu_noise': (grad_mu_noise / n_prompts_tensor).item(),
        'sigma_noise': (grad_sigma_noise / n_prompts_tensor).item(),
    }
    
    parameter_values = {
        'mu_x': mu_x.item(),
        'sigma_x': sigma_x.item(),
        'mu_theta': mu_theta.item(),
        'sigma_theta': sigma_theta.item(),
        'mu_noise': mu_noise.item(),
        'sigma_noise': sigma_noise.item(),
    }
    
    return {
        'gradients': gradients,
        'expected_loss': expected_loss,
        'parameter_values': parameter_values,
    }

