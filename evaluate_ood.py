"""
OOD evaluation function for in-context linear regression models.
Reuses existing evaluation code from ic_regression.py.
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ic_regression import (
    ICLinearRegressionTransformer,
    ICRegConfig,
    InContextLinearRegressionDataset,
    evaluate_ic_regression,
    load_checkpoint,
    recover_training_tasks,
)


def evaluate_ood_score(
    model: ICLinearRegressionTransformer,
    cfg: ICRegConfig,
    n_samples: int = 10000,
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> float:
    """
    Evaluate OOD score (M=inf) for a model.
    
    Creates a dataset with M="inf" (Gaussian task prior: theta ~ N(0, I_D))
    and evaluates the model's MSE/D loss on this dataset.
    
    Args:
        model: The trained model to evaluate
        cfg: Configuration object
        n_samples: Number of samples for evaluation
        batch_size: Batch size for evaluation
        device: Device to use (uses cfg.device if None)
    
    Returns:
        Average MSE/D loss (OOD score, normalized by dimension D)
    """
    if device is None:
        device = cfg.device
    
    # Create OOD dataset with M="inf" (Gaussian task prior)
    ood_dataset = InContextLinearRegressionDataset(
        cfg=cfg,
        tasks=None,  # ignore tasks, sample t ~ N(0, I_D)
        M="inf",
        num_samples=n_samples,
        device="cpu",  # Always CPU in dataset
    )
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    # Evaluate using existing function
    ood_score = evaluate_ic_regression(model, cfg, ood_loader, device)
    
    # Normalize by D to get MSE/D (matching reference implementation)
    ood_score = ood_score / cfg.D
    
    return ood_score


def evaluate_ridge_ood(
    cfg: ICRegConfig,
    n_samples: int = 10000,
    batch_size: int = 1024,
    device: Optional[str] = None,
    reg_lambda: Optional[float] = None,
) -> float:
    """
    Evaluate ridge regression on OOD data (M=inf).
    
    For each prompt, fits ridge regression on context (x_1, y_1, ..., x_{K-1}, y_{K-1})
    and predicts y_K for x_K.
    
    Lambda is computed as: lambda = sigma2 / task_scale^2
    For OOD (M=inf): task_scale = 1.0, so lambda = sigma2
    
    Args:
        cfg: Configuration object
        n_samples: Number of samples for evaluation
        batch_size: Batch size for evaluation
        device: Device to use (uses cfg.device if None)
        reg_lambda: Regularization parameter (default: None, computed as cfg.sigma2)
    
    Returns:
        Average MSE/D loss (OOD score, normalized by dimension D)
    """
    if device is None:
        device = cfg.device
    
    # Compute lambda from config if not provided
    # lambda = noise_scale^2 / task_scale^2
    # For OOD: task_scale = 1.0 (theta ~ N(0, I_D))
    # noise_scale^2 = sigma2
    # So lambda = sigma2 / 1.0 = sigma2
    if reg_lambda is None:
        reg_lambda = cfg.sigma2
    
    # Create OOD dataset with M="inf" (Gaussian task prior)
    ood_dataset = InContextLinearRegressionDataset(
        cfg=cfg,
        tasks=None,  # ignore tasks, sample t ~ N(0, I_D)
        M="inf",
        num_samples=n_samples,
        device="cpu",  # Always CPU in dataset
    )
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    D = cfg.D
    K = cfg.K
    
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for tokens, y_full in ood_loader:
            tokens = tokens.to(device)  # (B, 2K, D+1)
            y_full = y_full.to(device)  # (B, K)
            
            batch_size = tokens.shape[0]
            
            # Decode tokens to get x and y
            # x tokens are at positions 0, 2, 4, ..., 2K-2 (first D coords of token[1:])
            # y tokens are at positions 1, 3, 5, ..., 2K-1 (first coord of token)
            x_all = tokens[:, 0::2, 1:]  # (B, K, D) - extract x from x tokens
            y_all = tokens[:, 1::2, 0]  # (B, K) - extract y from y tokens
            
            # Use first K-1 examples as context, last one as query
            x_context = x_all[:, :-1, :]  # (B, K-1, D)
            y_context = y_all[:, :-1]  # (B, K-1)
            x_query = x_all[:, -1:, :]  # (B, 1, D)
            y_query = y_full[:, -1]  # (B,) - true y for query
            
            # Fit ridge regression for each prompt in batch
            # theta = (X^T X + lambda * I)^(-1) X^T Y
            XT_X = x_context.transpose(-2, -1) @ x_context  # (B, D, D)
            XT_Y = x_context.transpose(-2, -1) @ y_context.unsqueeze(-1)  # (B, D, 1)
            
            # Add regularization: XT_X + lambda * I
            lambda_eye = reg_lambda * torch.eye(D, device=device).unsqueeze(0)  # (1, D, D)
            ridge_matrix = XT_X + lambda_eye  # (B, D, D)
            
            # Solve for theta
            theta_ridge = torch.linalg.solve(ridge_matrix, XT_Y)  # (B, D, 1)
            
            # Predict
            y_pred = (x_query @ theta_ridge).squeeze(-1).squeeze(-1)  # (B,)
            
            # Compute loss
            loss = F.mse_loss(y_pred, y_query, reduction="mean")
            total_loss += loss.item()
            n_batches += 1
    
    # Normalize by D to get MSE/D (matching reference implementation)
    return (total_loss / n_batches) / D


def predict_dmmse(
    x_context: torch.Tensor,  # (B, K-1, D)
    y_context: torch.Tensor,  # (B, K-1)
    x_query: torch.Tensor,    # (B, 1, D)
    tasks: torch.Tensor,      # (M, D)
    sigma2: float,
) -> torch.Tensor:
    """
    Core dMMSE prediction function.
    
    Computes Bayesian posterior over M tasks and makes predictions using
    the discrete Maximum Mean Squared Error (dMMSE) algorithm.
    
    Args:
        x_context: Context x values (B, K-1, D)
        y_context: Context y values (B, K-1)
        x_query: Query x values (B, 1, D)
        tasks: M discrete tasks (M, D)
        sigma2: Noise variance
    
    Returns:
        y_pred: Predicted y values for queries (B,)
    """
    # Compute log-likelihoods for all tasks in parallel
    # For each task theta_i: log P(y_context | x_context, theta_i)
    # = -0.5 * ||y_context - x_context @ theta_i||^2 / sigma2 - (K-1)/2 * log(2*pi*sigma2)
    
    # x_context @ theta_i for all tasks: (B, K-1, D) @ (M, D)^T -> (B, K-1, M)
    # We want: (B, K-1, D) @ (D, M) -> (B, K-1, M)
    # tasks is (M, D), so tasks.T is (D, M)
    y_pred_context = x_context @ tasks.T  # (B, K-1, M)
    
    # Compute squared errors: (B, K-1, M)
    # y_context is (B, K-1), need to expand to (B, K-1, M)
    y_context_expanded = y_context.unsqueeze(-1)  # (B, K-1, 1)
    squared_errors = (y_context_expanded - y_pred_context) ** 2  # (B, K-1, M)
    
    # Sum over K-1 context examples: (B, M)
    sum_squared_errors = squared_errors.sum(dim=1)  # (B, M)
    
    # Log-likelihood (up to constant): (B, M)
    log_likelihood = -0.5 * sum_squared_errors / sigma2  # (B, M)
    # Note: We drop the constant term - (K-1)/2 * log(2*pi*sigma2) since it cancels in normalization
    
    # Normalize using log-sum-exp trick to get posterior probabilities
    # log_posterior_i = log_likelihood_i - log(sum_j exp(log_likelihood_j))
    log_sum_exp = torch.logsumexp(log_likelihood, dim=-1, keepdim=True)  # (B, 1)
    log_posterior = log_likelihood - log_sum_exp  # (B, M)
    posterior = torch.exp(log_posterior)  # (B, M)
    
    # Compute predictive mean: E[y | x_query, x_context, y_context]
    # = sum_i (x_query @ theta_i) * P(theta_i | x_context, y_context)
    # x_query @ tasks.T: (B, 1, D) @ (D, M) -> (B, 1, M)
    y_pred_query = x_query @ tasks.T  # (B, 1, M)
    
    # Weight by posterior: (B, 1, M) * (B, M) -> (B, 1, M), then sum over M
    posterior_expanded = posterior.unsqueeze(1)  # (B, 1, M)
    y_pred = (y_pred_query * posterior_expanded).sum(dim=-1).squeeze(-1)  # (B,)
    
    return y_pred


def evaluate_dmmse_ood(
    cfg: ICRegConfig,
    tasks: torch.Tensor,
    n_samples: int = 10000,
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> float:
    """
    Evaluate dMMSE (discrete Maximum Mean Squared Error) on OOD data.
    
    dMMSE is a Bayesian algorithm that:
    - Forms a uniform prior over M discrete tasks
    - Performs Bayesian updates on context data
    - Uses the Bayesian predictive distribution to make predictions
    
    Args:
        cfg: Configuration object
        tasks: M tasks used during training (shape: (M, D))
        n_samples: Number of samples for evaluation
        batch_size: Batch size for evaluation
        device: Device to use (uses cfg.device if None)
    
    Returns:
        Average MSE/D loss (OOD score, normalized by dimension D)
    """
    if device is None:
        device = cfg.device
    
    # Create OOD dataset with M="inf" (Gaussian task prior)
    ood_dataset = InContextLinearRegressionDataset(
        cfg=cfg,
        tasks=None,  # ignore tasks, sample t ~ N(0, I_D)
        M="inf",
        num_samples=n_samples,
        device="cpu",  # Always CPU in dataset
    )
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    D = cfg.D
    K = cfg.K
    sigma2 = cfg.sigma2
    M = tasks.shape[0]
    
    # Move tasks to device
    tasks = tasks.to(device)  # (M, D)
    
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for tokens, y_full in ood_loader:
            tokens = tokens.to(device)  # (B, 2K, D+1)
            y_full = y_full.to(device)  # (B, K)
            
            batch_size_actual = tokens.shape[0]
            
            # Decode tokens to get x and y
            # x tokens are at positions 0, 2, 4, ..., 2K-2 (first D coords of token[1:])
            # y tokens are at positions 1, 3, 5, ..., 2K-1 (first coord of token)
            x_all = tokens[:, 0::2, 1:]  # (B, K, D) - extract x from x tokens
            y_all = tokens[:, 1::2, 0]  # (B, K) - extract y from y tokens
            
            # Use first K-1 examples as context, last one as query
            x_context = x_all[:, :-1, :]  # (B, K-1, D)
            y_context = y_all[:, :-1]  # (B, K-1)
            x_query = x_all[:, -1:, :]  # (B, 1, D)
            y_query = y_full[:, -1]  # (B,) - true y for query
            
            # Use modular dMMSE prediction function
            y_pred = predict_dmmse(x_context, y_context, x_query, tasks, sigma2)
            
            # Compute loss
            loss = F.mse_loss(y_pred, y_query, reduction="mean")
            total_loss += loss.item()
            n_batches += 1
    
    # Normalize by D to get MSE/D (matching reference implementation)
    return (total_loss / n_batches) / D


def find_m_checkpoint_dirs(base_dir: str = ".") -> list:
    """
    Find all checkpoint directories matching pattern checkpoints_M*/.
    
    Args:
        base_dir: Base directory to search in
    
    Returns:
        List of (M, checkpoint_dir_path) tuples, sorted by M value
    """
    base_path = Path(base_dir)
    checkpoint_dirs = []
    
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoints_M"):
            # Extract M value from directory name
            try:
                M_str = item.name.replace("checkpoints_M", "")
                M = int(M_str)
                checkpoint_path = item / "checkpoint_final.pt"
                if checkpoint_path.exists():
                    checkpoint_dirs.append((M, str(checkpoint_path)))
            except ValueError:
                # Skip if M value can't be parsed
                continue
    
    # Sort by M value
    checkpoint_dirs.sort(key=lambda x: x[0])
    return checkpoint_dirs


def compare_ridge_vs_transformer_ood(
    checkpoint_base_dir: str = "checkpoints",
    n_samples: int = 10000,
    batch_size: int = 1024,
    output_plot: str = "plots/ridge_vs_transformer_ood.png",
    device: Optional[str] = None,
    reg_lambda: Optional[float] = None,
    task_recovery_max_M: int = 32768,
    task_recovery_seed: int = 0,
) -> None:
    """
    Compare ridge regression, dMMSE, and transformer OOD performance across M values.
    
    - Finds all final checkpoints for each M value
    - Evaluates transformer OOD loss for each M
    - Evaluates dMMSE OOD loss for each M (using exact tasks from training)
    - Evaluates ridge regression OOD loss (same for all M)
    - Plots M vs OOD loss for all three methods
    
    Args:
        checkpoint_base_dir: Base directory containing checkpoints_M*/ directories
        n_samples: Number of OOD samples for evaluation
        batch_size: Batch size for evaluation
        output_plot: Path to save the comparison plot
        device: Device to use (auto-detect if None)
        reg_lambda: Override lambda for ridge regression (default: None, compute from cfg.sigma2)
        task_recovery_max_M: max_M parameter for task recovery (default: 32768)
        task_recovery_seed: Seed parameter for task recovery (default: 0)
    """
    print("Finding checkpoints...")
    checkpoint_dirs = find_m_checkpoint_dirs(checkpoint_base_dir)
    
    if len(checkpoint_dirs) == 0:
        print(f"No final checkpoints found in {checkpoint_base_dir}")
        return
    
    print(f"Found {len(checkpoint_dirs)} checkpoints")
    
    # Create config (will be same for all, just need it for lambda computation)
    cfg = ICRegConfig()
    if device is None:
        device = cfg.device
    
    # Recover training tasks once (shared across all M values)
    print("\nRecovering training tasks...")
    all_tasks = recover_training_tasks(
        max_M=task_recovery_max_M,
        D=cfg.D,
        seed=task_recovery_seed,
    )
    print(f"Recovered {len(all_tasks)} tasks (max_M={task_recovery_max_M}, D={cfg.D}, seed={task_recovery_seed})")
    
    # Evaluate ridge regression once (same for all M)
    print("\nEvaluating ridge regression on OOD data...")
    ridge_ood_loss = evaluate_ridge_ood(
        cfg=cfg,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device,
        reg_lambda=reg_lambda,
    )
    print(f"Ridge regression OOD loss: {ridge_ood_loss:.6f}")
    
    # Evaluate transformers and dMMSE for each M
    print("\nEvaluating transformers and dMMSE on OOD data...")
    M_values = []
    transformer_ood_losses = []
    dmmse_ood_losses = []
    
    for M, checkpoint_path in checkpoint_dirs:
        print(f"  Processing M={M}...")
        try:
            model, _, _, step, cfg_checkpoint, M_checkpoint = load_checkpoint(
                checkpoint_path, device=device
            )
            model.to(device)
            
            # Evaluate transformer
            transformer_loss = evaluate_ood_score(
                model=model,
                cfg=cfg_checkpoint,
                n_samples=n_samples,
                batch_size=batch_size,
                device=device,
            )
            
            # Extract tasks for this M (first M tasks)
            tasks_M = all_tasks[:M]  # (M, D)
            
            # Evaluate dMMSE
            dmmse_loss = evaluate_dmmse_ood(
                cfg=cfg_checkpoint,
                tasks=tasks_M,
                n_samples=n_samples,
                batch_size=batch_size,
                device=device,
            )
            
            M_values.append(M)
            transformer_ood_losses.append(transformer_loss)
            dmmse_ood_losses.append(dmmse_loss)
            print(f"    M={M}, step={step}: Transformer OOD loss = {transformer_loss:.6f}, dMMSE OOD loss = {dmmse_loss:.6f}")
            
        except Exception as e:
            print(f"    ERROR: Failed to evaluate M={M}: {e}")
            continue
    
    if len(M_values) == 0:
        print("No checkpoints successfully evaluated")
        return
    
    # Create plot
    print(f"\nCreating comparison plot...")
    os.makedirs(os.path.dirname(output_plot) if os.path.dirname(output_plot) else ".", exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert M values to powers of 2 for x-axis
    # M = 2^power, so power = log2(M)
    powers = [np.log2(M) for M in M_values]
    
    # Plot transformer losses
    ax.plot(powers, transformer_ood_losses, "o-", label="Transformer", linewidth=2, markersize=8)
    
    # Plot dMMSE losses
    ax.plot(powers, dmmse_ood_losses, "s-", label="dMMSE", linewidth=2, markersize=8)
    
    # Plot ridge regression (constant line)
    ax.axhline(
        y=ridge_ood_loss,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Ridge Regression (λ={cfg.sigma2:.3f})",
    )
    
    ax.set_xlabel("Task Diversity M (power of 2)", fontsize=12)
    ax.set_ylabel("OOD Loss (MSE/D)", fontsize=12)
    ax.set_title("Ridge Regression vs dMMSE vs Transformer: OOD Performance", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show M values as powers of 2
    ax.set_xticks(powers)
    ax.set_xticklabels([f"2^{int(p)}" for p in powers])
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Plot saved to {output_plot}")
    print(f"\nSummary:")
    print(f"  Ridge regression OOD loss (MSE/D): {ridge_ood_loss:.6f}")
    print(f"  Transformer OOD losses (MSE/D):")
    for M, loss in zip(M_values, transformer_ood_losses):
        print(f"    M={M}: {loss:.6f}")
    print(f"  dMMSE OOD losses (MSE/D):")
    for M, loss in zip(M_values, dmmse_ood_losses):
        print(f"    M={M}: {loss:.6f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate OOD performance and compare ridge regression, dMMSE, and transformers"
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        type=str,
        default="checkpoints",
        help="Base directory containing checkpoints_M*/ directories",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of OOD samples for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="plots/ridge_vs_transformer_ood.png",
        help="Path to save the comparison plot",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detect if None)",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=None,
        help="Override lambda for ridge regression (default: compute from cfg.sigma2)",
    )
    parser.add_argument(
        "--task-recovery-max-M",
        type=int,
        default=32768,
        help="max_M parameter for task recovery (default: 32768)",
    )
    parser.add_argument(
        "--task-recovery-seed",
        type=int,
        default=0,
        help="Seed parameter for task recovery (default: 0)",
    )
    
    args = parser.parse_args()
    
    compare_ridge_vs_transformer_ood(
        checkpoint_base_dir=args.checkpoint_base_dir,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        output_plot=args.output_plot,
        device=args.device,
        reg_lambda=args.reg_lambda,
        task_recovery_max_M=args.task_recovery_max_M,
        task_recovery_seed=args.task_recovery_seed,
    )
