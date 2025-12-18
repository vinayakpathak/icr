#!/usr/bin/env python3
"""
Optimize prompts for transformer models using elastic net regularization.
Analyzes inner susceptibility - how model predictions respond to prompt perturbations.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Callable
import json

import torch
import matplotlib.pyplot as plt
import numpy as np

from ic_regression import (
    ICLinearRegressionTransformer,
    ICRegConfig,
    encode_sequence_tokens,
    load_checkpoint,
    recover_training_tasks,
)

from evaluate_ood import predict_dmmse


def find_checkpoints(
    checkpoint_dir: str = "checkpoints",
    checkpoint_type: str = "all",
) -> List[str]:
    """
    Find all checkpoint files in directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        checkpoint_type: 'all', 'final', or 'step' to filter checkpoint types
        
    Returns:
        List of checkpoint file paths
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []
    
    checkpoints = []
    for checkpoint_file in checkpoint_path.rglob("*.pt"):
        filename = checkpoint_file.name
        
        if checkpoint_type == "final" and "final" not in filename:
            continue
        elif checkpoint_type == "step" and "final" in filename:
            continue
        
        checkpoints.append(str(checkpoint_file))
    
    # Sort by M value and step number if possible
    def get_sort_key(path: str) -> Tuple[int, int]:
        """Extract M value and step number for sorting."""
        path_str = str(path)
        M = 0
        step = 0
        
        # Try to extract M from directory name (checkpoints_M{M})
        parts = path_str.split("/")
        for part in parts:
            if part.startswith("checkpoints_M"):
                try:
                    M = int(part.replace("checkpoints_M", ""))
                except:
                    pass
        
        # Try to extract step from filename
        if "step_" in path_str:
            try:
                step_str = path_str.split("step_")[1].split(".")[0]
                step = int(step_str)
            except:
                step = 0
        elif "final" in path_str:
            step = 999999  # Put final checkpoints at the end
        
        return (M, step)
    
    checkpoints.sort(key=get_sort_key)
    return checkpoints


def create_ridge_mapping(
    x_context: torch.Tensor,  # (K, D)
    y_context: torch.Tensor,  # (K,)
    cfg: ICRegConfig,
    reg_lambda: Optional[float] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a Ridge regression mapping function.
    
    Fits ridge regression on (x_context, y_context) and returns a function
    that predicts y_test from x_test.
    
    Args:
        x_context: Context x values (K, D)
        y_context: Context y values (K,)
        cfg: Configuration object
        reg_lambda: Regularization parameter (default: cfg.sigma2)
    
    Returns:
        Function that takes x_test (D,) and returns desired y_test (scalar tensor)
    """
    D = cfg.D
    K = x_context.shape[0]
    
    # Compute lambda from config if not provided
    if reg_lambda is None:
        reg_lambda = cfg.sigma2
    
    # Fit ridge regression on context
    # X^T X + lambda * I
    XT_X = x_context.T @ x_context  # (D, D)
    XT_Y = x_context.T @ y_context.unsqueeze(-1)  # (D, 1)
    
    # Add regularization
    lambda_eye = reg_lambda * torch.eye(D, device=x_context.device, dtype=x_context.dtype)
    ridge_matrix = XT_X + lambda_eye  # (D, D)
    
    # Solve for theta
    try:
        theta_ridge = torch.linalg.solve(ridge_matrix, XT_Y)  # (D, 1)
    except Exception as e:
        print(f"[DEBUG] Error in torch.linalg.solve: {e}")
        print(f"[DEBUG] ridge_matrix shape: {ridge_matrix.shape}, XT_Y shape: {XT_Y.shape}")
        print(f"[DEBUG] ridge_matrix has NaN: {torch.isnan(ridge_matrix).any()}")
        print(f"[DEBUG] XT_Y has NaN: {torch.isnan(XT_Y).any()}")
        raise
    
    # Return mapping function
    def mapping_fn(x_test: torch.Tensor) -> torch.Tensor:
        """
        Predict y_test from x_test using fitted ridge regression.
        
        Args:
            x_test: Query x value (D,)
        
        Returns:
            Predicted y_test (scalar tensor)
        """
        # x_test @ theta_ridge: (D,) @ (D, 1) -> (1,)
        y_pred = (x_test @ theta_ridge).squeeze()
        return y_pred
    
    return mapping_fn


def create_dmmse_mapping(
    x_context: torch.Tensor,  # (K, D)
    y_context: torch.Tensor,  # (K,)
    tasks: torch.Tensor,      # (M, D)
    cfg: ICRegConfig,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a dMMSE mapping function.
    
    Uses Bayesian posterior over M tasks to predict y_test from x_test.
    
    Args:
        x_context: Context x values (K, D)
        y_context: Context y values (K,)
        tasks: M discrete tasks (M, D)
        cfg: Configuration object
    
    Returns:
        Function that takes x_test (D,) and returns desired y_test (scalar tensor)
    """
    sigma2 = cfg.sigma2
    
    # Move tasks to same device as context
    tasks = tasks.to(x_context.device)
    
    # Return mapping function
    def mapping_fn(x_test: torch.Tensor) -> torch.Tensor:
        """
        Predict y_test from x_test using dMMSE.
        
        Args:
            x_test: Query x value (D,)
        
        Returns:
            Predicted y_test (scalar tensor)
        """
        # predict_dmmse expects batched inputs
        # x_context: (1, K, D), y_context: (1, K), x_query: (1, 1, D)
        x_context_batched = x_context.unsqueeze(0)  # (1, K, D)
        y_context_batched = y_context.unsqueeze(0)  # (1, K)
        x_query_batched = x_test.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        
        # Get prediction
        y_pred_batched = predict_dmmse(
            x_context_batched,
            y_context_batched,
            x_query_batched,
            tasks,
            sigma2,
        )  # (1,)
        
        return y_pred_batched.squeeze(0)  # scalar
    
    return mapping_fn


def create_constant_mapping(
    target_value: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a constant mapping function (ignores x_test).
    
    Args:
        target_value: Constant value to return
    
    Returns:
        Function that always returns target_value regardless of x_test
    """
    def mapping_fn(x_test: torch.Tensor) -> torch.Tensor:
        """
        Return constant target value.
        
        Args:
            x_test: Query x value (ignored)
        
        Returns:
            target_value as tensor
        """
        # Create tensor with same device and dtype as x_test
        return torch.tensor(
            target_value,
            device=x_test.device,
            dtype=x_test.dtype,
        )
    
    return mapping_fn


def create_mapping(
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    mapping_type: str,  # "ridge", "dmmse", or "constant"
    cfg: ICRegConfig,
    target_value: Optional[float] = None,
    tasks: Optional[torch.Tensor] = None,
    reg_lambda: Optional[float] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Factory function to create mapping based on type.
    
    Args:
        x_context: Context x values (K, D)
        y_context: Context y values (K,)
        mapping_type: Type of mapping - "ridge", "dmmse", or "constant"
        cfg: Configuration object
        target_value: Target value for constant mapping (required if mapping_type="constant")
        tasks: Tasks for dMMSE mapping (required if mapping_type="dmmse")
        reg_lambda: Lambda for Ridge regression (optional, uses cfg.sigma2 if None)
    
    Returns:
        Mapping function: x_test -> y_desired
    """
    if mapping_type == "ridge":
        return create_ridge_mapping(x_context, y_context, cfg, reg_lambda)
    elif mapping_type == "dmmse":
        if tasks is None:
            raise ValueError("tasks must be provided for dMMSE mapping")
        return create_dmmse_mapping(x_context, y_context, tasks, cfg)
    elif mapping_type == "constant":
        if target_value is None:
            raise ValueError("target_value must be provided for constant mapping")
        return create_constant_mapping(target_value)
    else:
        raise ValueError(f"Unknown mapping_type: {mapping_type}. Must be 'ridge', 'dmmse', or 'constant'")


def optimize_prompt_for_checkpoint(
    checkpoint_path: str,
    n_prompt: int = 20,
    n_x_test: int = 100,
    mapping_type: str = "ridge",
    target_value: Optional[float] = None,
    l1_penalty: float = 2.0,
    l2_penalty: float = 2.0,
    learning_rate: float = 0.1,
    num_steps: int = 100,
    optimize_x: bool = False,
    seed: int = 42,
    device: Optional[str] = None,
    tasks: Optional[torch.Tensor] = None,
    reg_lambda: Optional[float] = None,
) -> Dict:
    """
    Optimize prompt for a single checkpoint using elastic net regularization.
    
    Args:
        checkpoint_path: Path to checkpoint file
        n_prompt: Number of prompt examples
        n_x_test: Number of x_test values to generate (default: 100)
        mapping_type: Type of mapping - "ridge", "dmmse", or "constant" (default: "ridge")
        target_value: Target value for constant mapping (only used if mapping_type="constant")
        l1_penalty: L1 regularization strength (sparsity)
        l2_penalty: L2 regularization strength (magnitude control)
        learning_rate: Optimization learning rate
        num_steps: Number of optimization steps
        optimize_x: Whether to also optimize X values in prompt
        seed: Random seed for reproducibility
        device: Device to use (auto-detect if None)
        tasks: Tasks for dMMSE mapping (if None and mapping_type="dmmse", recover from checkpoint M)
        reg_lambda: Lambda for Ridge regression (if None, uses cfg.sigma2)
        
    Returns:
        Dictionary with optimization trajectory and results
    """
    torch.manual_seed(seed)
    
    # Load checkpoint
    model, _, _, step, cfg, M = load_checkpoint(checkpoint_path, device=device)
    if device is None:
        device = cfg.device
    
    model.to(device)
    model.eval()  # Set to eval mode, but we'll enable gradients for prompt values
    
    # Freeze model parameters (we only want gradients w.r.t. prompt)
    for param in model.parameters():
        param.requires_grad = False
    
    # Initialize prompt (matching notebook: Y = X @ theta + noise)
    D = cfg.D
    x_context = torch.randn(n_prompt, D, device=device)
    true_theta = torch.randn(D, 1, device=device)
    noise = 0.1 * torch.randn(n_prompt, 1, device=device)
    y_context = (x_context @ true_theta + noise).squeeze(1)  # (n_prompt,)
    
    # Generate multiple x_test values (same distribution as x_context)
    x_test_all = torch.randn(n_x_test, D, device=device)  # (n_x_test, D)
    
    # Store initial values
    x_context_initial = x_context.clone().detach()
    y_context_initial = y_context.clone().detach()
    
    # Enable gradients for prompt values
    if optimize_x:
        x_context = x_context.requires_grad_(True)
    y_context = y_context.requires_grad_(True)
    
    # Recover tasks for dMMSE if needed
    if mapping_type == "dmmse" and tasks is None:
        # Recover tasks from checkpoint M value
        if isinstance(M, int):
            all_tasks = recover_training_tasks(max_M=32768, D=cfg.D, seed=0)
            tasks = all_tasks[:M].to(device)  # (M, D)
            print(f"  Recovered {M} tasks from training (max_M=32768, seed=0)")
        else:
            raise ValueError(f"Cannot recover tasks for dMMSE: M={M} is not an integer")
    
    # Track trajectories
    prediction_trajectory = []
    perturbation_max_trajectory = []
    loss_trajectory = []
    
    print(f"Optimizing prompt for checkpoint: {checkpoint_path}")
    print(f"  M={M}, step={step}, mapping_type={mapping_type}, n_x_test={n_x_test}")
    if mapping_type == "constant":
        print(f"  target_value={target_value}")
    print(f"  L1={l1_penalty}, L2={l2_penalty}, LR={learning_rate}")
    print(f"  Optimizing {'X and Y' if optimize_x else 'Y only'}")
    print()
    
    print(f"[DEBUG] Starting optimization loop: {num_steps} steps")
    print(f"[DEBUG] Device: {device}, x_context shape: {x_context.shape}, y_context shape: {y_context.shape}")
    print(f"[DEBUG] x_test_all shape: {x_test_all.shape}")
    
    # Optimization loop
    for opt_step in range(num_steps):
        if opt_step == 0:
            print(f"[DEBUG] Step {opt_step}: Starting first optimization step...")
        # Zero gradients
        if optimize_x:
            x_context.grad = None
        y_context.grad = None
        
        # Encode tokens: we need to handle the model's expected K value
        # The model expects exactly cfg.K examples, so we may need to pad/truncate
        K_expected = cfg.K
        K_context = n_prompt
        
        if K_context + 1 > K_expected:
            # Truncate: use first K_expected - 1 context examples + 1 query
            K_context_use = K_expected - 1
            x_context_use = x_context[:K_context_use]
            y_context_use = y_context[:K_context_use]
        else:
            # Pad with zeros to reach K_expected
            K_pad = K_expected - K_context - 1
            x_context_use = x_context
            y_context_use = y_context
            if K_pad > 0:
                x_pad = torch.zeros(K_pad, D, device=device)
                y_pad = torch.zeros(K_pad, device=device)
                x_context_use = torch.cat([x_context, x_pad], dim=0)
                y_context_use = torch.cat([y_context, y_pad], dim=0)
        
        if opt_step == 0:
            print(f"[DEBUG] Step {opt_step}: Creating mapping function...")
        
        # Create mapping function from z (context)
        mapping_fn = create_mapping(
            x_context=x_context_use,
            y_context=y_context_use,
            mapping_type=mapping_type,
            cfg=cfg,
            target_value=target_value,
            tasks=tasks,
            reg_lambda=reg_lambda,
        )
        
        if opt_step == 0:
            print(f"[DEBUG] Step {opt_step}: Mapping function created, computing losses over {len(x_test_all)} x_test values...")
        
        # Compute loss over all x_test values
        losses = []
        for i, x_test in enumerate(x_test_all):
            if opt_step == 0 and i == 0:
                print(f"[DEBUG] Step {opt_step}: Processing x_test {i+1}/{len(x_test_all)}...")
            # Combine context and query
            x_full = torch.cat([x_context_use, x_test.unsqueeze(0)], dim=0)  # (K_expected, D)
            y_full = torch.cat([y_context_use, torch.zeros(1, device=device)], dim=0)  # (K_expected,)
            
            # Encode to tokens
            tokens = encode_sequence_tokens(x_full, y_full)  # (2*K_expected, D+1)
            tokens = tokens.unsqueeze(0).to(device)  # Add batch dimension: (1, 2*K_expected, D+1)
            
            if opt_step == 0 and i == 0:
                print(f"[DEBUG] Step {opt_step}: x_test {i+1} - tokens shape: {tokens.shape}")
            
            # Get model prediction for the query point
            y_pred_all = model.predict_y_from_x_tokens(tokens)  # (1, K_expected)
            y_pred = y_pred_all[0, -1]  # Prediction for query point (scalar)
            
            if opt_step == 0 and i == 0:
                print(f"[DEBUG] Step {opt_step}: x_test {i+1} - model prediction computed: {y_pred.item():.4f}")
            
            # Get desired value from mapping function
            y_desired = mapping_fn(x_test)  # Scalar tensor
            
            # Compute squared error
            losses.append((y_pred - y_desired) ** 2)
            
            if opt_step == 0 and i == 0:
                print(f"[DEBUG] Step {opt_step}: x_test {i+1} - y_pred={y_pred.item():.4f}, y_desired={y_desired.item():.4f}, loss={losses[-1].item():.4f}")
        
        # Average loss over all x_test values
        loss = torch.stack(losses).mean()
        
        if opt_step == 0:
            print(f"[DEBUG] Step {opt_step}: Average loss computed: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Get gradients
        y_grad = y_context.grad.clone()
        if optimize_x:
            x_grad = x_context.grad.clone()
        
        # Compute perturbations
        y_perturbation = y_context - y_context_initial
        if optimize_x:
            x_perturbation = x_context - x_context_initial
        
        # Elastic net regularization gradients
        # L1: sign of perturbation (sparsity)
        l1_grad_y = l1_penalty * torch.sign(y_perturbation)
        # L2: linear in perturbation (magnitude control)
        l2_grad_y = l2_penalty * y_perturbation
        
        # Update Y (matching notebook protocol)
        # Notebook uses: Y += lr * (chi_in - l1_grad - l2_grad)
        # Since chi_in = -loss_grad (gradient to maximize utility = -gradient to minimize loss)
        # This becomes: Y += lr * (-loss_grad - l1_grad - l2_grad)
        # Which is: Y -= lr * (loss_grad + l1_grad + l2_grad)
        # So we use: Y -= lr * (y_grad + l1_grad_y + l2_grad_y)
        with torch.no_grad():
            y_context.data -= learning_rate * (y_grad + l1_grad_y + l2_grad_y)
        
        # Update X if optimizing (same protocol)
        if optimize_x:
            l1_grad_x = l1_penalty * torch.sign(x_perturbation)
            l2_grad_x = l2_penalty * x_perturbation
            with torch.no_grad():
                x_context.data -= learning_rate * (x_grad + l1_grad_x + l2_grad_x)
        
        # Track metrics
        # For tracking, use average prediction over x_test values
        # (recompute for display purposes)
        with torch.no_grad():
            preds = []
            for x_test in x_test_all:
                x_full = torch.cat([x_context_use, x_test.unsqueeze(0)], dim=0)
                y_full = torch.cat([y_context_use, torch.zeros(1, device=device)], dim=0)
                tokens = encode_sequence_tokens(x_full, y_full).unsqueeze(0).to(device)
                y_pred_all = model.predict_y_from_x_tokens(tokens)
                preds.append(y_pred_all[0, -1].item())
            avg_pred = np.mean(preds)
        
        max_pert = torch.max(torch.abs(y_perturbation)).item()
        current_loss = loss.item()
        
        prediction_trajectory.append(avg_pred)
        perturbation_max_trajectory.append(max_pert)
        loss_trajectory.append(current_loss)
        
        if (opt_step + 1) % 20 == 0:
            print(f"  Step {opt_step + 1}/{num_steps}: avg_pred={avg_pred:.4f}, "
                  f"loss={current_loss:.4f}, max_pert={max_pert:.4f}")
    
    # Final perturbation
    final_y_perturbation = (y_context - y_context_initial).detach().cpu()
    if optimize_x:
        final_x_perturbation = (x_context - x_context_initial).detach().cpu()
    else:
        final_x_perturbation = None
    
    # Find active indices (significant perturbations)
    active_indices = torch.where(torch.abs(final_y_perturbation) > 0.5)[0]
    sparsity_count = n_prompt - len(active_indices)
    
    print(f"\nFinal: pred={prediction_trajectory[-1]:.4f}, "
          f"max_pert={perturbation_max_trajectory[-1]:.4f}, "
          f"sparsity={sparsity_count}/{n_prompt} unchanged")
    
    # Store X context values if optimized
    x_context_initial_result = x_context_initial.detach().cpu().numpy()
    x_context_final_result = x_context.detach().cpu().numpy() if optimize_x else None
    
    return {
        "checkpoint_path": checkpoint_path,
        "M": M,
        "step": step,
        "prediction_trajectory": prediction_trajectory,
        "perturbation_max_trajectory": perturbation_max_trajectory,
        "loss_trajectory": loss_trajectory,
        "final_y_perturbation": final_y_perturbation.numpy(),
        "final_x_perturbation": final_x_perturbation.numpy() if final_x_perturbation is not None else None,
        "y_context_initial": y_context_initial.detach().cpu().numpy(),
        "y_context_final": y_context.detach().cpu().numpy(),
        "x_context_initial": x_context_initial_result,
        "x_context_final": x_context_final_result,
        "active_indices": active_indices.numpy(),
        "sparsity_count": sparsity_count,
        "target_value": target_value,
        "n_prompt": n_prompt,
        "n_x_test": n_x_test,
        "mapping_type": mapping_type,
    }


def plot_optimization_results(
    results: Dict,
    checkpoint_name: str,
    output_path: str,
):
    """
    Generate plots similar to notebook.
    
    Args:
        results: Dictionary with optimization results
        checkpoint_name: Name for the checkpoint (for title)
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    prediction_traj = results["prediction_trajectory"]
    loss_traj = results["loss_trajectory"]
    pert_max_traj = results["perturbation_max_trajectory"]
    final_pert = results["final_y_perturbation"]
    active_indices = results["active_indices"]
    target = results.get("target_value")
    n_prompt = results["n_prompt"]
    mapping_type = results.get("mapping_type", "constant")
    n_x_test = results.get("n_x_test", 1)
    
    # Plot 1: Average Loss Trajectory
    ax = axes[0, 0]
    ax.plot(loss_traj, linewidth=3, label="Avg Loss", color="blue")
    ax.set_title(f"Loss Trajectory (mapping={mapping_type}, n_x_test={n_x_test})")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Average Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Max Perturbation (Magnitude Control)
    ax = axes[0, 1]
    ax.plot(pert_max_traj, linewidth=3, color="orange", label="Max |Change|")
    ax.set_title("Magnitude Control (Max Perturbation Size)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Max Absolute Change")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final Perturbation Vector (Elastic Net Style)
    ax = axes[1, 0]
    colors = [
        "grey" if i not in active_indices else "purple"
        for i in range(n_prompt)
    ]
    ax.bar(
        range(n_prompt),
        final_pert,
        color=colors,
        alpha=0.8,
    )
    ax.set_title("Y Perturbation Vector (Sparsity Pattern)")
    ax.set_xlabel("Prompt Index")
    ax.set_ylabel("Y Change Magnitude")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Plot 4: Before vs After
    ax = axes[1, 1]
    y_initial = results["y_context_initial"]
    y_final = results["y_context_final"]
    x_pos = np.arange(n_prompt)
    width = 0.35
    
    # Check if X was also optimized
    final_x_perturbation = results.get("final_x_perturbation")
    x_context_initial = results.get("x_context_initial")
    x_context_final = results.get("x_context_final")
    
    if final_x_perturbation is not None:
        # Show both X and Y perturbations
        # Use subplot or show X perturbation magnitude
        x_pert_magnitude = np.linalg.norm(final_x_perturbation, axis=1) if final_x_perturbation.ndim > 1 else np.abs(final_x_perturbation)
        y_pert_magnitude = np.abs(final_pert)
        
        ax.plot(range(n_prompt), x_pert_magnitude, 'o-', label="X Perturbation (L2 norm)", color="red", linewidth=2, markersize=4)
        ax.plot(range(n_prompt), y_pert_magnitude, 's-', label="Y Perturbation (abs)", color="blue", linewidth=2, markersize=4)
        ax.set_title("X and Y Perturbation Magnitudes")
        ax.set_xlabel("Prompt Index")
        ax.set_ylabel("Perturbation Magnitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Only Y was optimized - show before/after Y values
        ax.bar(x_pos - width / 2, y_initial, width, label="Initial Y", alpha=0.7)
        ax.bar(x_pos + width / 2, y_final, width, label="Optimized Y", alpha=0.7)
        ax.set_title("Before vs After (Y Values)")
        ax.set_xlabel("Prompt Index")
        ax.set_ylabel("Y Value")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    
    # Overall title
    M = results["M"]
    step = results["step"]
    mapping_type = results.get("mapping_type", "constant")
    n_x_test = results.get("n_x_test", 1)
    title_parts = [f"M={M}, Step={step}", f"mapping={mapping_type}, n_x_test={n_x_test}"]
    if mapping_type == "constant" and target is not None:
        title_parts.append(f"target={target}")
    title_parts.append(f"Sparsity={results['sparsity_count']}/{n_prompt}")
    fig.suptitle(
        f"Prompt Optimization: {checkpoint_name}\n" + ", ".join(title_parts),
        fontsize=14,
        y=0.995,
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved plot to {output_path}")


def main():
    """Main function: find checkpoints, optimize, plot, save."""
    parser = argparse.ArgumentParser(
        description="Optimize prompts for transformer models using elastic net regularization"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory with checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        default="all",
        choices=["all", "final", "step"],
        help="Type of checkpoints to process (default: all)",
    )
    parser.add_argument(
        "--n-prompt",
        type=int,
        default=20,
        help="Number of prompt examples (default: 20)",
    )
    parser.add_argument(
        "--n-x-test",
        type=int,
        default=100,
        help="Number of x_test values to generate (default: 100)",
    )
    parser.add_argument(
        "--mapping-type",
        type=str,
        default="ridge",
        choices=["ridge", "dmmse", "constant"],
        help="Type of mapping function: 'ridge', 'dmmse', or 'constant' (default: ridge)",
    )
    parser.add_argument(
        "--target-value",
        type=float,
        default=None,
        help="Target value for constant mapping (only used if mapping-type=constant, default: None)",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=None,
        help="Lambda for Ridge regression (default: None, uses cfg.sigma2)",
    )
    parser.add_argument(
        "--l1-penalty",
        type=float,
        default=2.0,
        help="L1 regularization strength (default: 2.0)",
    )
    parser.add_argument(
        "--l2-penalty",
        type=float,
        default=2.0,
        help="L2 regularization strength (default: 2.0)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Optimization learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of optimization steps (default: 100)",
    )
    parser.add_argument(
        "--optimize-x",
        action="store_true",
        help="Also optimize X values in prompt (default: False)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots)",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Maximum number of checkpoints to process (default: all)",
    )
    
    args = parser.parse_args()
    
    # Find checkpoints
    print(f"Searching for checkpoints in {args.checkpoint_dir}...")
    checkpoints = find_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_type=args.checkpoint_type,
    )
    
    if len(checkpoints) == 0:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return
    
    if args.max_checkpoints:
        checkpoints = checkpoints[: args.max_checkpoints]
    
    print(f"Found {len(checkpoints)} checkpoints to process")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each checkpoint
    all_results = []
    print(f"[DEBUG] Starting to process {len(checkpoints)} checkpoints")
    for i, checkpoint_path in enumerate(checkpoints):
        print(f"\n{'='*80}")
        print(f"Processing checkpoint {i+1}/{len(checkpoints)}")
        print(f"{'='*80}")
        print(f"[DEBUG] Checkpoint path: {checkpoint_path}")
        
        try:
            # Optimize prompt
            results = optimize_prompt_for_checkpoint(
                checkpoint_path=checkpoint_path,
                n_prompt=args.n_prompt,
                n_x_test=args.n_x_test,
                mapping_type=args.mapping_type,
                target_value=args.target_value,
                l1_penalty=args.l1_penalty,
                l2_penalty=args.l2_penalty,
                learning_rate=args.learning_rate,
                num_steps=args.num_steps,
                optimize_x=args.optimize_x,
                seed=args.seed,
                device=args.device,
                reg_lambda=args.reg_lambda,
            )
            
            # Generate checkpoint name for plot
            checkpoint_file = Path(checkpoint_path)
            checkpoint_name = checkpoint_file.stem
            M = results["M"]
            step = results["step"]
            
            # Save plot
            plot_filename = f"prompt_optimization_M{M}_step{step}.png"
            plot_path = output_dir / plot_filename
            
            plot_optimization_results(
                results=results,
                checkpoint_name=checkpoint_name,
                output_path=str(plot_path),
            )
            
            all_results.append(results)
            
        except Exception as e:
            print(f"ERROR: Failed to process {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    if all_results:
        summary_path = output_dir / "optimization_summary.json"
        summary = {
            "num_checkpoints": len(all_results),
            "config": {
                "n_prompt": args.n_prompt,
                "n_x_test": args.n_x_test,
                "mapping_type": args.mapping_type,
                "target_value": args.target_value,
                "reg_lambda": args.reg_lambda,
                "l1_penalty": args.l1_penalty,
                "l2_penalty": args.l2_penalty,
                "learning_rate": args.learning_rate,
                "num_steps": args.num_steps,
                "optimize_x": args.optimize_x,
                "seed": args.seed,
            },
            "results": [
                {
                    "checkpoint_path": r["checkpoint_path"],
                    "M": r["M"],
                    "step": r["step"],
                    "final_prediction": r["prediction_trajectory"][-1],
                    "final_loss": r["loss_trajectory"][-1],
                    "max_perturbation": r["perturbation_max_trajectory"][-1],
                    "sparsity_count": r["sparsity_count"],
                }
                for r in all_results
            ],
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Summary saved to {summary_path}")
        print(f"Processed {len(all_results)} checkpoints successfully")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
