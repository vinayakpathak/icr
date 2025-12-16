#!/usr/bin/env python3
"""
Optimize prompts for transformer models using elastic net regularization.
Analyzes inner susceptibility - how model predictions respond to prompt perturbations.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import json

import torch
import matplotlib.pyplot as plt
import numpy as np

from ic_regression import (
    ICLinearRegressionTransformer,
    ICRegConfig,
    encode_sequence_tokens,
    load_checkpoint,
)


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


def optimize_prompt_for_checkpoint(
    checkpoint_path: str,
    n_prompt: int = 20,
    target_value: float = 15.0,
    l1_penalty: float = 2.0,
    l2_penalty: float = 2.0,
    learning_rate: float = 0.1,
    num_steps: int = 100,
    optimize_x: bool = False,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict:
    """
    Optimize prompt for a single checkpoint using elastic net regularization.
    
    Args:
        checkpoint_path: Path to checkpoint file
        n_prompt: Number of prompt examples
        target_value: Target value for utility function
        l1_penalty: L1 regularization strength (sparsity)
        l2_penalty: L2 regularization strength (magnitude control)
        learning_rate: Optimization learning rate
        num_steps: Number of optimization steps
        optimize_x: Whether to also optimize X values in prompt
        seed: Random seed for reproducibility
        device: Device to use (auto-detect if None)
        
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
    x_query = torch.randn(1, D, device=device)  # Single query point
    
    # Store initial values
    x_context_initial = x_context.clone().detach()
    y_context_initial = y_context.clone().detach()
    
    # Enable gradients for prompt values
    if optimize_x:
        x_context = x_context.requires_grad_(True)
    y_context = y_context.requires_grad_(True)
    
    # Track trajectories
    prediction_trajectory = []
    perturbation_max_trajectory = []
    loss_trajectory = []
    
    print(f"Optimizing prompt for checkpoint: {checkpoint_path}")
    print(f"  M={M}, step={step}, target={target_value}")
    print(f"  L1={l1_penalty}, L2={l2_penalty}, LR={learning_rate}")
    print(f"  Optimizing {'X and Y' if optimize_x else 'Y only'}")
    print()
    
    # Optimization loop
    for opt_step in range(num_steps):
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
        
        # Combine context and query
        x_full = torch.cat([x_context_use, x_query], dim=0)  # (K_expected, D)
        y_full = torch.cat([y_context_use, torch.zeros(1, device=device)], dim=0)  # (K_expected,)
        
        # Encode to tokens
        tokens = encode_sequence_tokens(x_full, y_full)  # (2*K_expected, D+1)
        tokens = tokens.unsqueeze(0).to(device)  # Add batch dimension and move to device: (1, 2*K_expected, D+1)
        
        # Get prediction for the query point
        # The model predicts for all positions, we want the last one (query position)
        y_pred_all = model.predict_y_from_x_tokens(tokens)  # (1, K_expected)
        y_pred = y_pred_all[0, -1]  # Prediction for query point (scalar)
        
        # Loss: squared error from target
        # We interpret y_pred as mean of Gaussian with sd=1
        # Utility U(y) = -(y - target)^2, so we minimize (y_pred - target)^2
        loss = (y_pred - target_value) ** 2
        
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
        current_pred = y_pred.item()
        max_pert = torch.max(torch.abs(y_perturbation)).item()
        current_loss = loss.item()
        
        prediction_trajectory.append(current_pred)
        perturbation_max_trajectory.append(max_pert)
        loss_trajectory.append(current_loss)
        
        if (opt_step + 1) % 20 == 0:
            print(f"  Step {opt_step + 1}/{num_steps}: pred={current_pred:.4f}, "
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
        "active_indices": active_indices.numpy(),
        "sparsity_count": sparsity_count,
        "target_value": target_value,
        "n_prompt": n_prompt,
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
    pert_max_traj = results["perturbation_max_trajectory"]
    final_pert = results["final_y_perturbation"]
    active_indices = results["active_indices"]
    target = results["target_value"]
    n_prompt = results["n_prompt"]
    
    # Plot 1: Prediction Trajectory
    ax = axes[0, 0]
    ax.plot(prediction_traj, linewidth=3, label="Prediction")
    ax.axhline(y=target, color="r", linestyle="--", label=f"Target ({target})")
    ax.set_title("Optimization Trajectory")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Prediction")
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
    ax.set_title("Perturbation Vector (Sparsity Pattern)")
    ax.set_xlabel("Prompt Index")
    ax.set_ylabel("Change Magnitude")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Plot 4: Before vs After
    ax = axes[1, 1]
    y_initial = results["y_context_initial"]
    y_final = results["y_context_final"]
    x_pos = np.arange(n_prompt)
    width = 0.35
    
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
    fig.suptitle(
        f"Prompt Optimization: {checkpoint_name}\nM={M}, Step={step}, "
        f"Target={target}, Sparsity={results['sparsity_count']}/{n_prompt}",
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
        "--target-value",
        type=float,
        default=15.0,
        help="Target value for utility function (default: 15.0)",
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
    for i, checkpoint_path in enumerate(checkpoints):
        print(f"\n{'='*80}")
        print(f"Processing checkpoint {i+1}/{len(checkpoints)}")
        print(f"{'='*80}")
        
        try:
            # Optimize prompt
            results = optimize_prompt_for_checkpoint(
                checkpoint_path=checkpoint_path,
                n_prompt=args.n_prompt,
                target_value=args.target_value,
                l1_penalty=args.l1_penalty,
                l2_penalty=args.l2_penalty,
                learning_rate=args.learning_rate,
                num_steps=args.num_steps,
                optimize_x=args.optimize_x,
                seed=args.seed,
                device=args.device,
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
                "target_value": args.target_value,
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
