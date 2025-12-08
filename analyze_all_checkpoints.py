"""
Script to compute OOD scores and in-distribution gradients for all checkpoints
across all M values. Reuses existing functions from ic_regression.py, 
compute_gradients.py, gradient_analysis.py, and evaluate_ood.py.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch

from compute_gradients import find_checkpoints
from evaluate_ood import evaluate_ood_score
from gradient_analysis import compute_expected_loss_gradient
from ic_regression import ICRegConfig, load_checkpoint


def find_m_checkpoint_dirs(base_dir: str = ".") -> list:
    """
    Find all checkpoint directories matching pattern checkpoints_M*/.
    
    Args:
        base_dir: Base directory to search in
    
    Returns:
        List of checkpoint directory paths, sorted by M value
    """
    base_path = Path(base_dir)
    checkpoint_dirs = []
    
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoints_M"):
            # Extract M value from directory name
            try:
                M_str = item.name.replace("checkpoints_M", "")
                M = int(M_str)
                checkpoint_dirs.append((M, str(item)))
            except ValueError:
                # Skip if M value can't be parsed
                continue
    
    # Sort by M value
    checkpoint_dirs.sort(key=lambda x: x[0])
    return [dir_path for _, dir_path in checkpoint_dirs]


def analyze_all_checkpoints(
    checkpoint_base_dir: str = ".",
    output_file: str = "analysis_results.json",
    n_prompts: int = 1000,
    n_ood_samples: int = 10000,
    device: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Compute OOD scores and in-distribution gradients for all checkpoints.
    
    Args:
        checkpoint_base_dir: Base directory containing M-specific checkpoint dirs
        output_file: Output JSON file
        n_prompts: Number of prompts for gradient computation
        n_ood_samples: Number of samples for OOD evaluation
        device: Device to use (uses cfg.device if None)
        verbose: Whether to print progress
    """
    # Find all checkpoint directories
    checkpoint_dirs = find_m_checkpoint_dirs(checkpoint_base_dir)
    
    if len(checkpoint_dirs) == 0:
        print(f"No checkpoint directories found matching pattern checkpoints_M*/ in {checkpoint_base_dir}")
        return
    
    print(f"Found {len(checkpoint_dirs)} checkpoint directories")
    print(f"Using {n_prompts} prompts for gradient computation")
    print(f"Using {n_ood_samples} samples for OOD evaluation")
    print()
    
    # In-distribution gradient parameters (matching training distribution)
    mu_x = 0.0
    sigma_x = 1.0
    mu_theta = 0.0
    sigma_theta = 1.0
    mu_noise = 0.0
    sigma_noise = 0.354  # sqrt(0.125)
    
    if verbose:
        print(f"In-distribution gradient parameters:")
        print(f"  μ_x={mu_x}, σ_x={sigma_x}")
        print(f"  μ_θ={mu_theta}, σ_θ={sigma_theta}")
        print(f"  μ_noise={mu_noise}, σ_noise={sigma_noise}")
        print()
    
    results = []
    total_checkpoints = 0
    
    # Process each checkpoint directory
    for checkpoint_dir in checkpoint_dirs:
        # Extract M from directory name
        M = int(Path(checkpoint_dir).name.replace("checkpoints_M", ""))
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing checkpoints for M={M}")
            print(f"Directory: {checkpoint_dir}")
            print(f"{'='*80}")
        
        # Find all checkpoints in this directory
        checkpoints = find_checkpoints(checkpoint_dir)
        
        if len(checkpoints) == 0:
            if verbose:
                print(f"  No checkpoints found in {checkpoint_dir}")
            continue
        
        if verbose:
            print(f"  Found {len(checkpoints)} checkpoints")
        
        # Process each checkpoint
        for checkpoint_path in checkpoints:
            total_checkpoints += 1
            if verbose:
                print(f"\n  Processing: {Path(checkpoint_path).name}")
            
            try:
                # Load checkpoint
                model, _, _, step, cfg, M_loaded = load_checkpoint(checkpoint_path, device=device)
                if device is None:
                    device = cfg.device
                
                # Verify M matches directory
                if str(M_loaded) != str(M):
                    if verbose:
                        print(f"    Warning: M mismatch (directory={M}, checkpoint={M_loaded})")
                
                # Compute OOD score
                if verbose:
                    print(f"    Computing OOD score...")
                ood_score = evaluate_ood_score(
                    model=model,
                    cfg=cfg,
                    n_samples=n_ood_samples,
                    batch_size=1024,
                    device=device,
                )
                
                # Compute in-distribution gradients
                if verbose:
                    print(f"    Computing in-distribution gradients...")
                
                # Create parameter tensors with requires_grad=True
                mu_x_tensor = torch.tensor(mu_x, dtype=torch.float32, device=device, requires_grad=True)
                sigma_x_tensor = torch.tensor(max(sigma_x, 1e-6), dtype=torch.float32, device=device, requires_grad=True)
                mu_theta_tensor = torch.tensor(mu_theta, dtype=torch.float32, device=device, requires_grad=True)
                sigma_theta_tensor = torch.tensor(max(sigma_theta, 1e-6), dtype=torch.float32, device=device, requires_grad=True)
                mu_noise_tensor = torch.tensor(mu_noise, dtype=torch.float32, device=device, requires_grad=True)
                sigma_noise_tensor = torch.tensor(max(sigma_noise, 1e-6), dtype=torch.float32, device=device, requires_grad=True)
                
                gradient_result = compute_expected_loss_gradient(
                    model=model,
                    mu_x=mu_x_tensor,
                    sigma_x=sigma_x_tensor,
                    mu_theta=mu_theta_tensor,
                    sigma_theta=sigma_theta_tensor,
                    mu_noise=mu_noise_tensor,
                    sigma_noise=sigma_noise_tensor,
                    n_prompts=n_prompts,
                    D=cfg.D,
                    K=cfg.K,
                    device=device,
                    verbose=False,  # Reduce verbosity for individual checkpoints
                )
                
                # Store result
                result = {
                    "M": M,
                    "checkpoint": checkpoint_path,
                    "step": step,
                    "ood_score": ood_score,
                    "gradients": gradient_result["gradients"],
                    "parameter_values": gradient_result["parameter_values"],
                    "expected_loss": gradient_result["expected_loss"],
                }
                results.append(result)
                
                if verbose:
                    print(f"    OOD score: {ood_score:.6f}")
                    print(f"    Expected loss (in-distribution): {gradient_result['expected_loss']:.6f}")
                    print(f"    Gradients: {gradient_result['gradients']}")
                
            except Exception as e:
                print(f"    Error processing {checkpoint_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Processed {total_checkpoints} checkpoints")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute OOD scores and gradients for all checkpoints")
    parser.add_argument("--checkpoint_base_dir", type=str, default=".", help="Base directory containing M-specific checkpoint dirs")
    parser.add_argument("--output", type=str, default="analysis_results.json", help="Output JSON file")
    parser.add_argument("--n_prompts", type=int, default=1000, help="Number of prompts for gradient computation")
    parser.add_argument("--n_ood_samples", type=int, default=10000, help="Number of samples for OOD evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, defaults to cfg.device)")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    
    args = parser.parse_args()
    
    analyze_all_checkpoints(
        checkpoint_base_dir=args.checkpoint_base_dir,
        output_file=args.output,
        n_prompts=args.n_prompts,
        n_ood_samples=args.n_ood_samples,
        device=args.device,
        verbose=not args.quiet,
    )

