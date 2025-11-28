"""
Script to compute gradients of expected loss w.r.t. distribution parameters
for all saved checkpoints.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch

from gradient_analysis import compute_expected_loss_gradient
from ic_regression import ICRegConfig, load_checkpoint


def find_checkpoints(checkpoint_dir: str = "checkpoints") -> list:
    """Find all checkpoint files in the directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []
    
    checkpoints = []
    for file in checkpoint_path.glob("*.pt"):
        checkpoints.append(str(file))
    
    # Sort by step number if possible, otherwise by filename
    def get_step(path):
        try:
            if "step_" in path:
                step_str = path.split("step_")[1].split(".")[0]
                return int(step_str)
            elif "final" in path:
                return float('inf')  # Put final at the end
            else:
                return -1
        except:
            return -1
    
    checkpoints.sort(key=get_step)
    return checkpoints


def compute_gradients_for_checkpoints(
    checkpoint_dir: str = "checkpoints",
    output_file: str = "gradient_results.json",
    mu_x: float = 0.0,
    sigma_x: float = 1.0,
    mu_theta: float = 0.0,
    sigma_theta: float = 1.0,
    mu_noise: float = 0.0,
    sigma_noise: float = 0.354,  # sqrt(0.125) to match default sigma2
    n_prompts: int = 1000,
    device: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Compute gradients for all checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        output_file: File to save results (JSON format)
        mu_x, sigma_x, mu_theta, sigma_theta, mu_noise, sigma_noise: Initial parameter values
        n_prompts: Number of prompts for Monte Carlo estimation
        device: Device to use (uses cfg.device if None)
        verbose: Whether to print progress
    """
    checkpoints = find_checkpoints(checkpoint_dir)
    
    if len(checkpoints) == 0:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Parameter values: μ_x={mu_x}, σ_x={sigma_x}, μ_θ={mu_theta}, σ_θ={sigma_theta}, μ_noise={mu_noise}, σ_noise={sigma_noise}")
    print(f"Using {n_prompts} prompts for Monte Carlo estimation\n")
    
    results = []
    
    for i, checkpoint_path in enumerate(checkpoints):
        print(f"[{i+1}/{len(checkpoints)}] Processing {checkpoint_path}")
        
        try:
            # Load checkpoint
            model, _, _, step, cfg, M = load_checkpoint(checkpoint_path, device=device)
            if device is None:
                device = cfg.device
            
            # Create parameter tensors with requires_grad=True
            # Use softplus to ensure positive values for sigma parameters
            mu_x_tensor = torch.tensor(mu_x, dtype=torch.float32, device=device, requires_grad=True)
            # Use softplus to ensure sigma > 0: sigma = softplus(log_sigma) where log_sigma can be any real
            # For simplicity, we'll use the raw value and assume it's positive (user's responsibility)
            # Or we can use: sigma = torch.nn.functional.softplus(log_sigma_param)
            # For now, let's use raw values but add a small epsilon to ensure positivity
            sigma_x_tensor = torch.tensor(max(sigma_x, 1e-6), dtype=torch.float32, device=device, requires_grad=True)
            
            mu_theta_tensor = torch.tensor(mu_theta, dtype=torch.float32, device=device, requires_grad=True)
            sigma_theta_tensor = torch.tensor(max(sigma_theta, 1e-6), dtype=torch.float32, device=device, requires_grad=True)
            
            mu_noise_tensor = torch.tensor(mu_noise, dtype=torch.float32, device=device, requires_grad=True)
            sigma_noise_tensor = torch.tensor(max(sigma_noise, 1e-6), dtype=torch.float32, device=device, requires_grad=True)
            
            # Compute gradients
            result = compute_expected_loss_gradient(
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
                verbose=verbose,
            )
            
            # Store result with checkpoint info
            checkpoint_result = {
                'checkpoint': checkpoint_path,
                'step': step,
                'M': str(M),
                'expected_loss': result['expected_loss'],
                'parameter_values': result['parameter_values'],
                'gradients': result['gradients'],
            }
            
            results.append(checkpoint_result)
            
            print(f"  Expected loss: {result['expected_loss']:.6f}")
            print(f"  Gradients: {result['gradients']}\n")
            
        except Exception as e:
            print(f"  Error processing {checkpoint_path}: {e}\n")
            continue
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"Processed {len(results)} checkpoints successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute gradients for all checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory with checkpoints")
    parser.add_argument("--output", type=str, default="gradient_results.json", help="Output JSON file")
    parser.add_argument("--mu_x", type=float, default=0.0, help="Initial μ_x")
    parser.add_argument("--sigma_x", type=float, default=1.0, help="Initial σ_x")
    parser.add_argument("--mu_theta", type=float, default=0.0, help="Initial μ_θ")
    parser.add_argument("--sigma_theta", type=float, default=1.0, help="Initial σ_θ")
    parser.add_argument("--mu_noise", type=float, default=0.0, help="Initial μ_noise")
    parser.add_argument("--sigma_noise", type=float, default=0.354, help="Initial σ_noise (sqrt(0.125))")
    parser.add_argument("--n_prompts", type=int, default=1000, help="Number of prompts for Monte Carlo")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, defaults to cfg.device)")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    
    args = parser.parse_args()
    
    compute_gradients_for_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        output_file=args.output,
        mu_x=args.mu_x,
        sigma_x=args.sigma_x,
        mu_theta=args.mu_theta,
        sigma_theta=args.sigma_theta,
        mu_noise=args.mu_noise,
        sigma_noise=args.sigma_noise,
        n_prompts=args.n_prompts,
        device=args.device,
        verbose=not args.quiet,
    )

