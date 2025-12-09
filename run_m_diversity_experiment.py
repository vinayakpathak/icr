"""
Experiment runner for training models with exponentially increasing M values.
Reuses train_ic_regression() from ic_regression.py.
"""

import json
import os
from pathlib import Path
from typing import Optional

from ic_regression import ICRegConfig, train_ic_regression


def run_m_diversity_experiment(
    max_power: int = 20,
    num_steps: int = 150_000,
    batch_size: int = 2048,  # Increased from 1024 for better GPU utilization
    checkpoint_every: Optional[int] = None,
    learning_rate: float = 1e-3,
    grad_clip: Optional[float] = 1.0,
    warmup_steps: Optional[int] = None,
    print_every: int = 1000,
    eval_every: Optional[int] = None,
    skip_first_prediction: bool = False,
    base_checkpoint_dir: str = "checkpoints",
    early_stopping_patience: Optional[int] = None,  # Stop if loss doesn't improve for N evaluations
    early_stopping_min_delta: float = 1e-6,
) -> None:
    """
    Train models with exponentially increasing task diversity M.
    
    Args:
        max_power: Maximum power of 2 (M will range from 2^1 to 2^max_power)
        num_steps: Number of training steps per model
        batch_size: Batch size for training
        checkpoint_every: Save checkpoint every N steps (None = only final)
        learning_rate: Learning rate
        grad_clip: Gradient clipping value (None = no clipping)
        warmup_steps: Number of warmup steps for triangle LR schedule (None = constant LR)
        print_every: Print loss every N steps
        eval_every: Evaluate OOD every N steps (None = no evaluation during training)
        skip_first_prediction: Whether to skip first prediction in loss computation
        base_checkpoint_dir: Base directory for checkpoints (will create subdirectories)
        early_stopping_patience: Stop if loss doesn't improve for N evaluations
        early_stopping_min_delta: Minimum change to qualify as improvement
    """
    # Generate M values: 2^1, 2^2, ..., 2^max_power
    M_values = [2**i for i in range(1, max_power + 1)]
    
    print(f"Training models for M values: {M_values}")
    print(f"Total models to train: {len(M_values)}")
    print(f"Training hyperparameters:")
    print(f"  num_steps: {num_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"  checkpoint_every: {checkpoint_every}")
    print()
    
    # Create base checkpoint directory
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    
    # Store experiment metadata
    experiment_metadata = {
        "M_values": M_values,
        "max_power": max_power,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "checkpoint_every": checkpoint_every,
        "learning_rate": learning_rate,
        "grad_clip": grad_clip,
        "warmup_steps": warmup_steps,
        "print_every": print_every,
        "eval_every": eval_every,
        "skip_first_prediction": skip_first_prediction,
        "base_checkpoint_dir": base_checkpoint_dir,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
    }
    
    # Train model for each M value
    for i, M in enumerate(M_values):
        print(f"\n{'='*80}")
        print(f"Training model {i+1}/{len(M_values)}: M={M}")
        print(f"{'='*80}\n")
        
        # Create checkpoint directory for this M
        checkpoint_dir = os.path.join(base_checkpoint_dir, f"checkpoints_M{M}")
        
        try:
            # Create config
            cfg = ICRegConfig()
            
            # Train model (M is passed as parameter, not in config)
            train_ic_regression(
                cfg=cfg,
                M=M,
                num_steps=num_steps,
                batch_size=batch_size,
                print_every=print_every,
                eval_every=eval_every,
                learning_rate=learning_rate,
                grad_clip=grad_clip,
                warmup_steps=warmup_steps,
                skip_first_prediction=skip_first_prediction,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
            )
            
            print(f"\nCompleted training for M={M}")
            
        except Exception as e:
            print(f"\nERROR: Training failed for M={M}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nContinuing with next M value...")
            continue
    
    # Save experiment metadata
    metadata_file = "experiment_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    print(f"\nExperiment metadata saved to {metadata_file}")
    print(f"All training completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models with exponentially increasing M values (2^1 to 2^max_power)")
    parser.add_argument("--max_power", type=int, default=20, help="Maximum power of 2 (M ranges from 2^1 to 2^max_power)")
    parser.add_argument("--num_steps", type=int, default=150_000, help="Number of training steps per model")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size (increased default for better GPU utilization)")
    parser.add_argument("--checkpoint_every", type=int, default=None, help="Save checkpoint every N steps (None = only final)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (None = no clipping)")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps for triangle LR schedule (None = constant LR)")
    parser.add_argument("--print_every", type=int, default=1000, help="Print loss every N steps")
    parser.add_argument("--eval_every", type=int, default=None, help="Evaluate OOD every N steps (None = no evaluation)")
    parser.add_argument("--skip_first_prediction", action="store_true", help="Skip first prediction in loss computation")
    parser.add_argument("--base_checkpoint_dir", type=str, default="checkpoints", help="Base directory for checkpoints")
    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Stop if loss doesn't improve for N evaluations (None = no early stopping)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-6, help="Minimum change to qualify as improvement")
    
    args = parser.parse_args()
    
    run_m_diversity_experiment(
        max_power=args.max_power,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip if args.grad_clip is not None else None,
        warmup_steps=args.warmup_steps,
        print_every=args.print_every,
        eval_every=args.eval_every,
        skip_first_prediction=args.skip_first_prediction,
        base_checkpoint_dir=args.base_checkpoint_dir,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

