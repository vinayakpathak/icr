"""
Experiment runner for training models with exponentially increasing M values.
Reuses train_ic_regression() from ic_regression.py.
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

from ic_regression import ICRegConfig, train_ic_regression


def run_m_diversity_experiment(
    max_power: int = 20,
    num_steps: int = 500_000,  # Matching reference: 500k steps
    batch_size: int = 2048,  # Increased from 1024 for better GPU utilization
    checkpoint_every: Optional[int] = None,
    learning_rate: float = 1e-3,
    grad_clip: Optional[float] = 1.0,
    warmup_steps: Optional[int] = 250_000,  # Matching reference: 250k warmup steps
    print_every: int = 1000,
    eval_every: Optional[int] = None,
    skip_first_prediction: bool = False,
    base_checkpoint_dir: str = "checkpoints",
    version_suffix: Optional[str] = "v3",  # Version suffix for directories (e.g., "v3" -> "checkpoints_v3", None -> "checkpoints")
    random_order: bool = True,  # If True, train models in random order; if False, train in increasing order
    random_seed: Optional[int] = None,  # Random seed for shuffling order (None = no seed, order varies each run)
    early_stopping_patience: Optional[int] = None,  # Stop if loss doesn't improve for N evaluations
    early_stopping_min_delta: float = 1e-6,
    save_task_vectors: bool = True,  # Save task vectors used during training
    upload_task_vectors_to_r2: bool = True,  # Upload task vectors to R2 (requires R2 credentials)
    upload_checkpoints_to_r2: bool = True,  # Upload checkpoints to R2 (requires R2 credentials)
    r2_config_path: str = "r2_config.json",  # Path to R2 config file
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
        version_suffix: Version suffix for directories (e.g., "v3" -> "checkpoints_v3", None -> "checkpoints")
        random_order: If True, train models in random order; if False, train in increasing order
        random_seed: Random seed for shuffling order (None = no seed, order varies each run)
        early_stopping_patience: Stop if loss doesn't improve for N evaluations
        early_stopping_min_delta: Minimum change to qualify as improvement
        save_task_vectors: Whether to save task vectors used during training
        upload_task_vectors_to_r2: Whether to upload task vectors to R2 (requires R2 credentials)
        upload_checkpoints_to_r2: Whether to upload checkpoints to R2 (requires R2 credentials)
        r2_config_path: Path to R2 config JSON file
    """
    # Generate M values: 2^1, 2^2, ..., 2^max_power
    M_values_original = [2**i for i in range(1, max_power + 1)]
    
    # Shuffle order if requested
    if random_order:
        if random_seed is not None:
            random.seed(random_seed)
        M_values = M_values_original.copy()
        random.shuffle(M_values)
        print(f"Training models in random order (seed={random_seed})")
        print(f"Original order: {M_values_original}")
        print(f"Randomized order: {M_values}")
    else:
        M_values = M_values_original
        print(f"Training models in increasing order")
    
    print(f"Training models for M values: {M_values}")
    print(f"Total models to train: {len(M_values)}")
    print(f"Training hyperparameters:")
    print(f"  num_steps: {num_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"  checkpoint_every: {checkpoint_every}")
    if early_stopping_patience is not None:
        print(f"  early_stopping_patience: {early_stopping_patience}")
        print(f"  early_stopping_min_delta: {early_stopping_min_delta}")
    print()
    
    # Construct checkpoint directory name with version suffix if provided
    if version_suffix:
        checkpoint_dir_name = f"{base_checkpoint_dir}_{version_suffix}"
    else:
        checkpoint_dir_name = base_checkpoint_dir
    
    # Create base checkpoint directory
    os.makedirs(checkpoint_dir_name, exist_ok=True)
    
    # Store experiment metadata
    experiment_metadata = {
        "M_values_original": M_values_original,
        "M_values_training_order": M_values,
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
        "version_suffix": version_suffix,
        "random_order": random_order,
        "random_seed": random_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "save_task_vectors": save_task_vectors,
        "upload_task_vectors_to_r2": upload_task_vectors_to_r2,
        "upload_checkpoints_to_r2": upload_checkpoints_to_r2,
        "r2_config_path": r2_config_path,
    }
    
    # Train model for each M value
    for i, M in enumerate(M_values):
        print(f"\n{'='*80}")
        print(f"Training model {i+1}/{len(M_values)}: M={M}")
        print(f"{'='*80}\n")
        
        # Create checkpoint directory for this M
        checkpoint_dir = os.path.join(checkpoint_dir_name, f"checkpoints_M{M}")
        
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
                save_task_vectors=save_task_vectors,
                upload_task_vectors_to_r2=upload_task_vectors_to_r2,
                upload_checkpoints_to_r2=upload_checkpoints_to_r2,
                version_suffix=version_suffix,
                r2_config_path=r2_config_path,
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
    parser.add_argument("--num_steps", type=int, default=500_000, help="Number of training steps per model (default: 500k, matching reference)")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size (increased default for better GPU utilization)")
    parser.add_argument("--checkpoint_every", type=int, default=None, help="Save checkpoint every N steps (None = only final)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (None = no clipping)")
    parser.add_argument("--warmup_steps", type=int, default=250_000, help="Warmup steps for triangle LR schedule (default: 250k, matching reference; None = constant LR)")
    parser.add_argument("--print_every", type=int, default=1000, help="Print loss every N steps")
    parser.add_argument("--eval_every", type=int, default=None, help="Evaluate OOD every N steps (None = no evaluation)")
    parser.add_argument("--skip_first_prediction", action="store_true", help="Skip first prediction in loss computation")
    parser.add_argument("--base_checkpoint_dir", type=str, default="checkpoints", help="Base directory for checkpoints")
    parser.add_argument("--version_suffix", type=str, default="v3", help="Version suffix for directories (e.g., 'v3' -> 'checkpoints_v3', use '' for no suffix)")
    parser.add_argument("--random_order", action="store_true", default=True, help="Train models in random order (default: True)")
    parser.add_argument("--no_random_order", dest="random_order", action="store_false", help="Train models in increasing order")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for shuffling order (None = no seed, order varies each run)")
    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Stop if loss doesn't improve for N evaluations (None = no early stopping)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-6, help="Minimum change to qualify as improvement")
    parser.add_argument("--save_task_vectors", action="store_true", default=True, help="Save task vectors used during training (default: True)")
    parser.add_argument("--no_save_task_vectors", dest="save_task_vectors", action="store_false", help="Don't save task vectors")
    parser.add_argument("--upload_task_vectors_to_r2", action="store_true", default=True, help="Upload task vectors to R2 (default: True, requires R2 credentials)")
    parser.add_argument("--no_upload_task_vectors_to_r2", dest="upload_task_vectors_to_r2", action="store_false", help="Don't upload task vectors to R2")
    parser.add_argument("--upload_checkpoints_to_r2", action="store_true", default=True, help="Upload checkpoints to R2 (default: True, requires R2 credentials)")
    parser.add_argument("--no_upload_checkpoints_to_r2", dest="upload_checkpoints_to_r2", action="store_false", help="Don't upload checkpoints to R2")
    parser.add_argument("--r2_config_path", type=str, default="r2_config.json", help="Path to R2 config JSON file")
    
    args = parser.parse_args()
    
    # Handle empty string as None for version_suffix
    version_suffix = args.version_suffix if args.version_suffix else None
    
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
        version_suffix=version_suffix,
        random_order=args.random_order,
        random_seed=args.random_seed,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        save_task_vectors=args.save_task_vectors,
        upload_task_vectors_to_r2=args.upload_task_vectors_to_r2,
        upload_checkpoints_to_r2=args.upload_checkpoints_to_r2,
        r2_config_path=args.r2_config_path,
    )

