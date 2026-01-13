#!/usr/bin/env python3
"""
Compare task vectors from training against recovered task vectors.

This script compares task vectors saved during training (in tasks_v{version}/)
against the recovered task vectors (in task_vectors/).

Usage:
    python compare_task_vectors.py --version-suffix v4
    python compare_task_vectors.py --version-suffix v4 --check-all
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def find_task_vector_files(base_dir: str, pattern: str = "tasks_M*.pt") -> Dict[int, Path]:
    """
    Find all task vector files in a directory structure.
    
    Args:
        base_dir: Base directory to search
        pattern: Filename pattern to match
        
    Returns:
        Dictionary mapping M values to file paths
    """
    task_files = {}
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return task_files
    
    # Search for files matching the pattern
    for file_path in base_path.rglob(pattern):
        # Extract M from filename (e.g., "tasks_M64.pt" -> 64)
        filename = file_path.name
        try:
            M = int(filename.replace("tasks_M", "").replace(".pt", ""))
            task_files[M] = file_path
        except ValueError:
            continue
    
    return task_files


def compare_task_vectors(
    training_dir: str,
    recovered_dir: str = "task_vectors",
    version_suffix: str = "v4",
    check_all: bool = False,
    tolerance: float = 1e-6,
) -> None:
    """
    Compare task vectors from training against recovered task vectors.
    
    Args:
        training_dir: Base directory for training task vectors (e.g., "tasks_v4")
        recovered_dir: Directory containing recovered task vectors
        version_suffix: Version suffix used in training
        check_all: If True, check all M values found; if False, only check common M values
        tolerance: Tolerance for floating point comparison
    """
    print("=" * 80)
    print("Task Vector Comparison")
    print("=" * 80)
    print()
    print(f"Training task vectors: {training_dir}")
    print(f"Recovered task vectors: {recovered_dir}")
    print(f"Version suffix: {version_suffix}")
    print(f"Tolerance: {tolerance}")
    print()
    
    # Find task vector files
    training_files = find_task_vector_files(training_dir, "tasks_M*.pt")
    recovered_files = find_task_vector_files(recovered_dir, "tasks_M*.pt")
    
    print(f"Found {len(training_files)} training task vector files")
    print(f"Found {len(recovered_files)} recovered task vector files")
    print()
    
    if not training_files:
        print(f"ERROR: No training task vector files found in {training_dir}")
        return
    
    if not recovered_files:
        print(f"ERROR: No recovered task vector files found in {recovered_dir}")
        return
    
    # Determine which M values to check
    if check_all:
        M_values_to_check = sorted(set(training_files.keys()) | set(recovered_files.keys()))
    else:
        # Only check M values that exist in both, prioritizing training files (completed models)
        M_values_to_check = sorted(set(training_files.keys()) & set(recovered_files.keys()))
    
    if not M_values_to_check:
        if training_files:
            print(f"INFO: Found {len(training_files)} training task vector files, but none match recovered files yet.")
            print(f"      Training M values: {sorted(training_files.keys())}")
            print(f"      Recovered M values: {sorted(recovered_files.keys())}")
        else:
            print("INFO: No training task vector files found yet. Training may still be in progress.")
        return
    
    print(f"Checking {len(M_values_to_check)} M value(s): {M_values_to_check}")
    print()
    
    # Compare task vectors
    results = []
    all_match = True
    
    for M in M_values_to_check:
        training_path = training_files.get(M)
        recovered_path = recovered_files.get(M)
        
        if training_path is None:
            print(f"M={M}: ✗ Missing in training task vectors")
            results.append((M, False, "Missing in training"))
            all_match = False
            continue
        
        if recovered_path is None:
            print(f"M={M}: ✗ Missing in recovered task vectors")
            results.append((M, False, "Missing in recovered"))
            all_match = False
            continue
        
        # Load and compare
        try:
            training_tasks = torch.load(training_path, map_location="cpu")
            recovered_tasks = torch.load(recovered_path, map_location="cpu")
            
            # Check shapes
            if training_tasks.shape != recovered_tasks.shape:
                print(f"M={M}: ✗ Shape mismatch - training: {training_tasks.shape}, recovered: {recovered_tasks.shape}")
                results.append((M, False, f"Shape mismatch: {training_tasks.shape} vs {recovered_tasks.shape}"))
                all_match = False
                continue
            
            # Compare values
            if torch.allclose(training_tasks, recovered_tasks, atol=tolerance, rtol=tolerance):
                max_diff = (training_tasks - recovered_tasks).abs().max().item()
                print(f"M={M}: ✓ Match (max difference: {max_diff:.2e})")
                results.append((M, True, f"Match (max diff: {max_diff:.2e})"))
            else:
                max_diff = (training_tasks - recovered_tasks).abs().max().item()
                mean_diff = (training_tasks - recovered_tasks).abs().mean().item()
                print(f"M={M}: ✗ Mismatch (max difference: {max_diff:.2e}, mean: {mean_diff:.2e})")
                results.append((M, False, f"Mismatch (max diff: {max_diff:.2e})"))
                all_match = False
                
        except Exception as e:
            print(f"M={M}: ✗ Error loading/comparing: {e}")
            results.append((M, False, f"Error: {e}"))
            all_match = False
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    matched = sum(1 for _, match, _ in results if match)
    total = len(results)
    
    print(f"Total checked: {total}")
    print(f"Matched: {matched}")
    print(f"Mismatched/Missing: {total - matched}")
    print()
    
    if all_match:
        print("✓ All task vectors match!")
    else:
        print("✗ Some task vectors do not match or are missing")
        print()
        print("Details:")
        for M, match, msg in results:
            status = "✓" if match else "✗"
            print(f"  {status} M={M}: {msg}")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare task vectors from training against recovered task vectors."
    )
    
    parser.add_argument(
        "--training-dir",
        type=str,
        default=None,
        help="Base directory for training task vectors (default: tasks_v{version_suffix})",
    )
    parser.add_argument(
        "--recovered-dir",
        type=str,
        default="task_vectors",
        help="Directory containing recovered task vectors (default: task_vectors)",
    )
    parser.add_argument(
        "--version-suffix",
        type=str,
        default="v4",
        help="Version suffix used in training (default: v4)",
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Check all M values found in either directory (default: only check common M values)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for floating point comparison (default: 1e-6)",
    )
    
    args = parser.parse_args()
    
    # Determine training directory
    if args.training_dir is None:
        training_dir = f"tasks_{args.version_suffix}"
    else:
        training_dir = args.training_dir
    
    compare_task_vectors(
        training_dir=training_dir,
        recovered_dir=args.recovered_dir,
        version_suffix=args.version_suffix,
        check_all=args.check_all,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
