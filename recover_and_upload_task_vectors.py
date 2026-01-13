#!/usr/bin/env python3
"""
Recover task vectors that were used during training and upload them to R2.

Since task vectors are generated deterministically from a fixed seed (seed=0),
we can regenerate the exact same task vectors that were used during training.

This script:
1. Recovers task vectors using the same seed and config as training
2. For each M value, extracts the first M task vectors
3. Saves them locally
4. Uploads them to R2 in the tasks_v3 folder (or specified version)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict

import torch

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config
except ImportError:
    print("ERROR: boto3 is not installed. Please install it with:")
    print("  pip install boto3")
    sys.exit(1)

from ic_regression import ICRegConfig, recover_training_tasks, load_checkpoint


def load_r2_config(config_path: str = "r2_config.json") -> Dict:
    """
    Load R2 credentials from a JSON config file.
    
    Args:
        config_path: Path to the JSON config file
        
    Returns:
        Dictionary with R2 credentials
    """
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse config file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read config file {config_path}: {e}")
        sys.exit(1)


def get_r2_client(
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    endpoint_url: Optional[str] = None,
) -> boto3.client:
    """
    Create and return an R2 S3-compatible client.
    
    Args:
        account_id: Cloudflare account ID
        access_key_id: R2 access key ID
        secret_access_key: R2 secret access key
        endpoint_url: Optional custom endpoint URL. If None, uses default R2 endpoint.
        
    Returns:
        Configured boto3 S3 client for R2
    """
    if endpoint_url is None:
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    
    # Configure for R2
    config = Config(
        signature_version='s3v4',
        s3={
            'addressing_style': 'path'
        }
    )
    
    client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=config,
    )
    
    return client


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def upload_file(
    client: boto3.client,
    bucket_name: str,
    local_file_path: Path,
    s3_key: str,
    skip_existing: bool = False,
    verbose: bool = True,
) -> tuple[bool, int]:
    """
    Upload a single file to R2.
    
    Args:
        client: Boto3 S3 client
        bucket_name: R2 bucket name
        local_file_path: Local file path
        s3_key: S3 key (path in bucket)
        skip_existing: If True, skip files that already exist in R2
        verbose: If True, print progress
        
    Returns:
        Tuple of (success: bool, file_size: int)
    """
    file_size = local_file_path.stat().st_size
    
    # Check if file already exists
    if skip_existing:
        try:
            client.head_object(Bucket=bucket_name, Key=s3_key)
            if verbose:
                print(f"  ⊘ Skipped (already exists): {s3_key}")
            return True, file_size
        except ClientError as e:
            # File doesn't exist, proceed with upload
            if e.response['Error']['Code'] != '404':
                raise
    
    try:
        if verbose:
            print(f"  ↑ Uploading: {s3_key} ({format_size(file_size)})")
        
        # Upload file
        client.upload_file(
            str(local_file_path),
            bucket_name,
            s3_key,
        )
        
        if verbose:
            print(f"  ✓ Uploaded: {s3_key}")
        
        return True, file_size
        
    except ClientError as e:
        print(f"  ✗ Error uploading {s3_key}: {e}")
        return False, file_size
    except Exception as e:
        print(f"  ✗ Unexpected error uploading {s3_key}: {e}")
        return False, file_size


def recover_and_save_task_vectors(
    max_power: int = 20,
    checkpoint_path: Optional[str] = None,
    max_M: Optional[int] = None,
    D: Optional[int] = None,
    seed: int = 0,
    output_dir: str = "task_vectors",
) -> Dict[int, Path]:
    """
    Recover task vectors for each M value and save them locally.
    
    Args:
        max_power: Maximum power of 2 (M will range from 2^1 to 2^max_power)
        checkpoint_path: Path to a checkpoint file to load config from (optional)
        max_M: Maximum number of tasks to generate (overrides checkpoint if provided)
        D: Task dimension (overrides checkpoint if provided)
        seed: Random seed used during training (default: 0)
        output_dir: Directory to save task vector files
        
    Returns:
        Dictionary mapping M values to file paths
    """
    # Load config from checkpoint if provided
    if checkpoint_path is not None:
        try:
            _, _, _, _, cfg, _ = load_checkpoint(checkpoint_path, device="cpu")
            if max_M is None:
                max_M = cfg.max_M
            if D is None:
                D = cfg.D
            print(f"Loaded config from checkpoint: D={D}, max_M={max_M}")
        except Exception as e:
            print(f"WARNING: Failed to load checkpoint {checkpoint_path}: {e}")
            print("Using provided/default values instead")
    else:
        # Use defaults if not provided
        if max_M is None:
            max_M = 33554432  # Default from ICRegConfig
        if D is None:
            D = 8  # Default from ICRegConfig
        print(f"Using default config: D={D}, max_M={max_M}")
    
    # Generate M values: 2^1, 2^2, ..., 2^max_power
    M_values = [2**i for i in range(1, max_power + 1)]
    
    # Ensure max_M is at least as large as the largest M value
    max_M_needed = max(M_values)
    if max_M < max_M_needed:
        print(f"WARNING: max_M={max_M} is less than largest M={max_M_needed}")
        print(f"  Increasing max_M to {max_M_needed}")
        max_M = max_M_needed
    
    print(f"\nRecovering task vectors with seed={seed}...")
    print(f"  max_M: {max_M}")
    print(f"  D: {D}")
    print(f"  M values: {M_values}")
    print()
    
    # Recover all task vectors
    print("Generating task sequence...")
    all_tasks = recover_training_tasks(max_M=max_M, D=D, seed=seed)
    print(f"✓ Generated {all_tasks.shape[0]} task vectors of dimension {all_tasks.shape[1]}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save task vectors for each M value
    saved_files = {}
    print("Saving task vectors for each M value...")
    for M in M_values:
        # Extract first M task vectors
        tasks_M = all_tasks[:M]  # Shape: (M, D)
        
        # Save to file in structure matching training: output_dir/checkpoints_M{M}/tasks_M{M}.pt
        checkpoint_dir = os.path.join(output_dir, f"checkpoints_M{M}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        filename = f"tasks_M{M}.pt"
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(tasks_M, filepath)
        saved_files[M] = Path(filepath)
        
        file_size = os.path.getsize(filepath)
        print(f"  ✓ Saved M={M}: {filepath} ({format_size(file_size)})")
    
    print()
    return saved_files


def upload_task_vectors(
    task_vector_files: Dict[int, Path],
    bucket_name: str,
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    endpoint_url: Optional[str] = None,
    base_s3_prefix: str = "tasks_v2",
    skip_existing: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """
    Upload task vector files to R2.
    
    Args:
        task_vector_files: Dictionary mapping M values to file paths
        bucket_name: R2 bucket name
        account_id: Cloudflare account ID
        access_key_id: R2 access key ID
        secret_access_key: R2 secret access key
        endpoint_url: Optional custom endpoint URL
        base_s3_prefix: Base prefix for S3 keys (default: "tasks_v3")
        skip_existing: If True, skip files that already exist in R2
        dry_run: If True, show what would be uploaded without uploading
        verbose: If True, print detailed progress
    """
    print("=" * 80)
    print("Cloudflare R2 Task Vector Upload")
    print("=" * 80)
    print()
    print(f"Bucket: {bucket_name}")
    print(f"Account ID: {account_id}")
    print(f"Endpoint: {endpoint_url or f'https://{account_id}.r2.cloudflarestorage.com'}")
    print(f"Base S3 prefix: {base_s3_prefix}")
    print(f"Skip existing: {skip_existing}")
    print(f"Mode: {'DRY RUN' if dry_run else 'UPLOAD'}")
    print()
    
    # Create R2 client
    if not dry_run:
        try:
            client = get_r2_client(
                account_id=account_id,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                endpoint_url=endpoint_url,
            )
            
            # Test connection by checking if bucket exists
            try:
                client.head_bucket(Bucket=bucket_name)
                if verbose:
                    print(f"✓ Connected to bucket '{bucket_name}'")
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == '404':
                    print(f"ERROR: Bucket '{bucket_name}' not found")
                    sys.exit(1)
                elif error_code == '403':
                    print(f"ERROR: Access denied to bucket '{bucket_name}'. Check your credentials.")
                    sys.exit(1)
                else:
                    raise
            print()
        except Exception as e:
            print(f"ERROR: Failed to connect to R2: {e}")
            sys.exit(1)
    else:
        client = None
    
    # Track statistics
    total_files = len(task_vector_files)
    total_uploaded = 0
    total_skipped = 0
    total_failed = 0
    total_size = 0
    total_uploaded_size = 0
    
    # Upload each file
    for idx, (M, filepath) in enumerate(sorted(task_vector_files.items()), 1):
        print(f"[{idx}/{total_files}] Processing M={M}")
        
        if not filepath.exists():
            print(f"  ✗ File not found: {filepath}")
            total_failed += 1
            continue
        
        # Construct S3 key: tasks_v3/checkpoints_M{M}/tasks_M{M}.pt
        # Match the structure used in training
        filename = filepath.name
        # Extract M from filename (e.g., "tasks_M64.pt" -> "checkpoints_M64")
        checkpoint_dir_name = filename.replace("tasks_", "checkpoints_").replace(".pt", "")
        s3_key = f"{base_s3_prefix}/{checkpoint_dir_name}/{filename}"
        
        file_size = filepath.stat().st_size
        total_size += file_size
        
        if dry_run:
            print(f"  [DRY RUN] Would upload: {s3_key} ({format_size(file_size)})")
            total_uploaded += 1
            total_uploaded_size += file_size
        else:
            success, uploaded_size = upload_file(
                client=client,
                bucket_name=bucket_name,
                local_file_path=filepath,
                s3_key=s3_key,
                skip_existing=skip_existing,
                verbose=verbose,
            )
            
            if success:
                if skip_existing and uploaded_size == 0:
                    total_skipped += 1
                else:
                    total_uploaded += 1
                    total_uploaded_size += uploaded_size
            else:
                total_failed += 1
        
        print()
    
    # Print final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    
    if dry_run:
        print(f"✓ Dry run completed")
        print(f"  Files that would be uploaded: {total_uploaded}")
        print(f"  Total size: {format_size(total_size)}")
    else:
        print(f"✓ Upload completed")
        print()
        print("Statistics:")
        print(f"  Total files processed: {total_files}")
        if total_uploaded > 0:
            print(f"  ✓ Files uploaded: {total_uploaded} ({format_size(total_uploaded_size)})")
        if total_skipped > 0:
            print(f"  ⊘ Files skipped (already exist): {total_skipped}")
        if total_failed > 0:
            print(f"  ✗ Files failed: {total_failed}")
        print(f"  Total size: {format_size(total_size)}")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Recover task vectors used during training and upload them to R2."
    )
    
    # Recovery options
    parser.add_argument(
        "--max-power",
        type=int,
        default=20,
        help="Maximum power of 2 (M ranges from 2^1 to 2^max_power, default: 20)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to load config from (optional)",
    )
    parser.add_argument(
        "--max-M",
        type=int,
        default=None,
        help="Maximum number of tasks to generate (overrides checkpoint if provided)",
    )
    parser.add_argument(
        "--D",
        type=int,
        default=None,
        help="Task dimension (overrides checkpoint if provided)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used during training (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tasks_v3",
        help="Directory to save task vector files (default: tasks_v3, will create checkpoints_M{M} subdirectories)",
    )
    
    # R2 credentials (can be provided via env vars or command line)
    parser.add_argument(
        "--account-id",
        type=str,
        default=None,
        help="Cloudflare account ID (or set in r2_config.json or R2_ACCOUNT_ID env var)",
    )
    parser.add_argument(
        "--access-key-id",
        type=str,
        default=None,
        help="R2 access key ID (or set in r2_config.json or R2_ACCESS_KEY_ID env var)",
    )
    parser.add_argument(
        "--secret-access-key",
        type=str,
        default=None,
        help="R2 secret access key (or set in r2_config.json or R2_SECRET_ACCESS_KEY env var)",
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default=None,
        help="R2 bucket name (or set in r2_config.json or R2_BUCKET_NAME env var, default: elicitation)",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="Custom R2 endpoint URL (optional, or set in r2_config.json or R2_ENDPOINT_URL env var)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="r2_config.json",
        help="Path to R2 config JSON file (default: r2_config.json)",
    )
    
    # Upload options
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="tasks_v3",
        help="Base prefix for S3 keys (default: tasks_v3)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in R2 (checks by key name)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only recover and save task vectors locally, skip R2 upload",
    )
    
    args = parser.parse_args()
    
    # Load config from file if it exists
    config = load_r2_config(args.config)
    
    # Use config file values as defaults, but command line args and env vars take precedence
    account_id = args.account_id or config.get("account_id") or os.getenv("R2_ACCOUNT_ID")
    access_key_id = args.access_key_id or config.get("access_key_id") or os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = args.secret_access_key or config.get("secret_access_key") or os.getenv("R2_SECRET_ACCESS_KEY")
    bucket_name = args.bucket_name or config.get("bucket_name") or os.getenv("R2_BUCKET_NAME", "elicitation")
    endpoint_url = args.endpoint_url or config.get("endpoint_url") or os.getenv("R2_ENDPOINT_URL")
    
    # Recover and save task vectors
    print("=" * 80)
    print("Recovering Task Vectors")
    print("=" * 80)
    print()
    
    task_vector_files = recover_and_save_task_vectors(
        max_power=args.max_power,
        checkpoint_path=args.checkpoint,
        max_M=args.max_M,
        D=args.D,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    # Upload to R2 if not skipped
    if not args.skip_upload:
        # Validate required credentials
        if not account_id:
            print("ERROR: Account ID is required. Set in r2_config.json, R2_ACCOUNT_ID env var, or use --account-id")
            sys.exit(1)
        
        if not access_key_id:
            print("ERROR: Access Key ID is required. Set in r2_config.json, R2_ACCESS_KEY_ID env var, or use --access-key-id")
            sys.exit(1)
        
        if not secret_access_key:
            print("ERROR: Secret Access Key is required. Set in r2_config.json, R2_SECRET_ACCESS_KEY env var, or use --secret-access-key")
            sys.exit(1)
        
        upload_task_vectors(
            task_vector_files=task_vector_files,
            bucket_name=bucket_name,
            account_id=account_id,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            endpoint_url=endpoint_url,
            base_s3_prefix=args.s3_prefix,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
    else:
        print("Skipping R2 upload (--skip-upload specified)")
        print()


if __name__ == "__main__":
    main()
