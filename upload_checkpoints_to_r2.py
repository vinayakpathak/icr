#!/usr/bin/env python3
"""
Upload all checkpoints to Cloudflare R2 object storage.
Uses boto3 (S3-compatible API) to upload files.

Required credentials (set as environment variables or pass via command line):
- R2_ACCOUNT_ID: Your Cloudflare account ID
- R2_ACCESS_KEY_ID: Your R2 access key ID
- R2_SECRET_ACCESS_KEY: Your R2 secret access key
- R2_BUCKET_NAME: The bucket name (default: elicitation)
- R2_ENDPOINT_URL: Optional, defaults to https://<account-id>.r2.cloudflarestorage.com
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    from botocore.config import Config
except ImportError:
    print("ERROR: boto3 is not installed. Please install it with:")
    print("  pip install boto3")
    sys.exit(1)


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
        
        # Upload file with progress callback
        client.upload_file(
            str(local_file_path),
            bucket_name,
            s3_key,
            Callback=ProgressPercentage(local_file_path, verbose) if verbose else None,
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


class ProgressPercentage:
    """Progress callback for file uploads."""
    
    def __init__(self, filename: Path, verbose: bool = True):
        self._filename = filename
        self._size = filename.stat().st_size
        self._seen_so_far = 0
        self._verbose = verbose
        
    def __call__(self, bytes_amount: int):
        self._seen_so_far += bytes_amount
        if self._verbose and self._size > 0:
            percentage = (self._seen_so_far / self._size) * 100
            print(f"\r    Progress: {percentage:.1f}% ({format_size(self._seen_so_far)}/{format_size(self._size)})", end='', flush=True)


def find_checkpoint_files(base_dir: Path) -> list[Path]:
    """
    Find all checkpoint files (.pt) in the directory tree.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of checkpoint file paths
    """
    checkpoint_files = []
    for ext in ['*.pt', '*.pth']:
        checkpoint_files.extend(base_dir.rglob(ext))
    return sorted(checkpoint_files)


def upload_checkpoints(
    checkpoint_dirs: list[str],
    bucket_name: str,
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    endpoint_url: Optional[str] = None,
    base_s3_prefix: str = "",
    skip_existing: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """
    Upload all checkpoints from specified directories to R2.
    
    Args:
        checkpoint_dirs: List of local checkpoint directory paths
        bucket_name: R2 bucket name
        account_id: Cloudflare account ID
        access_key_id: R2 access key ID
        secret_access_key: R2 secret access key
        endpoint_url: Optional custom endpoint URL
        base_s3_prefix: Base prefix for S3 keys (e.g., "checkpoints/")
        skip_existing: If True, skip files that already exist in R2
        dry_run: If True, show what would be uploaded without uploading
        verbose: If True, print detailed progress
    """
    print("=" * 80)
    print("Cloudflare R2 Checkpoint Upload")
    print("=" * 80)
    print()
    print(f"Bucket: {bucket_name}")
    print(f"Account ID: {account_id}")
    print(f"Endpoint: {endpoint_url or f'https://{account_id}.r2.cloudflarestorage.com'}")
    print(f"Base S3 prefix: {base_s3_prefix or '(root)'}")
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
    total_files = 0
    total_uploaded = 0
    total_skipped = 0
    total_failed = 0
    total_size = 0
    total_uploaded_size = 0
    
    # Process each checkpoint directory
    for dir_idx, checkpoint_dir in enumerate(checkpoint_dirs, 1):
        checkpoint_path = Path(checkpoint_dir)
        
        if not checkpoint_path.exists():
            print(f"WARNING: Directory not found: {checkpoint_dir}")
            continue
        
        if not checkpoint_path.is_dir():
            print(f"WARNING: Not a directory: {checkpoint_dir}")
            continue
        
        print("=" * 80)
        print(f"Processing directory {dir_idx}/{len(checkpoint_dirs)}: {checkpoint_dir}")
        print("=" * 80)
        print()
        
        # Find all checkpoint files
        checkpoint_files = find_checkpoint_files(checkpoint_path)
        
        if not checkpoint_files:
            print(f"  No checkpoint files found in {checkpoint_dir}")
            print()
            continue
        
        print(f"  Found {len(checkpoint_files)} checkpoint file(s)")
        print()
        
        # Upload each file
        for file_idx, local_file in enumerate(checkpoint_files, 1):
            # Calculate relative path from checkpoint directory
            try:
                relative_path = local_file.relative_to(checkpoint_path)
            except ValueError:
                # If file is not relative to checkpoint_path, use its name
                relative_path = local_file.name
            
            # Get the parent directory name (checkpoints or checkpoints_v2) to include in S3 key
            # This ensures files from different checkpoint directories don't overwrite each other
            parent_dir_name = checkpoint_path.name
            
            # Construct S3 key with parent directory name
            if base_s3_prefix:
                s3_key = f"{base_s3_prefix.rstrip('/')}/{parent_dir_name}/{relative_path.as_posix()}"
            else:
                s3_key = f"{parent_dir_name}/{relative_path.as_posix()}"
            
            total_files += 1
            file_size = local_file.stat().st_size
            total_size += file_size
            
            if dry_run:
                print(f"  [DRY RUN] Would upload: {s3_key} ({format_size(file_size)})")
                total_uploaded += 1
                total_uploaded_size += file_size
            else:
                success, uploaded_size = upload_file(
                    client=client,
                    bucket_name=bucket_name,
                    local_file_path=local_file,
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
        print(f"✓ Completed directory: {checkpoint_dir}")
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
        description="Upload all checkpoints to Cloudflare R2 object storage. "
                    "By default, uploads both checkpoints/ and checkpoints_v2/ directories."
    )
    
    # Credentials (can be provided via env vars or command line)
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
        "--checkpoint-dir",
        type=str,
        action="append",
        help="Checkpoint directory to upload (can be specified multiple times). "
             "If not specified, defaults to both checkpoints/ and checkpoints_v2/",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="",
        help="Base prefix for S3 keys (e.g., 'checkpoints/' to organize files in bucket)",
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
    
    args = parser.parse_args()
    
    # Load config from file if it exists
    config = load_r2_config(args.config)
    
    # Use config file values as defaults, but command line args and env vars take precedence
    account_id = args.account_id or config.get("account_id") or os.getenv("R2_ACCOUNT_ID")
    access_key_id = args.access_key_id or config.get("access_key_id") or os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = args.secret_access_key or config.get("secret_access_key") or os.getenv("R2_SECRET_ACCESS_KEY")
    bucket_name = args.bucket_name or config.get("bucket_name") or os.getenv("R2_BUCKET_NAME", "elicitation")
    endpoint_url = args.endpoint_url or config.get("endpoint_url") or os.getenv("R2_ENDPOINT_URL")
    
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
    
    # Default checkpoint directories
    if args.checkpoint_dir is None:
        checkpoint_dirs = ["checkpoints", "checkpoints_v2"]
    else:
        checkpoint_dirs = args.checkpoint_dir
    
    # Convert to absolute paths
    checkpoint_dirs = [os.path.abspath(d) for d in checkpoint_dirs]
    
    upload_checkpoints(
        checkpoint_dirs=checkpoint_dirs,
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


if __name__ == "__main__":
    main()
