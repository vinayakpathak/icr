#!/usr/bin/env python3
"""
Compare task vectors from different R2 folders (e.g., tasks_v3 vs tasks_v4).

This script downloads task vectors from R2 and compares them to verify they match.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config
except ImportError:
    print("ERROR: boto3 is not installed. Please install it with:")
    print("  pip install boto3")
    sys.exit(1)


def load_r2_config(config_path: str = "r2_config.json") -> dict:
    """Load R2 credentials from a JSON config file."""
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
    """Create and return an R2 S3-compatible client."""
    if endpoint_url is None:
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    
    config = Config(
        signature_version='s3v4',
        s3={'addressing_style': 'path'}
    )
    
    client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=config,
    )
    
    return client


def list_r2_objects(client: boto3.client, bucket_name: str, prefix: str) -> List[str]:
    """List all objects in R2 with the given prefix."""
    objects = []
    paginator = client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                objects.append(obj['Key'])
    
    return sorted(objects)


def download_file(client: boto3.client, bucket_name: str, s3_key: str, local_path: str) -> bool:
    """Download a file from R2."""
    try:
        client.download_file(bucket_name, s3_key, local_path)
        return True
    except ClientError as e:
        print(f"  ✗ Error downloading {s3_key}: {e}")
        return False


def extract_m_from_key(s3_key: str) -> Optional[int]:
    """Extract M value from S3 key (e.g., 'tasks_v3/checkpoints_M64/tasks_M64.pt' -> 64)."""
    try:
        # Extract from path like: tasks_v3/checkpoints_M64/tasks_M64.pt
        parts = s3_key.split('/')
        for part in parts:
            if part.startswith('checkpoints_M'):
                M = int(part.replace('checkpoints_M', ''))
                return M
            elif part.startswith('tasks_M') and part.endswith('.pt'):
                M = int(part.replace('tasks_M', '').replace('.pt', ''))
                return M
    except (ValueError, AttributeError):
        pass
    return None


def compare_r2_task_vectors(
    folder1: str,
    folder2: str,
    bucket_name: str,
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    endpoint_url: Optional[str] = None,
    r2_config_path: str = "r2_config.json",
    tolerance: float = 1e-6,
) -> None:
    """
    Compare task vectors from two R2 folders.
    
    Args:
        folder1: First R2 folder prefix (e.g., "tasks_v3")
        folder2: Second R2 folder prefix (e.g., "tasks_v4")
        bucket_name: R2 bucket name
        account_id: Cloudflare account ID
        access_key_id: R2 access key ID
        secret_access_key: R2 secret access key
        endpoint_url: Optional custom endpoint URL
        r2_config_path: Path to R2 config JSON file
        tolerance: Tolerance for floating point comparison
    """
    print("=" * 80)
    print("R2 Task Vector Comparison")
    print("=" * 80)
    print()
    print(f"Folder 1: {folder1}")
    print(f"Folder 2: {folder2}")
    print(f"Bucket: {bucket_name}")
    print(f"Tolerance: {tolerance}")
    print()
    
    # Create R2 client
    client = get_r2_client(
        account_id=account_id,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        endpoint_url=endpoint_url,
    )
    
    # Test connection
    try:
        client.head_bucket(Bucket=bucket_name)
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
    
    # List objects in both folders
    print("Listing objects in R2...")
    objects1 = list_r2_objects(client, bucket_name, folder1 + "/")
    objects2 = list_r2_objects(client, bucket_name, folder2 + "/")
    
    print(f"Found {len(objects1)} objects in {folder1}/")
    print(f"Found {len(objects2)} objects in {folder2}/")
    print()
    
    # Extract M values and create mapping
    def create_mapping(objects: List[str], folder: str) -> Dict[int, str]:
        mapping = {}
        for obj_key in objects:
            M = extract_m_from_key(obj_key)
            if M is not None:
                mapping[M] = obj_key
        return mapping
    
    mapping1 = create_mapping(objects1, folder1)
    mapping2 = create_mapping(objects2, folder2)
    
    print(f"Extracted {len(mapping1)} M values from {folder1}")
    print(f"Extracted {len(mapping2)} M values from {folder2}")
    print()
    
    # Find common M values
    common_M = sorted(set(mapping1.keys()) & set(mapping2.keys()))
    
    if not common_M:
        print("ERROR: No common M values found between the two folders")
        print(f"  {folder1} M values: {sorted(mapping1.keys())}")
        print(f"  {folder2} M values: {sorted(mapping2.keys())}")
        return
    
    print(f"Comparing {len(common_M)} common M values: {common_M}")
    print()
    
    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        results = []
        all_match = True
        
        for M in common_M:
            key1 = mapping1[M]
            key2 = mapping2[M]
            
            # Download files
            file1_path = temp_path / f"folder1_M{M}.pt"
            file2_path = temp_path / f"folder2_M{M}.pt"
            
            print(f"M={M}:")
            print(f"  Downloading {key1}...")
            if not download_file(client, bucket_name, key1, str(file1_path)):
                results.append((M, False, "Failed to download from folder1"))
                all_match = False
                continue
            
            print(f"  Downloading {key2}...")
            if not download_file(client, bucket_name, key2, str(file2_path)):
                results.append((M, False, "Failed to download from folder2"))
                all_match = False
                continue
            
            # Load and compare
            try:
                tasks1 = torch.load(str(file1_path), map_location="cpu", weights_only=False)
                tasks2 = torch.load(str(file2_path), map_location="cpu", weights_only=False)
                
                # Check shapes
                if tasks1.shape != tasks2.shape:
                    print(f"  ✗ Shape mismatch: {tasks1.shape} vs {tasks2.shape}")
                    results.append((M, False, f"Shape mismatch: {tasks1.shape} vs {tasks2.shape}"))
                    all_match = False
                    continue
                
                # Compare values
                if torch.allclose(tasks1, tasks2, atol=tolerance, rtol=tolerance):
                    max_diff = (tasks1 - tasks2).abs().max().item()
                    print(f"  ✓ Match (max difference: {max_diff:.2e})")
                    results.append((M, True, f"Match (max diff: {max_diff:.2e})"))
                else:
                    max_diff = (tasks1 - tasks2).abs().max().item()
                    mean_diff = (tasks1 - tasks2).abs().mean().item()
                    print(f"  ✗ Mismatch (max difference: {max_diff:.2e}, mean: {mean_diff:.2e})")
                    results.append((M, False, f"Mismatch (max diff: {max_diff:.2e})"))
                    all_match = False
                    
            except Exception as e:
                print(f"  ✗ Error loading/comparing: {e}")
                results.append((M, False, f"Error: {e}"))
                all_match = False
            
            print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    matched = sum(1 for _, match, _ in results if match)
    total = len(results)
    
    print(f"Total compared: {total}")
    print(f"Matched: {matched}")
    print(f"Mismatched/Errors: {total - matched}")
    print()
    
    if all_match:
        print("✓ All task vectors match between the two folders!")
    else:
        print("✗ Some task vectors do not match")
        print()
        print("Details:")
        for M, match, msg in results:
            status = "✓" if match else "✗"
            print(f"  {status} M={M}: {msg}")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare task vectors from two R2 folders."
    )
    
    parser.add_argument(
        "--folder1",
        type=str,
        required=True,
        help="First R2 folder prefix (e.g., 'tasks_v3')",
    )
    parser.add_argument(
        "--folder2",
        type=str,
        required=True,
        help="Second R2 folder prefix (e.g., 'tasks_v4')",
    )
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
        help="Custom R2 endpoint URL (optional)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="r2_config.json",
        help="Path to R2 config JSON file (default: r2_config.json)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for floating point comparison (default: 1e-6)",
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
    
    compare_r2_task_vectors(
        folder1=args.folder1,
        folder2=args.folder2,
        bucket_name=bucket_name,
        account_id=account_id,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        endpoint_url=endpoint_url,
        r2_config_path=args.config,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
