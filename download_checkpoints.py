#!/usr/bin/env python3
"""
Download all checkpoints from remote server to local machine.
Uses rsync via subprocess - zero Python dependencies required.
Parses SSH config to get connection details automatically.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def parse_ssh_config(host_alias: str = "vast-gpu") -> Dict[str, Optional[str]]:
    """
    Parse ~/.ssh/config and extract connection details for host.
    
    Args:
        host_alias: SSH host alias to look up
        
    Returns:
        Dictionary with keys: HostName, Port, User, IdentityFile
    """
    ssh_config_path = Path.home() / ".ssh" / "config"
    
    if not ssh_config_path.exists():
        raise FileNotFoundError(f"SSH config file not found: {ssh_config_path}")
    
    if not ssh_config_path.is_file():
        raise ValueError(f"SSH config path is not a file: {ssh_config_path}")
    
    config = {
        "HostName": None,
        "Port": None,
        "User": None,
        "IdentityFile": None,
    }
    
    in_host_block = False
    current_indent = 0
    
    with open(ssh_config_path, "r") as f:
        for line in f:
            # Strip leading/trailing whitespace
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                continue
            
            # Check if this is a Host directive
            if stripped.lower().startswith("host "):
                # Check if this matches our host alias
                host_names = stripped[5:].strip().split()
                in_host_block = host_alias in host_names
                if in_host_block:
                    current_indent = len(line) - len(line.lstrip())
                continue
            
            # Only process lines within the matching Host block
            if not in_host_block:
                continue
            
            # Check if we've left the Host block (new Host directive or different indentation)
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= current_indent and not line.startswith(" ") and not line.startswith("\t"):
                # We've left the Host block
                break
            
            # Parse config directives
            parts = stripped.split(None, 1)
            if len(parts) != 2:
                continue
            
            key = parts[0]
            value = parts[1].strip()
            
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            
            if key == "HostName":
                config["HostName"] = value
            elif key == "Port":
                config["Port"] = value
            elif key == "User":
                config["User"] = value
            elif key == "IdentityFile":
                config["IdentityFile"] = value
    
    # Validate that we found the host
    if config["HostName"] is None:
        raise ValueError(f"Host '{host_alias}' not found in SSH config or missing HostName")
    
    return config


def build_rsync_command(
    remote_host: str,
    remote_path: str,
    local_path: str,
    ssh_config: Dict[str, Optional[str]],
    dry_run: bool = False,
) -> list:
    """
    Build rsync command with SSH options from config.
    
    Args:
        remote_host: SSH host alias (for display purposes)
        remote_path: Remote checkpoint directory path
        local_path: Local download directory path
        ssh_config: Parsed SSH config dictionary
        dry_run: If True, add --dry-run flag
        
    Returns:
        List of command arguments for subprocess
    """
    # Build SSH command string
    ssh_parts = ["ssh"]
    
    # Add port if specified
    if ssh_config["Port"]:
        ssh_parts.extend(["-p", ssh_config["Port"]])
    
    # Add identity file if specified
    if ssh_config["IdentityFile"]:
        identity_file = ssh_config["IdentityFile"]
        # Expand ~ in identity file path
        if identity_file.startswith("~"):
            identity_file = os.path.expanduser(identity_file)
        ssh_parts.extend(["-i", identity_file])
    
    ssh_cmd = " ".join(ssh_parts)
    
    # Build remote path with user@hostname
    user = ssh_config["User"] or "root"  # Default to root based on user's config
    hostname = ssh_config["HostName"]
    remote_spec = f"{user}@{hostname}:{remote_path}"
    
    # Ensure remote path ends with / for rsync directory sync
    if not remote_spec.endswith("/"):
        remote_spec += "/"
    
    # Build rsync command
    rsync_cmd = [
        "rsync",
        "-avz",  # Archive, verbose, compress
        "--progress",  # Show progress
        "--partial",  # Keep partial files on interruption
        "--partial-dir=.rsync-partial",  # Directory for partial files
        "-e", ssh_cmd,  # SSH command
    ]
    
    if dry_run:
        rsync_cmd.append("--dry-run")
    
    rsync_cmd.extend([remote_spec, local_path])
    
    return rsync_cmd


def check_rsync_available() -> bool:
    """Check if rsync is available on the system."""
    try:
        result = subprocess.run(
            ["which", "rsync"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def download_checkpoints(
    remote_host: str = "vast-gpu",
    remote_path: str = "/root/icr/checkpoints/",
    local_path: str = "~/icr_checkpoints",
    dry_run: bool = False,
) -> None:
    """
    Download all checkpoints from remote server to local machine.
    
    Args:
        remote_host: SSH host alias to look up in SSH config
        remote_path: Remote checkpoint directory path
        local_path: Local download directory path
        dry_run: If True, show what would be downloaded without downloading
    """
    print(f"Downloading checkpoints from {remote_host}...")
    print(f"  Remote path: {remote_path}")
    print(f"  Local path: {local_path}")
    if dry_run:
        print("  Mode: DRY RUN (no files will be downloaded)")
    print()
    
    # Check if rsync is available
    if not check_rsync_available():
        print("ERROR: rsync is not available on this system.")
        print("Please install rsync (usually pre-installed on macOS/Linux)")
        sys.exit(1)
    
    # Parse SSH config
    try:
        print(f"Parsing SSH config for host '{remote_host}'...")
        ssh_config = parse_ssh_config(remote_host)
        print(f"  HostName: {ssh_config['HostName']}")
        if ssh_config["Port"]:
            print(f"  Port: {ssh_config['Port']}")
        if ssh_config["User"]:
            print(f"  User: {ssh_config['User']}")
        if ssh_config["IdentityFile"]:
            print(f"  IdentityFile: {ssh_config['IdentityFile']}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to parse SSH config: {e}")
        sys.exit(1)
    
    # Expand local path
    local_path_expanded = os.path.expanduser(local_path)
    local_path_obj = Path(local_path_expanded)
    
    # Create local directory if it doesn't exist
    if not dry_run:
        local_path_obj.mkdir(parents=True, exist_ok=True)
        print(f"Local directory: {local_path_expanded}")
    else:
        print(f"Local directory (would be created): {local_path_expanded}")
    print()
    
    # Build rsync command
    try:
        rsync_cmd = build_rsync_command(
            remote_host=remote_host,
            remote_path=remote_path,
            local_path=local_path_expanded,
            ssh_config=ssh_config,
            dry_run=dry_run,
        )
    except Exception as e:
        print(f"ERROR: Failed to build rsync command: {e}")
        sys.exit(1)
    
    # Execute rsync
    print("Executing rsync...")
    if dry_run:
        print("Command:", " ".join(rsync_cmd))
    print()
    
    try:
        # Run rsync with real-time output
        process = subprocess.Popen(
            rsync_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end="")
        
        process.wait()
        
        if process.returncode == 0:
            print()
            if dry_run:
                print("✓ Dry run completed successfully")
            else:
                print("✓ Download completed successfully")
        else:
            print()
            print(f"ERROR: rsync failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
    except KeyboardInterrupt:
        print()
        print("\nDownload interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: Failed to execute rsync: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download all checkpoints from remote server to local machine. "
                    "Uses rsync and parses SSH config automatically."
    )
    parser.add_argument(
        "--remote-host",
        type=str,
        default="vast-gpu",
        help="SSH host alias to look up in SSH config (default: vast-gpu)",
    )
    parser.add_argument(
        "--remote-path",
        type=str,
        default="/root/icr/checkpoints/",
        help="Remote checkpoint directory path (default: /root/icr/checkpoints/)",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default="~/icr_checkpoints",
        help="Local download directory path (default: ~/icr_checkpoints)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    
    args = parser.parse_args()
    
    download_checkpoints(
        remote_host=args.remote_host,
        remote_path=args.remote_path,
        local_path=args.local_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
