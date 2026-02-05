#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Sync sensitive (non-git) files from a remote server to this laptop.

Usage:
  sync_sensitive_from_server.sh [--host HOST] [--remote-root PATH] \
    [--backup-root PATH] [--manifest FILE]

Defaults:
  --host        vast-gpu
  --remote-root /root/icr
  --backup-root ~/icr_sensitive_backup
  --manifest    <repo>/scripts/sensitive_paths.txt

Notes:
  - Relative paths in the manifest are resolved from the remote repo root.
  - Absolute paths in the manifest are synced into <backup-root>/absolute/.
  - Repo-relative files are synced into <backup-root>/repo/.
USAGE
}

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_ROOT="${REMOTE_ROOT:-}"
BACKUP_ROOT="${BACKUP_ROOT:-}"
MANIFEST="${MANIFEST:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --remote-root)
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --backup-root)
      BACKUP_ROOT="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$REMOTE_HOST" ]]; then
  REMOTE_HOST="vast-gpu"
fi
if [[ -z "$REMOTE_ROOT" ]]; then
  REMOTE_ROOT="/root/icr"
fi
if [[ -z "$BACKUP_ROOT" ]]; then
  BACKUP_ROOT="$HOME/icr_sensitive_backup"
fi
if [[ -z "$MANIFEST" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  MANIFEST="$SCRIPT_DIR/sensitive_paths.txt"
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync is required but not found." >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest not found: $MANIFEST" >&2
  exit 1
fi

REMOTE_ROOT="${REMOTE_ROOT%/}"
BACKUP_ROOT="${BACKUP_ROOT%/}"
REPO_BACKUP_ROOT="$BACKUP_ROOT/repo"
ABS_BACKUP_ROOT="$BACKUP_ROOT/absolute"

TMP_REL="$(mktemp)"
TMP_ABS_REMOTE="$(mktemp)"
trap 'rm -f "$TMP_REL" "$TMP_ABS_REMOTE"' EXIT

while IFS= read -r line; do
  line="${line%%#*}"
  line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  if [[ -z "$line" ]]; then
    continue
  fi

  if [[ "$line" == /* ]]; then
    printf '%s\n' "$line" >> "$TMP_ABS_REMOTE"
  else
    printf '%s\n' "./$line" >> "$TMP_REL"
  fi
done < "$MANIFEST"

if [[ ! -s "$TMP_REL" && ! -s "$TMP_ABS_REMOTE" ]]; then
  echo "ERROR: No paths found in manifest: $MANIFEST" >&2
  exit 1
fi

if ! ssh "$REMOTE_HOST" "test -d '$REMOTE_ROOT'" >/dev/null 2>&1; then
  echo "ERROR: Remote repo root not found: $REMOTE_HOST:$REMOTE_ROOT" >&2
  exit 1
fi

mkdir -p "$REPO_BACKUP_ROOT" "$ABS_BACKUP_ROOT"

if [[ -s "$TMP_REL" ]]; then
  rsync -a --no-owner --no-group --human-readable --progress --partial --inplace \
    --files-from="$TMP_REL" --relative \
    -e ssh \
    "$REMOTE_HOST:$REMOTE_ROOT/" "$REPO_BACKUP_ROOT/"
fi

if [[ -s "$TMP_ABS_REMOTE" ]]; then
  rsync -a --no-owner --no-group --human-readable --progress --partial --inplace \
    --files-from="$TMP_ABS_REMOTE" --relative \
    -e ssh \
    "$REMOTE_HOST:/" "$ABS_BACKUP_ROOT/"
fi

echo "Done. Backup saved to: $BACKUP_ROOT"
