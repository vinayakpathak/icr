#!/usr/bin/env python3
"""Convenience wrapper for running the full k=1..7 order-k Markov suite."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run full Markov-k suite using markov_k_transformer.py")
    p.add_argument("--k-list", type=str, default="1,2,3,4,5,6,7")
    p.add_argument("--extra-args", type=str, default="", help="Extra args forwarded to markov_k_transformer.py")
    return p


def main() -> None:
    args = build_parser().parse_args()
    script = Path(__file__).resolve().parent / "markov_k_transformer.py"
    cmd = ["python", str(script), "--k-list", args.k_list]
    if args.extra_args.strip():
        cmd.extend(args.extra_args.strip().split())
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
