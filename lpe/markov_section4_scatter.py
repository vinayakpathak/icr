#!/usr/bin/env python3
"""Generate mixed-length next-token calibration scatter for Markov-k checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lpe.markov_k_transformer import ModelConfig, bayes_next_prob, build_model, sample_markov_k_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Section-4 style Markov-k next-token scatter.")
    p.add_argument("--summary-path", type=str, required=True, help="Path to k-summary.json containing checkpoint info.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for figure/csv/summary.")
    p.add_argument("--n-contexts", type=int, default=200)
    p.add_argument("--len-min", type=int, default=64)
    p.add_argument("--len-max", type=int, default=None)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))

    summary_path = resolve_path(args.summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    k = int(summary["k"])
    cfg_raw = summary["model_config"]
    cfg = ModelConfig(
        n_layers=int(cfg_raw["n_layers"]),
        d_model=int(cfg_raw["d_model"]),
        n_heads=int(cfg_raw["n_heads"]),
        d_mlp=int(cfg_raw["d_mlp"]),
    )
    max_seq_len = int(summary.get("max_seq_len", 4096))
    ckpt = resolve_path(str(summary["checkpoint_path"]))
    use_pos = bool(summary.get("use_positional_encoding", True))

    model = build_model(cfg, max_seq_len=max_seq_len, use_positional_encoding=use_pos).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"section4_k{k}_nextprob_scatter_varlen.png"
    csv_path = out_dir / f"section4_k{k}_nextprob_scatter_varlen.csv"
    summary_path_out = out_dir / f"section4_k{k}_nextprob_scatter_summary.txt"

    len_min = max(int(args.len_min), k + 1)
    inferred_len_max = max_seq_len - 1
    len_max = int(args.len_max) if args.len_max is not None else inferred_len_max
    len_max = min(len_max, inferred_len_max)
    if len_max < len_min:
        raise ValueError(f"Invalid length range [{len_min}, {len_max}] for max_seq_len={max_seq_len}")

    rows: List[Dict[str, object]] = []
    bayes_vals: List[float] = []
    model_vals: List[float] = []

    rng = np.random.default_rng(int(args.seed))
    with torch.no_grad():
        for i in range(int(args.n_contexts)):
            clen = int(rng.integers(len_min, len_max + 1))
            context = sample_markov_k_batch(
                batch_size=1,
                seq_len=clen,
                k=k,
                alpha=float(args.alpha),
                beta=float(args.beta),
                device=device,
            )[0].detach().cpu().long()
            p_bayes = float(bayes_next_prob(context, k, alpha=float(args.alpha), beta=float(args.beta)))
            logits = model.predict_next_logits(context.to(device).unsqueeze(0))
            p_model = float(F.softmax(logits, dim=-1)[0, 1].item())
            rows.append(
                {
                    "context_id": i + 1,
                    "context_len": clen,
                    "bayes_p_next_1": p_bayes,
                    "model_p_next_1": p_model,
                }
            )
            bayes_vals.append(p_bayes)
            model_vals.append(p_model)

    bayes_np = np.array(bayes_vals, dtype=np.float64)
    model_np = np.array(model_vals, dtype=np.float64)
    corr = float(np.corrcoef(bayes_np, model_np)[0, 1]) if bayes_np.size > 1 else float("nan")
    mae = float(np.mean(np.abs(model_np - bayes_np)))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(bayes_np, model_np, s=20, alpha=0.45, edgecolors="none")
    ax.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Bayes next-bit P(1)")
    ax.set_ylabel("Model next-bit P(1)")
    ax.set_title(f"k={k}: Pearson r={corr:.4f}, MAE={mae:.5f}")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)

    summary_path_out.write_text(
        "\n".join(
            [
                f"n_contexts={int(args.n_contexts)}",
                f"len_min={len_min}",
                f"len_max={len_max}",
                f"pearson={corr:.6f}",
                f"mae={mae:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote figure: {fig_path}")
    print(f"Wrote summary: {summary_path_out}")


if __name__ == "__main__":
    main()
