#!/usr/bin/env python3
"""Postprocess Markov Step-3 detailed outputs using teacher-forced true probability."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lpe.markov_k_transformer import ModelConfig, build_model, compute_posterior_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher-forcing analysis for Markov Step-3 detailed artifacts.")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--detailed-out-dir", type=str, required=True)
    p.add_argument("--summary-path", type=str, required=True, help="k-summary.json from checkpoint run.")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def teacher_forced_target_logprob(
    model: torch.nn.Module,
    context: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> float:
    with torch.no_grad():
        seq = context.to(device=device, dtype=torch.long).unsqueeze(0)
        logp = 0.0
        for bit in target.tolist():
            logits = model.predict_next_logits(seq)
            log_probs = F.log_softmax(logits, dim=-1)
            logp += float(log_probs[0, int(bit)].item())
            nxt = torch.tensor([[int(bit)]], dtype=torch.long, device=device)
            seq = torch.cat([seq, nxt], dim=1)
    return float(logp)


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def main() -> None:
    args = parse_args()
    k = int(args.k)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    summary_path = resolve_path(args.summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    cfg_raw = summary["model_config"]
    cfg = ModelConfig(
        n_layers=int(cfg_raw["n_layers"]),
        d_model=int(cfg_raw["d_model"]),
        n_heads=int(cfg_raw["n_heads"]),
        d_mlp=int(cfg_raw["d_mlp"]),
    )
    max_seq_len = int(summary.get("max_seq_len", 4096))
    use_pos = bool(summary.get("use_positional_encoding", True))
    ckpt = resolve_path(str(summary["checkpoint_path"]))

    model = build_model(cfg, max_seq_len=max_seq_len, use_positional_encoding=use_pos).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    detailed_root = resolve_path(args.detailed_out_dir)
    k_dir = detailed_root / f"k{k}"
    fig_dir = detailed_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = k_dir / "step3_lpe_metrics_detailed.csv"
    contexts_path = k_dir / "step3_contexts.csv"
    target_path = k_dir / "step3_target_info.json"
    samples_path = k_dir / "step3_posterior_samples.npy"

    metric_rows_raw = read_csv(metrics_path)
    context_rows = read_csv(contexts_path)
    target_info = json.loads(target_path.read_text(encoding="utf-8"))
    sample_tensor = np.load(samples_path)

    target_bits = [int(c) for c in str(target_info["target_bits"]).strip()]
    target = torch.tensor(target_bits, dtype=torch.long)
    target_len = int(target.numel())
    if sample_tensor.ndim != 3:
        raise RuntimeError(f"Unexpected sample tensor shape: {sample_tensor.shape}")
    if sample_tensor.shape[0] != len(metric_rows_raw):
        raise RuntimeError("Mismatch between metrics rows and sample tensor context count.")
    if sample_tensor.shape[0] != len(context_rows):
        raise RuntimeError("Mismatch between metrics rows and contexts rows.")

    out_rows: List[Dict[str, object]] = []
    table_rows: List[Dict[str, object]] = []

    teacher_probs: List[float] = []
    estimates: List[float] = []

    num_samples = int(sample_tensor.shape[1])
    rollout_len = int(json.loads((k_dir / "step3_detailed_summary.json").read_text(encoding="utf-8"))["rollout_len"])
    r_eq = max(1, int((num_samples * rollout_len) // max(1, target_len)))

    for i, raw in enumerate(metric_rows_raw):
        ctx_id = int(raw["context_id"])
        bits = [int(c) for c in str(context_rows[i]["context_bits"]).strip()]
        context = torch.tensor(bits, dtype=torch.long)

        tf_logp = teacher_forced_target_logprob(model, context, target, device=device)
        tf_prob = float(math.exp(tf_logp))

        est_prob = float(raw["posterior_estimate"])
        rel_tf = float(abs(est_prob - tf_prob) / max(tf_prob, 1e-300) * 100.0)

        raw_out: Dict[str, object] = dict(raw)
        raw_out["teacher_forced_true_prob"] = tf_prob
        raw_out["teacher_forced_true_log10_prob"] = float(tf_logp / math.log(10.0))
        raw_out["relative_error_teacher_forced_pct"] = rel_tf
        out_rows.append(raw_out)

        alpha_post, beta_post = compute_posterior_params(
            context=context,
            k=k,
            alpha=float(args.alpha),
            beta=float(args.beta),
        )
        a = alpha_post.detach().cpu().numpy().astype(np.float64)
        b = beta_post.detach().cpu().numpy().astype(np.float64)
        true_mean = a / (a + b)
        true_sd = np.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1.0)))
        sample_mean = sample_tensor[i].mean(axis=0)
        sample_sd = sample_tensor[i].std(axis=0, ddof=1)

        row: Dict[str, object] = {
            "context_id": ctx_id,
            "teacher_forced_true_prob": tf_prob,
            "posterior_estimate": est_prob,
        }
        for s in range(1 << k):
            row[f"true_post_mean_state_{s}"] = float(true_mean[s])
            row[f"true_post_sd_state_{s}"] = float(true_sd[s])
            row[f"sampled_post_mean_state_{s}"] = float(sample_mean[s])
            row[f"sampled_post_sd_state_{s}"] = float(sample_sd[s])
        table_rows.append(row)

        teacher_probs.append(tf_prob)
        estimates.append(est_prob)

    out_rows_sorted = sorted(out_rows, key=lambda r: int(r["context_id"]))
    table_rows_sorted = sorted(table_rows, key=lambda r: int(r["context_id"]))

    teacher_csv = k_dir / "step3_lpe_metrics_teacher_forcing.csv"
    table_csv = k_dir / "step3_teacher_forcing_context_table.csv"
    write_csv(teacher_csv, out_rows_sorted)
    write_csv(table_csv, table_rows_sorted)

    tp = np.array(teacher_probs, dtype=np.float64)
    est = np.array(estimates, dtype=np.float64)
    rel = np.abs(est - tp) / np.maximum(tp, 1e-300) * 100.0
    naive_rel_se = np.sqrt((1.0 - tp) / np.maximum(float(r_eq) * tp, 1e-300)) * 100.0
    zero_hit = np.exp(np.log1p(-np.clip(tp, 1e-300, 1.0 - 1e-18)) * float(r_eq))

    p50_rel = percentile(rel, 50)
    p90_rel = percentile(rel, 90)
    p95_rel = percentile(rel, 95)
    p50_se = percentile(naive_rel_se, 50)
    p90_se = percentile(naive_rel_se, 90)
    p95_se = percentile(naive_rel_se, 95)
    p50_zero = percentile(zero_hit, 50)
    p90_zero = percentile(zero_hit, 90)
    p95_zero = percentile(zero_hit, 95)

    summary_txt = fig_dir / f"section6_k{k}_summary_teacher_forcing.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"R_eq={r_eq}",
                f"p50_rel={p50_rel}",
                f"p90_rel={p90_rel}",
                f"p95_rel={p95_rel}",
                f"p50_se={p50_se}",
                f"p90_se={p90_se}",
                f"p95_se={p95_se}",
                f"p50_zero={p50_zero}",
                f"p90_zero={p90_zero}",
                f"p95_zero={p95_zero}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    percentile_fig = fig_dir / f"posterior_vs_naive_percentiles_k{k}_teacher_forcing.png"
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    labels = ["p50", "p90", "p95"]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.35
    ax.bar(x - width / 2, [p50_rel, p90_rel, p95_rel], width=width, label="Posterior |rel err|%")
    ax.bar(x + width / 2, [p50_se, p90_se, p95_se], width=width, label="Naive expected rel SE%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Percent")
    ax.set_title(f"k={k}: posterior vs naive percentiles (teacher-forced truth)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(percentile_fig, dpi=180)
    plt.close(fig)

    scatter_fig = fig_dir / f"relative_error_vs_true_prob_k{k}_teacher_forcing.png"
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(tp, rel, s=44, alpha=0.85, label="Posterior |rel err|%")
    ax.scatter(tp, naive_rel_se, s=44, alpha=0.85, marker="x", label="Naive expected rel SE%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Teacher-forced true event probability")
    ax.set_ylabel("Relative error / expected SE (%)")
    ax.set_title(f"k={k}: per-context relative error vs true probability")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(scatter_fig, dpi=180)
    plt.close(fig)

    print(f"Wrote CSV: {teacher_csv}")
    print(f"Wrote table CSV: {table_csv}")
    print(f"Wrote summary: {summary_txt}")
    print(f"Wrote figure: {percentile_fig}")
    print(f"Wrote figure: {scatter_fig}")


if __name__ == "__main__":
    main()
