#!/usr/bin/env python3
"""Run Markov Step-3 LPE with detailed logging for each context."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

# Make imports work both as `python -m lpe.markov_step3_detailed` and
# `python lpe/markov_step3_detailed.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lpe.markov_k_transformer import (
    ModelConfig,
    bayes_target_logprob,
    build_model,
    generate_contexts_and_target,
    markov_sequence_logprob_given_theta,
    posterior_samples_from_rollouts,
)


def count_transitions_binary(seq: torch.Tensor) -> Dict[str, int]:
    bits = seq.detach().cpu().long().tolist()
    c00 = 0
    c01 = 0
    c10 = 0
    c11 = 0
    for i in range(1, len(bits)):
        a = bits[i - 1]
        b = bits[i]
        if a == 0 and b == 0:
            c00 += 1
        elif a == 0 and b == 1:
            c01 += 1
        elif a == 1 and b == 0:
            c10 += 1
        else:
            c11 += 1
    return {
        "t00": c00,
        "t01": c01,
        "t10": c10,
        "t11": c11,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detailed Step-3 rerun for k-Markov transformer.")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--checkpoint-path", type=str, default="checkpoints/markov_k1_gaplt1_highcap_run2/k1/best.pt")
    p.add_argument("--max-seq-len", type=int, default=2032)
    p.add_argument("--use-positional-encoding", action="store_true", default=True)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-mlp", type=int, default=512)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-contexts", type=int, default=8)
    p.add_argument("--context-len", type=int, default=1000)
    p.add_argument("--target-len", type=int, default=100)
    p.add_argument("--target-mode", type=str, choices=["random", "balanced"], default="balanced")
    p.add_argument("--num-posterior-samples", type=int, default=200)
    p.add_argument("--rollout-length", type=int, default=1000)
    p.add_argument("--posterior-rollout-batch-size", type=int, default=8)
    p.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/markov_k1_gaplt1_highcap_lpeonly_ctx1000_roll1000_detailed",
    )
    return p.parse_args()


def choose_device(arg_device: str) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = Path(args.out_dir)
    k_dir = out_dir / f"k{args.k}"
    k_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(
        n_layers=int(args.n_layers),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        d_mlp=int(args.d_mlp),
    )
    model = build_model(
        model_cfg,
        max_seq_len=int(args.max_seq_len),
        use_positional_encoding=bool(args.use_positional_encoding),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    contexts, target = generate_contexts_and_target(
        k=int(args.k),
        num_contexts=int(args.num_contexts),
        min_context_len=int(args.context_len),
        max_context_len=int(args.context_len),
        target_len=int(args.target_len),
        alpha=float(args.alpha),
        beta=float(args.beta),
        device=device,
        seed=int(args.seed) + 999 * int(args.k),
        target_mode=str(args.target_mode),
    )

    target_counts = count_transitions_binary(target)

    metrics_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    context_rows: List[Dict[str, object]] = []
    sample_tensor = np.zeros((int(args.num_contexts), int(args.num_posterior_samples), 1 << int(args.k)), dtype=np.float64)

    for i, context in enumerate(contexts):
        ctx_id = i
        print(f"[detailed step3] context {ctx_id + 1}/{len(contexts)} (len={int(context.numel())})", flush=True)
        context_counts = count_transitions_binary(context)
        context_rows.append(
            {
                "context_id": ctx_id,
                "context_len": int(context.numel()),
                "context_bits": "".join(str(int(x)) for x in context.tolist()),
            }
        )

        progress_every = max(1, int(math.ceil(int(args.num_posterior_samples) / max(1, int(args.posterior_rollout_batch_size)) / 5.0)))
        samples = posterior_samples_from_rollouts(
            model=model,
            context=context,
            k=int(args.k),
            num_samples=int(args.num_posterior_samples),
            rollout_length=int(args.rollout_length),
            device=device,
            rollout_batch_size=int(args.posterior_rollout_batch_size),
            progress_label=f"[detailed step3] ctx {ctx_id + 1}/{len(contexts)}",
            progress_every_batches=progress_every,
        )
        sample_tensor[ctx_id] = samples

        for s_idx in range(samples.shape[0]):
            row: Dict[str, object] = {
                "context_id": ctx_id,
                "sample_id": s_idx,
            }
            for state in range(samples.shape[1]):
                row[f"theta_state_{state}"] = float(samples[s_idx, state])
            sample_rows.append(row)

        sample_logps: List[float] = []
        for s_idx in range(samples.shape[0]):
            theta = torch.tensor(samples[s_idx], dtype=torch.float64)
            sample_logps.append(markov_sequence_logprob_given_theta(context, target, theta, int(args.k)))
        sample_logps_np = np.array(sample_logps, dtype=np.float64)
        max_lp = float(sample_logps_np.max())
        est_prob = float(np.exp(max_lp) * np.mean(np.exp(sample_logps_np - max_lp)))
        est_std = float(np.std(np.exp(sample_logps_np), ddof=1)) if sample_logps_np.size > 1 else 0.0
        est_stderr = est_std / math.sqrt(max(1, sample_logps_np.size))

        true_logp = bayes_target_logprob(
            context=context,
            target=target,
            k=int(args.k),
            alpha=float(args.alpha),
            beta=float(args.beta),
        )
        true_prob = float(math.exp(true_logp))
        rel_err_pct = float(abs(est_prob - true_prob) / max(true_prob, 1e-300) * 100.0)

        row = {
            "k": int(args.k),
            "context_id": ctx_id,
            "context_len": int(context.numel()),
            "target_len": int(target.numel()),
            "context_t01": context_counts["t01"],
            "context_t10": context_counts["t10"],
            "target_t01": target_counts["t01"],
            "target_t10": target_counts["t10"],
            "context_t00": context_counts["t00"],
            "context_t11": context_counts["t11"],
            "target_t00": target_counts["t00"],
            "target_t11": target_counts["t11"],
            "true_prob": true_prob,
            "true_log10_prob": float(true_logp / math.log(10.0)),
            "posterior_estimate": est_prob,
            "posterior_std_error": est_stderr,
            "relative_error_pct": rel_err_pct,
        }
        metrics_rows.append(row)

    target_bits = "".join(str(int(x)) for x in target.tolist())
    target_info = {
        "target_len": int(target.numel()),
        "target_bits": target_bits,
        "target_t00": target_counts["t00"],
        "target_t01": target_counts["t01"],
        "target_t10": target_counts["t10"],
        "target_t11": target_counts["t11"],
    }

    metrics_csv = k_dir / "step3_lpe_metrics_detailed.csv"
    samples_csv = k_dir / "step3_posterior_samples.csv"
    contexts_csv = k_dir / "step3_contexts.csv"
    samples_npy = k_dir / "step3_posterior_samples.npy"
    target_json = k_dir / "step3_target_info.json"
    summary_json = k_dir / "step3_detailed_summary.json"

    write_csv(metrics_csv, metrics_rows)
    write_csv(samples_csv, sample_rows)
    write_csv(contexts_csv, context_rows)
    np.save(samples_npy, sample_tensor)
    target_json.write_text(json.dumps(target_info, indent=2), encoding="utf-8")

    rel = np.array([float(r["relative_error_pct"]) for r in metrics_rows], dtype=np.float64)
    summary = {
        "k": int(args.k),
        "num_contexts": int(args.num_contexts),
        "context_len": int(args.context_len),
        "rollout_len": int(args.rollout_length),
        "num_posterior_samples": int(args.num_posterior_samples),
        "target_len": int(args.target_len),
        "target_mode": str(args.target_mode),
        "checkpoint_path": str(checkpoint_path),
        "metrics_csv": str(metrics_csv),
        "samples_csv": str(samples_csv),
        "samples_npy": str(samples_npy),
        "contexts_csv": str(contexts_csv),
        "target_json": str(target_json),
        "rel_err_median_pct": float(np.median(rel)) if rel.size else float("nan"),
        "rel_err_mean_pct": float(np.mean(rel)) if rel.size else float("nan"),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
