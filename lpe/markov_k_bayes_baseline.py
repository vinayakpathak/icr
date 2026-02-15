#!/usr/bin/env python3
"""Bayes-predictive baseline for Markov-k LPE steps (post-Step-1 only).

This runs the same Step 2/3 procedure used for transformer checkpoints, but with
an oracle true Bayes predictive process in place of the transformer generator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from markov_k_transformer import (  # noqa: E402
    bayes_target_logprob,
    compute_posterior_params,
    default_context_length_for_k,
    markov_sequence_logprob_given_theta,
    parse_k_list,
    save_csv,
    sample_markov_k_batch,
    set_seed,
)


def _state_from_list(bits: List[int], k: int) -> int:
    state = 0
    for b in bits[-k:]:
        state = (state << 1) | int(b)
    return state


def bayes_predictive_rollout(
    context: torch.Tensor,
    k: int,
    length: int,
    alpha: float,
    beta: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate one rollout from the true Bayes predictive process."""
    if length <= 0:
        return torch.empty(0, dtype=torch.long)

    alpha_post, beta_post = compute_posterior_params(context, k=k, alpha=alpha, beta=beta)
    alpha_work = alpha_post.clone().double()
    beta_work = beta_post.clone().double()

    hist = context.detach().cpu().long().tolist()
    generated = torch.empty(length, dtype=torch.long)

    for t in range(length):
        if len(hist) < k:
            p = 0.5
            state = None
        else:
            state = _state_from_list(hist, k)
            denom = float(alpha_work[state] + beta_work[state])
            p = float(alpha_work[state] / denom)

        bit = 1 if torch.rand((), generator=rng).item() < p else 0
        generated[t] = bit
        hist.append(bit)

        if state is not None:
            if bit == 1:
                alpha_work[state] += 1.0
            else:
                beta_work[state] += 1.0

    return generated


def posterior_samples_from_bayes_rollouts(
    context: torch.Tensor,
    k: int,
    num_samples: int,
    rollout_length: int,
    alpha: float,
    beta: float,
    seed: int,
) -> np.ndarray:
    """Estimate posterior samples via rollout transition frequencies (baseline oracle)."""
    num_states = 1 << k
    context_cpu = context.detach().cpu().long()

    samples = np.full((num_samples, num_states), 0.5, dtype=np.float64)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    for i in range(num_samples):
        generated = bayes_predictive_rollout(
            context=context_cpu,
            k=k,
            length=rollout_length,
            alpha=alpha,
            beta=beta,
            rng=rng,
        )
        full = torch.cat([context_cpu, generated], dim=0)

        ones = np.zeros(num_states, dtype=np.float64)
        zeros = np.zeros(num_states, dtype=np.float64)

        start_t = max(k, int(context_cpu.numel()))
        for t in range(start_t, int(full.numel())):
            window = full[t - k : t].tolist()
            state = _state_from_list(window, k)
            bit = int(full[t].item())
            if bit == 1:
                ones[state] += 1.0
            else:
                zeros[state] += 1.0

        denom = ones + zeros
        samples[i] = np.where(denom > 0.0, ones / np.maximum(denom, 1e-12), 0.5)

    return samples


def generate_contexts_and_target(
    k: int,
    num_contexts: int,
    min_context_len: int,
    max_context_len: int,
    target_len: int,
    alpha: float,
    beta: float,
    seed: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Deterministic context/target generator (matches transformer script intent)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    contexts: list[torch.Tensor] = []
    for _ in range(num_contexts):
        clen = int(torch.randint(min_context_len, max_context_len + 1, (1,), generator=g).item())
        seq = sample_markov_k_batch(
            batch_size=1,
            seq_len=clen,
            k=k,
            alpha=alpha,
            beta=beta,
            device=torch.device("cpu"),
        )[0].detach().cpu().long()
        contexts.append(seq)

    target = torch.randint(0, 2, (target_len,), generator=g, dtype=torch.long).detach().cpu().long()
    return contexts, target


def run_one_k(k: int, args: argparse.Namespace) -> Dict[str, object]:
    set_seed(int(args.seed) + 1000 * k)

    out_root = Path(args.out_dir)
    k_dir = out_root / f"k{k}"
    fig_dir = k_dir / "figures"
    k_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rollout_len = int(args.rollout_length_mult) * (2**k)
    context_len = int(args.context_length) if args.context_length is not None else default_context_length_for_k(k)

    context_for_posterior = sample_markov_k_batch(
        batch_size=1,
        seq_len=max(context_len, k + 1),
        k=k,
        alpha=float(args.alpha),
        beta=float(args.beta),
        device=torch.device("cpu"),
    )[0].detach().cpu().long()

    posterior_samples = posterior_samples_from_bayes_rollouts(
        context=context_for_posterior,
        k=k,
        num_samples=int(args.num_posterior_samples),
        rollout_length=rollout_len,
        alpha=float(args.alpha),
        beta=float(args.beta),
        seed=int(args.seed) + 10000 * k + 1,
    )

    alpha_post, beta_post = compute_posterior_params(
        context_for_posterior,
        k=k,
        alpha=float(args.alpha),
        beta=float(args.beta),
    )
    true_mean = (alpha_post / (alpha_post + beta_post)).numpy()
    sample_mean = posterior_samples.mean(axis=0)
    sample_std = posterior_samples.std(axis=0)

    posterior_mae = float(np.mean(np.abs(sample_mean - true_mean)))
    posterior_rmse = float(np.sqrt(np.mean((sample_mean - true_mean) ** 2)))

    post_rows: List[Dict[str, object]] = []
    for state in range(1 << k):
        post_rows.append(
            {
                "k": k,
                "state": state,
                "true_mean": float(true_mean[state]),
                "sample_mean": float(sample_mean[state]),
                "sample_std": float(sample_std[state]),
                "abs_err": float(abs(sample_mean[state] - true_mean[state])),
            }
        )
    save_csv(k_dir / "step2_posterior_state_metrics.csv", post_rows)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(true_mean, sample_mean, s=22, alpha=0.75)
    ax.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.5)
    ax.set_xlabel("Analytic posterior mean")
    ax.set_ylabel("Bayes-rollout sampled posterior mean")
    ax.set_title(f"k={k} Bayes baseline posterior mean by state")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    posterior_plot = fig_dir / "step2_posterior_mean_scatter_bayes_baseline.png"
    plt.savefig(posterior_plot, dpi=150)
    plt.close(fig)

    contexts, target = generate_contexts_and_target(
        k=k,
        num_contexts=int(args.num_lpe_contexts),
        min_context_len=max(k, int(args.lpe_context_min_len)),
        max_context_len=max(k, int(args.lpe_context_max_len)),
        target_len=int(args.lpe_target_len),
        alpha=float(args.alpha),
        beta=float(args.beta),
        seed=int(args.seed) + 999 * k,
    )

    lpe_rows: List[Dict[str, object]] = []
    for i, context in enumerate(contexts):
        samples = posterior_samples_from_bayes_rollouts(
            context=context,
            k=k,
            num_samples=int(args.num_posterior_samples),
            rollout_length=rollout_len,
            alpha=float(args.alpha),
            beta=float(args.beta),
            seed=int(args.seed) + 20000 * k + i,
        )

        sample_logps = []
        for s_idx in range(samples.shape[0]):
            theta = torch.tensor(samples[s_idx], dtype=torch.float64)
            sample_logps.append(markov_sequence_logprob_given_theta(context, target, theta, k))
        sample_logps_np = np.array(sample_logps, dtype=np.float64)

        max_lp = float(sample_logps_np.max())
        est_prob = float(np.exp(max_lp) * np.mean(np.exp(sample_logps_np - max_lp)))
        est_std = float(np.std(np.exp(sample_logps_np), ddof=1)) if sample_logps_np.size > 1 else 0.0
        std_err = est_std / math.sqrt(max(1, sample_logps_np.size))

        true_logp = bayes_target_logprob(
            context=context,
            target=target,
            k=k,
            alpha=float(args.alpha),
            beta=float(args.beta),
        )
        true_prob = float(math.exp(true_logp))
        rel_err_pct = float(abs(est_prob - true_prob) / max(true_prob, 1e-300) * 100.0)

        lpe_rows.append(
            {
                "k": k,
                "context_id": i,
                "context_len": int(context.numel()),
                "target_len": int(target.numel()),
                "true_prob": true_prob,
                "true_log10_prob": float(true_logp / math.log(10.0)),
                "posterior_estimate": est_prob,
                "posterior_std_error": std_err,
                "relative_error_pct": rel_err_pct,
            }
        )

    save_csv(k_dir / "step3_lpe_metrics.csv", lpe_rows)

    rel_errors = np.array([float(r["relative_error_pct"]) for r in lpe_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.hist(rel_errors, bins=min(20, max(5, int(len(rel_errors) // 2))), color="seagreen", alpha=0.85)
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"k={k} Bayes baseline LPE relative-error histogram")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    lpe_plot = fig_dir / "step3_lpe_rel_error_hist_bayes_baseline.png"
    plt.savefig(lpe_plot, dpi=150)
    plt.close(fig)

    summary: Dict[str, object] = {
        "k": k,
        "baseline_name": "true_bayes_predictive",
        "rollout_length": rollout_len,
        "num_posterior_samples": int(args.num_posterior_samples),
        "step2_posterior_mae": posterior_mae,
        "step2_posterior_rmse": posterior_rmse,
        "step3_lpe_rel_error_median_pct": float(np.median(rel_errors)) if rel_errors.size else float("nan"),
        "step3_lpe_rel_error_mean_pct": float(np.mean(rel_errors)) if rel_errors.size else float("nan"),
        "posterior_plot": str(posterior_plot),
        "lpe_hist_plot": str(lpe_plot),
        "posterior_csv": str(k_dir / "step2_posterior_state_metrics.csv"),
        "lpe_csv": str(k_dir / "step3_lpe_metrics.csv"),
        "timestamp": time.time(),
    }

    with (k_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_comparison(
    baseline_summaries: List[Dict[str, object]],
    transformer_dir: Path,
    out_dir: Path,
) -> None:
    rows: List[Dict[str, object]] = []

    for b in sorted(baseline_summaries, key=lambda x: int(x["k"])):
        k = int(b["k"])
        t_summary_path = transformer_dir / f"k{k}" / "summary.json"
        if not t_summary_path.exists():
            continue
        t = json.loads(t_summary_path.read_text())

        t_step2 = float(t.get("step2_posterior_mae", float("nan")))
        b_step2 = float(b.get("step2_posterior_mae", float("nan")))
        t_step3 = float(t.get("step3_lpe_rel_error_median_pct", float("nan")))
        b_step3 = float(b.get("step3_lpe_rel_error_median_pct", float("nan")))

        rows.append(
            {
                "k": k,
                "transformer_step2_posterior_mae": t_step2,
                "bayes_baseline_step2_posterior_mae": b_step2,
                "step2_mae_diff": t_step2 - b_step2,
                "step2_mae_ratio": t_step2 / max(b_step2, 1e-12),
                "transformer_step3_lpe_median_rel_error_pct": t_step3,
                "bayes_baseline_step3_lpe_median_rel_error_pct": b_step3,
                "step3_rel_error_diff_pct": t_step3 - b_step3,
                "step3_rel_error_ratio": t_step3 / max(b_step3, 1e-12),
            }
        )

    comp_csv = out_dir / "transformer_vs_bayes_baseline.csv"
    if rows:
        save_csv(comp_csv, rows)

        ks = [int(r["k"]) for r in rows]
        t2 = [float(r["transformer_step2_posterior_mae"]) for r in rows]
        b2 = [float(r["bayes_baseline_step2_posterior_mae"]) for r in rows]
        t3 = [float(r["transformer_step3_lpe_median_rel_error_pct"]) for r in rows]
        b3 = [float(r["bayes_baseline_step3_lpe_median_rel_error_pct"]) for r in rows]

        x = np.arange(len(ks))
        width = 0.36

        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.bar(x - width / 2, t2, width=width, label="Transformer")
        ax.bar(x + width / 2, b2, width=width, label="True Bayes baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("k")
        ax.set_ylabel("Posterior MAE")
        ax.set_title("Step 2: Posterior MAE comparison")
        ax.grid(alpha=0.25, axis="y")
        ax.legend()
        plt.tight_layout()
        fig_path = out_dir / "step2_mae_transformer_vs_bayes.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.bar(x - width / 2, t3, width=width, label="Transformer")
        ax.bar(x + width / 2, b3, width=width, label="True Bayes baseline")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("k")
        ax.set_ylabel("Median relative error (%) [log scale]")
        ax.set_title("Step 3: LPE relative-error comparison")
        ax.grid(alpha=0.25, axis="y")
        ax.legend()
        plt.tight_layout()
        fig_path = out_dir / "step3_rel_error_transformer_vs_bayes.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Bayes-predictive baseline for Markov-k LPE steps")
    p.add_argument("--k-list", type=str, default="1,2,3")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--rollout-length-mult", type=int, default=100)
    p.add_argument("--context-length", type=int, default=None)
    p.add_argument("--num-posterior-samples", type=int, default=200)
    p.add_argument("--num-lpe-contexts", type=int, default=8)
    p.add_argument("--lpe-context-min-len", type=int, default=10)
    p.add_argument("--lpe-context-max-len", type=int, default=20)
    p.add_argument("--lpe-target-len", type=int, default=100)
    p.add_argument("--out-dir", type=str, default="artifacts/markov_k123_bayes_baseline")
    p.add_argument("--transformer-dir", type=str, default="artifacts/markov_k123_full")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_list = parse_k_list(args.k_list)
    summaries: List[Dict[str, object]] = []

    for k in k_list:
        print("\n" + "=" * 80)
        print(f"Bayes baseline for k={k}")
        print("=" * 80)
        summaries.append(run_one_k(k, args=args))

    summaries = sorted(summaries, key=lambda x: int(x["k"]))

    suite_json = out_dir / "suite_summary.json"
    suite_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    rows = []
    for s in summaries:
        rows.append(
            {
                "k": int(s["k"]),
                "baseline_name": str(s["baseline_name"]),
                "rollout_length": int(s["rollout_length"]),
                "num_posterior_samples": int(s["num_posterior_samples"]),
                "step2_posterior_mae": float(s["step2_posterior_mae"]),
                "step2_posterior_rmse": float(s["step2_posterior_rmse"]),
                "step3_lpe_rel_error_median_pct": float(s["step3_lpe_rel_error_median_pct"]),
                "step3_lpe_rel_error_mean_pct": float(s["step3_lpe_rel_error_mean_pct"]),
            }
        )
    suite_csv = out_dir / "suite_summary.csv"
    save_csv(suite_csv, rows)

    transformer_dir = Path(args.transformer_dir)
    if transformer_dir.exists():
        build_comparison(summaries, transformer_dir=transformer_dir, out_dir=out_dir)

    manifest = {
        "timestamp": time.time(),
        "k_list": k_list,
        "suite_csv": str(suite_csv),
        "suite_json": str(suite_json),
        "transformer_dir": str(transformer_dir),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nRun complete.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
