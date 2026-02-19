#!/usr/bin/env python3
"""Generate k=1 Markov posterior diagnostics mirroring the Bernoulli report style."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from lpe.bernoulli_posterior_grid import context_distribution_metrics, plot_metric_boxplots, summarize_metric
    from lpe.markov_k_transformer import (
        ModelConfig,
        build_model,
        compute_posterior_params,
        compute_transition_counts,
        posterior_samples_from_rollouts,
        sample_markov_k_batch,
    )
except ModuleNotFoundError:
    from bernoulli_posterior_grid import context_distribution_metrics, plot_metric_boxplots, summarize_metric
    from markov_k_transformer import (  # type: ignore
        ModelConfig,
        build_model,
        compute_posterior_params,
        compute_transition_counts,
        posterior_samples_from_rollouts,
        sample_markov_k_batch,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent


def beta_pdf(alpha: float, beta: float, x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, 1e-6, 1.0 - 1e-6)
    log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_pdf = (alpha - 1.0) * np.log(x_clip) + (beta - 1.0) * np.log(1.0 - x_clip) - log_beta
    return np.exp(np.clip(log_pdf, -745.0, 60.0))


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def load_model_from_summary(summary_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg_raw = summary["model_config"]
    cfg = ModelConfig(
        n_layers=int(cfg_raw["n_layers"]),
        d_model=int(cfg_raw["d_model"]),
        n_heads=int(cfg_raw["n_heads"]),
        d_mlp=int(cfg_raw["d_mlp"]),
    )
    max_seq_len = int(summary.get("max_seq_len", 2032))
    use_pos = bool(summary.get("use_positional_encoding", False))
    ckpt_path = _resolve_path(str(summary["checkpoint_path"]))
    state_dict = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg, max_seq_len=max_seq_len, use_positional_encoding=use_pos).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, summary


def metric_keys() -> List[str]:
    return [
        "wasserstein1",
        "ks_cdf",
        "cvm_int",
        "pit_ks",
        "pit_cvm",
        "quantile_rmse",
        "coverage_mae",
    ]


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))

    summary_path = Path(args.summary_path)
    if not summary_path.is_absolute():
        summary_path = REPO_ROOT / summary_path
    model, summary = load_model_from_summary(summary_path, device=device)

    k = int(args.k)
    rows = int(args.rows)
    cols = int(args.cols)
    n_contexts = int(args.num_contexts)
    if rows * cols != n_contexts:
        raise ValueError(f"num-contexts ({n_contexts}) must equal rows*cols ({rows * cols}).")

    max_seq_len = int(summary.get("max_seq_len", 2032))
    context_len = int(args.context_length)
    rollout_len = int(args.rollout_length)
    if context_len + rollout_len > max_seq_len:
        raise ValueError(
            f"context_length + rollout_length = {context_len + rollout_len} exceeds model max_seq_len={max_seq_len}."
        )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_grid = out_dir / f"posterior_context_grid_5x6_k{k}.png"
    fig_box = out_dir / f"posterior_context_metrics_boxplot_5x6_k{k}.png"
    context_csv = out_dir / f"posterior_context_grid_5x6_k{k}.csv"
    sample_csv = out_dir / f"posterior_context_samples_5x6_k{k}.csv"
    metric_csv = out_dir / f"posterior_context_metrics_5x6_k{k}.csv"
    metric_summary_csv = out_dir / f"posterior_context_metrics_summary_5x6_k{k}.csv"

    metric_names = metric_keys()
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), sharex=True, sharey=False)
    axes_arr = np.array(axes).reshape(-1)
    x = np.linspace(0.001, 0.999, 400, dtype=np.float64)

    context_rows: List[Dict[str, float | int]] = []
    sample_rows: List[Dict[str, float | int]] = []
    metric_rows: List[Dict[str, float | int]] = []
    baseline_rows: List[Dict[str, float | int]] = []

    for idx in range(n_contexts):
        with torch.no_grad():
            context = sample_markov_k_batch(
                batch_size=1,
                seq_len=context_len,
                k=k,
                alpha=1.0,
                beta=1.0,
                device=device,
            )[0].detach().cpu().long()

            alpha_post, beta_post = compute_posterior_params(context, k=k, alpha=1.0, beta=1.0)
            alpha_np = alpha_post.detach().cpu().numpy().astype(np.float64)
            beta_np = beta_post.detach().cpu().numpy().astype(np.float64)

            post_samples = posterior_samples_from_rollouts(
                model=model,
                context=context,
                k=k,
                num_samples=int(args.num_posterior_samples),
                rollout_length=rollout_len,
                device=device,
                rollout_batch_size=int(args.posterior_rollout_batch_size),
            )

        ones, zeros = compute_transition_counts(context, k=k)
        ones_np = ones.detach().cpu().numpy().astype(np.float64)
        zeros_np = zeros.detach().cpu().numpy().astype(np.float64)

        ax = axes_arr[idx]
        colors = ["tab:blue", "tab:orange"]
        state_metrics: List[Dict[str, float]] = []
        state_baselines: List[Dict[str, float]] = []
        for state in range(1 << k):
            svals = np.clip(post_samples[:, state].astype(np.float64), 0.0, 1.0)
            a = float(alpha_np[state])
            b = float(beta_np[state])
            pdf = beta_pdf(a, b, x)
            ax.hist(
                svals,
                bins=16,
                density=True,
                alpha=0.32,
                color=colors[state % len(colors)],
                edgecolor="black",
                linewidth=0.25,
            )
            ax.plot(x, pdf, color=colors[state % len(colors)], linewidth=1.3)
            ax.axvline(a / (a + b), color=colors[state % len(colors)], linestyle="--", linewidth=0.9, alpha=0.8)
            ax.axvline(float(np.mean(svals)), color=colors[state % len(colors)], linestyle=":", linewidth=0.9, alpha=0.8)

            m = context_distribution_metrics(svals, a, b)
            state_metrics.append(m)

            baseline_acc: Dict[str, float] = {name: 0.0 for name in metric_names}
            for _ in range(int(args.baseline_reps)):
                ref = np.random.beta(a, b, size=int(args.num_posterior_samples)).astype(np.float64)
                bm = context_distribution_metrics(ref, a, b)
                for name in metric_names:
                    baseline_acc[name] += float(bm[name])
            state_baselines.append({name: baseline_acc[name] / float(args.baseline_reps) for name in metric_names})

            for s_idx, p_hat in enumerate(svals, start=1):
                sample_rows.append(
                    {
                        "context_id": idx + 1,
                        "state": state,
                        "sample_id": s_idx,
                        "theta_hat": float(p_hat),
                    }
                )

        agg: Dict[str, float] = {name: float(np.mean([sm[name] for sm in state_metrics])) for name in metric_names}
        agg_baseline: Dict[str, float] = {
            name: float(np.mean([sb[name] for sb in state_baselines])) for name in metric_names
        }

        metric_rows.append(
            {
                "context_id": idx + 1,
                **agg,
                "state0_abs_mean_error": float(state_metrics[0]["abs_mean_error"]),
                "state1_abs_mean_error": float(state_metrics[1]["abs_mean_error"]),
            }
        )
        baseline_rows.append({"context_id": idx + 1, **agg_baseline})

        context_rows.append(
            {
                "context_id": idx + 1,
                "context_len": int(context_len),
                "n00": int(zeros_np[0]),
                "n01": int(ones_np[0]),
                "n10": int(zeros_np[1]),
                "n11": int(ones_np[1]),
                "alpha0_post": float(alpha_np[0]),
                "beta0_post": float(beta_np[0]),
                "alpha1_post": float(alpha_np[1]),
                "beta1_post": float(beta_np[1]),
                "rollout_length": int(rollout_len),
                "num_posterior_samples": int(args.num_posterior_samples),
            }
        )

        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.18)
        ax.set_title(
            f"ctx {idx + 1}: n00={int(zeros_np[0])}, n01={int(ones_np[0])}\n"
            f"n10={int(zeros_np[1])}, n11={int(ones_np[1])}",
            fontsize=8,
        )

        if (idx + 1) % 5 == 0 or (idx + 1) == n_contexts:
            print(f"completed contexts: {idx + 1}/{n_contexts}", flush=True)

    for c in range(cols):
        axes_arr[(rows - 1) * cols + c].set_xlabel(r"$\theta$")
    for r in range(rows):
        axes_arr[r * cols].set_ylabel("density")

    fig.suptitle(
        f"Markov-{k} posterior diagnostics across {n_contexts} contexts "
        "(histograms: model-implied samples; curves: true Bayes marginals)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.0, 1, 0.97])
    plt.savefig(fig_grid, dpi=220)
    plt.close(fig)

    with context_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(context_rows[0].keys()))
        writer.writeheader()
        writer.writerows(context_rows)
    with sample_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)
    with metric_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary_rows: List[Dict[str, float | str]] = []
    for name in metric_names:
        vals = np.array([float(r[name]) for r in metric_rows], dtype=np.float64)
        s = summarize_metric(vals)
        perfect = float(np.mean([float(r[name]) for r in baseline_rows]))
        summary_rows.append({"metric": name, **s, "perfect_match_expected_mean": perfect})

    with metric_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_metric_boxplots(metric_rows, fig_box)

    print(f"Wrote figure: {fig_grid}")
    print(f"Wrote boxplot: {fig_box}")
    print(f"Wrote context CSV: {context_csv}")
    print(f"Wrote sample CSV: {sample_csv}")
    print(f"Wrote metric CSV: {metric_csv}")
    print(f"Wrote metric summary CSV: {metric_summary_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate 5x6 posterior diagnostics for Markov-k checkpoint.")
    p.add_argument(
        "--summary-path",
        type=str,
        default="artifacts/markov_k123_transformer_500_ctx1000x2k_balanced/k1/summary.json",
    )
    p.add_argument("--out-dir", type=str, default="artifacts/markov_k123_transformer_report_ctx1000x2k/figures")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--num-contexts", type=int, default=30)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--context-length", type=int, default=1000)
    p.add_argument("--num-posterior-samples", type=int, default=200)
    p.add_argument("--rollout-length", type=int, default=1000)
    p.add_argument("--posterior-rollout-batch-size", type=int, default=8)
    p.add_argument("--baseline-reps", type=int, default=60)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
