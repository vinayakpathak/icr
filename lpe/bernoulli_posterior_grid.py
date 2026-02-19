#!/usr/bin/env python3
"""Generate posterior-comparison grid and distribution-similarity metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from lpe.bernoulli_transformer import BernoulliTransformer, compute_bernoulli_posterior
    from lpe.markov_k_transformer import rollout_with_cache_batch
except ModuleNotFoundError:
    from bernoulli_transformer import BernoulliTransformer, compute_bernoulli_posterior
    from markov_k_transformer import rollout_with_cache_batch


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int, int]:
    d_model = int(state_dict["token_emb.weight"].shape[1])
    layer_ids = {
        int(parts[1])
        for key in state_dict
        for parts in [key.split(".")]
        if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit()
    }
    n_layers = len(layer_ids)
    d_mlp = int(state_dict["blocks.0.mlp.0.weight"].shape[0]) if n_layers > 0 else d_model
    n_heads = 1
    return d_model, n_layers, n_heads, d_mlp


def beta_pdf_cdf_grid(alpha: float, beta: float, num_points: int = 2001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically build Beta(alpha,beta) PDF/CDF on [0,1] grid."""
    eps = 1e-6
    x_mid = np.linspace(eps, 1.0 - eps, num_points)
    log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_pdf = (alpha - 1.0) * np.log(x_mid) + (beta - 1.0) * np.log(1.0 - x_mid) - log_beta
    log_pdf = np.clip(log_pdf, -745.0, 60.0)
    pdf_mid = np.exp(log_pdf)

    cdf_mid = np.zeros_like(x_mid)
    dx_mid = np.diff(x_mid)
    cdf_mid[1:] = np.cumsum(0.5 * (pdf_mid[:-1] + pdf_mid[1:]) * dx_mid)
    if cdf_mid[-1] > 0:
        cdf_mid = cdf_mid / cdf_mid[-1]

    x = np.concatenate(([0.0], x_mid, [1.0]))
    cdf = np.concatenate(([0.0], cdf_mid, [1.0]))
    pdf = np.concatenate(([pdf_mid[0]], pdf_mid, [pdf_mid[-1]]))
    return x, pdf, cdf


def empirical_cdf_on_grid(samples_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = samples_sorted.size
    if n == 0:
        return np.zeros_like(x)
    return np.searchsorted(samples_sorted, x, side="right") / float(n)


def pit_ks_stat(u: np.ndarray) -> float:
    u_sorted = np.sort(np.clip(u, 0.0, 1.0))
    n = u_sorted.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = np.max(i / n - u_sorted)
    d_minus = np.max(u_sorted - (i - 1) / n)
    return float(max(d_plus, d_minus))


def pit_cvm_stat(u: np.ndarray) -> float:
    u_sorted = np.sort(np.clip(u, 0.0, 1.0))
    n = u_sorted.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(1.0 / (12.0 * n) + np.sum((u_sorted - (2.0 * i - 1.0) / (2.0 * n)) ** 2))


def context_distribution_metrics(samples: np.ndarray, alpha_post: float, beta_post: float) -> dict[str, float]:
    x, _pdf, cdf_true = beta_pdf_cdf_grid(alpha_post, beta_post, num_points=2001)
    s = np.sort(np.clip(samples.astype(np.float64), 0.0, 1.0))
    cdf_emp = empirical_cdf_on_grid(s, x)

    abs_diff = np.abs(cdf_emp - cdf_true)
    sq_diff = (cdf_emp - cdf_true) ** 2
    wasserstein1 = float(np.trapz(abs_diff, x))
    ks_cdf = float(np.max(abs_diff))
    cvm_int = float(np.trapz(sq_diff, x))

    u = np.interp(s, x, cdf_true)
    pit_ks = pit_ks_stat(u)
    pit_cvm = pit_cvm_stat(u)

    q_levels = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype=np.float64)
    sample_q = np.quantile(s, q_levels)
    true_q = np.interp(q_levels, cdf_true, x)
    quantile_rmse = float(np.sqrt(np.mean((sample_q - true_q) ** 2)))

    coverage_levels = [0.50, 0.80, 0.90, 0.95]
    cov_errs = []
    for level in coverage_levels:
        lo_q = np.interp((1.0 - level) / 2.0, cdf_true, x)
        hi_q = np.interp(1.0 - (1.0 - level) / 2.0, cdf_true, x)
        cov = float(np.mean((s >= lo_q) & (s <= hi_q)))
        cov_errs.append(abs(cov - level))
    coverage_mae = float(np.mean(cov_errs))
    coverage_max_abs = float(np.max(cov_errs))

    true_mean = float(alpha_post / (alpha_post + beta_post))
    true_std = float(
        math.sqrt((alpha_post * beta_post) / (((alpha_post + beta_post) ** 2) * (alpha_post + beta_post + 1.0)))
    )
    sample_mean = float(np.mean(s))
    sample_std = float(np.std(s, ddof=1)) if s.size > 1 else 0.0

    return {
        "true_mean": true_mean,
        "sample_mean": sample_mean,
        "abs_mean_error": abs(sample_mean - true_mean),
        "true_std": true_std,
        "sample_std": sample_std,
        "abs_std_error": abs(sample_std - true_std),
        "wasserstein1": wasserstein1,
        "ks_cdf": ks_cdf,
        "cvm_int": cvm_int,
        "pit_ks": pit_ks,
        "pit_cvm": pit_cvm,
        "quantile_rmse": quantile_rmse,
        "coverage_mae": coverage_mae,
        "coverage_max_abs": coverage_max_abs,
    }


def summarize_metric(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def plot_metric_boxplots(metric_rows: list[dict[str, float | int]], out_path: Path) -> None:
    metrics = [
        "wasserstein1",
        "ks_cdf",
        "cvm_int",
        "pit_ks",
        "pit_cvm",
        "quantile_rmse",
        "coverage_mae",
    ]
    data = [np.array([float(r[m]) for r in metric_rows], dtype=np.float64) for m in metrics]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.boxplot(data, labels=metrics, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("Value (log scale)")
    ax.set_title("Posterior similarity metrics across 30 contexts")
    ax.grid(alpha=0.25, axis="y")
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rank_panels(metric_rows: list[dict[str, float | int]], out_path: Path) -> None:
    w1 = np.array([float(r["wasserstein1"]) for r in metric_rows], dtype=np.float64)
    order = np.argsort(w1)
    rank = np.arange(1, len(order) + 1)

    def series(name: str) -> np.ndarray:
        arr = np.array([float(r[name]) for r in metric_rows], dtype=np.float64)
        return arr[order]

    fig, axs = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    axs = axs.reshape(-1)

    axs[0].plot(rank, series("wasserstein1"), marker="o", linewidth=1.4)
    axs[0].set_yscale("log")
    axs[0].set_title("W1 by rank (sorted by W1)")
    axs[0].set_ylabel("W1")
    axs[0].grid(alpha=0.25)

    axs[1].plot(rank, series("ks_cdf"), marker="o", linewidth=1.2, label="KS(CDF)")
    axs[1].plot(rank, series("pit_ks"), marker="s", linewidth=1.2, label="PIT-KS")
    axs[1].set_yscale("log")
    axs[1].set_title("KS-type metrics by rank")
    axs[1].grid(alpha=0.25)
    axs[1].legend()

    axs[2].plot(rank, series("cvm_int"), marker="o", linewidth=1.2, label="CvM-int")
    axs[2].plot(rank, series("pit_cvm"), marker="s", linewidth=1.2, label="PIT-CvM")
    axs[2].set_yscale("log")
    axs[2].set_title("CvM-type metrics by rank")
    axs[2].set_xlabel("Context rank (1=best by W1)")
    axs[2].set_ylabel("Metric value")
    axs[2].grid(alpha=0.25)
    axs[2].legend()

    axs[3].plot(rank, series("quantile_rmse"), marker="o", linewidth=1.2, label="Quantile RMSE")
    axs[3].plot(rank, series("coverage_mae"), marker="s", linewidth=1.2, label="Coverage MAE")
    axs[3].set_yscale("log")
    axs[3].set_title("Quantile/Coverage metrics by rank")
    axs[3].set_xlabel("Context rank (1=best by W1)")
    axs[3].grid(alpha=0.25)
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_corr_heatmap(metric_rows: list[dict[str, float | int]], out_path: Path) -> None:
    metric_names = [
        "wasserstein1",
        "ks_cdf",
        "cvm_int",
        "pit_ks",
        "pit_cvm",
        "quantile_rmse",
        "coverage_mae",
    ]
    X = np.stack(
        [np.array([float(r[name]) for r in metric_rows], dtype=np.float64) for name in metric_names],
        axis=1,
    )
    corr = np.corrcoef(X, rowvar=False)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    ax.set_yticklabels(metric_names)
    ax.set_title("Correlation between posterior similarity metrics")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson r")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))

    checkpoint = Path(args.model_path)
    state_dict = torch.load(checkpoint, map_location=device)
    d_model, n_layers, n_heads, d_mlp = infer_model_config(state_dict)

    model = BernoulliTransformer(
        max_seq_len=None,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_mlp=d_mlp,
        use_prenorm=True,
        attention_mode="causal",
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "posterior_context_grid_5x6.png"
    context_csv_path = out_dir / "posterior_context_grid_5x6.csv"
    sample_csv_path = out_dir / "posterior_context_samples_5x6.csv"
    metric_csv_path = out_dir / "posterior_context_metrics_5x6.csv"
    metric_summary_csv_path = out_dir / "posterior_context_metrics_summary_5x6.csv"
    top10_csv_path = out_dir / "posterior_context_metrics_top10_w1_5x6.csv"
    boxplot_path = out_dir / "posterior_context_metrics_boxplot_5x6.png"
    rankplot_path = out_dir / "posterior_context_metrics_rank_5x6.png"
    corrplot_path = out_dir / "posterior_context_metrics_corr_5x6.png"

    beta_prior = torch.distributions.Beta(torch.tensor(1.0, device=device), torch.tensor(1.0, device=device))
    p_axis = torch.linspace(0.001, 0.999, 400, device=device)

    rows = args.rows
    cols = args.cols
    n_contexts = rows * cols
    if args.num_contexts != n_contexts:
        raise ValueError(f"num_contexts ({args.num_contexts}) must equal rows*cols ({n_contexts}).")

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), sharex=True, sharey=False)
    axes_arr = np.array(axes).reshape(-1)

    context_rows: list[dict[str, float | int]] = []
    sample_rows: list[dict[str, float | int]] = []
    metric_rows: list[dict[str, float | int]] = []

    with torch.no_grad():
        for idx in range(n_contexts):
            p_true = float(beta_prior.sample().item())  # Beta(1,1)
            context = (torch.rand(args.context_length, device=device) < p_true).long()
            n_ones = int(context.sum().item())
            n_zeros = int(args.context_length - n_ones)

            alpha_post, beta_post = compute_bernoulli_posterior(n_ones, n_zeros, alpha=1.0, beta=1.0)
            beta_dist = torch.distributions.Beta(
                torch.tensor(alpha_post, device=device), torch.tensor(beta_post, device=device)
            )
            beta_pdf = torch.exp(beta_dist.log_prob(p_axis)).detach().cpu().numpy()

            rollout_batch = rollout_with_cache_batch(
                model=model,
                prefix=context,
                length=args.rollout_length,
                batch_size=args.num_posterior_samples,
                temperature=1.0,
            )
            if rollout_batch.numel() == 0:
                p_samples_np = np.full((args.num_posterior_samples,), 0.5, dtype=np.float64)
            else:
                p_samples_np = rollout_batch.float().mean(dim=1).detach().cpu().numpy().astype(np.float64)

            sample_mean = float(p_samples_np.mean())
            sample_std = float(p_samples_np.std(ddof=1)) if p_samples_np.size > 1 else 0.0
            bayes_mean = float(alpha_post / (alpha_post + beta_post))

            ax = axes_arr[idx]
            ax.hist(
                p_samples_np,
                bins=20,
                density=True,
                alpha=0.65,
                color="steelblue",
                edgecolor="black",
                linewidth=0.3,
            )
            ax.plot(p_axis.detach().cpu().numpy(), beta_pdf, color="crimson", linewidth=1.5)
            ax.axvline(bayes_mean, color="crimson", linestyle="--", linewidth=1.0, alpha=0.9)
            ax.axvline(sample_mean, color="black", linestyle=":", linewidth=1.0, alpha=0.9)
            ax.set_xlim(0.0, 1.0)
            ax.grid(alpha=0.18)
            ax.set_title(
                f"ctx {idx + 1}: ones={n_ones}, zeros={n_zeros}\n"
                f"true p={p_true:.2f}, mean={sample_mean:.2f}",
                fontsize=8,
            )

            context_rows.append(
                {
                    "context_id": idx + 1,
                    "p_true": p_true,
                    "context_ones": n_ones,
                    "context_zeros": n_zeros,
                    "alpha_post": float(alpha_post),
                    "beta_post": float(beta_post),
                    "num_samples": args.num_posterior_samples,
                    "rollout_length": args.rollout_length,
                }
            )
            for s_idx, p_hat in enumerate(p_samples_np, start=1):
                sample_rows.append(
                    {
                        "context_id": idx + 1,
                        "sample_id": s_idx,
                        "p_hat": float(p_hat),
                    }
                )

            m = context_distribution_metrics(p_samples_np, float(alpha_post), float(beta_post))
            m.update(
                {
                    "context_id": idx + 1,
                    "p_true": p_true,
                    "context_ones": n_ones,
                    "context_zeros": n_zeros,
                    "alpha_post": float(alpha_post),
                    "beta_post": float(beta_post),
                }
            )
            metric_rows.append(m)

            if (idx + 1) % 5 == 0 or (idx + 1) == n_contexts:
                print(f"completed contexts: {idx + 1}/{n_contexts}", flush=True)

    for c in range(cols):
        axes_arr[(rows - 1) * cols + c].set_xlabel("p")
    for r in range(rows):
        axes_arr[r * cols].set_ylabel("density")

    fig.suptitle(
        "Bernoulli posterior diagnostics across 30 input contexts "
        "(hist: model-implied posterior samples, curve: true Beta posterior)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.0, 1, 0.97])
    plt.savefig(fig_path, dpi=220)
    plt.close(fig)

    with context_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(context_rows[0].keys()))
        writer.writeheader()
        writer.writerows(context_rows)

    with sample_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    with metric_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary_metric_names = [
        "wasserstein1",
        "ks_cdf",
        "cvm_int",
        "pit_ks",
        "pit_cvm",
        "quantile_rmse",
        "coverage_mae",
        "coverage_max_abs",
        "abs_mean_error",
        "abs_std_error",
    ]
    summary_rows: list[dict[str, float | str]] = []
    for name in summary_metric_names:
        vals = np.array([float(r[name]) for r in metric_rows], dtype=np.float64)
        s = summarize_metric(vals)
        summary_rows.append({"metric": name, **s})
    with metric_summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    sorted_by_w1 = sorted(metric_rows, key=lambda r: float(r["wasserstein1"]), reverse=True)
    with top10_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sorted_by_w1[0].keys()))
        writer.writeheader()
        writer.writerows(sorted_by_w1[:10])

    plot_metric_boxplots(metric_rows, boxplot_path)
    plot_rank_panels(metric_rows, rankplot_path)
    plot_corr_heatmap(metric_rows, corrplot_path)

    print(f"Wrote figure: {fig_path}")
    print(f"Wrote context CSV: {context_csv_path}")
    print(f"Wrote sample CSV: {sample_csv_path}")
    print(f"Wrote metric CSV: {metric_csv_path}")
    print(f"Wrote metric summary CSV: {metric_summary_csv_path}")
    print(f"Wrote top-10 (W1) CSV: {top10_csv_path}")
    print(f"Wrote boxplot: {boxplot_path}")
    print(f"Wrote rank plot: {rankplot_path}")
    print(f"Wrote correlation heatmap: {corrplot_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate 5x6 posterior comparison grid for Bernoulli contexts.")
    p.add_argument("--model-path", type=str, default="checkpoints/bernoulli_transformer_L1_D16_seq1024.pt")
    p.add_argument("--out-dir", type=str, default="artifacts/bernoulli_transformer_report/figures")
    p.add_argument("--num-contexts", type=int, default=30)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--context-length", type=int, default=50)
    p.add_argument("--num-posterior-samples", type=int, default=200)
    p.add_argument("--rollout-length", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
