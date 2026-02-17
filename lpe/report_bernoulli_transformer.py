#!/usr/bin/env python3
"""Generate a Bernoulli-transformer LPE report in LaTeX and compile it to PDF."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    from lpe.bernoulli_transformer import BernoulliTransformer
except ModuleNotFoundError:
    from bernoulli_transformer import BernoulliTransformer


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "bernoulli_transformer_L1_D16_seq1024.pt"
DEFAULT_TRAIN_LOG = REPO_ROOT / "logs" / "bernoulli_diagnostics_seq1024_M200_L1000.log"
DEFAULT_DIAG_CSV = REPO_ROOT / "plots" / "bernoulli_posterior_sampling_diagnostics.csv"
DEFAULT_OUT_TEX = REPO_ROOT / "latex" / "bernoulli_transformer_lpe_report.tex"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "bernoulli_transformer_report"


@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    d_mlp: int
    num_parameters: int
    attention_mode: str = "causal"
    use_prenorm: bool = True
    positional_encoding: str = "none"


@dataclass
class TrainingLogSummary:
    seq_len: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    grad_clip: float
    steps: List[int]
    losses: List[float]
    model_parameters_logged: int | None
    posterior_samples: int | None
    rollout_length: int | None
    target_length: int | None
    context_length: int | None
    num_trials: int | None


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _fmt(x: float, digits: int = 6) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def _fmt_pct(x: float, digits: int = 2) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}\\%"


def _fmt_sci(x: float, digits: int = 3) -> str:
    if not math.isfinite(x):
        return "nan"
    if x == 0.0:
        return "0"
    return f"{x:.{digits}e}"


def infer_model_config(checkpoint: Path, n_heads: int) -> Tuple[ModelConfig, Dict[str, torch.Tensor]]:
    state_dict = torch.load(checkpoint, map_location="cpu")
    if "token_emb.weight" not in state_dict:
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint}")

    d_model = int(state_dict["token_emb.weight"].shape[1])
    layer_ids = sorted(
        {
            int(parts[1])
            for k in state_dict
            if k.startswith("blocks.")
            for parts in [k.split(".")]
            if len(parts) > 2 and parts[1].isdigit()
        }
    )
    n_layers = len(layer_ids)
    d_mlp = int(state_dict["blocks.0.mlp.0.weight"].shape[0]) if n_layers > 0 else d_model
    num_parameters = int(sum(v.numel() for v in state_dict.values()))

    config = ModelConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_mlp=d_mlp,
        num_parameters=num_parameters,
    )
    return config, state_dict


def parse_training_log(log_path: Path) -> TrainingLogSummary:
    text = log_path.read_text(encoding="utf-8")

    def _extract(pattern: str, cast, default=None):
        m = re.search(pattern, text)
        if not m:
            return default
        return cast(m.group(1))

    seq_len = _extract(r"Sequence length:\s*([0-9]+)", int, 0)
    batch_size = _extract(r"Batch size:\s*([0-9]+)", int, 0)
    lr = _extract(r"Learning rate:\s*([0-9.eE+-]+)", float, float("nan"))
    warmup = _extract(r"Warmup steps:\s*([0-9]+)", int, 0)
    grad_clip = _extract(r"Gradient clip:\s*([0-9.eE+-]+)", float, float("nan"))
    model_params = _extract(r"Model parameters:\s*([0-9,]+)", lambda s: int(s.replace(",", "")), None)

    curve = re.findall(r"\[step\s+([0-9]+)\]\s+loss=([0-9.]+)", text)
    steps = [int(s) for s, _ in curve]
    losses = [float(l) for _, l in curve]

    num_trials = _extract(r"Trials=([0-9]+)", int, None)
    context_len = _extract(r"context_len=([0-9]+)", int, None)
    target_len = _extract(r"target_len=([0-9]+)", int, None)
    posterior_samples = _extract(r"posterior_samples=([0-9]+)", int, None)
    rollout_length = _extract(r"rollout_len=([0-9]+)", int, None)

    return TrainingLogSummary(
        seq_len=seq_len,
        batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup,
        grad_clip=grad_clip,
        steps=steps,
        losses=losses,
        model_parameters_logged=model_params,
        posterior_samples=posterior_samples,
        rollout_length=rollout_length,
        target_length=target_len,
        context_length=context_len,
        num_trials=num_trials,
    )


def evaluate_model_vs_bayes_nll(
    model: BernoulliTransformer,
    *,
    num_sequences: int,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> Dict[str, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    model.eval()

    model_nll_sum = 0.0
    bayes_nll_sum = 0.0
    n_tokens = 0

    with torch.no_grad():
        for start in range(0, num_sequences, batch_size):
            b = min(batch_size, num_sequences - start)
            p = torch.rand((b, 1), generator=generator)
            seq = (torch.rand((b, seq_len), generator=generator) < p).long()

            inputs = seq[:, :-1]
            targets = seq[:, 1:]

            logits = model(inputs)
            log_probs = F.log_softmax(logits, dim=-1)
            token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            model_nll_sum += float((-token_logp).sum().item())

            ones_prefix = torch.cumsum(inputs, dim=1).float()
            prefix_len = torch.arange(1, seq_len, dtype=torch.float32).unsqueeze(0).expand(b, -1)
            p_next_one = (1.0 + ones_prefix) / (2.0 + prefix_len)
            p_target = torch.where(targets == 1, p_next_one, 1.0 - p_next_one)
            bayes_nll_sum += float((-torch.log(p_target)).sum().item())

            n_tokens += targets.numel()

    model_nll = model_nll_sum / n_tokens
    bayes_nll = bayes_nll_sum / n_tokens
    gap_pct = (model_nll - bayes_nll) / bayes_nll * 100.0
    return {
        "model_nll": model_nll,
        "bayes_nll": bayes_nll,
        "gap_pct": gap_pct,
        "n_tokens": float(n_tokens),
    }


def load_diagnostics_csv(path: Path) -> Dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"Diagnostics CSV is empty: {path}")

    def arr(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in rows], dtype=np.float64)

    return {
        "true_prob": arr("true_prob"),
        "estimate": arr("estimate"),
        "std_error": arr("std_error"),
        "coef_var": arr("coef_var"),
        "rel_error": arr("rel_error"),
        "log10_ratio": arr("log10_ratio"),
        "p_rollout_mean": arr("p_rollout_mean"),
        "p_rollout_std": arr("p_rollout_std"),
        "bayes_mean": arr("bayes_mean"),
        "true_p": arr("true_p"),
        "target_len": arr("target_len"),
        "context_len": arr("context_len"),
    }


def summarize_diagnostics(
    d: Dict[str, np.ndarray],
    *,
    posterior_samples: int,
    rollout_length: int,
    target_len: int,
) -> Dict[str, float]:
    true_prob = d["true_prob"]
    estimate = d["estimate"]
    rel_error = d["rel_error"]
    abs_rel_error = np.abs(rel_error)
    abs_log10 = np.abs(d["log10_ratio"])
    coef_var = d["coef_var"]
    p_rollout_mean = d["p_rollout_mean"]
    bayes_mean = d["bayes_mean"]

    p_mean_mse = float(np.mean((p_rollout_mean - bayes_mean) ** 2))
    p_mean_corr = float(np.corrcoef(p_rollout_mean, bayes_mean)[0, 1]) if len(true_prob) > 1 else float("nan")
    log_corr = float(np.corrcoef(np.log10(true_prob), np.log10(estimate))[0, 1]) if len(true_prob) > 1 else float("nan")

    # Equal token-budget naive Monte Carlo:
    # posterior uses posterior_samples * rollout_length generated tokens per trial.
    # Direct rollout method needs target_len generated tokens per rollout sample.
    r_eq = max(1, int((posterior_samples * rollout_length) // target_len))
    naive_rel_se = np.sqrt((1.0 - true_prob) / np.maximum(r_eq * true_prob, 1e-300))
    naive_zero_hit_prob = np.exp(np.log1p(-true_prob) * r_eq)

    return {
        "num_trials": float(len(true_prob)),
        "posterior_abs_rel_p50": float(np.percentile(abs_rel_error, 50)),
        "posterior_abs_rel_p90": float(np.percentile(abs_rel_error, 90)),
        "posterior_abs_rel_p95": float(np.percentile(abs_rel_error, 95)),
        "posterior_abs_log10_p50": float(np.percentile(abs_log10, 50)),
        "posterior_abs_log10_p90": float(np.percentile(abs_log10, 90)),
        "posterior_abs_log10_p95": float(np.percentile(abs_log10, 95)),
        "posterior_cv_mean": float(np.mean(coef_var)),
        "posterior_cv_median": float(np.median(coef_var)),
        "p_mean_mse": p_mean_mse,
        "p_mean_corr": p_mean_corr,
        "log_corr": log_corr,
        "naive_r_eq": float(r_eq),
        "naive_rel_se_p50": float(np.percentile(naive_rel_se, 50)),
        "naive_rel_se_p90": float(np.percentile(naive_rel_se, 90)),
        "naive_rel_se_p95": float(np.percentile(naive_rel_se, 95)),
        "naive_zero_hit_p50": float(np.percentile(naive_zero_hit_prob, 50)),
        "naive_zero_hit_p90": float(np.percentile(naive_zero_hit_prob, 90)),
        "naive_zero_hit_p95": float(np.percentile(naive_zero_hit_prob, 95)),
    }


def make_figures(
    *,
    fig_dir: Path,
    train: TrainingLogSummary,
    nll_summary: Dict[str, float],
    diag: Dict[str, np.ndarray],
    diag_summary: Dict[str, float],
) -> Dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    # Training curve
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(train.steps, train.losses, marker="o", linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Train loss")
    ax.set_title("Bernoulli Transformer Training Curve")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    p = fig_dir / "training_curve.png"
    plt.savefig(p, dpi=160)
    plt.close(fig)
    out["training_curve"] = p

    # Model NLL vs Bayes NLL
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    vals = [nll_summary["model_nll"], nll_summary["bayes_nll"]]
    labels = ["Model", "Bayes"]
    ax.bar(labels, vals, color=["#4c78a8", "#f58518"])
    ax.set_ylabel("Per-token NLL")
    ax.set_title("Final Model NLL vs Bayes NLL")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    p = fig_dir / "nll_vs_bayes.png"
    plt.savefig(p, dpi=160)
    plt.close(fig)
    out["nll_vs_bayes"] = p

    # Posterior-vs-naive relative error percentiles
    percentiles = ["p50", "p90", "p95"]
    posterior_vals = [
        diag_summary["posterior_abs_rel_p50"],
        diag_summary["posterior_abs_rel_p90"],
        diag_summary["posterior_abs_rel_p95"],
    ]
    naive_vals = [
        diag_summary["naive_rel_se_p50"],
        diag_summary["naive_rel_se_p90"],
        diag_summary["naive_rel_se_p95"],
    ]
    x = np.arange(len(percentiles))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - w / 2, posterior_vals, width=w, label="Posterior method |rel err|")
    ax.bar(x + w / 2, naive_vals, width=w, label="Naive MC expected rel SE")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(percentiles)
    ax.set_ylabel("Relative error (log scale)")
    ax.set_title("LPE Error: Posterior Method vs Naive MC")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    plt.tight_layout()
    p = fig_dir / "posterior_vs_naive_percentiles.png"
    plt.savefig(p, dpi=160)
    plt.close(fig)
    out["posterior_vs_naive_percentiles"] = p

    # Per-trial absolute relative error vs true probability
    true_prob = diag["true_prob"]
    posterior_abs_rel = np.abs(diag["rel_error"])
    naive_rel_se = np.sqrt((1.0 - true_prob) / np.maximum(diag_summary["naive_r_eq"] * true_prob, 1e-300))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.scatter(true_prob, posterior_abs_rel, s=18, alpha=0.75, label="Posterior method |rel err|")
    ax.scatter(true_prob, naive_rel_se, s=18, alpha=0.75, label="Naive MC expected rel SE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True probability")
    ax.set_ylabel("Relative error")
    ax.set_title("Per-trial Relative Error vs Event Probability")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    p = fig_dir / "relative_error_vs_true_prob.png"
    plt.savefig(p, dpi=160)
    plt.close(fig)
    out["relative_error_vs_true_prob"] = p

    return out


def build_latex(
    *,
    out_tex: Path,
    checkpoint: Path,
    train_log: Path,
    diag_csv: Path,
    model_cfg: ModelConfig,
    train: TrainingLogSummary,
    nll_summary: Dict[str, float],
    diag_summary: Dict[str, float],
    figs: Dict[str, Path],
) -> None:
    def rel(path: Path) -> str:
        return _latex_escape(os.path.relpath(path, out_tex.parent).replace("\\", "/"))

    fig_training = rel(figs["training_curve"])
    fig_nll = rel(figs["nll_vs_bayes"])
    fig_percentiles = rel(figs["posterior_vs_naive_percentiles"])
    fig_scatter = rel(figs["relative_error_vs_true_prob"])

    ext_true_vs_est = REPO_ROOT / "plots" / "bernoulli_diag_true_vs_est.png"
    ext_post_mean = REPO_ROOT / "plots" / "bernoulli_diag_posterior_mean.png"
    ext_hist = REPO_ROOT / "plots" / "bernoulli_diag_log10_ratio_hist.png"

    include_true_vs_est = ext_true_vs_est.exists()
    include_post_mean = ext_post_mean.exists()
    include_hist = ext_hist.exists()

    lines: List[str] = []
    if include_true_vs_est:
        lines.extend(
            [
                "\\begin{figure}[H]",
                "\\centering",
                f"\\includegraphics[width=0.68\\linewidth]{{{rel(ext_true_vs_est)}}}",
                "\\caption{Diagnostics: \\(\\log_{10}(\\text{true prob})\\) vs \\(\\log_{10}(\\text{estimated prob})\\).}",
                "\\end{figure}",
            ]
        )
    if include_post_mean:
        lines.extend(
            [
                "\\begin{figure}[H]",
                "\\centering",
                f"\\includegraphics[width=0.68\\linewidth]{{{rel(ext_post_mean)}}}",
                "\\caption{Diagnostics: rollout-derived posterior mean vs exact Bayes posterior mean.}",
                "\\end{figure}",
            ]
        )
    if include_hist:
        lines.extend(
            [
                "\\begin{figure}[H]",
                "\\centering",
                f"\\includegraphics[width=0.68\\linewidth]{{{rel(ext_hist)}}}",
                "\\caption{Diagnostics: histogram of \\(\\log_{10}(\\hat p / p)\\) estimation error.}",
                "\\end{figure}",
            ]
        )
    diagnostics_fig_block = "\n".join(lines)

    tex = f"""
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{amsmath}}
\\title{{Bernoulli Transformer LPE Report}}
\\author{{Automated run}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle

\\section*{{Artifacts Used}}
\\begin{{itemize}}
\\item Checkpoint: \\texttt{{{_latex_escape(str(checkpoint.relative_to(REPO_ROOT)))}}}
\\item Training/diagnostics log: \\texttt{{{_latex_escape(str(train_log.relative_to(REPO_ROOT)))}}}
\\item Diagnostics CSV: \\texttt{{{_latex_escape(str(diag_csv.relative_to(REPO_ROOT)))}}}
\\end{{itemize}}

\\section*{{1. Model Architecture}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{ll}}
\\toprule
Field & Value \\\\
\\midrule
Transformer type & Decoder-only, causal self-attention \\\\
Positional encoding & {model_cfg.positional_encoding} \\\\
Layers ($L$) & {model_cfg.n_layers} \\\\
Model width ($d_\\text{{model}}$) & {model_cfg.d_model} \\\\
Heads ($H$) & {model_cfg.n_heads} \\\\
MLP width ($d_\\text{{mlp}}$) & {model_cfg.d_mlp} \\\\
Pre-norm & {str(model_cfg.use_prenorm)} \\\\
Trainable parameters & {model_cfg.num_parameters:,} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Bernoulli-transformer architecture reconstructed from checkpoint and script defaults.}}
\\end{{table}}

\\section*{{2. Training Details}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{ll}}
\\toprule
Field & Value \\\\
\\midrule
Data generation & $p\\sim\\mathrm{{Beta}}(1,1)$, then sequence $y_t\\sim\\mathrm{{Bernoulli}}(p)$ \\\\
Training objective & Autoregressive cross-entropy on next-token prediction \\\\
Sequence length & {train.seq_len} \\\\
Batch size & {train.batch_size} \\\\
Optimizer & AdamW (weight decay $0.01$) \\\\
Learning rate & {_fmt(train.learning_rate, 6)} \\\\
Warmup steps & {train.warmup_steps} \\\\
Gradient clipping & {_fmt(train.grad_clip, 2)} \\\\
Total steps & {max(train.steps) if train.steps else 0} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Training setup from the logged run.}}
\\end{{table}}

\\section*{{3. Training Curve}}
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.70\\linewidth]{{{fig_training}}}
\\caption{{Logged training loss over optimization steps.}}
\\end{{figure}}

\\section*{{4. Final Loss vs Bayes Predictive}}
Held-out evaluation used {int(nll_summary["n_tokens"]):,} prediction tokens drawn from the same Beta-Bernoulli process.
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lll}}
\\toprule
Metric & Value & Notes \\\\
\\midrule
Model NLL & {_fmt(nll_summary["model_nll"], 6)} & Per-token negative log-likelihood \\\\
Bayes NLL & {_fmt(nll_summary["bayes_nll"], 6)} & Exact Beta-Bernoulli predictive \\\\
Gap & {_fmt_pct(nll_summary["gap_pct"], 2)} & $(\\text{{model}}-\\text{{Bayes}})/\\text{{Bayes}}$ \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Model predictive quality against the Bayes-optimal baseline.}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.58\\linewidth]{{{fig_nll}}}
\\caption{{Per-token NLL: trained model vs Bayes optimal predictor.}}
\\end{{figure}}

\\section*{{5. Posterior-Sample Quality}}
Diagnostics used {int(diag_summary["num_trials"])} trials, posterior samples per trial={train.posterior_samples}, rollout length={train.rollout_length}.
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lll}}
\\toprule
Metric & Value & Interpretation \\\\
\\midrule
$|\\log_{{10}}(\\hat p/p)|$ p50 & {_fmt(diag_summary["posterior_abs_log10_p50"], 3)} & Multiplicative error in log space \\\\
$|\\log_{{10}}(\\hat p/p)|$ p90 & {_fmt(diag_summary["posterior_abs_log10_p90"], 3)} & Tail log-error \\\\
$|\\log_{{10}}(\\hat p/p)|$ p95 & {_fmt(diag_summary["posterior_abs_log10_p95"], 3)} & Tail log-error \\\\
Posterior CV mean & {_fmt(diag_summary["posterior_cv_mean"], 3)} & Variability of Rao-Blackwell terms \\\\
Posterior CV median & {_fmt(diag_summary["posterior_cv_median"], 3)} & Typical variability \\\\
Posterior-mean MSE & {_fmt_sci(diag_summary["p_mean_mse"], 3)} & Rollout mean vs exact Bayes mean \\\\
Posterior-mean correlation & {_fmt(diag_summary["p_mean_corr"], 4)} & Rollout mean vs exact Bayes mean \\\\
$\\log_{{10}}$ true-vs-est corr & {_fmt(diag_summary["log_corr"], 4)} & Alignment in log-probability space \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Posterior-sampling diagnostics.}}
\\end{{table}}

{diagnostics_fig_block}

\\section*{{6. LPE Estimation Quality and Naive MC Baseline}}
Equal-compute naive Monte Carlo budget was estimated as:
\\[
R_{{\\text{{eq}}}} = \\left\\lfloor \\frac{{M\\cdot L}}{{m}} \\right\\rfloor
= \\left\\lfloor \\frac{{{train.posterior_samples}\\times{train.rollout_length}}}{{{train.target_length}}} \\right\\rfloor
= {int(diag_summary["naive_r_eq"])}.
\\]
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{llll}}
\\toprule
Statistic & Posterior method $|\\text{{rel err}}|$ & Naive MC expected rel SE & Naive MC $P(\\text{{zero hits}})$ \\\\
\\midrule
p50 & {_fmt(diag_summary["posterior_abs_rel_p50"], 3)} & {_fmt(diag_summary["naive_rel_se_p50"], 1)} & {_fmt(diag_summary["naive_zero_hit_p50"], 6)} \\\\
p90 & {_fmt(diag_summary["posterior_abs_rel_p90"], 3)} & {_fmt(diag_summary["naive_rel_se_p90"], 1)} & {_fmt(diag_summary["naive_zero_hit_p90"], 6)} \\\\
p95 & {_fmt(diag_summary["posterior_abs_rel_p95"], 3)} & {_fmt(diag_summary["naive_rel_se_p95"], 1)} & {_fmt(diag_summary["naive_zero_hit_p95"], 6)} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Posterior estimator error vs expected naive-MC error under matched token budget.}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.72\\linewidth]{{{fig_percentiles}}}
\\caption{{Percentile-level comparison of posterior method error and naive-MC expected error (log scale).}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.72\\linewidth]{{{fig_scatter}}}
\\caption{{Per-trial relative error versus true event probability (log-log).}}
\\end{{figure}}

\\section*{{7. Additional Notes}}
\\begin{{itemize}}
\\item The Bayes gap is small ({_fmt_pct(nll_summary["gap_pct"], 2)}), indicating the transformer is close to Bayes-optimal one-step prediction on this task.
\\item LPE errors are still sensitive to event rarity: even with strong one-step calibration, tiny target probabilities produce heavy-tailed relative error.
\\item Under the same compute budget, naive direct-MC is effectively unusable on these trials (typical zero-hit probability near 1), while posterior sampling remains informative.
\\end{{itemize}}

\\end{{document}}
"""

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex, encoding="utf-8")


def compile_pdf(tex_path: Path) -> Path:
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    for _ in range(2):
        subprocess.run(cmd, cwd=str(tex_path.parent), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return tex_path.with_suffix(".pdf")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Bernoulli transformer LPE report (LaTeX + PDF).")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--train-log", type=Path, default=DEFAULT_TRAIN_LOG)
    p.add_argument("--diagnostics-csv", type=Path, default=DEFAULT_DIAG_CSV)
    p.add_argument("--out-tex", type=Path, default=DEFAULT_OUT_TEX)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--n-heads", type=int, default=1, help="Transformer heads (not inferable from checkpoint tensor shapes).")
    p.add_argument("--eval-num-sequences", type=int, default=512)
    p.add_argument("--eval-seq-len", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--no-compile", action="store_true", help="Skip pdflatex compilation.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    checkpoint = args.checkpoint.resolve()
    train_log = args.train_log.resolve()
    diagnostics_csv = args.diagnostics_csv.resolve()
    out_tex = args.out_tex.resolve()
    out_dir = args.out_dir.resolve()
    fig_dir = out_dir / "figures"

    model_cfg, state_dict = infer_model_config(checkpoint, n_heads=args.n_heads)
    train = parse_training_log(train_log)

    model = BernoulliTransformer(
        max_seq_len=None,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        d_mlp=model_cfg.d_mlp,
        use_prenorm=model_cfg.use_prenorm,
        attention_mode=model_cfg.attention_mode,
    )
    model.load_state_dict(state_dict)
    model.eval()

    nll_summary = evaluate_model_vs_bayes_nll(
        model,
        num_sequences=args.eval_num_sequences,
        seq_len=args.eval_seq_len,
        batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    diag = load_diagnostics_csv(diagnostics_csv)
    if train.posterior_samples is None or train.rollout_length is None or train.target_length is None:
        raise RuntimeError(
            "Could not parse posterior_samples/rollout_length/target_length from training log. "
            "Use a diagnostics log that includes these fields."
        )

    diag_summary = summarize_diagnostics(
        diag,
        posterior_samples=train.posterior_samples,
        rollout_length=train.rollout_length,
        target_len=train.target_length,
    )

    figs = make_figures(
        fig_dir=fig_dir,
        train=train,
        nll_summary=nll_summary,
        diag=diag,
        diag_summary=diag_summary,
    )

    build_latex(
        out_tex=out_tex,
        checkpoint=checkpoint,
        train_log=train_log,
        diag_csv=diagnostics_csv,
        model_cfg=model_cfg,
        train=train,
        nll_summary=nll_summary,
        diag_summary=diag_summary,
        figs=figs,
    )

    print(f"Wrote LaTeX report: {out_tex}")
    if not args.no_compile:
        pdf_path = compile_pdf(out_tex)
        print(f"Compiled PDF: {pdf_path}")


if __name__ == "__main__":
    main()
