#!/usr/bin/env python3
"""Build combined transformer-vs-Bayes report for k=1,2,3 (balanced target, ctx=100)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

TRANSFORMER_ROOT = REPO_ROOT / "artifacts" / "markov_k123_transformer_500_ctx100_balanced"
BAYES_ROOT = REPO_ROOT / "artifacts" / "markov_k123_bayes_baseline_500_ctx100_balanced"

TRAINING_SUMMARIES = {
    1: REPO_ROOT / "artifacts" / "markov_k12_step1_pos" / "k1" / "summary.json",
    2: REPO_ROOT / "artifacts" / "markov_k12_step1_pos" / "k2" / "summary.json",
    3: REPO_ROOT / "artifacts" / "markov_k3_step1_pos" / "k3" / "summary.json",
}

OUT_TEX = REPO_ROOT / "latex" / "markov_k123_balanced_ctx100_report.tex"


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: float, digits: int = 6) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def _fmt_sci(v: float) -> str:
    if not math.isfinite(v):
        return "nan"
    if v == 0.0:
        return "0"
    if abs(v) < 1e-3 or abs(v) >= 1e4:
        return f"{v:.3e}"
    return f"{v:.6f}"


def _latex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def build() -> None:
    ks = [1, 2, 3]

    transformer = {
        k: _read_json(TRANSFORMER_ROOT / f"k{k}" / "summary.json")
        for k in ks
    }
    bayes = {
        k: _read_json(BAYES_ROOT / f"k{k}" / "summary.json")
        for k in ks
    }
    training = {k: _read_json(TRAINING_SUMMARIES[k]) for k in ks}

    fig_dir = TRANSFORMER_ROOT / "figures_combined"
    fig_dir.mkdir(parents=True, exist_ok=True)

    step2_t = [float(transformer[k]["step2_posterior_mae"]) for k in ks]
    step2_b = [float(bayes[k]["step2_posterior_mae"]) for k in ks]
    step3_t = [float(transformer[k]["step3_lpe_rel_error_median_pct"]) for k in ks]
    step3_b = [float(bayes[k]["step3_lpe_rel_error_median_pct"]) for k in ks]

    x = np.arange(len(ks))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(x - width / 2, step2_t, width=width, label="Transformer")
    ax.bar(x + width / 2, step2_b, width=width, label="True Bayes predictive")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k")
    ax.set_ylabel("Posterior MAE")
    ax.set_title("Step 2 Posterior MAE: Transformer vs True Bayes")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    plt.tight_layout()
    step2_fig = fig_dir / "step2_transformer_vs_bayes.png"
    plt.savefig(step2_fig, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(x - width / 2, step3_t, width=width, label="Transformer")
    ax.bar(x + width / 2, step3_b, width=width, label="True Bayes predictive")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k")
    ax.set_ylabel("Median relative error (%) [log scale]")
    ax.set_title("Step 3 Median Relative Error: Transformer vs True Bayes")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    plt.tight_layout()
    step3_fig = fig_dir / "step3_transformer_vs_bayes.png"
    plt.savefig(step3_fig, dpi=150)
    plt.close(fig)

    # Training-step summary plot (not a full learning curve).
    steps_run = [int(training[k].get("train_stats", {}).get("steps_run", 0)) for k in ks]
    best_steps = [int(training[k].get("train_stats", {}).get("best_step", 0)) for k in ks]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(ks, steps_run, marker="o", label="steps_run")
    ax.plot(ks, best_steps, marker="s", label="best_step")
    ax.set_xlabel("k")
    ax.set_ylabel("Training steps")
    ax.set_title("Training Step Summary")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    train_steps_fig = fig_dir / "training_steps_summary.png"
    plt.savefig(train_steps_fig, dpi=150)
    plt.close(fig)

    result_rows: List[str] = []
    for k in ks:
        t = transformer[k]
        b = bayes[k]
        result_rows.append(
            " & ".join(
                [
                    str(k),
                    _fmt(float(t["step1_eval_model_nll"])),
                    _fmt(float(t["step1_eval_bayes_nll"])),
                    _fmt(float(t["step1_gap_pct"]), 3),
                    _fmt_sci(float(t["step2_posterior_mae"])),
                    _fmt_sci(float(b["step2_posterior_mae"])),
                    _fmt_sci(float(t["step3_lpe_rel_error_median_pct"])),
                    _fmt_sci(float(b["step3_lpe_rel_error_median_pct"])),
                ]
            )
            + " \\\\"
        )

    train_rows: List[str] = []
    for k in ks:
        ts = training[k]
        m = ts["model_config"]
        st = ts.get("train_stats", {})
        train_rows.append(
            " & ".join(
                [
                    str(k),
                    f"L{int(m['n_layers'])}/D{int(m['d_model'])}/H{int(m['n_heads'])}/M{int(m['d_mlp'])}",
                    str(int(ts["batch_size"])),
                    str(int(ts["train_seq_len"])),
                    str(int(st.get("steps_run", 0))),
                    str(int(st.get("best_step", 0))),
                    _fmt(float(st.get("best_gap_ratio", float("nan"))) * 100.0, 3),
                    _fmt(float(st.get("final_train_loss", float("nan"))), 6),
                ]
            )
            + " \\\\"
        )

    rel_step2 = os.path.relpath(step2_fig, OUT_TEX.parent).replace("\\", "/")
    rel_step3 = os.path.relpath(step3_fig, OUT_TEX.parent).replace("\\", "/")
    rel_train = os.path.relpath(train_steps_fig, OUT_TEX.parent).replace("\\", "/")

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\title{{Markov-$k$ LPE Report (k=1,2,3): Balanced Target, Long Context}}
\author{{Automated run}}
\date{{\today}}
\begin{{document}}
\maketitle

\section*{{Setup}}
\begin{{itemize}}
\item Context length fixed to 100 for Step 2/3.
\item Target string mode: \texttt{{balanced}} (de Bruijn-style state-balanced construction).
\item Target length: 100.
\item Posterior samples per estimate: 500.
\item Rollout length per $k$: $100 \cdot 2^k$.
\item Transformer checkpoints reused from earlier Step-1 training (no retraining in this rerun).
\end{{itemize}}

\section*{{Transformer vs True Bayes Predictive}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{rrrrrrrr}}
\toprule
$k$ & Model NLL & Bayes NLL & Gap(\%) & Step2 MAE (T) & Step2 MAE (B) & Step3 MedErr\% (T) & Step3 MedErr\% (B) \\
\midrule
{os.linesep.join(result_rows)}
\bottomrule
\end{{tabular}}
\caption{{Side-by-side metrics for transformer (T) and true Bayes predictive baseline (B).}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\linewidth]{{{_latex_escape(rel_step2)}}}
\caption{{Step 2 posterior MAE comparison.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\linewidth]{{{_latex_escape(rel_step3)}}}
\caption{{Step 3 median relative-error comparison (log scale).}}
\end{{figure}}

\section*{{Training Details (Checkpoint Runs)}}
Transformer hyperparameters per $k$:
\begin{{table}}[H]
\centering
\begin{{tabular}}{{rrrrrrrr}}
\toprule
$k$ & Model (L/D/H/M) & Batch & SeqLen & StepsRun & BestStep & BestGap(\%) & FinalTrainLoss \\
\midrule
{os.linesep.join(train_rows)}
\bottomrule
\end{{tabular}}
\caption{{Training summary from Step-1 checkpoint runs used in this report.}}
\end{{table}}

Optimization process (from training script defaults/recorded setup):
\begin{{itemize}}
\item Optimizer: AdamW, learning rate $3\times10^{{-4}}$, weight decay $0.01$.
\item Gradient clipping: $1.0$; gradient accumulation: $1$.
\item Token-budget-driven steps with early stopping on Bayes-gap target.
\item Evaluation cadence during training: periodic held-out NLL checks vs Bayes-optimal predictor.
\item Training sequence length matched rollout length ($100\cdot 2^k$).
\end{{itemize}}

\section*{{Learning-Curve Note}}
Full per-step learning curves were not persisted in the original checkpoint artifacts.
Available training progress fields include \texttt{{steps\_run}}, \texttt{{best\_step}},
\texttt{{final\_train\_loss}}, and final/selected eval metrics.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\linewidth]{{{_latex_escape(rel_train)}}}
\caption{{Training-step summary (available checkpoint metadata; not a full loss curve).}}
\end{{figure}}

\end{{document}}
"""

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex, encoding="utf-8")

    print(f"Wrote report TeX: {OUT_TEX}")


def main() -> None:
    build()


if __name__ == "__main__":
    main()

