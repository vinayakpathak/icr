#!/usr/bin/env python3
"""Build a k=2 Markov transformer report in the same style as the k=1 report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lpe.markov_k_transformer import ModelConfig, build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Markov-k2 transformer report.")
    p.add_argument("--summary-path", type=str, required=True)
    p.add_argument("--section4-dir", type=str, required=True)
    p.add_argument("--posterior-dir", type=str, required=True)
    p.add_argument("--step3-dir", type=str, required=True)
    p.add_argument("--out-tex", type=str, default="latex/markov_k2_transformer_lpe_report.tex")
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_keyvals(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out


def fmt(v: float, digits: int = 6) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def fmt_pct(v: float, digits: int = 2) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}\\%"


def fmt_sci(v: float, digits: int = 3) -> str:
    if not math.isfinite(v):
        return "nan"
    if v == 0.0:
        return "0"
    return f"{v:.{digits}e}"


def fmt_sci_latex(v: float, digits: int = 2) -> str:
    if not math.isfinite(v):
        return "nan"
    if v == 0.0:
        return "0"
    s = f"{v:.{digits}e}"
    mant, exp = s.split("e")
    return f"{mant}\\times 10^{{{int(exp)}}}"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def rel_path(out_tex: Path, path: Path) -> str:
    return latex_escape(os.path.relpath(path, out_tex.parent).replace("\\", "/"))


def rel_repo(path: Path) -> str:
    p = path.resolve()
    try:
        return latex_escape(str(p.relative_to(REPO_ROOT)))
    except ValueError:
        return latex_escape(str(p))


def parse_train_command(train_log_path: Path) -> Dict[str, str]:
    text = train_log_path.read_text(encoding="utf-8")
    cmd_line = ""
    for line in text.splitlines():
        if line.startswith("/") and "markov_k_transformer.py" in line:
            cmd_line = line.strip()
            break
    if not cmd_line:
        return {}

    def arg(name: str) -> str:
        m = re.search(rf"{re.escape(name)}\s+([^\s]+)", cmd_line)
        return m.group(1) if m else ""

    return {
        "batch_size": arg("--batch-size"),
        "grad_accum": arg("--grad-accum-steps"),
        "learning_rate": arg("--learning-rate"),
        "min_lr": arg("--min-lr"),
        "weight_decay": arg("--weight-decay"),
        "warmup_steps": arg("--warmup-steps"),
        "grad_clip": arg("--grad-clip"),
        "num_steps": arg("--num-steps"),
        "target_gap_pct": arg("--target-gap-pct"),
    }


def model_param_count(summary: Dict[str, object], device: torch.device) -> int:
    cfg_raw = summary["model_config"]
    cfg = ModelConfig(
        n_layers=int(cfg_raw["n_layers"]),
        d_model=int(cfg_raw["d_model"]),
        n_heads=int(cfg_raw["n_heads"]),
        d_mlp=int(cfg_raw["d_mlp"]),
    )
    max_seq_len = int(summary.get("max_seq_len", 4096))
    use_pos = bool(summary.get("use_positional_encoding", True))
    model = build_model(cfg, max_seq_len=max_seq_len, use_positional_encoding=use_pos).to(device)
    return int(sum(p.numel() for p in model.parameters()))


def metric_label(name: str) -> str:
    labels = {
        "wasserstein1": "Wasserstein-1",
        "ks_cdf": "KS(CDF)",
        "cvm_int": "CvM-int",
        "pit_ks": "PIT-KS",
        "pit_cvm": "PIT-CvM",
        "quantile_rmse": "Quantile RMSE",
        "coverage_mae": "Coverage MAE",
    }
    return labels.get(name, name)


def vector_string(row: Dict[str, str], prefix: str, n_states: int) -> str:
    vals: List[str] = []
    for s in range(n_states):
        key = f"{prefix}_state_{s}"
        vals.append(f"{float(row[key]):.3f}")
    return "(" + ",".join(vals) + ")"


def compile_pdf(tex_path: Path) -> Path:
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    for _ in range(2):
        subprocess.run(cmd, cwd=str(tex_path.parent), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return tex_path.with_suffix(".pdf")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_path = resolve_path(args.summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    k = int(summary["k"])
    if k != 2:
        raise ValueError(f"Expected k=2 summary, got k={k}.")

    attempt_dir = summary_path.parent.parent
    train_log_path = attempt_dir / "train.log"
    train_cfg = parse_train_command(train_log_path) if train_log_path.exists() else {}

    section4_dir = resolve_path(args.section4_dir)
    posterior_dir = resolve_path(args.posterior_dir)
    step3_dir = resolve_path(args.step3_dir)
    step3_k_dir = step3_dir / "k2"
    step3_fig_dir = step3_dir / "figures"

    section4_summary = read_keyvals(section4_dir / "section4_k2_nextprob_scatter_summary.txt")
    posterior_metric_rows = read_csv(posterior_dir / "posterior_context_metrics_summary_5x6_k2.csv")
    step3_summary = read_keyvals(step3_fig_dir / "section6_k2_summary_teacher_forcing.txt")
    step3_table_rows = read_csv(step3_k_dir / "step3_teacher_forcing_context_table.csv")
    step3_teacher_rows = read_csv(step3_k_dir / "step3_lpe_metrics_teacher_forcing.csv")
    target_info = json.loads((step3_k_dir / "step3_target_info.json").read_text(encoding="utf-8"))

    out_tex = resolve_path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    n_params = model_param_count(summary, device=device)

    cfg = summary["model_config"]
    steps_run = int(summary["train_stats"]["steps_run"])
    best_step = int(summary["train_stats"]["best_step"])
    early_gap = float(train_cfg.get("target_gap_pct", summary.get("quality_gate_required_gap_pct", 1.0)))
    lr_display = fmt_sci_latex(float(train_cfg["learning_rate"])) if train_cfg.get("learning_rate") else "2.5\\times 10^{-4}"
    min_lr_display = fmt_sci_latex(float(train_cfg["min_lr"])) if train_cfg.get("min_lr") else "2.0\\times 10^{-6}"

    metric_rows_tex: List[str] = []
    for r in posterior_metric_rows:
        metric_rows_tex.append(
            " & ".join(
                [
                    metric_label(r["metric"]),
                    fmt(float(r["mean"]), 4),
                    fmt(float(r["median"]), 4),
                    fmt(float(r["p90"]), 4),
                    fmt(float(r["p95"]), 4),
                    fmt(float(r["max"]), 4),
                    fmt(float(r["perfect_match_expected_mean"]), 4),
                ]
            )
            + r" \\"
        )

    n_states = 1 << k
    ctx_rows_tex: List[str] = []
    for r in sorted(step3_table_rows, key=lambda x: int(x["context_id"])):
        ctx_id = int(r["context_id"])
        ctx_rows_tex.append(
            " & ".join(
                [
                    str(ctx_id),
                    fmt_sci(float(r["teacher_forced_true_prob"]), 3),
                    fmt_sci(float(r["posterior_estimate"]), 3),
                    vector_string(r, "true_post_mean", n_states),
                    vector_string(r, "true_post_sd", n_states),
                    vector_string(r, "sampled_post_mean", n_states),
                    vector_string(r, "sampled_post_sd", n_states),
                ]
            )
            + r" \\"
        )

    p50_rel = float(step3_summary["p50_rel"])
    p90_rel = float(step3_summary["p90_rel"])
    p95_rel = float(step3_summary["p95_rel"])
    p50_se = float(step3_summary["p50_se"])
    p90_se = float(step3_summary["p90_se"])
    p95_se = float(step3_summary["p95_se"])
    p50_zero = float(step3_summary["p50_zero"])
    p90_zero = float(step3_summary["p90_zero"])
    p95_zero = float(step3_summary["p95_zero"])
    r_eq = int(step3_summary["R_eq"])

    step3_num_contexts = len(step3_teacher_rows)
    step3_context_len = int(float(step3_teacher_rows[0]["context_len"])) if step3_teacher_rows else 0
    step3_target_len = int(float(step3_teacher_rows[0]["target_len"])) if step3_teacher_rows else 0

    training_curve = resolve_path(str(summary["training_curve_plot"]))
    section4_fig = section4_dir / "section4_k2_nextprob_scatter_varlen.png"
    section5_grid = posterior_dir / "posterior_context_grid_5x6_k2.png"
    section5_box = posterior_dir / "posterior_context_metrics_boxplot_5x6_k2.png"
    section6_percentile = step3_fig_dir / "posterior_vs_naive_percentiles_k2_teacher_forcing.png"
    section6_scatter = step3_fig_dir / "relative_error_vs_true_prob_k2_teacher_forcing.png"

    tex = f"""
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{amsmath}}
\\title{{Markov-$k$ Transformer LPE Report (k=2)}}
\\author{{Automated run}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle

\\section*{{Artifacts Used}}
\\begin{{itemize}}
\\item Transformer artifacts: \\texttt{{{rel_repo(attempt_dir)}}}
\\item Report figures and diagnostics: \\texttt{{{rel_repo(posterior_dir)}}} and \\texttt{{{rel_repo(step3_dir / "figures")}}}
\\end{{itemize}}

\\section*{{1. Model Architecture}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{ll}}
\\toprule
Field & Value \\\\
\\midrule
Transformer type & Decoder-only, causal self-attention \\\\
Positional encoding & Learned absolute positional embeddings \\\\
Markov order & $k=2$ \\\\
Layers ($L$) & {int(cfg["n_layers"])} \\\\
Model width ($d_\\text{{model}}$) & {int(cfg["d_model"])} \\\\
Heads ($H$) & {int(cfg["n_heads"])} \\\\
MLP width ($d_\\text{{mlp}}$) & {int(cfg["d_mlp"])} \\\\
Pre-norm & True \\\\
Trainable parameters & {n_params:,} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{k=2 transformer architecture used for this report.}}
\\end{{table}}

\\section*{{2. Training Details}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{ll}}
\\toprule
Field & Value \\\\
\\midrule
Training objective & Autoregressive cross-entropy on next-token prediction \\\\
Sequence length & {int(summary["train_seq_len"])} \\\\
Batch size & {train_cfg.get("batch_size", str(summary.get("batch_size", "")))} \\\\
Gradient accumulation & {train_cfg.get("grad_accum", "8")} \\\\
Optimizer & AdamW \\\\
Learning rate & ${lr_display}$ (cosine decay to ${min_lr_display}$) \\\\
Weight decay & {train_cfg.get("weight_decay", "0.001")} \\\\
Warmup schedule & Linear warmup for {train_cfg.get("warmup_steps", "500")} steps \\\\
Gradient clipping & {train_cfg.get("grad_clip", "0.8")} \\\\
Total steps run & {steps_run} (early-stop target reached) \\\\
Early-stop target & Step-1 gap $\\le {fmt(early_gap, 2)}\\%$ \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Training setup for the high-capacity k=2 run.}}
\\end{{table}}

Data generation process (explicit Markov parameterization):
\\begin{{itemize}}
\\item General $k$-Markov parameterization:
\\[
\\theta_s \\equiv P(x_t=1\\mid x_{{t-k:t-1}}=s),\\qquad s\\in\\{{0,1\\}}^k.
\\]
\\item Prior over parameters: independent Beta per state
\\[
\\theta_s \\sim \\mathrm{{Beta}}(1,1)\\;\\;\\text{{independently for all }}s\\in\\{{0,1\\}}^k.
\\]
\\item Sequence generation for one training example of length $T={int(summary["train_seq_len"])}$:
\\[
x_1,\\dots,x_k \\overset{{\\text{{i.i.d.}}}}{{\\sim}}\\mathrm{{Bernoulli}}(0.5),
\\]
\\[
x_t\\mid x_{{t-k:t-1}}\\sim \\mathrm{{Bernoulli}}\\!\\left(\\theta_{{x_{{t-k:t-1}}}}\\right),\\qquad t\\ge k+1.
\\]
\\item For this report ($k=2$), the latent parameters are
\\[
\\theta_{{00}},\\theta_{{01}},\\theta_{{10}},\\theta_{{11}}\\overset{{\\text{{i.i.d.}}}}{{\\sim}}\\mathrm{{Beta}}(1,1),
\\]
with
\\[
P(x_t=1\\mid x_{{t-2:t-1}}=ab)=\\theta_{{ab}},\\qquad ab\\in\\{{00,01,10,11\\}}.
\\]
\\end{{itemize}}

\\section*{{3. Training Curve}}
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.72\\linewidth]{{{rel_path(out_tex, training_curve)}}}
\\caption{{Training curve for k=2. Training loss is logged every gradient step. Evaluation curves are drawn only at true eval checkpoints and linearly connected between checkpoints.}}
\\end{{figure}}

\\section*{{4. Final Loss vs Bayes Predictive}}
Held-out evaluation uses sequences from the same Beta-Bernoulli Markov-2 process.

For a prefix $x_{{1:t-1}}$ and current state $s_t\\in\\{{00,01,10,11\\}}$, with transition counts
$n_{{s,0}}, n_{{s,1}}$, the Bayes predictive is
\\[
P(x_t=1\\mid x_{{1:t-1}})=\\frac{{1+n_{{s_t,1}}}}{{2+n_{{s_t,0}}+n_{{s_t,1}}}}.
\\]
The per-token Bayes NLL is
\\[
\\ell_t^{{\\text{{Bayes}}}} = -\\big[x_t\\log q_t + (1-x_t)\\log(1-q_t)\\big],
\\]
where $q_t=P(x_t=1\\mid x_{{1:t-1}})$.

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lll}}
\\toprule
Metric & Value & Notes \\\\
\\midrule
Model NLL & {fmt(float(summary["step1_eval_model_nll"]), 6)} & Per-token negative log-likelihood \\\\
Bayes NLL & {fmt(float(summary["step1_eval_bayes_nll"]), 6)} & Exact Bayes predictive on held-out samples \\\\
Gap & {fmt_pct(float(summary["step1_gap_pct"]), 2)} & $(\\text{{model}}-\\text{{Bayes}})/\\text{{Bayes}}$ \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Model predictive quality against Bayes baseline (k=2).}}
\\end{{table}}

To compare next-token probabilities directly, we generated {int(section4_summary.get("n_contexts", 0))} mixed-length contexts:
\\begin{{itemize}}
\\item Sample context length uniformly from $[{int(section4_summary.get("len_min", 0))},{int(section4_summary.get("len_max", 0))}]$.
\\item Sample one latent Markov-2 process from the priors above and draw context bits.
\\item Compute Bayes and model $P(x_{{n+1}}=1\\mid x_{{1:n}})$.
\\end{{itemize}}
Summary: Pearson $r={fmt(float(section4_summary.get("pearson", float("nan"))), 4)}$, MAE $={fmt(float(section4_summary.get("mae", float("nan"))), 5)}$.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.72\\linewidth]{{{rel_path(out_tex, section4_fig)}}}
\\caption{{Model vs Bayes next-token probability on mixed-length contexts for k=2.}}
\\end{{figure}}

\\section*{{5. Posterior-Sample Quality}}
Diagnostics used 30 trials, posterior samples per trial=200, rollout length=1000.
For this section, each trial corresponds to one generated input context. Contexts were generated as:
\\begin{{itemize}}
\\item Sample latent transitions independently: $\\theta_{{00}},\\theta_{{01}},\\theta_{{10}},\\theta_{{11}}\\sim\\mathrm{{Beta}}(1,1)$.
\\item Sample one context of length 1000 from the Markov-2 process.
\\item Compute exact Bayes posterior marginals from context transition counts:
\\[
\\theta_s\\mid x_{{1:n}}\\sim\\mathrm{{Beta}}(n_{{s,1}}+1, n_{{s,0}}+1),\\quad s\\in\\{{00,01,10,11\\}}.
\\]
\\item Estimate model-implied posterior by 200 rollouts of length 1000 per context; each rollout yields one
$\\hat\\theta\\in[0,1]^4$ from continuation transition frequencies.
\\end{{itemize}}

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
Metric & Mean & Median & p90 & p95 & Max & Perfect-match expected mean \\\\
\\midrule
{os.linesep.join(metric_rows_tex)}
\\bottomrule
\\end{{tabular}}
\\caption{{Summary of posterior-distribution similarity metrics across 30 Markov-2 contexts, with finite-sample perfect-match baselines.}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.98\\linewidth]{{{rel_path(out_tex, section5_grid)}}}
\\caption{{Posterior-shape diagnostics for 30 contexts (5x6 grid). Histograms are model-implied samples by state; curves are true Bayes posterior marginals.}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.88\\linewidth]{{{rel_path(out_tex, section5_box)}}}
\\caption{{Boxplots of per-context posterior similarity metrics (30 values per metric, log scale).}}
\\end{{figure}}

For each metric, the reported context-level value is computed per state and then averaged, preserving the same metric family and table structure as the Bernoulli and k=1 reports while adapting to the 4-parameter posterior.

\\section*{{6. LPE Estimation Quality and Naive MC Baseline}}
For this k=2 LPE rerun (Step 3 only; same checkpoint as Sections 1--5):
\\begin{{itemize}}
\\item Number of contexts: {step3_num_contexts} (all length {step3_context_len}).
\\item Target string length: {step3_target_len}.
\\item Target mode: \\texttt{{balanced}} (de Bruijn-based construction).
\\item The same fixed target string is used for all {step3_num_contexts} contexts in this run.
\\item Posterior samples per context: $M=200$; rollout length $L=1000$.
\\end{{itemize}}

A binary de Bruijn cycle $B(2,n)$ is a cyclic bit string of length $2^n$ such that, with wrap-around, every length-$n$ binary substring appears exactly once. For k=2 we use order $n=k+1=3$, so the base cycle length is $2^3=8$.

Target generation matches \\texttt{{make\\_target\\_bits}}:
\\begin{{enumerate}}
\\item Build $B(2,3)$.
\\item Sample a random cyclic offset.
\\item Rotate, repeat, and truncate to length $m={step3_target_len}$.
\\end{{enumerate}}

True event probability is computed by teacher forcing under the trained transformer:
\\[
P_{{\\mathrm{{TF}}}}(y_{{1:m}}\\mid x_{{1:n}})
=\\prod_{{t=1}}^{{m}}P_{{\\mathrm{{model}}}}\\!\\left(y_t\\mid x_{{1:n}},y_{{1:t-1}}\\right).
\\]
Here $x_{{1:n}}$ is the observed context (length $n$) and $y_{{1:m}}$ is the fixed target (length $m$). We use this teacher-forced model probability as the ``true'' probability for LPE relative-error evaluation.

The rollout-posterior estimator is
\\[
\\widehat P_{{\\text{{post}}}}(y_{{1:m}}\\mid x_{{1:n}})
=\\frac1M\\sum_{{j=1}}^M f(\\hat\\theta^{{(j)}}),
\\]
where
\\[
f(\\theta)\\equiv P_{{\\theta}}(y_{{1:m}}\\mid x_{{1:n}})
=\\prod_{{t=1}}^{{m}}\\theta_{{s_t}}^{{\\,y_t}}\\bigl(1-\\theta_{{s_t}}\\bigr)^{{1-y_t}}.
\\]

Equal-compute naive MC uses
\\[
R_{{\\mathrm{{eq}}}}=\\left\\lfloor\\frac{{ML}}{{m}}\\right\\rfloor
=\\left\\lfloor\\frac{{200\\times 1000}}{{{step3_target_len}}}\\right\\rfloor={r_eq},
\\]
and
\\[
\\text{{relative SE}}_{{\\text{{naive}}}}=\\sqrt{{\\frac{{1-P}}{{R_{{\\mathrm{{eq}}}}P}}}},
\\qquad
P(\\text{{zero hits}})=(1-P)^{{R_{{\\mathrm{{eq}}}}}}.
\\]

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{llll}}
\\toprule
Statistic & Posterior method $|\\text{{rel err}}|$ & Naive MC expected rel SE & Naive MC $P(\\text{{zero hits}})$ \\\\
\\midrule
p50 & ${fmt_sci_latex(p50_rel, 2)}$ & ${fmt_sci_latex(p50_se, 2)}$ & {fmt(p50_zero, 6)} \\\\
p90 & ${fmt_sci_latex(p90_rel, 2)}$ & ${fmt_sci_latex(p90_se, 2)}$ & {fmt(p90_zero, 6)} \\\\
p95 & ${fmt_sci_latex(p95_rel, 2)}$ & ${fmt_sci_latex(p95_se, 2)}$ & {fmt(p95_zero, 6)} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{k=2 posterior estimator error vs expected naive-MC error under matched token budget.}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.72\\linewidth]{{{rel_path(out_tex, section6_percentile)}}}
\\caption{{Percentile-level comparison of posterior method error and naive-MC expected error (k=2, teacher-forced truth, log scale).}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.72\\linewidth]{{{rel_path(out_tex, section6_scatter)}}}
\\caption{{Per-context relative error versus teacher-forced true event probability (k=2, log-log).}}
\\end{{figure}}

\\begin{{table}}[H]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{rcccccc}}
\\toprule
Context & True prob (teacher-forced) & Predicted prob & True posterior mean & True posterior sd & Sampled posterior mean & Sampled posterior sd \\\\
\\midrule
{os.linesep.join(ctx_rows_tex)}
\\bottomrule
\\end{{tabular}}
\\caption{{Per-context posterior diagnostics for the Step-3 rerun. Entries are vectors over states $(00,01,10,11)$.}}
\\end{{table}}

With teacher-forced model probability as truth, the posterior method remains far better than equal-compute naive MC across percentiles, but absolute errors are still large for difficult contexts.

\\end{{document}}
"""

    out_tex.write_text(tex, encoding="utf-8")
    print(f"Wrote LaTeX: {out_tex}")
    if not args.no_compile:
        pdf = compile_pdf(out_tex)
        print(f"Compiled PDF: {pdf}")


if __name__ == "__main__":
    main()
