#!/usr/bin/env python3
"""Build a detailed Markov-k (k=1,2,3) report with LaTeX + PDF output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

try:
    from lpe.markov_k_transformer import (
        ModelConfig,
        bayes_next_prob,
        build_model,
        sample_markov_k_batch,
    )
except ModuleNotFoundError:
    from markov_k_transformer import (  # type: ignore
        ModelConfig,
        bayes_next_prob,
        build_model,
        sample_markov_k_batch,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TRANSFORMER_ROOT = REPO_ROOT / "artifacts" / "markov_k123_transformer_500_ctx1000x2k_balanced"
DEFAULT_BAYES_ROOT = REPO_ROOT / "artifacts" / "markov_k123_bayes_baseline_ctx1000x2k_balanced"

OUT_TEX = REPO_ROOT / "latex" / "markov_k123_transformer_lpe_report.tex"
OUT_DIR = REPO_ROOT / "artifacts" / "markov_k123_transformer_report"


@dataclass
class Step2Stats:
    mae: float
    rmse: float
    max_abs: float
    corr: float
    mean_sample_std: float


@dataclass
class Step3Stats:
    p50: float
    p90: float
    p95: float
    mean: float
    maxv: float


@dataclass
class NaiveStats:
    r_eq: int
    rel_se_p50_pct: float
    rel_se_p90_pct: float
    rel_se_p95_pct: float
    zero_hit_p50: float
    zero_hit_p90: float
    zero_hit_p95: float


@dataclass
class Section4Stats:
    k: int
    n_contexts: int
    len_min: int
    len_max: int
    corr: float
    mae: float


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: List[Dict[str, float]] = []
        for r in reader:
            row: Dict[str, float] = {}
            for k, v in r.items():
                if v is None or v == "":
                    row[k] = float("nan")
                else:
                    row[k] = float(v)
            out.append(row)
    if not out:
        raise RuntimeError(f"Empty CSV: {path}")
    return out


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _fmt(v: float, digits: int = 6) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def _fmt_pct(v: float, digits: int = 2) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}\\%"


def _fmt_sci(v: float, digits: int = 3) -> str:
    if not math.isfinite(v):
        return "nan"
    if v == 0.0:
        return "0"
    return f"{v:.{digits}e}"


def step2_stats(rows: List[Dict[str, float]]) -> Step2Stats:
    true = np.array([r["true_mean"] for r in rows], dtype=np.float64)
    sample = np.array([r["sample_mean"] for r in rows], dtype=np.float64)
    std = np.array([r["sample_std"] for r in rows], dtype=np.float64)
    abs_err = np.abs(sample - true)
    corr = float(np.corrcoef(true, sample)[0, 1]) if len(true) > 1 else float("nan")
    return Step2Stats(
        mae=float(abs_err.mean()),
        rmse=float(np.sqrt(np.mean((sample - true) ** 2))),
        max_abs=float(abs_err.max()),
        corr=corr,
        mean_sample_std=float(np.mean(std)),
    )


def step3_stats(rows: List[Dict[str, float]]) -> Step3Stats:
    rel = np.array([abs(r["relative_error_pct"]) for r in rows], dtype=np.float64)
    return Step3Stats(
        p50=float(np.percentile(rel, 50)),
        p90=float(np.percentile(rel, 90)),
        p95=float(np.percentile(rel, 95)),
        mean=float(np.mean(rel)),
        maxv=float(np.max(rel)),
    )


def naive_stats_from_step3(
    rows: List[Dict[str, float]],
    *,
    posterior_samples: int,
    rollout_length: int,
) -> NaiveStats:
    true_prob = np.array([r["true_prob"] for r in rows], dtype=np.float64)
    target_len = int(rows[0]["target_len"])
    r_eq = max(1, int((posterior_samples * rollout_length) // max(1, target_len)))

    rel_se = np.sqrt((1.0 - true_prob) / np.maximum(r_eq * true_prob, 1e-300))
    zero_hit = np.exp(np.log1p(-true_prob) * r_eq)

    return NaiveStats(
        r_eq=r_eq,
        rel_se_p50_pct=float(np.percentile(rel_se * 100.0, 50)),
        rel_se_p90_pct=float(np.percentile(rel_se * 100.0, 90)),
        rel_se_p95_pct=float(np.percentile(rel_se * 100.0, 95)),
        zero_hit_p50=float(np.percentile(zero_hit, 50)),
        zero_hit_p90=float(np.percentile(zero_hit, 90)),
        zero_hit_p95=float(np.percentile(zero_hit, 95)),
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def infer_train_stats(summary: Dict[str, object]) -> Dict[str, object]:
    """Best-effort training stats; falls back to parsing training_history.csv."""
    tr_raw = summary.get("train_stats", {})
    if isinstance(tr_raw, dict):
        tr = tr_raw
        if int(tr.get("steps_run", 0) or 0) > 0:
            return {
                "final_train_loss": float(tr.get("final_train_loss", float("nan"))),
                "best_gap_ratio": float(tr.get("best_gap_ratio", float("nan"))),
                "best_step": int(tr.get("best_step", 0) or 0),
                "steps_run": int(tr.get("steps_run", 0) or 0),
            }

    hist_str = str(summary.get("training_history_csv", "") or "")
    if not hist_str:
        return {
            "final_train_loss": float("nan"),
            "best_gap_ratio": float("nan"),
            "best_step": 0,
            "steps_run": 0,
        }

    hist_path = Path(hist_str)
    if not hist_path.is_absolute():
        hist_path = REPO_ROOT / hist_path
    if not hist_path.exists():
        return {
            "final_train_loss": float("nan"),
            "best_gap_ratio": float("nan"),
            "best_step": 0,
            "steps_run": 0,
        }

    with hist_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {
            "final_train_loss": float("nan"),
            "best_gap_ratio": float("nan"),
            "best_step": 0,
            "steps_run": 0,
        }

    steps: List[int] = []
    train_losses: List[float] = []
    eval_pairs: List[Tuple[int, float]] = []
    for r in rows:
        step = int(float(r.get("step", "0") or 0))
        tl = float(r.get("train_loss", "nan") or "nan")
        eg = float(r.get("eval_gap_pct", "nan") or "nan")
        steps.append(step)
        train_losses.append(tl)
        if math.isfinite(eg):
            eval_pairs.append((step, eg / 100.0))

    steps_run = max(steps) if steps else 0
    final_train_loss = float("nan")
    for tl in reversed(train_losses):
        if math.isfinite(tl):
            final_train_loss = tl
            break

    if eval_pairs:
        best_step, best_gap_ratio = min(eval_pairs, key=lambda x: x[1])
    else:
        best_step, best_gap_ratio = 0, float("nan")

    return {
        "final_train_loss": float(final_train_loss),
        "best_gap_ratio": float(best_gap_ratio),
        "best_step": int(best_step),
        "steps_run": int(steps_run),
    }


def make_training_progress_figure(step1: Dict[int, Dict[str, object]], out_path: Path) -> None:
    ks = sorted(step1.keys())
    steps_run = [int(step1[k].get("train_stats", {}).get("steps_run", 0)) for k in ks]
    best_step = [int(step1[k].get("train_stats", {}).get("best_step", 0)) for k in ks]
    final_train = [float(step1[k].get("train_stats", {}).get("final_train_loss", float("nan"))) for k in ks]
    best_gap_pct = [
        100.0 * float(step1[k].get("train_stats", {}).get("best_gap_ratio", float("nan")))
        for k in ks
    ]

    ensure_parent(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    ax = axes[0]
    ax.plot(ks, steps_run, marker="o", label="steps run")
    ax.plot(ks, best_step, marker="s", label="best step")
    ax.set_xlabel("k")
    ax.set_ylabel("Step")
    ax.set_title("Training Step Progress")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(ks, final_train, marker="o", label="final train loss")
    ax.plot(ks, best_gap_pct, marker="s", label="best eval gap (%)")
    ax.set_xlabel("k")
    ax.set_title("Training End Metrics")
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def make_step2_state_scatter(
    t_rows: Dict[int, List[Dict[str, float]]],
    b_rows: Dict[int, List[Dict[str, float]]],
    out_path: Path,
) -> None:
    ks = sorted(t_rows.keys())
    ensure_parent(out_path)
    fig, axes = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.6))
    if len(ks) == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        t_true = np.array([r["true_mean"] for r in t_rows[k]], dtype=np.float64)
        t_sample = np.array([r["sample_mean"] for r in t_rows[k]], dtype=np.float64)
        b_true = np.array([r["true_mean"] for r in b_rows[k]], dtype=np.float64)
        b_sample = np.array([r["sample_mean"] for r in b_rows[k]], dtype=np.float64)

        ax.scatter(t_true, t_sample, s=45, alpha=0.85, label="Transformer", marker="o")
        ax.scatter(b_true, b_sample, s=45, alpha=0.85, label="True Bayes baseline", marker="x")
        ax.plot([0, 1], [0, 1], "r--", linewidth=1.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("True posterior mean by state")
        ax.set_ylabel("Sample-implied mean")
        ax.set_title(f"k={k} (states={2**k})")
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def make_step3_error_scatter(
    t_rows: Dict[int, List[Dict[str, float]]],
    b_rows: Dict[int, List[Dict[str, float]]],
    t_naive: Dict[int, NaiveStats],
    b_naive: Dict[int, NaiveStats],
    out_path: Path,
) -> None:
    ks = sorted(t_rows.keys())
    ensure_parent(out_path)
    fig, axes = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.6))
    if len(ks) == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        tp_t = np.array([r["true_prob"] for r in t_rows[k]], dtype=np.float64)
        rel_t = np.array([abs(r["relative_error_pct"]) for r in t_rows[k]], dtype=np.float64)
        tp_b = np.array([r["true_prob"] for r in b_rows[k]], dtype=np.float64)
        rel_b = np.array([abs(r["relative_error_pct"]) for r in b_rows[k]], dtype=np.float64)

        r_eq_t = t_naive[k].r_eq
        naive_t = np.sqrt((1.0 - tp_t) / np.maximum(r_eq_t * tp_t, 1e-300)) * 100.0
        r_eq_b = b_naive[k].r_eq
        naive_b = np.sqrt((1.0 - tp_b) / np.maximum(r_eq_b * tp_b, 1e-300)) * 100.0

        ax.scatter(tp_t, rel_t, s=35, alpha=0.85, label="Transformer |rel err|%", marker="o")
        ax.scatter(tp_t, naive_t, s=30, alpha=0.8, label="Naive rel SE% (T budget)", marker="s")
        ax.scatter(tp_b, rel_b, s=35, alpha=0.85, label="Bayes-baseline |rel err|%", marker="x")
        ax.scatter(tp_b, naive_b, s=30, alpha=0.8, label="Naive rel SE% (B budget)", marker="^")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("True event probability")
        ax.set_ylabel("Relative error / SE (%)")
        ax.set_title(f"k={k}")
        ax.grid(alpha=0.25)

    axes[0].legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def generate_section4_scatter(
    transformer_summary: Dict[int, Dict[str, object]],
    *,
    out_dir: Path,
    n_contexts_per_k: int,
    len_min: int,
    len_max: int,
    seed: int,
    alpha: float,
    beta: float,
    device: torch.device,
) -> Tuple[Path, Path, List[Section4Stats]]:
    ensure_parent(out_dir / "dummy")
    fig_path = out_dir / "figures" / "section4_k123_nextprob_scatter_varlen.png"
    csv_path = out_dir / "section4_k123_nextprob_scatter_varlen.csv"

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    ks = sorted(transformer_summary.keys())
    fig, axes = plt.subplots(1, len(ks), figsize=(5.3 * len(ks), 4.8))
    if len(ks) == 1:
        axes = [axes]

    all_rows: List[Dict[str, object]] = []
    stats: List[Section4Stats] = []

    with torch.no_grad():
        for ax, k in zip(axes, ks):
            s = transformer_summary[k]
            model_cfg = s["model_config"]
            cfg = ModelConfig(
                n_layers=int(model_cfg["n_layers"]),
                d_model=int(model_cfg["d_model"]),
                n_heads=int(model_cfg["n_heads"]),
                d_mlp=int(model_cfg["d_mlp"]),
            )
            use_pos = bool(s.get("use_positional_encoding", True))
            max_seq_len = int(s.get("max_seq_len", len_max + 32))
            checkpoint = Path(str(s["checkpoint_path"]))

            model = build_model(cfg, max_seq_len=max_seq_len, use_positional_encoding=use_pos)
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model = model.to(device)
            model.eval()

            bayes_vals: List[float] = []
            model_vals: List[float] = []
            lengths: List[int] = []

            lo = max(k + 1, len_min)
            hi = min(len_max, max_seq_len - 1)
            if hi < lo:
                raise RuntimeError(f"Invalid length range for k={k}: [{lo},{hi}]")

            for i in range(n_contexts_per_k):
                clen = int(rng.integers(lo, hi + 1))
                seq = sample_markov_k_batch(1, clen, k, alpha=alpha, beta=beta, device=device)[0]
                context = seq.detach().cpu().long()

                p_bayes = float(bayes_next_prob(context, k, alpha=alpha, beta=beta))
                logits = model.predict_next_logits(context.to(device).unsqueeze(0))
                p_model = float(F.softmax(logits, dim=-1)[0, 1].item())

                bayes_vals.append(p_bayes)
                model_vals.append(p_model)
                lengths.append(clen)

                all_rows.append(
                    {
                        "k": k,
                        "context_id": i + 1,
                        "context_len": clen,
                        "bayes_p_next_1": p_bayes,
                        "model_p_next_1": p_model,
                    }
                )
                progress_every = max(1, n_contexts_per_k // 4)
                if ((i + 1) % progress_every == 0) or (i + 1 == n_contexts_per_k):
                    print(
                        f"[section4 k={k}] processed {i + 1}/{n_contexts_per_k} mixed-length contexts",
                        flush=True,
                    )

            bayes_np = np.array(bayes_vals, dtype=np.float64)
            model_np = np.array(model_vals, dtype=np.float64)
            corr = float(np.corrcoef(bayes_np, model_np)[0, 1]) if len(bayes_np) > 1 else float("nan")
            mae = float(np.mean(np.abs(model_np - bayes_np)))

            stats.append(
                Section4Stats(
                    k=k,
                    n_contexts=n_contexts_per_k,
                    len_min=lo,
                    len_max=hi,
                    corr=corr,
                    mae=mae,
                )
            )

            ax.scatter(bayes_np, model_np, s=10, alpha=0.45, edgecolors="none")
            ax.plot([0, 1], [0, 1], "r--", linewidth=1.4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Bayes next-bit P(1)")
            ax.set_ylabel("Model next-bit P(1)")
            ax.set_title(f"k={k}: r={corr:.4f}, MAE={mae:.4f}")
            ax.grid(alpha=0.25)

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    return fig_path, csv_path, stats


def build_latex(
    *,
    out_tex: Path,
    transformer_root: Path,
    bayes_root: Path,
    transformer_summary: Dict[int, Dict[str, object]],
    bayes_summary: Dict[int, Dict[str, object]],
    step1_summary: Dict[int, Dict[str, object]],
    t_step2_stats: Dict[int, Step2Stats],
    b_step2_stats: Dict[int, Step2Stats],
    t_step3_stats: Dict[int, Step3Stats],
    b_step3_stats: Dict[int, Step3Stats],
    t_naive_stats: Dict[int, NaiveStats],
    b_naive_stats: Dict[int, NaiveStats],
    section4_stats: List[Section4Stats],
    fig_training: Path,
    fig_step2_state: Path,
    fig_step3_scatter: Path,
    fig_section4: Path,
) -> None:
    def rel(path: Path) -> str:
        return _latex_escape(os.path.relpath(path, out_tex.parent).replace("\\", "/"))

    def relrepo_text(path: Path) -> str:
        p = path
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        try:
            return _latex_escape(str(p.relative_to(REPO_ROOT)))
        except ValueError:
            return _latex_escape(str(p))

    ks = [1, 2, 3]
    if section4_stats:
        section4_n_contexts = str(section4_stats[0].n_contexts)
    else:
        section4_n_contexts = "n/a"

    rollout_mult_text = "configured per k (see table)"
    try:
        rollout_mults: List[Optional[int]] = []
        for k in ks:
            rl = int(transformer_summary[k]["rollout_length"])
            if rl % (2**k) == 0:
                rollout_mults.append(rl // (2**k))
            else:
                rollout_mults.append(None)
        if rollout_mults and all(m is not None for m in rollout_mults) and len(set(rollout_mults)) == 1:
            rollout_mult_text = f"{int(rollout_mults[0])}\\cdot 2^k"
    except Exception:
        pass

    model_rows = []
    for k in ks:
        s = transformer_summary[k]
        m = s["model_config"]
        model_rows.append(
            " & ".join(
                [
                    str(k),
                    f"L{int(m['n_layers'])}/D{int(m['d_model'])}/H{int(m['n_heads'])}/M{int(m['d_mlp'])}",
                    str(bool(s.get("use_positional_encoding", True))),
                    str(int(s["train_seq_len"])),
                    str(int(s["rollout_length"])),
                    str(int(s["batch_size"])),
                ]
            )
            + r" \\" 
        )

    train_rows = []
    for k in ks:
        ts = step1_summary[k]
        tr = ts.get("train_stats", {})
        best_gap_pct = 100.0 * float(tr.get("best_gap_ratio", float("nan")))
        train_rows.append(
            " & ".join(
                [
                    str(k),
                    str(int(tr.get("steps_run", 0))),
                    str(int(tr.get("best_step", 0))),
                    _fmt(best_gap_pct, 3),
                    _fmt(float(tr.get("final_train_loss", float("nan")),), 6),
                ]
            )
            + r" \\" 
        )

    nll_rows = []
    for k in ks:
        s = transformer_summary[k]
        nll_rows.append(
            " & ".join(
                [
                    str(k),
                    _fmt(float(s["step1_eval_model_nll"]), 6),
                    _fmt(float(s["step1_eval_bayes_nll"]), 6),
                    _fmt_pct(float(s["step1_gap_pct"]), 2),
                ]
            )
            + r" \\" 
        )

    sec4_rows = []
    for s in section4_stats:
        sec4_rows.append(
            " & ".join(
                [
                    str(s.k),
                    str(s.n_contexts),
                    f"[{s.len_min}, {s.len_max}]",
                    _fmt(s.corr, 4),
                    _fmt(s.mae, 5),
                ]
            )
            + r" \\" 
        )

    step2_rows = []
    for k in ks:
        t = t_step2_stats[k]
        b = b_step2_stats[k]
        step2_rows.append(
            " & ".join(
                [
                    str(k),
                    _fmt(t.mae, 6),
                    _fmt(b.mae, 6),
                    _fmt(t.rmse, 6),
                    _fmt(b.rmse, 6),
                    _fmt(t.max_abs, 6),
                    _fmt(b.max_abs, 6),
                ]
            )
            + r" \\" 
        )

    step3_rows = []
    for k in ks:
        t = t_step3_stats[k]
        b = b_step3_stats[k]
        tn = t_naive_stats[k]
        bn = b_naive_stats[k]
        step3_rows.append(
            " & ".join(
                [
                    str(k),
                    _fmt(t.p50, 3),
                    _fmt(t.p90, 3),
                    _fmt_sci(t.mean, 3),
                    _fmt(tn.rel_se_p50_pct, 1),
                    _fmt_sci(tn.rel_se_p90_pct, 2),
                    _fmt(b.p50, 3),
                    _fmt(b.p90, 3),
                    _fmt_sci(b.mean, 3),
                    _fmt(bn.rel_se_p50_pct, 1),
                    _fmt_sci(bn.rel_se_p90_pct, 2),
                ]
            )
            + r" \\" 
        )

    per_k_fig_rows = []
    for k in ks:
        pred = Path(str(transformer_summary[k]["prediction_plot"]))
        post_t = Path(str(transformer_summary[k]["posterior_plot"]))
        post_b = Path(str(bayes_summary[k]["posterior_plot"]))
        lpe_t = Path(str(transformer_summary[k]["lpe_hist_plot"]))
        lpe_b = Path(str(bayes_summary[k]["lpe_hist_plot"]))

        for path, cap in [
            (pred, f"k={k} Step 1: model vs Bayes next-bit predictions (context length fixed by run setup)."),
            (post_t, f"k={k} Step 2 transformer posterior mean by state."),
            (post_b, f"k={k} Step 2 true-Bayes baseline posterior mean by state."),
            (lpe_t, f"k={k} Step 3 transformer LPE relative-error histogram."),
            (lpe_b, f"k={k} Step 3 true-Bayes baseline LPE relative-error histogram."),
        ]:
            if path.exists():
                per_k_fig_rows.append(
                    "\\begin{figure}[H]"
                    "\\centering"
                    f"\\includegraphics[width=0.66\\linewidth]{{{rel(path)}}}"
                    f"\\caption{{{cap}}}"
                    "\\end{figure}"
                )

    tex = f"""
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{amsmath}}
\\title{{Markov-$k$ Transformer LPE Report (k=1,2,3)}}
\\author{{Automated run}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle

\\section*{{Artifacts Used}}
\\begin{{itemize}}
\\item Transformer artifacts: \\texttt{{{relrepo_text(transformer_root)}}}
\\item True-Bayes baseline artifacts: \\texttt{{{relrepo_text(bayes_root)}}}
\\item Step-1 training metadata source: per-\\(k\\) \\texttt{{training\\_history.csv}} and \\texttt{{summary.json}} inside transformer artifacts.
\\end{{itemize}}

\\section*{{1. Model Architecture and Training Setup}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{rrrrrr}}
\\toprule
$k$ & Model (L/D/H/M) & PosEnc & Train SeqLen & Rollout Len & Batch \\\\
\\midrule
{os.linesep.join(model_rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Model and sequence configuration by $k$.}}
\\end{{table}}

Common optimization settings (from \\texttt{{lpe/markov\\_k\\_transformer.py}}):
\\begin{{itemize}}
\\item Optimizer: AdamW, learning rate $3\\times10^{{-4}}$, weight decay $0.01$.
\\item Warmup: 2000 steps; gradient clipping: $1.0$.
\\item Training data: Beta-Bernoulli order-$k$ Markov process with $\\alpha=\\beta=1$ per state.
\\item Training length matched inference rollout length: ${rollout_mult_text}$.
\\end{{itemize}}

\\section*{{2. Training Progress Summary}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{rrrrr}}
\\toprule
$k$ & Steps Run & Best Step & Best Eval Gap (\\%) & Final Train Loss \\\\
\\midrule
{os.linesep.join(train_rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Available training-progress metadata from Step-1 runs.}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.78\\linewidth]{{{rel(fig_training)}}}
\\caption{{Training progress summary across $k=1,2,3$ (available metadata fields).}}
\\end{{figure}}

\\section*{{3. Final Loss vs Bayes Predictive}}
For each $k$, the Bayes-optimal predictor uses independent Beta posteriors per state $s\\in\\{{0,1\\}}^k$:
\\[
\\theta_s \\mid \\mathcal{{D}} \\sim \\mathrm{{Beta}}(\\alpha+n_{{s,1}},\\beta+n_{{s,0}}),
\\quad
P(y_{{t+1}}=1 \\mid y_{{1:t}})=\\frac{{\\alpha+n_{{s_t,1}}}}{{\\alpha+\\beta+n_{{s_t,1}}+n_{{s_t,0}}}}.
\\]
The reported Bayes NLL is computed on held-out sampled sequences from the same Markov-$k$ data process.

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{rrrr}}
\\toprule
$k$ & Model NLL & Bayes NLL & Gap \\\\
\\midrule
{os.linesep.join(nll_rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Step-1 held-out next-token NLL comparison.}}
\\end{{table}}

\\section*{{4. Mixed-Length Context Calibration (New Diagnostic)}}
To mirror the Bernoulli report's next-token calibration check, we generated fresh mixed-length contexts for each $k$:
\\begin{{itemize}}
\\item Number of contexts per $k$: {section4_n_contexts}.
\\item Length distribution: uniform integer over a valid range per $k$ (shown in the table).
\\item Context generation: sampled from the same Beta-Bernoulli order-$k$ Markov process.
\\item For each context, we computed Bayes $P(y_{{n+1}}=1\\mid x_{{1:n}})$ and model $P_\\theta(y_{{n+1}}=1\\mid x_{{1:n}})$.
\\end{{itemize}}

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{rrrrr}}
\\toprule
$k$ & Contexts & Length Range & Pearson $r$ & MAE \\\\
\\midrule
{os.linesep.join(sec4_rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Mixed-length next-token calibration summary.}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.95\\linewidth]{{{rel(fig_section4)}}}
\\caption{{Model vs Bayes next-token probability on mixed-length contexts (one panel per $k$).}}
\\end{{figure}}

\\section*{{5. Posterior-Sample Quality (Step 2)}}
Step 2 estimates posterior means for each state transition parameter using rollout-derived samples,
then compares to analytic posterior means.

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{rrrrrrr}}
\\toprule
$k$ & MAE (T) & MAE (B) & RMSE (T) & RMSE (B) & MaxAbs (T) & MaxAbs (B) \\\\
\\midrule
{os.linesep.join(step2_rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{State-wise posterior-mean error summary. T=transformer posterior sampler, B=true-Bayes predictive sampler.}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.95\\linewidth]{{{rel(fig_step2_state)}}}
\\caption{{Step 2 posterior mean by state: transformer vs true-Bayes baseline.}}
\\end{{figure}}

\\section*{{6. LPE Quality and Naive-MC Equal-Compute Baseline (Step 3)}}
For a fixed target sequence $y_{{1:m}}$, the true event probability under the Bayes predictive can be written
in closed form as a product of Beta-function ratios over states:
\\[
P(y_{{1:m}}\\mid x_{{1:n}})
= \\prod_{{s\\in\\{{0,1\\}}^k}}
\\frac{{B(\\alpha+n_{{s,1}}+m_{{s,1}},\\;\\beta+n_{{s,0}}+m_{{s,0}})}}
{{B(\\alpha+n_{{s,1}},\\;\\beta+n_{{s,0}})}},
\\]
where $n_{{s,\\cdot}}$ are transition counts from context and $m_{{s,\\cdot}}$ are additional counts induced by the fixed target rollout path.

Equal-compute naive MC uses token budget matching:
\\[
R_{{\\text{{eq}}}}=\\left\\lfloor\\frac{{M\\cdot L}}{{m}}\\right\\rfloor,
\\]
with posterior-sampling count $M$, rollout length $L$, and target length $m$.
Naive expected relative SE for event probability $P$ is
\\[
\\sqrt{{\\frac{{1-P}}{{R_{{\\text{{eq}}}}P}}}}.
\\]

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{rrrrrrrrrrr}}
\\toprule
$k$ & p50(T) & p90(T) & mean(T) & naive p50(T) & naive p90(T) & p50(B) & p90(B) & mean(B) & naive p50(B) & naive p90(B) \\\\
\\midrule
{os.linesep.join(step3_rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Step 3 absolute relative error (\\%) and naive-MC expected relative SE (\\%) summaries.}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.95\\linewidth]{{{rel(fig_step3_scatter)}}}
\\caption{{Per-context Step 3 error vs true probability, with equal-compute naive-MC expected relative SE.}}
\\end{{figure}}

\\section*{{7. Per-k Diagnostic Figures}}
{os.linesep.join(per_k_fig_rows)}

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
    p = argparse.ArgumentParser(description="Build k=1,2,3 Markov transformer report (LaTeX + PDF).")
    p.add_argument("--transformer-root", type=Path, default=DEFAULT_TRANSFORMER_ROOT)
    p.add_argument("--bayes-root", type=Path, default=DEFAULT_BAYES_ROOT)
    p.add_argument("--out-tex", type=Path, default=OUT_TEX)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-contexts-per-k", type=int, default=1500)
    p.add_argument("--section4-len-min", type=int, default=16)
    p.add_argument("--section4-len-max", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-compile", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ks = [1, 2, 3]
    t_root = Path(args.transformer_root)
    b_root = Path(args.bayes_root)
    t_summary = {k: _read_json(t_root / f"k{k}" / "summary.json") for k in ks}
    b_summary = {k: _read_json(b_root / f"k{k}" / "summary.json") for k in ks}
    step1_summary = {
        k: {
            "train_stats": infer_train_stats(t_summary[k]),
        }
        for k in ks
    }

    t_step2_rows = {
        k: _read_csv_rows(t_root / f"k{k}" / "step2_posterior_state_metrics.csv")
        for k in ks
    }
    b_step2_rows = {
        k: _read_csv_rows(b_root / f"k{k}" / "step2_posterior_state_metrics.csv")
        for k in ks
    }
    t_step3_rows = {
        k: _read_csv_rows(t_root / f"k{k}" / "step3_lpe_metrics.csv")
        for k in ks
    }
    b_step3_rows = {
        k: _read_csv_rows(b_root / f"k{k}" / "step3_lpe_metrics.csv")
        for k in ks
    }

    t_step2_stats = {k: step2_stats(t_step2_rows[k]) for k in ks}
    b_step2_stats = {k: step2_stats(b_step2_rows[k]) for k in ks}
    t_step3_stats = {k: step3_stats(t_step3_rows[k]) for k in ks}
    b_step3_stats = {k: step3_stats(b_step3_rows[k]) for k in ks}

    t_naive_stats: Dict[int, NaiveStats] = {}
    b_naive_stats: Dict[int, NaiveStats] = {}
    for k in ks:
        posterior_samples_t = int(
            float(
                t_summary[k].get(
                    "num_posterior_samples",
                    b_summary[k].get("num_posterior_samples", 200),
                )
            )
        )
        rollout_len_t = int(float(t_summary[k]["rollout_length"]))
        t_naive_stats[k] = naive_stats_from_step3(
            t_step3_rows[k],
            posterior_samples=posterior_samples_t,
            rollout_length=rollout_len_t,
        )

        posterior_samples_b = int(float(b_summary[k].get("num_posterior_samples", 200)))
        rollout_len_b = int(float(b_summary[k]["rollout_length"]))
        b_naive_stats[k] = naive_stats_from_step3(
            b_step3_rows[k],
            posterior_samples=posterior_samples_b,
            rollout_length=rollout_len_b,
        )

    fig_training = args.out_dir / "figures" / "training_progress_summary_k123.png"
    fig_step2_state = args.out_dir / "figures" / "step2_state_scatter_transformer_vs_bayes_k123.png"
    fig_step3_scatter = args.out_dir / "figures" / "step3_relerr_vs_naive_k123.png"

    make_training_progress_figure(step1_summary, fig_training)
    make_step2_state_scatter(t_step2_rows, b_step2_rows, fig_step2_state)
    make_step3_error_scatter(t_step3_rows, b_step3_rows, t_naive_stats, b_naive_stats, fig_step3_scatter)

    fig_section4, section4_csv, section4_stats = generate_section4_scatter(
        t_summary,
        out_dir=args.out_dir,
        n_contexts_per_k=int(args.n_contexts_per_k),
        len_min=int(args.section4_len_min),
        len_max=int(args.section4_len_max),
        seed=int(args.seed),
        alpha=float(args.alpha),
        beta=float(args.beta),
        device=device,
    )

    build_latex(
        out_tex=args.out_tex,
        transformer_root=t_root,
        bayes_root=b_root,
        transformer_summary=t_summary,
        bayes_summary=b_summary,
        step1_summary=step1_summary,
        t_step2_stats=t_step2_stats,
        b_step2_stats=b_step2_stats,
        t_step3_stats=t_step3_stats,
        b_step3_stats=b_step3_stats,
        t_naive_stats=t_naive_stats,
        b_naive_stats=b_naive_stats,
        section4_stats=section4_stats,
        fig_training=fig_training,
        fig_step2_state=fig_step2_state,
        fig_step3_scatter=fig_step3_scatter,
        fig_section4=fig_section4,
    )

    print(f"Wrote section4 CSV: {section4_csv}")
    print(f"Wrote LaTeX report: {args.out_tex}")
    if not args.no_compile:
        pdf_path = compile_pdf(args.out_tex)
        print(f"Compiled PDF: {pdf_path}")


if __name__ == "__main__":
    main()
