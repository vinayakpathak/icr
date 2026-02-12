#!/usr/bin/env python3
"""Run token stability experiments for small-vocabulary Hugging Face causal LMs.

This script performs long autoregressive rollouts from a fixed start token and tracks
stability of:
- Unigram token frequencies
- Bigram conditional transitions P(y_t=b | y_{t-1}=a)
- Trigram-next conditionals P(y_t=c | y_{t-2}=a, y_{t-1}=b)

It supports multi-seed parallel execution and writes per-seed plus aggregate outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    max_order: int
    start_token: str
    device: str
    max_tokens: int
    min_tokens_before_convergence: int
    checkpoint_interval: int
    window_checkpoints: int
    unigram_max_delta_threshold: float
    bigram_tv_p95_threshold: float
    trigram_tv_p95_threshold: float
    bigram_min_support: int
    trigram_min_support: int
    save_every_tokens: int
    trajectory_interval: int
    out_dir: str
    plot_dir: str
    num_threads_per_worker: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run token-frequency and transition-stability experiments for a small-vocab HF causal LM."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="phonemetransformers/GPT2-85M-CHAR-PHON",
        help="Hugging Face model id.",
    )
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of rollout seeds when --seeds is not provided.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated list of integer seeds (overrides --num-seeds when non-empty).",
    )
    parser.add_argument("--max-order", type=int, default=3, choices=[1, 2, 3], help="Maximum context order to monitor.")
    parser.add_argument(
        "--start-token",
        type=str,
        default="bos",
        choices=["bos", "eos"],
        help="Fixed initial token for each rollout.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for workers (cpu or cuda).")
    parser.add_argument("--parallel-workers", type=int, default=5, help="Number of worker processes.")

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2_000_000,
        help="Maximum total tokens in each chain, including the fixed start token.",
    )
    parser.add_argument(
        "--min-tokens-before-convergence",
        type=int,
        default=200_000,
        help="Minimum tokens before convergence checks can trigger stopping.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20_000,
        help="Emit metrics every N total tokens.",
    )
    parser.add_argument(
        "--window-checkpoints",
        type=int,
        default=5,
        help="Number of consecutive converged checkpoints required.",
    )
    parser.add_argument(
        "--unigram-max-delta-threshold",
        type=float,
        default=0.002,
        help="Convergence threshold for max absolute unigram delta.",
    )
    parser.add_argument(
        "--bigram-tv-p95-threshold",
        type=float,
        default=0.03,
        help="Convergence threshold for bigram context-level TV p95.",
    )
    parser.add_argument(
        "--trigram-tv-p95-threshold",
        type=float,
        default=0.05,
        help="Convergence threshold for trigram context-level TV p95.",
    )
    parser.add_argument(
        "--bigram-min-support",
        type=int,
        default=500,
        help="Minimum context count for bigram contexts to be eligible.",
    )
    parser.add_argument(
        "--trigram-min-support",
        type=int,
        default=200,
        help="Minimum context count for trigram contexts to be eligible.",
    )
    parser.add_argument(
        "--save-every-tokens",
        type=int,
        default=100_000,
        help="Write periodic state snapshots every N tokens.",
    )
    parser.add_argument(
        "--trajectory-interval",
        type=int,
        default=10,
        help="Record unigram trajectory points every N generated tokens for token_trajectories plots.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="language_models/results/token_stability",
        help="Directory for per-seed and aggregate numeric outputs.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots/language_models/token_stability",
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--num-threads-per-worker",
        type=int,
        default=8,
        help="CPU threads assigned to each worker process.",
    )

    return parser.parse_args()


def parse_seed_list(seed_arg: str, num_seeds: int) -> List[int]:
    cleaned = seed_arg.strip()
    if cleaned:
        seeds = [int(x.strip()) for x in cleaned.split(",") if x.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but empty after parsing")
        return seeds
    return list(range(num_seeds))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def maybe_nan_to_none(x: float) -> Optional[float]:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


def resolve_start_token_id(tokenizer: Any, mode: str) -> int:
    if mode == "bos":
        token_id = tokenizer.bos_token_id
        token_name = "BOS"
    elif mode == "eos":
        token_id = tokenizer.eos_token_id
        token_name = "EOS"
    else:
        raise ValueError(f"Unsupported start token mode: {mode}")

    if token_id is None:
        raise ValueError(f"Tokenizer does not define {token_name} token id; cannot run fixed-token rollout.")
    return int(token_id)


def conditional_from_bigram_counts(counts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    support = counts2.sum(axis=1)
    denom = support[:, None]
    cond = np.divide(counts2, denom, out=np.zeros_like(counts2, dtype=np.float64), where=denom > 0)
    return cond, support


def conditional_from_trigram_counts(counts3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    support = counts3.sum(axis=2)
    denom = support[:, :, None]
    cond = np.divide(counts3, denom, out=np.zeros_like(counts3, dtype=np.float64), where=denom > 0)
    return cond, support


def p95_tv_bigram(curr: np.ndarray, prev: np.ndarray, support: np.ndarray, min_support: int) -> Tuple[float, int]:
    eligible = support >= min_support
    eligible_count = int(eligible.sum())
    if eligible_count == 0:
        return float("inf"), 0

    tv_all = 0.5 * np.abs(curr - prev).sum(axis=1)
    return float(np.percentile(tv_all[eligible], 95)), eligible_count


def p95_tv_trigram(curr: np.ndarray, prev: np.ndarray, support: np.ndarray, min_support: int) -> Tuple[float, int]:
    eligible = support >= min_support
    eligible_count = int(eligible.sum())
    if eligible_count == 0:
        return float("inf"), 0

    tv_all = 0.5 * np.abs(curr - prev).sum(axis=2)
    return float(np.percentile(tv_all[eligible], 95)), eligible_count


def crop_past_key_values(past: Any, keep_len: int) -> Any:
    if past is None:
        return None
    if keep_len <= 0:
        return None

    # transformers>=5 returns a DynamicCache-like object with in-place crop().
    if hasattr(past, "crop") and callable(past.crop):
        past.crop(keep_len)
        return past

    # Backward-compatible path for legacy tuple-based caches.
    cropped = []
    for layer_cache in past:
        if len(layer_cache) < 2:
            raise ValueError("Unsupported legacy cache format: expected at least key/value tensors per layer.")
        layer_k, layer_v, *rest = layer_cache
        if layer_k.shape[-2] > keep_len:
            layer_k = layer_k[:, :, -keep_len:, :]
            layer_v = layer_v[:, :, -keep_len:, :]
        cropped.append((layer_k, layer_v, *rest))
    return tuple(cropped)


def plot_seed_outputs(seed_dir: Path, plot_seed_dir: Path, seed: int, vocab_size: int) -> None:
    ckpt_path = seed_dir / "checkpoints.jsonl"
    ckpt_rows: List[Dict[str, Any]] = []
    with ckpt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ckpt_rows.append(json.loads(line))

    if not ckpt_rows:
        return

    tokens = np.array([row["tokens_generated"] for row in ckpt_rows], dtype=np.int64)
    unigram_d = np.array([row.get("unigram_max_delta", np.nan) for row in ckpt_rows], dtype=np.float64)
    bigram_d = np.array([row.get("bigram_tv_p95", np.nan) for row in ckpt_rows], dtype=np.float64)
    trigram_d = np.array([row.get("trigram_tv_p95", np.nan) for row in ckpt_rows], dtype=np.float64)

    seed_summary = json.loads((seed_dir / "summary.json").read_text(encoding="utf-8"))
    thresholds = seed_summary["thresholds"]

    ensure_dir(plot_seed_dir)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(tokens, unigram_d, color="tab:blue")
    axes[0].axhline(thresholds["unigram_max_delta"], color="tab:blue", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Unigram max |Δ|")
    axes[0].grid(alpha=0.25)

    axes[1].plot(tokens, bigram_d, color="tab:orange")
    axes[1].axhline(thresholds["bigram_tv_p95"], color="tab:orange", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Bigram TV p95")
    axes[1].grid(alpha=0.25)

    axes[2].plot(tokens, trigram_d, color="tab:green")
    axes[2].axhline(thresholds["trigram_tv_p95"], color="tab:green", linestyle="--", alpha=0.7)
    axes[2].set_ylabel("Trigram TV p95")
    axes[2].set_xlabel("Total tokens in chain")
    axes[2].grid(alpha=0.25)

    fig.suptitle(f"Seed {seed}: Stability Metrics")
    fig.tight_layout()
    fig.savefig(plot_seed_dir / "stability_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    traj_path = seed_dir / "token_trajectories.npz"
    if not traj_path.exists():
        return

    data = np.load(traj_path)
    traj_tokens = data["tokens"]
    traj_freqs = data["unigram_freqs"]

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    for tok_idx in range(min(vocab_size, traj_freqs.shape[1])):
        ax2.plot(traj_tokens, traj_freqs[:, tok_idx], alpha=0.5, linewidth=0.8)
    ax2.set_xlabel("Total tokens in chain")
    ax2.set_ylabel("Token frequency")
    ax2.set_title(f"Seed {seed}: Unigram trajectories (all {vocab_size} tokens)")
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(plot_seed_dir / "token_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    if "next_prob_tokens" in data.files and "next_token_probs" in data.files:
        prob_tokens = data["next_prob_tokens"]
        prob_traj = data["next_token_probs"].astype(np.float32, copy=False)

        fig3, ax3 = plt.subplots(figsize=(12, 7))
        for tok_idx in range(min(vocab_size, prob_traj.shape[1])):
            ax3.plot(prob_tokens, prob_traj[:, tok_idx], alpha=0.5, linewidth=0.8)
        ax3.set_xlabel("Total tokens in chain")
        ax3.set_ylabel("P(next token)")
        ax3.set_title(f"Seed {seed}: Next-token probability trajectories (all {vocab_size} tokens)")
        ax3.grid(alpha=0.25)
        fig3.tight_layout()
        fig3.savefig(plot_seed_dir / "next_token_prob_trajectories.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)


def run_single_seed(seed: int, cfg: ExperimentConfig) -> Dict[str, Any]:
    t0 = time.time()

    os.environ["OMP_NUM_THREADS"] = str(cfg.num_threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(cfg.num_threads_per_worker)

    torch.set_num_threads(cfg.num_threads_per_worker)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    model.to(device)
    model.eval()

    model_cfg = model.config
    vocab_size = int(model_cfg.vocab_size)
    max_positions = int(getattr(model_cfg, "max_position_embeddings", 0) or 0)
    if max_positions <= 1:
        raise ValueError(
            f"Model {cfg.model_id} does not provide usable max_position_embeddings (got {max_positions})."
        )

    start_token_id = resolve_start_token_id(tokenizer, cfg.start_token)
    if not (0 <= start_token_id < vocab_size):
        raise ValueError(f"Start token id {start_token_id} out of range for vocab size {vocab_size}")

    seed_dir = Path(cfg.out_dir) / f"seed_{seed}"
    ensure_dir(seed_dir)

    ckpt_file = seed_dir / "checkpoints.jsonl"
    if ckpt_file.exists():
        ckpt_file.unlink()

    counts1 = np.zeros(vocab_size, dtype=np.int64)
    counts2 = np.zeros((vocab_size, vocab_size), dtype=np.int64) if cfg.max_order >= 2 else None
    counts3 = np.zeros((vocab_size, vocab_size, vocab_size), dtype=np.int64) if cfg.max_order >= 3 else None

    counts1[start_token_id] = 1

    prev2: Optional[int] = None
    prev1: int = start_token_id

    prev_unigram: Optional[np.ndarray] = None
    prev_bigram_cond: Optional[np.ndarray] = None
    prev_trigram_cond: Optional[np.ndarray] = None

    consecutive_converged = 0
    converged = False
    stop_reason = "max_tokens_reached"
    convergence_checkpoint: Optional[int] = None

    # Pre-allocate dense trajectory buffers to avoid Python-object overhead when interval is small.
    max_traj_points = (cfg.max_tokens // cfg.trajectory_interval) + 2
    traj_tokens = np.empty(max_traj_points, dtype=np.int64)
    traj_unigram = np.empty((max_traj_points, vocab_size), dtype=np.float32)
    traj_size = 0
    next_prob_tokens = np.empty(max_traj_points, dtype=np.int64)
    # float16 keeps memory bounded for long runs while preserving trajectory shape.
    next_token_probs = np.empty((max_traj_points, vocab_size), dtype=np.float16)
    next_prob_size = 0

    def append_trajectory_point(chain_tokens: int) -> None:
        nonlocal traj_size
        traj_tokens[traj_size] = chain_tokens
        traj_unigram[traj_size, :] = counts1.astype(np.float64) / float(chain_tokens)
        traj_size += 1

    def append_next_prob_point(chain_tokens: int, probs_vec: np.ndarray) -> None:
        nonlocal next_prob_size
        next_prob_tokens[next_prob_size] = chain_tokens
        next_token_probs[next_prob_size, :] = probs_vec.astype(np.float16, copy=False)
        next_prob_size += 1

    next_snapshot_token = cfg.save_every_tokens

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    input_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    past_key_values = None

    append_trajectory_point(chain_tokens=1)

    with torch.inference_mode(), ckpt_file.open("a", encoding="utf-8") as ckpt_f:
        for total_tokens in range(1, cfg.max_tokens):
            outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)

            if total_tokens == 1 or (total_tokens % cfg.trajectory_interval == 0):
                append_next_prob_point(
                    chain_tokens=total_tokens,
                    probs_vec=probs[0].detach().cpu().numpy(),
                )

            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            token_id = int(next_token.item())

            counts1[token_id] += 1
            if counts2 is not None:
                counts2[prev1, token_id] += 1
            if counts3 is not None and prev2 is not None:
                counts3[prev2, prev1, token_id] += 1

            prev2 = prev1
            prev1 = token_id

            input_ids = next_token

            # Keep rolling context within model positional limit.
            past_key_values = crop_past_key_values(past_key_values, keep_len=max_positions - 1)

            chain_tokens = total_tokens + 1

            if chain_tokens >= next_snapshot_token:
                np.savez_compressed(
                    seed_dir / f"state_{chain_tokens}.npz",
                    seed=seed,
                    tokens_generated=chain_tokens,
                    counts1=counts1,
                    counts2=counts2 if counts2 is not None else np.array([], dtype=np.int64),
                    counts3=counts3 if counts3 is not None else np.array([], dtype=np.int64),
                )
                next_snapshot_token += cfg.save_every_tokens

            if chain_tokens % cfg.trajectory_interval == 0:
                append_trajectory_point(chain_tokens=chain_tokens)

            if chain_tokens % cfg.checkpoint_interval != 0 and chain_tokens != cfg.max_tokens:
                continue

            unigram = counts1.astype(np.float64) / float(chain_tokens)
            bigram_cond = None
            bigram_support = None
            trigram_cond = None
            trigram_support = None

            if counts2 is not None:
                bigram_cond, bigram_support = conditional_from_bigram_counts(counts2)
            if counts3 is not None:
                trigram_cond, trigram_support = conditional_from_trigram_counts(counts3)

            if prev_unigram is None:
                unigram_delta = float("inf")
                bigram_tv_p95 = float("nan")
                trigram_tv_p95 = float("nan")
                eligible_bigram_contexts = 0
                eligible_trigram_contexts = 0
            else:
                unigram_delta = float(np.max(np.abs(unigram - prev_unigram)))
                if counts2 is not None and prev_bigram_cond is not None:
                    bigram_tv_p95, eligible_bigram_contexts = p95_tv_bigram(
                        bigram_cond,
                        prev_bigram_cond,
                        bigram_support,
                        cfg.bigram_min_support,
                    )
                else:
                    bigram_tv_p95 = float("nan")
                    eligible_bigram_contexts = 0

                if counts3 is not None and prev_trigram_cond is not None:
                    trigram_tv_p95, eligible_trigram_contexts = p95_tv_trigram(
                        trigram_cond,
                        prev_trigram_cond,
                        trigram_support,
                        cfg.trigram_min_support,
                    )
                else:
                    trigram_tv_p95 = float("nan")
                    eligible_trigram_contexts = 0

            prev_unigram = unigram
            if counts2 is not None:
                prev_bigram_cond = bigram_cond
            if counts3 is not None:
                prev_trigram_cond = trigram_cond

            can_check_convergence = chain_tokens >= cfg.min_tokens_before_convergence
            passes = can_check_convergence and (unigram_delta <= cfg.unigram_max_delta_threshold)
            if counts2 is not None:
                passes = passes and (bigram_tv_p95 <= cfg.bigram_tv_p95_threshold)
            if counts3 is not None:
                passes = passes and (trigram_tv_p95 <= cfg.trigram_tv_p95_threshold)

            if passes:
                consecutive_converged += 1
            else:
                consecutive_converged = 0

            if consecutive_converged >= cfg.window_checkpoints:
                converged = True
                stop_reason = "converged"
                convergence_checkpoint = chain_tokens

            row = {
                "seed": seed,
                "tokens_generated": chain_tokens,
                "elapsed_sec": float(time.time() - t0),
                "unigram_max_delta": maybe_nan_to_none(unigram_delta),
                "bigram_tv_p95": maybe_nan_to_none(bigram_tv_p95),
                "trigram_tv_p95": maybe_nan_to_none(trigram_tv_p95),
                "eligible_bigram_contexts": int(eligible_bigram_contexts),
                "eligible_trigram_contexts": int(eligible_trigram_contexts),
                "converged": bool(converged),
                "stop_reason_candidate": stop_reason if converged else "continue",
            }
            ckpt_f.write(json.dumps(row, sort_keys=True) + "\n")
            ckpt_f.flush()

            print(
                f"[seed={seed}] tokens={chain_tokens} "
                f"uni_max_delta={row['unigram_max_delta']} "
                f"bi_tv_p95={row['bigram_tv_p95']} tri_tv_p95={row['trigram_tv_p95']} "
                f"eligible(bi,tri)=({eligible_bigram_contexts},{eligible_trigram_contexts}) "
                f"converged={converged}",
                flush=True,
            )

            if converged:
                break

    final_tokens = int(counts1.sum())
    final_unigram = counts1.astype(np.float64) / float(final_tokens)

    if traj_tokens[traj_size - 1] != final_tokens:
        append_trajectory_point(chain_tokens=final_tokens)

    if next_prob_size == 0 or next_prob_tokens[next_prob_size - 1] != final_tokens:
        with torch.inference_mode():
            final_outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            final_probs = torch.softmax(final_outputs.logits[:, -1, :].float(), dim=-1)
        append_next_prob_point(
            chain_tokens=final_tokens,
            probs_vec=final_probs[0].detach().cpu().numpy(),
        )

    np.savez_compressed(
        seed_dir / "token_trajectories.npz",
        tokens=traj_tokens[:traj_size],
        unigram_freqs=traj_unigram[:traj_size],
        next_prob_tokens=next_prob_tokens[:next_prob_size],
        next_token_probs=next_token_probs[:next_prob_size],
    )

    summary = {
        "model_id": cfg.model_id,
        "seed": seed,
        "start_token_mode": cfg.start_token,
        "start_token_id": start_token_id,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_positions,
        "effective_context_window": max_positions,
        "max_order": cfg.max_order,
        "final_tokens": final_tokens,
        "stop_reason": stop_reason,
        "runtime_sec": float(time.time() - t0),
        "converged": bool(converged),
        "convergence_checkpoint": convergence_checkpoint,
        "thresholds": {
            "unigram_max_delta": cfg.unigram_max_delta_threshold,
            "bigram_tv_p95": cfg.bigram_tv_p95_threshold,
            "trigram_tv_p95": cfg.trigram_tv_p95_threshold,
            "window_checkpoints": cfg.window_checkpoints,
        },
        "supports": {
            "bigram_min_support": cfg.bigram_min_support,
            "trigram_min_support": cfg.trigram_min_support,
        },
        "final_unigram": final_unigram.tolist(),
    }
    save_json(seed_dir / "summary.json", summary)

    plot_seed_outputs(seed_dir=seed_dir, plot_seed_dir=Path(cfg.plot_dir) / f"seed_{seed}", seed=seed, vocab_size=vocab_size)

    return summary


def write_seed_csv(summaries: List[Dict[str, Any]], csv_path: Path) -> None:
    fields = [
        "seed",
        "final_tokens",
        "converged",
        "stop_reason",
        "convergence_checkpoint",
        "runtime_sec",
        "vocab_size",
        "max_position_embeddings",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k) for k in fields})


def plot_aggregate_outputs(out_dir: Path, plot_dir: Path, summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        return

    ensure_dir(plot_dir)

    thresholds = summaries[0]["thresholds"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
    metric_names = ["unigram_max_delta", "bigram_tv_p95", "trigram_tv_p95"]
    ylabels = ["Unigram max |Δ|", "Bigram TV p95", "Trigram TV p95"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i, (metric_name, ylabel, color) in enumerate(zip(metric_names, ylabels, colors)):
        for summary in summaries:
            seed = summary["seed"]
            ckpt_file = out_dir / f"seed_{seed}" / "checkpoints.jsonl"
            xs: List[int] = []
            ys: List[float] = []
            with ckpt_file.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    xs.append(row["tokens_generated"])
                    val = row.get(metric_name)
                    ys.append(float("nan") if val is None else float(val))
            if xs:
                axes[i].plot(xs, ys, alpha=0.85, linewidth=1.2, label=f"seed {seed}")

        thr_key = metric_name
        axes[i].axhline(thresholds[thr_key], color=color, linestyle="--", alpha=0.7)
        axes[i].set_ylabel(ylabel)
        axes[i].grid(alpha=0.25)

    axes[2].set_xlabel("Total tokens in chain")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Aggregate Stability Curves Across Seeds")
    fig.tight_layout()
    fig.savefig(plot_dir / "aggregate_stability_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    seed_ids = [s["seed"] for s in summaries]
    unigram_matrix = np.array([s["final_unigram"] for s in summaries], dtype=np.float64)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    im = ax2.imshow(unigram_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax2.set_xlabel("Token id")
    ax2.set_ylabel("Seed index")
    ax2.set_yticks(np.arange(len(seed_ids)))
    ax2.set_yticklabels([str(x) for x in seed_ids])
    ax2.set_title("Final unigram frequencies by seed")
    cbar = fig2.colorbar(im, ax=ax2)
    cbar.set_label("Frequency")
    fig2.tight_layout()
    fig2.savefig(plot_dir / "aggregate_final_unigram_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    args = parse_args()

    seeds = parse_seed_list(args.seeds, args.num_seeds)
    if args.parallel_workers < 1:
        raise ValueError("--parallel-workers must be >= 1")
    if args.trajectory_interval < 1:
        raise ValueError("--trajectory-interval must be >= 1")
    if args.parallel_workers > len(seeds):
        print(
            f"[info] reducing parallel workers from {args.parallel_workers} to {len(seeds)} to match number of seeds",
            flush=True,
        )
        args.parallel_workers = len(seeds)

    ensure_dir(Path(args.out_dir))
    ensure_dir(Path(args.plot_dir))

    cfg = ExperimentConfig(
        model_id=args.model_id,
        max_order=args.max_order,
        start_token=args.start_token,
        device=args.device,
        max_tokens=args.max_tokens,
        min_tokens_before_convergence=args.min_tokens_before_convergence,
        checkpoint_interval=args.checkpoint_interval,
        window_checkpoints=args.window_checkpoints,
        unigram_max_delta_threshold=args.unigram_max_delta_threshold,
        bigram_tv_p95_threshold=args.bigram_tv_p95_threshold,
        trigram_tv_p95_threshold=args.trigram_tv_p95_threshold,
        bigram_min_support=args.bigram_min_support,
        trigram_min_support=args.trigram_min_support,
        save_every_tokens=args.save_every_tokens,
        trajectory_interval=args.trajectory_interval,
        out_dir=args.out_dir,
        plot_dir=args.plot_dir,
        num_threads_per_worker=args.num_threads_per_worker,
    )

    config_out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "parallel_workers": args.parallel_workers,
        "config": asdict(cfg),
    }
    save_json(Path(args.out_dir) / "run_config.json", config_out)

    print("=== Token Stability Experiment ===", flush=True)
    print(json.dumps(config_out, indent=2), flush=True)

    summaries: List[Dict[str, Any]] = []

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.parallel_workers, mp_context=ctx) as executor:
        fut_to_seed = {executor.submit(run_single_seed, seed, cfg): seed for seed in seeds}
        for fut in as_completed(fut_to_seed):
            seed = fut_to_seed[fut]
            summary = fut.result()
            summaries.append(summary)
            print(
                f"[done] seed={seed} stop_reason={summary['stop_reason']} "
                f"final_tokens={summary['final_tokens']} runtime_sec={summary['runtime_sec']:.2f}",
                flush=True,
            )

    summaries = sorted(summaries, key=lambda x: int(x["seed"]))
    for summary in summaries:
        save_json(Path(args.out_dir) / f"seed_{summary['seed']}" / "summary.json", summary)

    write_seed_csv(summaries, Path(args.out_dir) / "seed_summaries.csv")
    plot_aggregate_outputs(Path(args.out_dir), Path(args.plot_dir), summaries)

    converged_count = sum(1 for s in summaries if s.get("converged"))
    final_tokens = [int(s["final_tokens"]) for s in summaries]

    aggregate = {
        "model_id": cfg.model_id,
        "vocab_size": int(summaries[0]["vocab_size"] if summaries else 0),
        "seeds": [int(s["seed"]) for s in summaries],
        "num_seeds": len(summaries),
        "converged_seeds": converged_count,
        "convergence_rate": float(converged_count / len(summaries)) if summaries else 0.0,
        "final_tokens": {
            "min": int(min(final_tokens)) if final_tokens else 0,
            "max": int(max(final_tokens)) if final_tokens else 0,
            "mean": float(np.mean(final_tokens)) if final_tokens else 0.0,
            "median": float(np.median(final_tokens)) if final_tokens else 0.0,
        },
        "seed_summaries": summaries,
    }
    save_json(Path(args.out_dir) / "aggregate_summary.json", aggregate)

    print("=== Aggregate Summary ===", flush=True)
    print(json.dumps({k: v for k, v in aggregate.items() if k != "seed_summaries"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
