#!/usr/bin/env python3
"""
General order-k Markov transformer experiments for LPE.

This script implements:
- Step 1: Train/evaluate a transformer against Bayes-optimal next-token loss
- Step 2: Posterior sampling via model rollouts and comparison to analytic posterior
- Step 3: LPE probability estimates for fixed target strings from posterior samples
- Step 4: Artifact generation suitable for LaTeX reporting

Design constraints from lpe/todo.md:
- Supports optional positional encoding (disabled by default)
- Training sequence length equals inference rollout length (100 * 2^k)
- No sliding-window generation for posterior rollouts
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from markov_transformer import MarkovTransformer  # noqa: E402


@dataclass
class ModelConfig:
    n_layers: int
    d_model: int
    n_heads: int
    d_mlp: int

    def label(self) -> str:
        return f"L{self.n_layers}_D{self.d_model}_H{self.n_heads}_M{self.d_mlp}"


class PositionalMarkovTransformer(MarkovTransformer):
    """MarkovTransformer variant with learned absolute positional embeddings."""

    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_mlp: int,
        use_prenorm: bool = True,
    ):
        if max_seq_len is None:
            raise ValueError("PositionalMarkovTransformer requires max_seq_len.")
        super().__init__(
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_mlp=d_mlp,
            use_prenorm=use_prenorm,
        )
        self.pos_emb = torch.nn.Embedding(max_seq_len, d_model)
        torch.nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        if self.max_seq_len is not None and T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}")
        tok = self.token_emb(x)
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = tok + self.pos_emb(pos_ids)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.output_proj(x)
        return logits


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


def parse_k_list(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k < 1:
            raise ValueError("k must be >= 1")
        out.append(k)
    if not out:
        raise ValueError("empty --k-list")
    return sorted(set(out))


def default_batch_size_for_k(k: int) -> int:
    # Keeps batch_size * seq_len roughly constant because seq_len = 100*2^k.
    return max(1, 64 // (2 ** (k - 1)))


def default_context_length_for_k(k: int) -> int:
    return max(2 * k, 16)


def default_max_seq_len_for_k(k: int, context_margin: int = 32) -> int:
    return (100 * (2**k)) + context_margin


def default_model_grid() -> List[ModelConfig]:
    return [
        ModelConfig(2, 64, 4, 256),
        ModelConfig(3, 96, 6, 384),
        ModelConfig(4, 128, 8, 512),
        ModelConfig(6, 128, 8, 512),
    ]


def build_override_config(args: argparse.Namespace, k: int) -> Optional[ModelConfig]:
    if args.n_layers is None and args.d_model is None and args.n_heads is None and args.d_mlp is None:
        return None
    if args.n_layers is None or args.d_model is None:
        raise ValueError("When overriding model config, provide both --n-layers and --d-model.")
    d_model = int(args.d_model)
    n_layers = int(args.n_layers)
    n_heads = int(args.n_heads) if args.n_heads is not None else max(1, d_model // 32)
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
    d_mlp = int(args.d_mlp) if args.d_mlp is not None else d_model * 4
    if k >= 5 and d_model < 128:
        print(f"[k={k}] Warning: d_model={d_model} may be too small for higher-order setting.", flush=True)
    return ModelConfig(n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp)


def model_config_for_k(k: int) -> ModelConfig:
    if k <= 2:
        return ModelConfig(2, 64, 4, 256)
    if k <= 4:
        return ModelConfig(3, 96, 6, 384)
    return ModelConfig(4, 128, 8, 512)


def _state_powers(k: int, device: torch.device) -> torch.Tensor:
    return (2 ** torch.arange(k - 1, -1, -1, device=device, dtype=torch.long)).long()


def encode_state_windows(windows: torch.Tensor) -> torch.Tensor:
    """Encode (..., k) binary windows to integer state IDs."""
    if windows.numel() == 0:
        return torch.zeros(windows.shape[:-1], dtype=torch.long, device=windows.device)
    k = windows.shape[-1]
    powers = _state_powers(k, windows.device)
    return (windows.long() * powers).sum(dim=-1).long()


def sample_theta_batch(
    batch_size: int,
    num_states: int,
    alpha: float,
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    if abs(alpha - 1.0) < 1e-12 and abs(beta - 1.0) < 1e-12:
        return torch.rand(batch_size, num_states, device=device)
    a = torch.full((num_states,), alpha, device=device)
    b = torch.full((num_states,), beta, device=device)
    dist = torch.distributions.Beta(a, b)
    return dist.sample((batch_size,))


def sample_markov_k_batch(
    batch_size: int,
    seq_len: int,
    k: int,
    alpha: float,
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample sequences from an order-k binary Markov process.

    First k bits are Bernoulli(0.5), then x_t ~ Bernoulli(theta[state(x_{t-k:t}))).
    """
    num_states = 1 << k
    seq = torch.randint(0, 2, (batch_size, seq_len), device=device, dtype=torch.long)
    if seq_len <= k:
        return seq

    theta = sample_theta_batch(batch_size, num_states, alpha=alpha, beta=beta, device=device)
    mask = num_states - 1
    state = encode_state_windows(seq[:, :k])
    batch_idx = torch.arange(batch_size, device=device)

    for t in range(k, seq_len):
        probs = theta[batch_idx, state]
        seq[:, t] = (torch.rand(batch_size, device=device) < probs).long()
        state = ((state << 1) & mask) | seq[:, t]

    return seq


def compute_transition_counts(sequence: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Counts transitions for each k-bit state into next bit 1/0."""
    num_states = 1 << k
    ones = torch.zeros(num_states, dtype=torch.float64)
    zeros = torch.zeros(num_states, dtype=torch.float64)
    seq = sequence.detach().cpu().long()
    T = int(seq.numel())
    if T <= k:
        return ones, zeros

    for t in range(k, T):
        window = seq[t - k : t]
        state = int(encode_state_windows(window.unsqueeze(0)).item())
        bit = int(seq[t].item())
        if bit == 1:
            ones[state] += 1.0
        else:
            zeros[state] += 1.0
    return ones, zeros


def compute_posterior_params(
    context: torch.Tensor,
    k: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ones, zeros = compute_transition_counts(context, k)
    alpha_post = ones + alpha
    beta_post = zeros + beta
    return alpha_post, beta_post


def bayes_next_prob(
    context: torch.Tensor,
    k: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    if context.numel() < k:
        return 0.5
    alpha_post, beta_post = compute_posterior_params(context, k, alpha=alpha, beta=beta)
    state = int(encode_state_windows(context[-k:].unsqueeze(0)).item())
    a = float(alpha_post[state].item())
    b = float(beta_post[state].item())
    return a / (a + b)


def markov_sequence_logprob_given_theta(
    context: torch.Tensor,
    target: torch.Tensor,
    theta: torch.Tensor,
    k: int,
) -> float:
    hist = context.detach().cpu().long().tolist()
    tgt = target.detach().cpu().long().tolist()
    theta_np = theta.detach().cpu().double().numpy()
    logp = 0.0
    for bit in tgt:
        if len(hist) < k:
            p = 0.5
        else:
            state = 0
            for b in hist[-k:]:
                state = (state << 1) | int(b)
            p = float(theta_np[state])
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        logp += math.log(p) if bit == 1 else math.log1p(-p)
        hist.append(int(bit))
    return logp


def bayes_target_logprob(
    context: torch.Tensor,
    target: torch.Tensor,
    k: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    alpha_post, beta_post = compute_posterior_params(context, k, alpha=alpha, beta=beta)
    alpha_work = alpha_post.clone().double()
    beta_work = beta_post.clone().double()

    hist = context.detach().cpu().long().tolist()
    tgt = target.detach().cpu().long().tolist()

    logp = 0.0
    for bit in tgt:
        if len(hist) < k:
            p = 0.5
        else:
            state = 0
            for b in hist[-k:]:
                state = (state << 1) | int(b)
            denom = float(alpha_work[state] + beta_work[state])
            if int(bit) == 1:
                p = float(alpha_work[state] / denom)
                alpha_work[state] += 1.0
            else:
                p = float(beta_work[state] / denom)
                beta_work[state] += 1.0
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        logp += math.log(p) if int(bit) == 1 else math.log1p(-p)
        hist.append(int(bit))
    return logp


def bayes_optimal_nll_batch(
    sequences: torch.Tensor,
    k: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Average token NLL for Bayes-optimal predictor under the same prior."""
    seqs = sequences.long()
    B, T = seqs.shape
    if T <= 1:
        return 0.0

    num_states = 1 << k
    device = seqs.device
    alpha_tab = torch.full((B, num_states), alpha, device=device, dtype=torch.float64)
    beta_tab = torch.full((B, num_states), beta, device=device, dtype=torch.float64)
    total_nll = torch.zeros((), device=device, dtype=torch.float64)

    mask = num_states - 1
    batch_idx = torch.arange(B, device=device)
    state = None

    for t in range(1, T):
        y = seqs[:, t].double()
        if t < k:
            p = torch.full((B,), 0.5, device=device, dtype=torch.float64)
        else:
            if t == k:
                state = encode_state_windows(seqs[:, t - k : t])
            assert state is not None
            a = alpha_tab[batch_idx, state]
            b = beta_tab[batch_idx, state]
            p = a / (a + b)

        p = p.clamp(1e-12, 1.0 - 1e-12)
        total_nll += (-(y * torch.log(p) + (1.0 - y) * torch.log1p(-p))).sum()

        if t >= k:
            y_int = seqs[:, t]
            ones_mask = y_int == 1
            zeros_mask = ~ones_mask
            if ones_mask.any():
                alpha_tab[batch_idx[ones_mask], state[ones_mask]] += 1.0
            if zeros_mask.any():
                beta_tab[batch_idx[zeros_mask], state[zeros_mask]] += 1.0
            state = ((state << 1) & mask) | y_int

    return float((total_nll / ((T - 1) * B)).item())


def build_model(config: ModelConfig, max_seq_len: int, use_positional_encoding: bool) -> MarkovTransformer:
    if use_positional_encoding:
        return PositionalMarkovTransformer(
            max_seq_len=max_seq_len,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_mlp=config.d_mlp,
            use_prenorm=True,
        )
    return MarkovTransformer(
        max_seq_len=max_seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_mlp=config.d_mlp,
        use_prenorm=True,
    )


def train_model(
    model: MarkovTransformer,
    k: int,
    seq_len: int,
    batch_size: int,
    grad_accum_steps: int,
    num_steps: int,
    learning_rate: float,
    warmup_steps: int,
    grad_clip: float,
    alpha: float,
    beta: float,
    device: torch.device,
    print_every: int,
    eval_every: int,
    eval_batches: int,
    checkpoint_path: Optional[Path],
    target_gap_ratio: Optional[float] = None,
    min_steps_before_stop: int = 0,
    history_csv_path: Optional[Path] = None,
) -> Dict[str, object]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    grad_accum_steps = max(1, int(grad_accum_steps))

    def get_lr(step: int) -> float:
        if warmup_steps <= 0:
            return learning_rate
        if step < warmup_steps:
            return learning_rate * float(step + 1) / float(warmup_steps)
        return learning_rate

    t0 = time.time()
    recent_losses: List[float] = []
    last_eval_model_nll = float("nan")
    last_eval_bayes_nll = float("nan")
    best_gap_ratio = float("inf")
    best_step = -1
    steps_run = 0
    early_stopped = False
    history_rows: List[Dict[str, object]] = []

    for step in range(num_steps):
        steps_run = step + 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses: List[float] = []
        for _ in range(grad_accum_steps):
            seqs = sample_markov_k_batch(batch_size, seq_len, k, alpha=alpha, beta=beta, device=device)
            inputs = seqs[:, :-1]
            targets = seqs[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
            micro_losses.append(float(loss.item()))
            (loss / grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()

        step_loss = float(np.mean(micro_losses)) if micro_losses else float("nan")
        recent_losses.append(step_loss)
        if len(recent_losses) > 100:
            recent_losses.pop(0)

        if eval_every > 0 and (step + 1) % eval_every == 0:
            eval_stats = evaluate_model_vs_bayes(
                model=model,
                k=k,
                seq_len=seq_len,
                batch_size=max(1, min(batch_size, 8)),
                num_batches=eval_batches,
                alpha=alpha,
                beta=beta,
                device=device,
            )
            last_eval_model_nll = eval_stats["model_nll"]
            last_eval_bayes_nll = eval_stats["bayes_nll"]
            current_gap = float(eval_stats["gap_ratio"])
            if current_gap < best_gap_ratio:
                best_gap_ratio = current_gap
                best_step = step + 1
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), checkpoint_path)
            if (
                target_gap_ratio is not None
                and step + 1 >= int(min_steps_before_stop)
                and current_gap <= float(target_gap_ratio)
            ):
                early_stopped = True
                print(
                    f"[k={k}] Early stop at step {step + 1}: gap={100.0*current_gap:.2f}% "
                    f"(target <= {100.0*float(target_gap_ratio):.2f}%)",
                    flush=True,
                )
                history_rows.append(
                    {
                        "step": int(step + 1),
                        "train_loss": float(step_loss),
                        "lr": float(lr),
                        "elapsed_sec": float(time.time() - t0),
                        "eval_model_nll": float(last_eval_model_nll),
                        "eval_bayes_nll": float(last_eval_bayes_nll),
                        "eval_gap_pct": float(100.0 * current_gap),
                    }
                )
                break

        current_gap_pct = float("nan")
        if not math.isnan(last_eval_model_nll) and not math.isnan(last_eval_bayes_nll):
            current_gap_pct = float(
                100.0 * (last_eval_model_nll - last_eval_bayes_nll) / max(last_eval_bayes_nll, 1e-12)
            )
        history_rows.append(
            {
                "step": int(step + 1),
                "train_loss": float(step_loss),
                "lr": float(lr),
                "elapsed_sec": float(time.time() - t0),
                "eval_model_nll": float(last_eval_model_nll),
                "eval_bayes_nll": float(last_eval_bayes_nll),
                "eval_gap_pct": current_gap_pct,
            }
        )

        if print_every > 0 and (step + 1) % print_every == 0:
            avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            elapsed = time.time() - t0
            msg = (
                f"step {step + 1:6d}/{num_steps} | train_loss={avg_loss:.6f} | lr={lr:.6e} "
                f"| elapsed={elapsed/60.0:.1f}m"
            )
            if not math.isnan(last_eval_model_nll) and not math.isnan(last_eval_bayes_nll):
                gap = (last_eval_model_nll - last_eval_bayes_nll) / max(last_eval_bayes_nll, 1e-12)
                msg += (
                    f" | eval_model_nll={last_eval_model_nll:.6f} | eval_bayes_nll={last_eval_bayes_nll:.6f}"
                    f" | gap={100.0*gap:.2f}%"
                )
            print(msg, flush=True)

    if checkpoint_path is not None and best_step < 0:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    history_csv = ""
    if history_csv_path is not None:
        save_csv(history_csv_path, history_rows)
        history_csv = str(history_csv_path)

    return {
        "final_train_loss": float(np.mean(recent_losses)) if recent_losses else float("nan"),
        "last_eval_model_nll": float(last_eval_model_nll),
        "last_eval_bayes_nll": float(last_eval_bayes_nll),
        "best_gap_ratio": float(best_gap_ratio),
        "best_step": int(best_step),
        "steps_run": int(steps_run),
        "early_stopped": bool(early_stopped),
        "history_csv": history_csv,
    }


def evaluate_model_vs_bayes(
    model: MarkovTransformer,
    k: int,
    seq_len: int,
    batch_size: int,
    num_batches: int,
    alpha: float,
    beta: float,
    device: torch.device,
) -> Dict[str, float]:
    model = model.to(device)
    model.eval()

    total_model_nll = 0.0
    total_bayes_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(num_batches):
            seqs = sample_markov_k_batch(batch_size, seq_len, k, alpha=alpha, beta=beta, device=device)
            inputs = seqs[:, :-1]
            targets = seqs[:, 1:]

            logits = model(inputs)
            loss_sum = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1), reduction="sum")
            token_count = int(targets.numel())

            model_nll = float(loss_sum.item()) / token_count
            bayes_nll = bayes_optimal_nll_batch(seqs, k, alpha=alpha, beta=beta)

            total_model_nll += model_nll * token_count
            total_bayes_nll += bayes_nll * token_count
            total_tokens += token_count

    model_avg = total_model_nll / max(total_tokens, 1)
    bayes_avg = total_bayes_nll / max(total_tokens, 1)
    gap_ratio = (model_avg - bayes_avg) / max(bayes_avg, 1e-12)
    return {
        "model_nll": model_avg,
        "bayes_nll": bayes_avg,
        "gap_ratio": gap_ratio,
    }


def gather_prediction_pairs(
    model: MarkovTransformer,
    k: int,
    num_points: int,
    context_length: int,
    alpha: float,
    beta: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()

    bayes_vals: List[float] = []
    model_vals: List[float] = []
    with torch.no_grad():
        for _ in range(num_points):
            seq = sample_markov_k_batch(1, context_length, k, alpha=alpha, beta=beta, device=device)[0]
            context = seq[:-1]
            bayes_p = bayes_next_prob(context.detach().cpu(), k, alpha=alpha, beta=beta)

            logits = model.predict_next_logits(context.unsqueeze(0))
            p_model = float(F.softmax(logits, dim=-1)[0, 1].item())

            bayes_vals.append(bayes_p)
            model_vals.append(p_model)
    return np.array(bayes_vals), np.array(model_vals)


def plot_prediction_comparison(
    bayes_vals: np.ndarray,
    model_vals: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(bayes_vals, model_vals, s=18, alpha=0.65)
    ax.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.5)
    ax.set_xlabel("Bayes next-bit probability")
    ax.set_ylabel("Model next-bit probability")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_history(history_csv_path: Path, out_path: Path, k: int) -> None:
    if not history_csv_path.exists():
        return

    with history_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return

    steps: List[int] = []
    train_losses: List[float] = []
    eval_steps: List[int] = []
    eval_model_nll: List[float] = []
    eval_bayes_nll: List[float] = []
    eval_gap_pct: List[float] = []

    for r in rows:
        step = int(float(r.get("step", "0") or 0))
        tl = float(r.get("train_loss", "nan") or "nan")
        em = float(r.get("eval_model_nll", "nan") or "nan")
        eb = float(r.get("eval_bayes_nll", "nan") or "nan")
        eg = float(r.get("eval_gap_pct", "nan") or "nan")
        steps.append(step)
        train_losses.append(tl)
        if math.isfinite(em) and math.isfinite(eb):
            eval_steps.append(step)
            eval_model_nll.append(em)
            eval_bayes_nll.append(eb)
            eval_gap_pct.append(eg)

    if not steps:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(steps, train_losses, color="tab:blue", linewidth=1.2, label="train loss")
    if eval_steps:
        ax1.plot(eval_steps, eval_model_nll, color="tab:orange", linewidth=1.4, label="eval model NLL")
        ax1.plot(eval_steps, eval_bayes_nll, color="tab:green", linewidth=1.2, label="eval Bayes NLL")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Loss / NLL")
    ax1.set_title(f"k={k} learning curve")
    ax1.grid(alpha=0.25)

    handles1, labels1 = ax1.get_legend_handles_labels()
    if eval_steps:
        ax2 = ax1.twinx()
        ax2.plot(eval_steps, eval_gap_pct, color="tab:red", linestyle="--", linewidth=1.2, label="eval gap (%)")
        ax2.set_ylabel("Gap (%)")
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def posterior_samples_from_rollouts(
    model: MarkovTransformer,
    context: torch.Tensor,
    k: int,
    num_samples: int,
    rollout_length: int,
    device: torch.device,
    rollout_batch_size: int = 1,
) -> np.ndarray:
    """Posterior sampling proxy from rollout transition frequencies."""
    model = model.to(device)
    model.eval()

    num_states = 1 << k
    samples = np.full((num_samples, num_states), 0.5, dtype=np.float64)

    with torch.no_grad():
        rollout_batch_size = max(1, int(rollout_batch_size))
        offset = 0
        while offset < num_samples:
            bsz = min(rollout_batch_size, num_samples - offset)
            generated_batch = rollout_with_cache_batch(
                model=model,
                prefix=context.to(device),
                length=rollout_length,
                batch_size=bsz,
                temperature=1.0,
            )
            for j in range(bsz):
                generated = generated_batch[j]
                full = torch.cat([context.to(device), generated.long()], dim=0).detach().cpu().long()
                ones = np.zeros(num_states, dtype=np.float64)
                zeros = np.zeros(num_states, dtype=np.float64)

                start_t = max(k, int(context.numel()))
                for t in range(start_t, int(full.numel())):
                    state = 0
                    for b in full[t - k : t].tolist():
                        state = (state << 1) | int(b)
                    if int(full[t].item()) == 1:
                        ones[state] += 1.0
                    else:
                        zeros[state] += 1.0

                denom = ones + zeros
                row = np.where(denom > 0.0, ones / np.maximum(denom, 1e-12), 0.5)
                samples[offset + j] = row
            offset += bsz

    return samples


def rollout_with_cache(
    model: MarkovTransformer,
    prefix: torch.Tensor,
    length: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Incremental rollout with per-layer KV-like caches.

    This avoids recomputing the full forward pass from scratch at every token.
    It is equivalent to causal generation because each new query attends only
    to previous cached states and itself.
    """
    if prefix.dim() != 1:
        raise ValueError(f"Expected 1D prefix, got shape {tuple(prefix.shape)}")
    if prefix.numel() == 0:
        raise ValueError("Empty prefix is not supported for rollout.")
    if length <= 0:
        return torch.empty(0, dtype=prefix.dtype, device=prefix.device)

    model.eval()
    device = prefix.device
    prefix_len = int(prefix.numel())
    if model.max_seq_len is not None:
        max_allowed = int(model.max_seq_len) - prefix_len
        if max_allowed <= 0:
            return torch.empty(0, dtype=prefix.dtype, device=device)
        actual_length = min(length, max_allowed)
    else:
        actual_length = length
    if actual_length <= 0:
        return torch.empty(0, dtype=prefix.dtype, device=device)

    total_len = prefix_len + actual_length
    d_model = int(model.d_model)

    # Cache stores per-layer tensors used as K/V for attention.
    caches: List[torch.Tensor] = []
    for _ in range(len(model.blocks)):
        caches.append(torch.empty(total_len, d_model, device=device, dtype=model.token_emb.weight.dtype))

    def step_token(token_id: torch.Tensor, pos: int) -> torch.Tensor:
        x = model.token_emb(token_id.view(1, 1))
        if hasattr(model, "pos_emb"):
            pos_ids = torch.tensor([[pos]], device=device, dtype=torch.long)
            x = x + model.pos_emb(pos_ids)
        for layer_idx, block in enumerate(model.blocks):
            if model.use_prenorm:
                h = block.ln1(x)
                caches[layer_idx][pos] = h[0, 0]
                kv = caches[layer_idx][: pos + 1].unsqueeze(0)
                attn_out, _ = block.attn(h, kv, kv, attn_mask=None, need_weights=False)
                x = x + attn_out
                h2 = block.ln2(x)
                x = x + block.mlp(h2)
            else:
                caches[layer_idx][pos] = x[0, 0]
                kv = caches[layer_idx][: pos + 1].unsqueeze(0)
                attn_out, _ = block.attn(x, kv, kv, attn_mask=None, need_weights=False)
                x = x + attn_out
                x = block.ln1(x)
                h2 = block.mlp(x)
                x = x + h2
                x = block.ln2(x)
        x = model.ln_f(x)
        logits = model.output_proj(x)[:, -1, :]
        return logits

    logits = None
    for pos in range(prefix_len):
        logits = step_token(prefix[pos], pos)
    if logits is None:
        raise RuntimeError("Failed to compute initial logits from prefix.")

    generated: List[int] = []
    for i in range(actual_length):
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = int(torch.multinomial(probs[0], num_samples=1).item())
        generated.append(next_token)
        logits = step_token(torch.tensor(next_token, device=device, dtype=torch.long), prefix_len + i)

    return torch.tensor(generated, dtype=prefix.dtype, device=device)


def rollout_with_cache_batch(
    model: MarkovTransformer,
    prefix: torch.Tensor,
    length: int,
    batch_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Batched incremental rollout with per-layer caches.

    Returns tensor of shape (batch_size, generated_length).
    """
    if prefix.dim() != 1:
        raise ValueError(f"Expected 1D prefix, got shape {tuple(prefix.shape)}")
    if prefix.numel() == 0:
        raise ValueError("Empty prefix is not supported for rollout.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if length <= 0:
        return torch.empty(batch_size, 0, dtype=prefix.dtype, device=prefix.device)

    model.eval()
    device = prefix.device
    prefix_len = int(prefix.numel())
    if model.max_seq_len is not None:
        max_allowed = int(model.max_seq_len) - prefix_len
        if max_allowed <= 0:
            return torch.empty(batch_size, 0, dtype=prefix.dtype, device=device)
        actual_length = min(length, max_allowed)
    else:
        actual_length = length
    if actual_length <= 0:
        return torch.empty(batch_size, 0, dtype=prefix.dtype, device=device)

    total_len = prefix_len + actual_length
    d_model = int(model.d_model)
    cache_dtype = model.token_emb.weight.dtype

    caches: List[torch.Tensor] = []
    for _ in range(len(model.blocks)):
        caches.append(torch.empty(batch_size, total_len, d_model, device=device, dtype=cache_dtype))

    prefix_batch = prefix.unsqueeze(0).expand(batch_size, -1)

    def step_tokens(token_ids: torch.Tensor, pos: int) -> torch.Tensor:
        x = model.token_emb(token_ids.long()).unsqueeze(1)  # (B, 1, D)
        if hasattr(model, "pos_emb"):
            pos_ids = torch.full((batch_size, 1), pos, device=device, dtype=torch.long)
            x = x + model.pos_emb(pos_ids)
        for layer_idx, block in enumerate(model.blocks):
            if model.use_prenorm:
                h = block.ln1(x)
                caches[layer_idx][:, pos, :] = h[:, 0, :]
                kv = caches[layer_idx][:, : pos + 1, :]
                attn_out, _ = block.attn(h, kv, kv, attn_mask=None, need_weights=False)
                x = x + attn_out
                h2 = block.ln2(x)
                x = x + block.mlp(h2)
            else:
                caches[layer_idx][:, pos, :] = x[:, 0, :]
                kv = caches[layer_idx][:, : pos + 1, :]
                attn_out, _ = block.attn(x, kv, kv, attn_mask=None, need_weights=False)
                x = x + attn_out
                x = block.ln1(x)
                h2 = block.mlp(x)
                x = x + h2
                x = block.ln2(x)
        x = model.ln_f(x)
        logits = model.output_proj(x)[:, -1, :]  # (B, 2)
        return logits

    logits = None
    for pos in range(prefix_len):
        logits = step_tokens(prefix_batch[:, pos], pos)
    if logits is None:
        raise RuntimeError("Failed to compute initial logits from prefix.")

    generated = torch.empty(batch_size, actual_length, dtype=prefix.dtype, device=device)
    for i in range(actual_length):
        probs = F.softmax(logits / temperature, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1).long()
        generated[:, i] = next_tokens
        logits = step_tokens(next_tokens, prefix_len + i)

    return generated


def plot_posterior_mean_scatter(
    true_mean: np.ndarray,
    sample_mean: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(true_mean, sample_mean, s=22, alpha=0.75)
    ax.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.5)
    ax.set_xlabel("Analytic posterior mean")
    ax.set_ylabel("Rollout-sampled posterior mean")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_arch_sweep(
    k: int,
    seq_len: int,
    max_seq_len: int,
    use_positional_encoding: bool,
    grid: Sequence[ModelConfig],
    sweep_steps: int,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    warmup_steps: int,
    grad_clip: float,
    alpha: float,
    beta: float,
    device: torch.device,
    gap_target: float,
    artifacts_dir: Path,
    print_every: int,
    eval_every: int,
    eval_batches: int,
) -> Dict[str, object]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sweep_log_path = artifacts_dir / f"k{k}_sweep.jsonl"

    results: List[Dict[str, object]] = []
    chosen_idx = 0
    best_gap = float("inf")

    with sweep_log_path.open("w", encoding="utf-8") as fp:
        for idx, cfg in enumerate(grid):
            print(f"\n[k={k}] sweep {idx + 1}/{len(grid)}: {cfg.label()}", flush=True)
            model = build_model(cfg, max_seq_len=max_seq_len, use_positional_encoding=use_positional_encoding)
            sweep_warmup = min(max(1, warmup_steps // 2), max(1, sweep_steps // 10))
            train_stats = train_model(
                model=model,
                k=k,
                seq_len=seq_len,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                num_steps=sweep_steps,
                learning_rate=learning_rate,
                warmup_steps=sweep_warmup,
                grad_clip=grad_clip,
                alpha=alpha,
                beta=beta,
                device=device,
                print_every=print_every,
                eval_every=max(1, eval_every),
                eval_batches=eval_batches,
                checkpoint_path=None,
                target_gap_ratio=None,
                min_steps_before_stop=0,
            )
            eval_stats = evaluate_model_vs_bayes(
                model=model,
                k=k,
                seq_len=seq_len,
                batch_size=max(1, min(batch_size, 8)),
                num_batches=max(2, eval_batches),
                alpha=alpha,
                beta=beta,
                device=device,
            )

            record: Dict[str, object] = {
                "k": k,
                "config": asdict(cfg),
                "config_label": cfg.label(),
                "sweep_steps": sweep_steps,
                "batch_size": batch_size,
                "train_stats": train_stats,
                "eval_stats": eval_stats,
                "timestamp": time.time(),
            }
            results.append(record)
            fp.write(json.dumps(record) + "\n")
            fp.flush()

            gap = float(eval_stats["gap_ratio"])
            if gap < best_gap:
                best_gap = gap
                chosen_idx = idx
            if gap <= gap_target:
                chosen_idx = idx
                break

    return {
        "chosen_config": asdict(grid[chosen_idx]),
        "chosen_config_label": grid[chosen_idx].label(),
        "best_gap": best_gap,
        "results": results,
        "sweep_log_path": str(sweep_log_path),
    }


def binary_de_bruijn(order: int) -> List[int]:
    """Binary de Bruijn cycle B(2, order) as a bit list."""
    if order <= 0:
        return [0]

    a = [0] * (2 * order + 1)
    seq: List[int] = []

    def db(t: int, p: int) -> None:
        if t > order:
            if order % p == 0:
                seq.extend(a[1 : p + 1])
            return
        a[t] = a[t - p]
        db(t + 1, p)
        for j in range(a[t - p] + 1, 2):
            a[t] = j
            db(t + 1, t)

    db(1, 1)
    return seq


def make_target_bits(
    k: int,
    target_len: int,
    mode: str,
    rng: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Build target bits for LPE."""
    if target_len <= 0:
        return torch.empty(0, dtype=torch.long)

    if mode == "random":
        return (
            torch.randint(0, 2, (target_len,), generator=rng, device=device, dtype=torch.long)
            .detach()
            .cpu()
            .long()
        )

    if mode == "balanced":
        # For k>=1, B(2, k+1) balances 0/1 continuations per k-state over a full cycle.
        base = binary_de_bruijn(k + 1) if k > 0 else [0, 1]
        if not base:
            base = [0, 1]
        offset = int(torch.randint(0, len(base), (1,), generator=rng, device=device).item())
        cycle = base[offset:] + base[:offset]
        reps = (target_len + len(cycle) - 1) // len(cycle)
        bits = (cycle * reps)[:target_len]
        return torch.tensor(bits, dtype=torch.long).detach().cpu().long()

    raise ValueError(f"Unknown target mode: {mode}")


def generate_contexts_and_target(
    k: int,
    num_contexts: int,
    min_context_len: int,
    max_context_len: int,
    target_len: int,
    alpha: float,
    beta: float,
    device: torch.device,
    seed: int,
    target_mode: str,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    contexts: List[torch.Tensor] = []
    for _ in range(num_contexts):
        clen = int(torch.randint(min_context_len, max_context_len + 1, (1,), generator=g, device=device).item())
        seq = sample_markov_k_batch(1, clen, k, alpha=alpha, beta=beta, device=device)[0].detach().cpu().long()
        contexts.append(seq)

    target = make_target_bits(
        k=k,
        target_len=target_len,
        mode=target_mode,
        rng=g,
        device=device,
    )
    return contexts, target


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def run_single_k(
    k: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(int(args.seed) + 1000 * k)

    rollout_len = int(args.rollout_length_mult) * (2**k)
    train_seq_len = rollout_len
    if args.max_seq_len is not None:
        max_seq_len = int(args.max_seq_len)
    else:
        max_seq_len = max(default_max_seq_len_for_k(k, context_margin=int(args.context_margin)), train_seq_len + int(args.context_margin))

    if max_seq_len < train_seq_len:
        raise ValueError(f"max_seq_len ({max_seq_len}) must be >= train_seq_len ({train_seq_len})")

    batch_size = int(args.batch_size) if args.batch_size is not None else default_batch_size_for_k(k)
    context_len = int(args.context_length) if args.context_length is not None else default_context_length_for_k(k)

    out_root = Path(args.out_dir)
    k_dir = out_root / f"k{k}"
    fig_dir = k_dir / "figures"
    k_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint_root) / f"k{k}" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    override_config = build_override_config(args, k=k)
    if override_config is not None:
        config = override_config
        sweep = {
            "chosen_config": asdict(config),
            "chosen_config_label": config.label(),
            "best_gap": float("nan"),
            "results": [],
            "sweep_log_path": "",
            "override": True,
        }
    elif args.use_sweep:
        grid = default_model_grid()
        sweep = run_arch_sweep(
            k=k,
            seq_len=train_seq_len,
            max_seq_len=max_seq_len,
            use_positional_encoding=bool(args.use_positional_encoding),
            grid=grid,
            sweep_steps=int(args.sweep_steps),
            batch_size=batch_size,
            grad_accum_steps=int(args.grad_accum_steps),
            learning_rate=float(args.learning_rate),
            warmup_steps=int(args.warmup_steps),
            grad_clip=float(args.grad_clip),
            alpha=float(args.alpha),
            beta=float(args.beta),
            device=device,
            gap_target=float(args.gap_target),
            artifacts_dir=k_dir,
            print_every=int(args.print_every),
            eval_every=int(args.eval_every),
            eval_batches=int(args.eval_batches),
        )
        cfg_dict = sweep["chosen_config"]
        config = ModelConfig(**cfg_dict)
    else:
        config = model_config_for_k(k)
        sweep = {
            "chosen_config": asdict(config),
            "chosen_config_label": config.label(),
            "best_gap": float("nan"),
            "results": [],
            "sweep_log_path": "",
        }

    model = build_model(
        config,
        max_seq_len=max_seq_len,
        use_positional_encoding=bool(args.use_positional_encoding),
    )

    should_train = True
    if args.skip_existing and checkpoint_path.exists():
        print(f"[k={k}] using existing checkpoint: {checkpoint_path}", flush=True)
        should_train = False

    train_stats: Dict[str, object] = {}
    history_csv_path = k_dir / "training_history.csv"
    training_curve_path = fig_dir / "step1_training_curve.png"
    if should_train:
        # Use token budget unless explicit steps were set.
        if args.num_steps is not None:
            num_steps = int(args.num_steps)
        else:
            tokens_per_step = max(1, batch_size * (train_seq_len - 1))
            num_steps = max(1, int(math.ceil(float(args.target_train_tokens) / float(tokens_per_step))))
        effective_warmup_steps = min(max(1, int(args.warmup_steps)), max(1, num_steps // 10))
        print(
            f"[k={k}] training config={config.label()} seq_len={train_seq_len} rollout_len={rollout_len} "
            f"batch_size={batch_size} steps={num_steps} warmup_steps={effective_warmup_steps} "
            f"grad_accum_steps={int(args.grad_accum_steps)} max_seq_len={max_seq_len} "
            f"positional_encoding={bool(args.use_positional_encoding)}",
            flush=True,
        )
        train_stats = train_model(
            model=model,
            k=k,
            seq_len=train_seq_len,
            batch_size=batch_size,
            grad_accum_steps=int(args.grad_accum_steps),
            num_steps=num_steps,
            learning_rate=float(args.learning_rate),
            warmup_steps=effective_warmup_steps,
            grad_clip=float(args.grad_clip),
            alpha=float(args.alpha),
            beta=float(args.beta),
            device=device,
            print_every=int(args.print_every),
            eval_every=int(args.eval_every),
            eval_batches=int(args.eval_batches),
            checkpoint_path=checkpoint_path,
            target_gap_ratio=float(args.target_gap_pct) / 100.0 if args.target_gap_pct is not None else None,
            min_steps_before_stop=int(args.min_steps_before_stop),
            history_csv_path=history_csv_path,
        )
        plot_training_history(history_csv_path=history_csv_path, out_path=training_curve_path, k=k)
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        if history_csv_path.exists():
            plot_training_history(history_csv_path=history_csv_path, out_path=training_curve_path, k=k)

    if checkpoint_path.exists() and should_train:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    eval_stats = evaluate_model_vs_bayes(
        model=model,
        k=k,
        seq_len=train_seq_len,
        batch_size=max(1, min(batch_size, int(args.eval_batch_size))),
        num_batches=int(args.eval_batches),
        alpha=float(args.alpha),
        beta=float(args.beta),
        device=device,
    )

    bayes_vals, model_vals = gather_prediction_pairs(
        model=model,
        k=k,
        num_points=int(args.num_prediction_points),
        context_length=max(context_len, k + 1),
        alpha=float(args.alpha),
        beta=float(args.beta),
        device=device,
    )
    pred_plot_path = fig_dir / "step1_model_vs_bayes.png"
    plot_prediction_comparison(
        bayes_vals,
        model_vals,
        pred_plot_path,
        title=f"k={k} next-bit predictions",
    )

    gap_pct = float(eval_stats["gap_ratio"] * 100.0)
    gate_failed = False
    if args.require_gap_pct is not None:
        gate_failed = bool(gap_pct > float(args.require_gap_pct))

    if gate_failed and args.fail_on_gap:
        raise RuntimeError(
            f"k={k} quality gate failed: gap={gap_pct:.3f}% > required {float(args.require_gap_pct):.3f}%"
        )

    if args.step1_only:
        k_summary_step1: Dict[str, object] = {
            "k": k,
            "device": str(device),
            "model_config": asdict(config),
            "model_config_label": config.label(),
            "use_positional_encoding": bool(args.use_positional_encoding),
            "target_mode": str(args.target_mode),
            "quality_gate_required_gap_pct": float(args.require_gap_pct) if args.require_gap_pct is not None else None,
            "quality_gate_passed": not gate_failed,
            "rollout_length": rollout_len,
            "train_seq_len": train_seq_len,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "checkpoint_path": str(checkpoint_path),
            "step1_eval_model_nll": float(eval_stats["model_nll"]),
            "step1_eval_bayes_nll": float(eval_stats["bayes_nll"]),
            "step1_gap_pct": gap_pct,
            "step2_posterior_mae": float("nan"),
            "step2_posterior_rmse": float("nan"),
            "step3_lpe_rel_error_median_pct": float("nan"),
            "step3_lpe_rel_error_mean_pct": float("nan"),
            "prediction_plot": str(pred_plot_path),
            "training_history_csv": str(history_csv_path) if history_csv_path.exists() else "",
            "training_curve_plot": str(training_curve_path) if training_curve_path.exists() else "",
            "posterior_plot": "",
            "lpe_hist_plot": "",
            "posterior_csv": "",
            "lpe_csv": "",
            "train_stats": train_stats,
            "sweep": sweep,
            "timestamp": time.time(),
        }
        with (k_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(k_summary_step1, f, indent=2)
        return k_summary_step1

    if gate_failed:
        print(
            f"[k={k}] Skipping Step 2/3 because quality gate failed: "
            f"gap={gap_pct:.3f}% > required {float(args.require_gap_pct):.3f}%",
            flush=True,
        )
        k_summary_blocked: Dict[str, object] = {
            "k": k,
            "device": str(device),
            "model_config": asdict(config),
            "model_config_label": config.label(),
            "use_positional_encoding": bool(args.use_positional_encoding),
            "target_mode": str(args.target_mode),
            "quality_gate_required_gap_pct": float(args.require_gap_pct),
            "quality_gate_passed": False,
            "rollout_length": rollout_len,
            "train_seq_len": train_seq_len,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "checkpoint_path": str(checkpoint_path),
            "step1_eval_model_nll": float(eval_stats["model_nll"]),
            "step1_eval_bayes_nll": float(eval_stats["bayes_nll"]),
            "step1_gap_pct": gap_pct,
            "step2_posterior_mae": float("nan"),
            "step2_posterior_rmse": float("nan"),
            "step3_lpe_rel_error_median_pct": float("nan"),
            "step3_lpe_rel_error_mean_pct": float("nan"),
            "prediction_plot": str(pred_plot_path),
            "training_history_csv": str(history_csv_path) if history_csv_path.exists() else "",
            "training_curve_plot": str(training_curve_path) if training_curve_path.exists() else "",
            "posterior_plot": "",
            "lpe_hist_plot": "",
            "posterior_csv": "",
            "lpe_csv": "",
            "train_stats": train_stats,
            "sweep": sweep,
            "timestamp": time.time(),
        }
        with (k_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(k_summary_blocked, f, indent=2)
        return k_summary_blocked

    context_for_posterior = sample_markov_k_batch(
        1,
        max(context_len, k + 1),
        k,
        alpha=float(args.alpha),
        beta=float(args.beta),
        device=device,
    )[0].detach().cpu().long()

    posterior_samples = posterior_samples_from_rollouts(
        model=model,
        context=context_for_posterior,
        k=k,
        num_samples=int(args.num_posterior_samples),
        rollout_length=rollout_len,
        device=device,
        rollout_batch_size=int(args.posterior_rollout_batch_size),
    )

    alpha_post, beta_post = compute_posterior_params(
        context_for_posterior,
        k,
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

    post_plot_path = fig_dir / "step2_posterior_mean_scatter.png"
    plot_posterior_mean_scatter(
        true_mean,
        sample_mean,
        post_plot_path,
        title=f"k={k} posterior mean by state",
    )

    contexts, target = generate_contexts_and_target(
        k=k,
        num_contexts=int(args.num_lpe_contexts),
        min_context_len=max(k, int(args.lpe_context_min_len)),
        max_context_len=max(k, int(args.lpe_context_max_len)),
        target_len=int(args.lpe_target_len),
        alpha=float(args.alpha),
        beta=float(args.beta),
        device=device,
        seed=int(args.seed) + 999 * k,
        target_mode=str(args.target_mode),
    )

    lpe_rows: List[Dict[str, object]] = []
    for i, context in enumerate(contexts):
        samples = posterior_samples_from_rollouts(
            model=model,
            context=context,
            k=k,
            num_samples=int(args.num_posterior_samples),
            rollout_length=rollout_len,
            device=device,
            rollout_batch_size=int(args.posterior_rollout_batch_size),
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

    lpe_rel_errors = np.array([float(r["relative_error_pct"]) for r in lpe_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.hist(lpe_rel_errors, bins=min(20, max(5, int(len(lpe_rel_errors) // 2))), color="steelblue", alpha=0.85)
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"k={k} LPE relative-error histogram")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    lpe_hist_path = fig_dir / "step3_lpe_rel_error_hist.png"
    plt.savefig(lpe_hist_path, dpi=150)
    plt.close(fig)

    k_summary: Dict[str, object] = {
        "k": k,
        "device": str(device),
        "model_config": asdict(config),
        "model_config_label": config.label(),
        "use_positional_encoding": bool(args.use_positional_encoding),
        "target_mode": str(args.target_mode),
        "quality_gate_required_gap_pct": float(args.require_gap_pct) if args.require_gap_pct is not None else None,
        "quality_gate_passed": not gate_failed,
        "rollout_length": rollout_len,
        "train_seq_len": train_seq_len,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "checkpoint_path": str(checkpoint_path),
        "step1_eval_model_nll": float(eval_stats["model_nll"]),
        "step1_eval_bayes_nll": float(eval_stats["bayes_nll"]),
        "step1_gap_pct": gap_pct,
        "step2_posterior_mae": posterior_mae,
        "step2_posterior_rmse": posterior_rmse,
        "step3_lpe_rel_error_median_pct": float(np.median(lpe_rel_errors)) if lpe_rel_errors.size else float("nan"),
        "step3_lpe_rel_error_mean_pct": float(np.mean(lpe_rel_errors)) if lpe_rel_errors.size else float("nan"),
        "prediction_plot": str(pred_plot_path),
        "training_history_csv": str(history_csv_path) if history_csv_path.exists() else "",
        "training_curve_plot": str(training_curve_path) if training_curve_path.exists() else "",
        "posterior_plot": str(post_plot_path),
        "lpe_hist_plot": str(lpe_hist_path),
        "posterior_csv": str(k_dir / "step2_posterior_state_metrics.csv"),
        "lpe_csv": str(k_dir / "step3_lpe_metrics.csv"),
        "train_stats": train_stats,
        "sweep": sweep,
        "timestamp": time.time(),
    }

    with (k_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(k_summary, f, indent=2)

    return k_summary


def compile_report_tex(tex_path: Path) -> bool:
    cmd = f"cd {tex_path.parent} && pdflatex -interaction=nonstopmode -halt-on-error {tex_path.name} >/tmp/markov_k_report.log 2>&1 && pdflatex -interaction=nonstopmode -halt-on-error {tex_path.name} >>/tmp/markov_k_report.log 2>&1"
    rc = os.system(cmd)
    return rc == 0


def write_latex_report(
    all_summaries: List[Dict[str, object]],
    out_tex: Path,
    title: str = "Order-k Markov Transformer LPE Experiments",
) -> None:
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in sorted(all_summaries, key=lambda x: int(x["k"])):
        rows.append(
            "{} & {:.6f} & {:.6f} & {:.2f} & {:.4f} & {:.2f} \\\\".format(
                int(s["k"]),
                float(s["step1_eval_model_nll"]),
                float(s["step1_eval_bayes_nll"]),
                float(s["step1_gap_pct"]),
                float(s["step2_posterior_mae"]),
                float(s["step3_lpe_rel_error_median_pct"]),
            )
        )

    fig_lines: List[str] = []
    for s in sorted(all_summaries, key=lambda x: int(x["k"])):
        k = int(s["k"])
        for key, caption in [
            ("prediction_plot", "Step 1: next-bit prediction vs Bayes"),
            ("training_curve_plot", "Step 1: learning curve"),
            ("posterior_plot", "Step 2: posterior mean comparison"),
            ("lpe_hist_plot", "Step 3: LPE relative-error histogram"),
        ]:
            raw_path = str(s.get(key, ""))
            if not raw_path:
                continue
            p = Path(raw_path)
            if not p.exists():
                continue
            rel = os.path.relpath(p, out_tex.parent)
            fig_lines.append(
                "\\begin{figure}[ht]\\centering\\includegraphics[width=0.72\\linewidth]{"
                + rel.replace("\\", "/")
                + "}\\caption{k="
                + str(k)
                + ": "
                + caption
                + "}\\end{figure}"
            )

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\title{{{title}}}
\author{{Automated run via lpe/markov\_k\_transformer.py}}
\date{{\today}}
\begin{{document}}
\maketitle

\section*{{Summary}}
This report summarizes order-k experiments for $k\in\{{1,\dots,7\}}$ with independent Beta(1,1) priors per transition state.
Training used sequence length equal to rollout length: $100\cdot 2^k$.

\section*{{Core Metrics}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{rrrrrr}}
\toprule
$k$ & Model NLL & Bayes NLL & Gap (\%) & Posterior MAE & LPE Median RelErr (\%) \\
\midrule
{os.linesep.join(rows)}
\bottomrule
\end{{tabular}}
\caption{{Primary metrics by order $k$.}}
\end{{table}}

\section*{{Figures}}
{os.linesep.join(fig_lines)}

\end{{document}}
"""
    out_tex.write_text(tex, encoding="utf-8")


def maybe_upload_to_r2(args: argparse.Namespace) -> Optional[str]:
    if not args.upload_r2:
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"checkpoints/markov_k/{timestamp}"
    cmd = (
        "source venv/bin/activate && "
        "python upload_checkpoints_to_r2.py "
        f"--checkpoint-dir {Path(args.checkpoint_root).as_posix()} "
        f"--s3-prefix {prefix} --skip-existing"
    )
    print(f"Uploading checkpoints to R2 with prefix: {prefix}", flush=True)
    rc = os.system(f"cd {Path(__file__).resolve().parents[1]} && bash -lc '{cmd}'")
    if rc != 0:
        raise RuntimeError("R2 upload failed")
    return prefix


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Order-k Markov transformer LPE experiment")
    p.add_argument("--k-list", type=str, default="1,2,3,4,5,6,7")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use-positional-encoding", action="store_true")

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)

    p.add_argument("--rollout-length-mult", type=int, default=100)
    p.add_argument("--context-margin", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--n-layers", type=int, default=None, help="Override model layers (disables sweep).")
    p.add_argument("--d-model", type=int, default=None, help="Override model width (disables sweep).")
    p.add_argument("--n-heads", type=int, default=None, help="Override model attention heads.")
    p.add_argument("--d-mlp", type=int, default=None, help="Override model MLP hidden dimension.")
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--target-train-tokens", type=float, default=2e7)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument(
        "--target-gap-pct",
        type=float,
        default=3.0,
        help="Early-stop target for Step 1 gap in percent (model-vs-Bayes NLL).",
    )
    p.add_argument(
        "--min-steps-before-stop",
        type=int,
        default=1000,
        help="Minimum training steps before enabling early stop on target gap.",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)

    p.add_argument("--use-sweep", action="store_true")
    p.add_argument("--sweep-steps", type=int, default=500)
    p.add_argument("--gap-target", type=float, default=0.03)

    p.add_argument("--context-length", type=int, default=None)
    p.add_argument("--num-posterior-samples", type=int, default=200)
    p.add_argument("--posterior-rollout-batch-size", type=int, default=8)
    p.add_argument("--num-prediction-points", type=int, default=500)

    p.add_argument("--num-lpe-contexts", type=int, default=8)
    p.add_argument("--lpe-context-min-len", type=int, default=10)
    p.add_argument("--lpe-context-max-len", type=int, default=20)
    p.add_argument("--lpe-target-len", type=int, default=100)
    p.add_argument("--target-mode", type=str, choices=["random", "balanced"], default="random")

    p.add_argument("--out-dir", type=str, default="artifacts/markov_k")
    p.add_argument("--checkpoint-root", type=str, default="checkpoints/markov_k")
    p.add_argument("--report-tex", type=str, default="latex/markov_k_lpe_report.tex")

    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--upload-r2", action="store_true")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--step1-only", action="store_true")
    p.add_argument(
        "--require-gap-pct",
        type=float,
        default=None,
        help="If set, Step 2/3 run only when Step 1 gap <= this threshold.",
    )
    p.add_argument(
        "--fail-on-gap",
        action="store_true",
        help="If quality gate fails, raise an error instead of just skipping Step 2/3.",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    k_list = parse_k_list(args.k_list)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Device: {device}", flush=True)
    print(f"k list: {k_list}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, object]] = []
    for k in k_list:
        print("\n" + "=" * 80, flush=True)
        print(f"Running k={k}", flush=True)
        print("=" * 80, flush=True)
        summary = run_single_k(k, args=args, device=device)
        all_summaries.append(summary)

    suite_csv = out_dir / "suite_summary.csv"
    suite_rows = []
    for s in all_summaries:
        suite_rows.append(
            {
                "k": int(s["k"]),
                "model_config": str(s["model_config_label"]),
                "use_positional_encoding": bool(s.get("use_positional_encoding", False)),
                "target_mode": str(s.get("target_mode", "random")),
                "quality_gate_passed": bool(s.get("quality_gate_passed", True)),
                "rollout_length": int(s["rollout_length"]),
                "train_seq_len": int(s["train_seq_len"]),
                "max_seq_len": int(s["max_seq_len"]),
                "batch_size": int(s["batch_size"]),
                "step1_eval_model_nll": float(s["step1_eval_model_nll"]),
                "step1_eval_bayes_nll": float(s["step1_eval_bayes_nll"]),
                "step1_gap_pct": float(s["step1_gap_pct"]),
                "step2_posterior_mae": float(s["step2_posterior_mae"]),
                "step3_lpe_rel_error_median_pct": float(s["step3_lpe_rel_error_median_pct"]),
                "steps_run": int(s.get("train_stats", {}).get("steps_run", 0)),
                "best_step": int(s.get("train_stats", {}).get("best_step", -1)),
                "training_history_csv": str(s.get("training_history_csv", "")),
                "training_curve_plot": str(s.get("training_curve_plot", "")),
                "checkpoint_path": str(s["checkpoint_path"]),
            }
        )
    save_csv(suite_csv, suite_rows)

    suite_json = out_dir / "suite_summary.json"
    with suite_json.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    report_ok = None
    report_pdf = None
    if not args.no_report:
        report_tex = Path(args.report_tex)
        write_latex_report(all_summaries, out_tex=report_tex)
        report_ok = compile_report_tex(report_tex)
        report_pdf = str(report_tex.with_suffix(".pdf"))
        print(f"Report compile success: {report_ok}", flush=True)

    r2_prefix = maybe_upload_to_r2(args)

    final_manifest = {
        "timestamp": time.time(),
        "device": str(device),
        "k_list": k_list,
        "suite_csv": str(suite_csv),
        "suite_json": str(suite_json),
        "report_pdf": report_pdf,
        "report_compiled": report_ok,
        "r2_prefix": r2_prefix,
    }
    manifest_path = out_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2)

    print("\nRun complete.", flush=True)
    print(json.dumps(final_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
