#!/usr/bin/env python3
"""Prompt-tuning replication on BabyLM with exact-vs-covariance comparison.

Implements two optimization tracks from the same soft-prompt initialization:
1) Exact gradient descent on exact N=2 objective J.
2) Covariance-based J_epsilon update under a conditionally-iid proxy posterior.

Both tracks are evaluated on the exact N=2 objective for direct comparison.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-vs-covariance prompt tuning replication for a small-vocab BabyLM checkpoint."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="vesteinn/gpt2-dna",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--context-text",
        type=str,
        default="",
        help="Fixed context text. If empty, BOS token is used when available.",
    )
    parser.add_argument(
        "--prepend-bos",
        dest="prepend_bos",
        action="store_true",
        default=True,
        help="Prepend BOS token when tokenizer provides one (default: true).",
    )
    parser.add_argument(
        "--no-prepend-bos",
        dest="prepend_bos",
        action="store_false",
        help="Disable BOS prefixing.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=8,
        help="Number of soft prompt vectors.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
        choices=[2],
        help="Continuation horizon. This exact implementation currently supports only N=2.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of steps for exact-gradient track.",
    )
    parser.add_argument(
        "--cov-steps",
        type=int,
        default=-1,
        help="Number of covariance-track steps. If <0, uses --steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-2,
        help="Learning rate for exact-gradient optimizer (Adam).",
    )
    parser.add_argument(
        "--cov-lr",
        type=float,
        default=5e-2,
        help="Learning rate for covariance update.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set <=0 to disable clipping.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=4.0,
        help="Bandwidth for kappa_epsilon in covariance tilt gradient.",
    )
    parser.add_argument(
        "--pmc-samples",
        type=int,
        default=300,
        help="Number of predictive-MC samples for covariance track.",
    )
    parser.add_argument(
        "--pmc-rollout-len",
        type=int,
        default=500,
        help="Autoregressive rollout length used to estimate rollout token-frequency distribution in PMC.",
    )
    parser.add_argument(
        "--tau-alpha",
        type=float,
        default=0.3,
        help="Dirichlet concentration for target categorical distribution tau.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed.",
    )
    parser.add_argument(
        "--pmc-seed-offset",
        type=int,
        default=10_000,
        help="Offset added to --seed for PMC sampling generator.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log interval for optimization steps.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Top-K tokens by tau mass for token-distribution plots.",
    )
    parser.add_argument(
        "--skip-covariance",
        action="store_true",
        help="Skip covariance track and only run exact optimization.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="language_models/results/prompt_tuning_elicitation",
        help="Output root directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run subdirectory name. Defaults to timestamp-based name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, ...",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_context_ids(
    tokenizer: Any,
    context_text: str,
    prepend_bos: bool,
    device: torch.device,
) -> torch.Tensor:
    token_ids: List[int] = []

    if prepend_bos and tokenizer.bos_token_id is not None:
        token_ids.append(int(tokenizer.bos_token_id))

    if context_text:
        token_ids.extend(int(x) for x in tokenizer.encode(context_text, add_special_tokens=False))

    if not token_ids:
        if tokenizer.eos_token_id is not None:
            token_ids = [int(tokenizer.eos_token_id)]
        else:
            raise ValueError(
                "No context tokens available. Provide --context-text or use a tokenizer with BOS/EOS."
            )

    return torch.tensor(token_ids, dtype=torch.long, device=device)


def sanitize_token_label(token: str) -> str:
    token = token.replace("\n", "\\n").replace("\t", "\\t")
    if token == "":
        return "<empty>"
    if token == " ":
        return "<space>"
    return token


def format_token(tokenizer: Any, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        return f"id_{token_id}"
    return sanitize_token_label(str(token))


def crop_past_key_values(past: Any, keep_len: int) -> Any:
    if past is None:
        return None
    if keep_len <= 0:
        return None
    if hasattr(past, "crop") and callable(past.crop):
        past.crop(keep_len)
        return past

    cropped = []
    for layer_cache in past:
        if len(layer_cache) < 2:
            raise ValueError("Unsupported legacy cache format for past_key_values.")
        layer_k, layer_v, *rest = layer_cache
        if layer_k.shape[-2] > keep_len:
            layer_k = layer_k[:, :, -keep_len:, :]
            layer_v = layer_v[:, :, -keep_len:, :]
        cropped.append((layer_k, layer_v, *rest))
    return tuple(cropped)


def exact_two_step_stats(
    model: nn.Module,
    context_embeds: torch.Tensor,
    all_token_embeds: torch.Tensor,
    soft_prompt: torch.Tensor,
    tau: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Exact objective and marginals for N=2."""
    vocab_size = int(tau.shape[0])

    z = soft_prompt.unsqueeze(0)  # [1, M, D]
    ctx = context_embeds.unsqueeze(0)  # [1, T, D]
    base_inputs = torch.cat([z, ctx], dim=1)  # [1, M+T, D]

    out1 = model(inputs_embeds=base_inputs, use_cache=False, return_dict=True)
    logp1 = torch.log_softmax(out1.logits[:, -1, :], dim=-1).squeeze(0)  # [V]
    p1 = torch.exp(logp1)

    base_batch = base_inputs.expand(vocab_size, -1, -1)  # [V, M+T, D]
    next_token_embeds = all_token_embeds.unsqueeze(1)  # [V, 1, D]
    two_step_inputs = torch.cat([base_batch, next_token_embeds], dim=1)  # [V, M+T+1, D]

    out2 = model(inputs_embeds=two_step_inputs, use_cache=False, return_dict=True)
    logp2 = torch.log_softmax(out2.logits[:, -1, :], dim=-1)  # [V, V]
    p2_cond = torch.exp(logp2)

    joint = p1[:, None] * p2_cond
    p2_marg = joint.sum(dim=0)

    log_tau = torch.log(tau.clamp_min(1e-30))
    ce_step1 = -(p1 * log_tau).sum()
    ce_step2 = -(p2_marg * log_tau).sum()
    objective = 0.5 * (ce_step1 + ce_step2)

    return {
        "J": objective,
        "ce_step1": ce_step1,
        "ce_step2": ce_step2,
        "p1": p1,
        "p2_marg": p2_marg,
    }


def compute_state_for_reporting(
    model: nn.Module,
    context_embeds: torch.Tensor,
    all_token_embeds: torch.Tensor,
    soft_prompt: torch.Tensor,
    tau: torch.Tensor,
) -> Dict[str, np.ndarray | float]:
    with torch.no_grad():
        stats = exact_two_step_stats(
            model=model,
            context_embeds=context_embeds,
            all_token_embeds=all_token_embeds,
            soft_prompt=soft_prompt,
            tau=tau,
        )
    return {
        "J": float(stats["J"].item()),
        "ce_step1": float(stats["ce_step1"].item()),
        "ce_step2": float(stats["ce_step2"].item()),
        "p1": stats["p1"].detach().cpu().numpy(),
        "p2_marg": stats["p2_marg"].detach().cpu().numpy(),
    }


@torch.no_grad()
def sample_pmc_predictives(
    model: nn.Module,
    context_ids: torch.Tensor,
    num_samples: int,
    rollout_len: int,
    seed: int,
    max_positions: int,
    log_every: int,
) -> torch.Tensor:
    """Predictive-MC rows estimated from empirical token frequencies along rollouts."""
    if num_samples <= 0:
        raise ValueError("--pmc-samples must be positive")
    if rollout_len < 0:
        raise ValueError("--pmc-rollout-len must be non-negative")
    if rollout_len == 0:
        raise ValueError("--pmc-rollout-len must be > 0 for rollout-frequency estimation")

    device = context_ids.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    with torch.inference_mode():
        # Determine vocabulary size from one forward pass.
        warm = model(input_ids=context_ids.view(1, -1), use_cache=False, return_dict=True)
        vocab_size = int(warm.logits.shape[-1])
        pmc = torch.empty((num_samples, vocab_size), dtype=torch.float32, device=device)

        for idx in range(num_samples):
            input_ids = context_ids.view(1, -1)
            past_key_values = None
            token_counts = torch.zeros(vocab_size, dtype=torch.float32, device=device)

            for _ in range(rollout_len):
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                probs = torch.softmax(outputs.logits[:, -1, :].float(), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)
                input_ids = next_token
                token_counts.scatter_add_(
                    0,
                    next_token.view(-1),
                    torch.ones(1, dtype=torch.float32, device=device),
                )

                if max_positions > 1:
                    past_key_values = crop_past_key_values(past_key_values, keep_len=max_positions - 1)
            pmc[idx, :] = token_counts / float(rollout_len)

            if idx == 0 or (idx + 1) % log_every == 0 or idx + 1 == num_samples:
                print(
                    f"[pmc] sample={idx + 1:4d}/{num_samples} rollout_len={rollout_len}",
                    flush=True,
                )

    return pmc


@torch.no_grad()
def covariance_elicitation_grad(
    soft_prompt: torch.Tensor,
    pmc_probs: torch.Tensor,
    tau: torch.Tensor,
    token_embeds: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Covariance gradient for the J_epsilon track under conditionally-iid proxy."""
    if eps <= 0:
        raise ValueError("--eps must be positive")

    eps2 = float(eps * eps)

    # kappa_j(y) = exp(-||z_j - phi(y)||^2 / (2 eps^2))
    diff = soft_prompt.unsqueeze(1) - token_embeds.unsqueeze(0)  # [M, V, D]
    sq_norm = (diff * diff).sum(dim=-1)  # [M, V]
    kappa = torch.exp(-0.5 * sq_norm / eps2).clamp_min(1e-30)  # [M, V]

    # q_lj = sum_y p_l(y) * kappa_j(y)
    q = (pmc_probs @ kappa.transpose(0, 1)).clamp_min(1e-30)  # [L, M]
    logw = torch.log(q).sum(dim=1)  # [L]
    w = torch.softmax(logw, dim=0)  # [L]

    log_tau = torch.log(tau.clamp_min(1e-30))
    mu = -(pmc_probs * log_tau.unsqueeze(0)).sum(dim=1)  # [L]
    mu_bar = (w * mu).sum()

    # responsibilities r_ljy = p_l(y) kappa_j(y) / q_lj
    responsibilities = (pmc_probs.unsqueeze(1) * kappa.unsqueeze(0)) / q.unsqueeze(-1)  # [L, M, V]
    expected_phi = torch.einsum("lmv,vd->lmd", responsibilities, token_embeds)  # [L, M, D]

    score = -(soft_prompt.unsqueeze(0) - expected_phi) / eps2  # [L, M, D]
    score_bar = (w[:, None, None] * score).sum(dim=0, keepdim=True)  # [1, M, D]

    grad = (w[:, None, None] * (mu[:, None, None] - mu_bar) * (score - score_bar)).sum(dim=0)

    ess = 1.0 / (w.square().sum() + 1e-12)
    diag = {
        "ess": float(ess.item()),
        "ess_frac": float((ess / float(pmc_probs.shape[0])).item()),
        "max_w": float(w.max().item()),
        "mu_mean": float(mu.mean().item()),
        "mu_std": float(mu.std(unbiased=False).item()),
    }
    return grad, diag


def optimize_exact_track(
    model: nn.Module,
    context_embeds: torch.Tensor,
    all_token_embeds: torch.Tensor,
    tau: torch.Tensor,
    z_init: torch.Tensor,
    steps: int,
    lr: float,
    grad_clip: float,
    log_every: int,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], float]:
    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=lr)
    history: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(steps):
        stats = exact_two_step_stats(
            model=model,
            context_embeds=context_embeds,
            all_token_embeds=all_token_embeds,
            soft_prompt=z,
            tau=tau,
        )
        J = stats["J"]

        optimizer.zero_grad(set_to_none=True)
        J.backward()

        grad_norm = float(z.grad.norm().item())
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], max_norm=grad_clip)
        optimizer.step()

        row = {
            "step": int(step),
            "J": float(J.item()),
            "ce_step1": float(stats["ce_step1"].item()),
            "ce_step2": float(stats["ce_step2"].item()),
            "grad_norm": grad_norm,
            "elapsed_sec": float(time.time() - t0),
        }
        history.append(row)

        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            print(
                f"[exact] step={step + 1:4d}/{steps} "
                f"J={row['J']:.6f} ce1={row['ce_step1']:.6f} ce2={row['ce_step2']:.6f} "
                f"grad_norm={row['grad_norm']:.6f} elapsed={row['elapsed_sec']:.1f}s",
                flush=True,
            )

    final_state = compute_state_for_reporting(
        model=model,
        context_embeds=context_embeds,
        all_token_embeds=all_token_embeds,
        soft_prompt=z.detach(),
        tau=tau,
    )
    history.append(
        {
            "step": int(steps),
            "J": float(final_state["J"]),
            "ce_step1": float(final_state["ce_step1"]),
            "ce_step2": float(final_state["ce_step2"]),
            "grad_norm": None,
            "elapsed_sec": float(time.time() - t0),
        }
    )

    return z.detach().clone(), history, float(time.time() - t0)


def optimize_covariance_track(
    model: nn.Module,
    context_embeds: torch.Tensor,
    all_token_embeds: torch.Tensor,
    tau: torch.Tensor,
    pmc_probs: torch.Tensor,
    z_init: torch.Tensor,
    steps: int,
    lr: float,
    grad_clip: float,
    eps: float,
    log_every: int,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], float]:
    z = z_init.clone()
    history: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(steps):
        state = compute_state_for_reporting(
            model=model,
            context_embeds=context_embeds,
            all_token_embeds=all_token_embeds,
            soft_prompt=z,
            tau=tau,
        )
        grad, diag = covariance_elicitation_grad(
            soft_prompt=z,
            pmc_probs=pmc_probs,
            tau=tau,
            token_embeds=all_token_embeds,
            eps=eps,
        )
        grad_norm = float(grad.norm().item())
        if grad_clip > 0 and grad_norm > grad_clip:
            grad = grad * (grad_clip / (grad_norm + 1e-12))
            grad_norm = float(grad.norm().item())

        with torch.no_grad():
            z -= lr * grad

        row = {
            "step": int(step),
            "J": float(state["J"]),
            "ce_step1": float(state["ce_step1"]),
            "ce_step2": float(state["ce_step2"]),
            "grad_norm": grad_norm,
            "ess": float(diag["ess"]),
            "ess_frac": float(diag["ess_frac"]),
            "max_w": float(diag["max_w"]),
            "mu_mean": float(diag["mu_mean"]),
            "mu_std": float(diag["mu_std"]),
            "elapsed_sec": float(time.time() - t0),
        }
        history.append(row)

        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            print(
                f"[cov ] step={step + 1:4d}/{steps} "
                f"J={row['J']:.6f} ce1={row['ce_step1']:.6f} ce2={row['ce_step2']:.6f} "
                f"grad_norm={row['grad_norm']:.6f} ESS={row['ess']:.1f} "
                f"max_w={row['max_w']:.4f} elapsed={row['elapsed_sec']:.1f}s",
                flush=True,
            )

    final_state = compute_state_for_reporting(
        model=model,
        context_embeds=context_embeds,
        all_token_embeds=all_token_embeds,
        soft_prompt=z,
        tau=tau,
    )
    history.append(
        {
            "step": int(steps),
            "J": float(final_state["J"]),
            "ce_step1": float(final_state["ce_step1"]),
            "ce_step2": float(final_state["ce_step2"]),
            "grad_norm": None,
            "ess": None,
            "ess_frac": None,
            "max_w": None,
            "mu_mean": None,
            "mu_std": None,
            "elapsed_sec": float(time.time() - t0),
        }
    )

    return z.detach().clone(), history, float(time.time() - t0)


def history_to_arrays(history: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    steps = np.array([int(x["step"]) for x in history], dtype=np.int64)
    values = np.array([float(x["J"]) for x in history], dtype=np.float64)
    return steps, values


def plot_objective_single(
    steps: np.ndarray,
    values: np.ndarray,
    title: str,
    path: Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, values, linewidth=1.8, color=color)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Exact objective J (N=2)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_objective_comparison(
    exact_steps: np.ndarray,
    exact_values: np.ndarray,
    cov_steps: Optional[np.ndarray],
    cov_values: Optional[np.ndarray],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(exact_steps, exact_values, linewidth=1.8, color="tab:blue", label="Exact gradient")
    if cov_steps is not None and cov_values is not None:
        ax.plot(cov_steps, cov_values, linewidth=1.8, color="tab:orange", label="Covariance J_epsilon")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Exact objective J (N=2)")
    ax.set_title("Prompt tuning objective trajectory (exact vs covariance)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_multiseries(
    path: Path,
    token_labels: List[str],
    series: List[Tuple[str, np.ndarray]],
    title: str,
    y_label: str,
) -> None:
    x = np.arange(len(token_labels))
    n = len(series)
    width = min(0.8 / max(n, 1), 0.26)

    fig, ax = plt.subplots(figsize=(max(10, len(token_labels) * 0.8), 5.5))
    center_offset = (n - 1) * 0.5
    for idx, (label, values) in enumerate(series):
        xpos = x + (idx - center_offset) * width
        ax.bar(xpos, values, width=width, label=label, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(token_labels, rotation=45, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def method_summary(history: List[Dict[str, Any]]) -> Dict[str, float]:
    j_init = float(history[0]["J"])
    j_final = float(history[-1]["J"])
    abs_improvement = float(j_init - j_final)
    rel_improvement = float(abs_improvement / max(abs(j_init), 1e-12))
    return {
        "J_init": j_init,
        "J_final": j_final,
        "absolute_improvement": abs_improvement,
        "relative_improvement": rel_improvement,
    }


def run() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    cov_steps = args.steps if args.cov_steps < 0 else args.cov_steps
    if cov_steps < 0:
        raise ValueError("--cov-steps must be >= 0 when provided")
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    if args.pmc_samples <= 0 and not args.skip_covariance:
        raise ValueError("--pmc-samples must be positive unless --skip-covariance is set")

    out_root = Path(args.output_dir)
    ensure_dir(out_root)
    run_name = args.run_name.strip() or f"seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_name
    ensure_dir(run_dir)

    print(f"[run] output_dir={run_dir}")
    print(f"[run] loading tokenizer/model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    embed_layer = model.get_input_embeddings()
    hidden_size = int(embed_layer.embedding_dim)
    vocab_size = int(model.config.vocab_size)
    max_positions = int(getattr(model.config, "max_position_embeddings", 0) or 0)

    context_ids = build_context_ids(
        tokenizer=tokenizer,
        context_text=args.context_text,
        prepend_bos=args.prepend_bos,
        device=device,
    )

    with torch.no_grad():
        context_embeds = embed_layer(context_ids).detach()
        all_token_ids = torch.arange(vocab_size, dtype=torch.long, device=device)
        all_token_embeds = embed_layer(all_token_ids).detach()

    rng = np.random.default_rng(args.seed)
    tau_np = rng.dirichlet(np.full(vocab_size, args.tau_alpha, dtype=np.float64)).astype(np.float32)
    tau = torch.tensor(tau_np, dtype=torch.float32, device=device)

    z_init = torch.randn(args.prompt_len, hidden_size, device=device) * 0.02

    print(
        f"[run] device={device} vocab_size={vocab_size} hidden_size={hidden_size} "
        f"context_len={int(context_ids.numel())} prompt_len={args.prompt_len}"
    )
    print(f"[run] exact track: steps={args.steps} lr={args.lr}")
    z_exact_final, exact_history, exact_runtime = optimize_exact_track(
        model=model,
        context_embeds=context_embeds,
        all_token_embeds=all_token_embeds,
        tau=tau,
        z_init=z_init,
        steps=args.steps,
        lr=args.lr,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
    )

    pmc_probs: Optional[torch.Tensor] = None
    cov_history: Optional[List[Dict[str, Any]]] = None
    cov_runtime_sec: Optional[float] = None
    pmc_runtime_sec: Optional[float] = None
    z_cov_final: Optional[torch.Tensor] = None

    if not args.skip_covariance:
        print(
            f"[run] covariance track: steps={cov_steps} cov_lr={args.cov_lr} eps={args.eps} "
            f"pmc_samples={args.pmc_samples} rollout_len={args.pmc_rollout_len}"
        )
        pmc_t0 = time.time()
        pmc_probs = sample_pmc_predictives(
            model=model,
            context_ids=context_ids,
            num_samples=args.pmc_samples,
            rollout_len=args.pmc_rollout_len,
            seed=args.seed + args.pmc_seed_offset,
            max_positions=max_positions,
            log_every=max(1, args.pmc_samples // 10),
        )
        pmc_runtime_sec = float(time.time() - pmc_t0)
        print(f"[run] pmc sampling done in {pmc_runtime_sec:.2f}s")

        z_cov_final, cov_history, cov_runtime_sec = optimize_covariance_track(
            model=model,
            context_embeds=context_embeds,
            all_token_embeds=all_token_embeds,
            tau=tau,
            pmc_probs=pmc_probs,
            z_init=z_init,
            steps=cov_steps,
            lr=args.cov_lr,
            grad_clip=args.grad_clip,
            eps=args.eps,
            log_every=args.log_every,
        )

    init_report = compute_state_for_reporting(
        model=model,
        context_embeds=context_embeds,
        all_token_embeds=all_token_embeds,
        soft_prompt=z_init,
        tau=tau,
    )
    exact_final_report = compute_state_for_reporting(
        model=model,
        context_embeds=context_embeds,
        all_token_embeds=all_token_embeds,
        soft_prompt=z_exact_final,
        tau=tau,
    )
    cov_final_report = None
    if z_cov_final is not None:
        cov_final_report = compute_state_for_reporting(
            model=model,
            context_embeds=context_embeds,
            all_token_embeds=all_token_embeds,
            soft_prompt=z_cov_final,
            tau=tau,
        )

    exact_steps_arr, exact_values_arr = history_to_arrays(exact_history)
    cov_steps_arr = None
    cov_values_arr = None
    if cov_history is not None:
        cov_steps_arr, cov_values_arr = history_to_arrays(cov_history)

    plot_objective_single(
        steps=exact_steps_arr,
        values=exact_values_arr,
        title="Exact-gradient objective trajectory",
        path=run_dir / "objective_exact_vs_step.png",
        color="tab:blue",
    )
    if cov_steps_arr is not None and cov_values_arr is not None:
        plot_objective_single(
            steps=cov_steps_arr,
            values=cov_values_arr,
            title="Covariance-gradient objective trajectory (evaluated on exact J)",
            path=run_dir / "objective_covariance_vs_step.png",
            color="tab:orange",
        )

    # Keep legacy-ish filename, now as the comparison chart.
    plot_objective_comparison(
        exact_steps=exact_steps_arr,
        exact_values=exact_values_arr,
        cov_steps=cov_steps_arr,
        cov_values=cov_values_arr,
        path=run_dir / "objective_vs_step.png",
    )

    top_k = min(args.top_k, vocab_size)
    top_ids = np.argsort(-tau_np)[:top_k]
    token_labels = [format_token(tokenizer, int(i)) for i in top_ids]

    p1_series: List[Tuple[str, np.ndarray]] = [
        ("tau", tau_np[top_ids]),
        ("init", init_report["p1"][top_ids]),
        ("exact_final", exact_final_report["p1"][top_ids]),
    ]
    p2_series: List[Tuple[str, np.ndarray]] = [
        ("tau", tau_np[top_ids]),
        ("init", init_report["p2_marg"][top_ids]),
        ("exact_final", exact_final_report["p2_marg"][top_ids]),
    ]
    if cov_final_report is not None:
        p1_series.append(("cov_final", cov_final_report["p1"][top_ids]))
        p2_series.append(("cov_final", cov_final_report["p2_marg"][top_ids]))

    plot_distribution_multiseries(
        path=run_dir / "next_token_probs_init_exact_cov.png",
        token_labels=token_labels,
        series=p1_series,
        title="Next-token distribution: tau vs init vs exact vs covariance",
        y_label="Probability",
    )
    plot_distribution_multiseries(
        path=run_dir / "two_step_expected_token_mass_init_exact_cov.png",
        token_labels=token_labels,
        series=p2_series,
        title="Step-2 marginal distribution: tau vs init vs exact vs covariance",
        y_label="Probability",
    )

    top_tokens: List[Dict[str, Any]] = []
    for token_id in top_ids:
        tid = int(token_id)
        row = {
            "token_id": tid,
            "token": format_token(tokenizer, tid),
            "tau": float(tau_np[tid]),
            "p1_init": float(init_report["p1"][tid]),
            "p1_exact_final": float(exact_final_report["p1"][tid]),
            "p2_marg_init": float(init_report["p2_marg"][tid]),
            "p2_marg_exact_final": float(exact_final_report["p2_marg"][tid]),
        }
        if cov_final_report is not None:
            row["p1_cov_final"] = float(cov_final_report["p1"][tid])
            row["p2_marg_cov_final"] = float(cov_final_report["p2_marg"][tid])
        top_tokens.append(row)

    metrics_payload: Dict[str, Any] = {
        "objective": {
            "exact": method_summary(exact_history),
        },
        "cross_entropy_components": {
            "init": {
                "ce_step1": float(init_report["ce_step1"]),
                "ce_step2": float(init_report["ce_step2"]),
            },
            "exact_final": {
                "ce_step1": float(exact_final_report["ce_step1"]),
                "ce_step2": float(exact_final_report["ce_step2"]),
            },
        },
        "runtime_sec": {
            "exact_optimization": float(exact_runtime),
        },
        "tau": tau_np.tolist(),
        "histories": {
            "exact": exact_history,
        },
        "top_tokens": top_tokens,
    }
    if cov_history is not None:
        metrics_payload["objective"]["covariance"] = method_summary(cov_history)
        metrics_payload["histories"]["covariance"] = cov_history
        metrics_payload["runtime_sec"]["covariance_optimization"] = float(cov_runtime_sec)
        metrics_payload["runtime_sec"]["pmc_sampling"] = float(pmc_runtime_sec)
    if cov_final_report is not None:
        metrics_payload["cross_entropy_components"]["covariance_final"] = {
            "ce_step1": float(cov_final_report["ce_step1"]),
            "ce_step2": float(cov_final_report["ce_step2"]),
        }

    config_payload = {
        "model_id": args.model_id,
        "context_text": args.context_text,
        "prepend_bos": bool(args.prepend_bos),
        "context_token_ids": [int(x) for x in context_ids.detach().cpu().tolist()],
        "prompt_len": int(args.prompt_len),
        "horizon": int(args.horizon),
        "steps": int(args.steps),
        "cov_steps": int(cov_steps),
        "lr": float(args.lr),
        "cov_lr": float(args.cov_lr),
        "grad_clip": float(args.grad_clip),
        "eps": float(args.eps),
        "pmc_samples": int(args.pmc_samples),
        "pmc_rollout_len": int(args.pmc_rollout_len),
        "tau_alpha": float(args.tau_alpha),
        "seed": int(args.seed),
        "pmc_seed_offset": int(args.pmc_seed_offset),
        "log_every": int(args.log_every),
        "top_k": int(top_k),
        "skip_covariance": bool(args.skip_covariance),
        "device": str(device),
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "max_position_embeddings": int(max_positions),
        "run_name": run_name,
        "run_dir": str(run_dir),
    }

    np_save_payload = {
        "tau": tau_np,
        "p1_init": init_report["p1"],
        "p1_exact_final": exact_final_report["p1"],
        "p2_marg_init": init_report["p2_marg"],
        "p2_marg_exact_final": exact_final_report["p2_marg"],
        "objective_exact_steps": exact_steps_arr,
        "objective_exact_values": exact_values_arr,
    }
    if cov_final_report is not None and cov_steps_arr is not None and cov_values_arr is not None:
        np_save_payload["p1_cov_final"] = cov_final_report["p1"]
        np_save_payload["p2_marg_cov_final"] = cov_final_report["p2_marg"]
        np_save_payload["objective_cov_steps"] = cov_steps_arr
        np_save_payload["objective_cov_values"] = cov_values_arr
    if pmc_probs is not None:
        np_save_payload["pmc_probs"] = pmc_probs.detach().cpu().numpy()

    np.savez_compressed(run_dir / "distributions.npz", **np_save_payload)
    save_json(run_dir / "config.json", config_payload)
    save_json(run_dir / "metrics.json", metrics_payload)

    exact_summary = method_summary(exact_history)
    print(
        f"[done] exact: J_init={exact_summary['J_init']:.6f} "
        f"J_final={exact_summary['J_final']:.6f} "
        f"improvement={exact_summary['absolute_improvement']:.6f}"
    )
    if cov_history is not None:
        cov_summary = method_summary(cov_history)
        print(
            f"[done] cov  : J_init={cov_summary['J_init']:.6f} "
            f"J_final={cov_summary['J_final']:.6f} "
            f"improvement={cov_summary['absolute_improvement']:.6f}"
        )
    print(f"[done] wrote artifacts to {run_dir}")


if __name__ == "__main__":
    run()
