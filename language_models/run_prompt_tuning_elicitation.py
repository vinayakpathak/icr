#!/usr/bin/env python3
"""Replicate tiny-horizon prompt tuning on BabyLM with exact categorical objective.

This script generalizes the binary notebook experiment to a full vocabulary causal
language model and performs exact objective optimization for horizon N=2.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact tiny-N prompt tuning replication for BabyLM language model."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="phonemetransformers/GPT2-85M-CHAR-PHON",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--context-text",
        type=str,
        default="",
        help="Fixed context text. If empty, BOS token will be used when available.",
    )
    parser.add_argument(
        "--prepend-bos",
        dest="prepend_bos",
        action="store_true",
        default=True,
        help="Prepend BOS token to context when tokenizer defines one (default: true).",
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
        help="Rollout horizon. Only N=2 is supported in this exact phase.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of optimization steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-2,
        help="Adam learning rate for soft prompt optimization.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set <=0 to disable clipping.",
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
        help="Random seed.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print optimization logs every N steps.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Top-K tokens (by tau mass) for comparison plots.",
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
        help="Optional run subdirectory name. Defaults to timestamped name.",
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
        ctx_ids = tokenizer.encode(context_text, add_special_tokens=False)
        token_ids.extend(int(x) for x in ctx_ids)

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


def exact_two_step_stats(
    model: nn.Module,
    context_embeds: torch.Tensor,
    all_token_embeds: torch.Tensor,
    soft_prompt: torch.Tensor,
    tau: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute exact N=2 objective and relevant marginals."""
    vocab_size = int(tau.shape[0])

    z = soft_prompt.unsqueeze(0)  # [1, M, D]
    ctx = context_embeds.unsqueeze(0)  # [1, T, D]
    base_inputs = torch.cat([z, ctx], dim=1)  # [1, M+T, D]

    out1 = model(inputs_embeds=base_inputs, use_cache=False, return_dict=True)
    logp1 = torch.log_softmax(out1.logits[:, -1, :], dim=-1).squeeze(0)  # [V]
    p1 = torch.exp(logp1)

    batch = vocab_size
    base_batch = base_inputs.expand(batch, -1, -1)  # [V, M+T, D]
    next_token_embeds = all_token_embeds.unsqueeze(1)  # [V, 1, D]
    two_step_inputs = torch.cat([base_batch, next_token_embeds], dim=1)  # [V, M+T+1, D]

    out2 = model(inputs_embeds=two_step_inputs, use_cache=False, return_dict=True)
    logp2 = torch.log_softmax(out2.logits[:, -1, :], dim=-1)  # [V, V]
    p2_cond = torch.exp(logp2)

    joint = p1[:, None] * p2_cond  # [V, V]
    p2_marg = joint.sum(dim=0)  # [V]

    log_tau = torch.log(tau)
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


def plot_objective(steps: np.ndarray, values: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, values, linewidth=1.8, color="tab:blue")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Exact objective J (N=2)")
    ax.set_title("Prompt tuning objective trajectory")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_comparison(
    path: Path,
    token_labels: List[str],
    tau_vals: np.ndarray,
    init_vals: np.ndarray,
    final_vals: np.ndarray,
    title: str,
    y_label: str,
) -> None:
    x = np.arange(len(token_labels))
    w = 0.26

    fig, ax = plt.subplots(figsize=(max(10, len(token_labels) * 0.8), 5.5))
    ax.bar(x - w, tau_vals, width=w, label="tau", alpha=0.9)
    ax.bar(x, init_vals, width=w, label="init", alpha=0.9)
    ax.bar(x + w, final_vals, width=w, label="final", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(token_labels, rotation=45, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    out_root = Path(args.output_dir)
    ensure_dir(out_root)
    run_name = args.run_name.strip() or (
        f"seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
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
    tau_np = rng.dirichlet(np.full(vocab_size, args.tau_alpha, dtype=np.float64)).astype(
        np.float32
    )
    tau = torch.tensor(tau_np, dtype=torch.float32, device=device)

    z = nn.Parameter(torch.randn(args.prompt_len, hidden_size, device=device) * 0.02)
    z_init = z.detach().clone()
    optimizer = torch.optim.Adam([z], lr=args.lr)

    print(
        f"[run] device={device} vocab_size={vocab_size} hidden_size={hidden_size} "
        f"context_len={int(context_ids.numel())} prompt_len={args.prompt_len}"
    )
    print(f"[run] starting optimization: steps={args.steps} lr={args.lr}")

    step_list: List[int] = []
    objective_history: List[float] = []
    ce1_history: List[float] = []
    ce2_history: List[float] = []
    grad_norm_history: List[float] = []

    t0 = time.time()

    for step in range(args.steps):
        stats = exact_two_step_stats(
            model=model,
            context_embeds=context_embeds,
            all_token_embeds=all_token_embeds,
            soft_prompt=z,
            tau=tau,
        )

        J = stats["J"]
        ce1 = float(stats["ce_step1"].item())
        ce2 = float(stats["ce_step2"].item())

        optimizer.zero_grad(set_to_none=True)
        J.backward()

        grad_norm = float(z.grad.norm().item())
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], max_norm=args.grad_clip)
        optimizer.step()

        step_list.append(step)
        objective_history.append(float(J.item()))
        ce1_history.append(ce1)
        ce2_history.append(ce2)
        grad_norm_history.append(grad_norm)

        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            elapsed = time.time() - t0
            print(
                f"[opt] step={step + 1:4d}/{args.steps} "
                f"J={objective_history[-1]:.6f} "
                f"ce1={ce1:.6f} ce2={ce2:.6f} "
                f"grad_norm={grad_norm:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    with torch.no_grad():
        final_stats = exact_two_step_stats(
            model=model,
            context_embeds=context_embeds,
            all_token_embeds=all_token_embeds,
            soft_prompt=z,
            tau=tau,
        )

    step_list.append(args.steps)
    objective_history.append(float(final_stats["J"].item()))
    ce1_history.append(float(final_stats["ce_step1"].item()))
    ce2_history.append(float(final_stats["ce_step2"].item()))
    grad_norm_history.append(float("nan"))

    z_final = z.detach().clone()

    init_report = compute_state_for_reporting(
        model=model,
        context_embeds=context_embeds,
        all_token_embeds=all_token_embeds,
        soft_prompt=z_init,
        tau=tau,
    )
    final_report = compute_state_for_reporting(
        model=model,
        context_embeds=context_embeds,
        all_token_embeds=all_token_embeds,
        soft_prompt=z_final,
        tau=tau,
    )

    top_k = min(args.top_k, vocab_size)
    top_ids = np.argsort(-tau_np)[:top_k]
    token_labels = [format_token(tokenizer, int(i)) for i in top_ids]

    steps_arr = np.array(step_list, dtype=np.int64)
    obj_arr = np.array(objective_history, dtype=np.float64)

    plot_objective(steps_arr, obj_arr, run_dir / "objective_vs_step.png")

    plot_distribution_comparison(
        path=run_dir / "next_token_probs_init_vs_final.png",
        token_labels=token_labels,
        tau_vals=tau_np[top_ids],
        init_vals=init_report["p1"][top_ids],
        final_vals=final_report["p1"][top_ids],
        title="Next-token distribution: tau vs init vs final",
        y_label="Probability",
    )

    plot_distribution_comparison(
        path=run_dir / "two_step_expected_token_mass_init_vs_final.png",
        token_labels=token_labels,
        tau_vals=tau_np[top_ids],
        init_vals=init_report["p2_marg"][top_ids],
        final_vals=final_report["p2_marg"][top_ids],
        title="Step-2 marginal distribution: tau vs init vs final",
        y_label="Probability",
    )

    np.savez_compressed(
        run_dir / "distributions.npz",
        tau=tau_np,
        p1_init=init_report["p1"],
        p1_final=final_report["p1"],
        p2_marg_init=init_report["p2_marg"],
        p2_marg_final=final_report["p2_marg"],
        objective_steps=steps_arr,
        objective_values=obj_arr,
        ce_step1=np.array(ce1_history, dtype=np.float64),
        ce_step2=np.array(ce2_history, dtype=np.float64),
        grad_norm=np.array(grad_norm_history, dtype=np.float64),
    )

    top_tokens: List[Dict[str, Any]] = []
    for token_id in top_ids:
        tid = int(token_id)
        top_tokens.append(
            {
                "token_id": tid,
                "token": format_token(tokenizer, tid),
                "tau": float(tau_np[tid]),
                "p1_init": float(init_report["p1"][tid]),
                "p1_final": float(final_report["p1"][tid]),
                "p2_marg_init": float(init_report["p2_marg"][tid]),
                "p2_marg_final": float(final_report["p2_marg"][tid]),
            }
        )

    total_runtime_sec = float(time.time() - t0)
    j_init = float(init_report["J"])
    j_final = float(final_report["J"])
    abs_improvement = float(j_init - j_final)
    rel_improvement = float(abs_improvement / max(abs(j_init), 1e-12))

    config_payload = {
        "model_id": args.model_id,
        "context_text": args.context_text,
        "prepend_bos": bool(args.prepend_bos),
        "context_token_ids": [int(x) for x in context_ids.detach().cpu().tolist()],
        "prompt_len": int(args.prompt_len),
        "horizon": int(args.horizon),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "grad_clip": float(args.grad_clip),
        "tau_alpha": float(args.tau_alpha),
        "seed": int(args.seed),
        "log_every": int(args.log_every),
        "top_k": int(top_k),
        "device": str(device),
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "run_name": run_name,
        "run_dir": str(run_dir),
    }

    metrics_payload = {
        "objective": {
            "J_init": j_init,
            "J_final": j_final,
            "absolute_improvement": abs_improvement,
            "relative_improvement": rel_improvement,
        },
        "cross_entropy_components": {
            "ce_step1_init": float(init_report["ce_step1"]),
            "ce_step1_final": float(final_report["ce_step1"]),
            "ce_step2_init": float(init_report["ce_step2"]),
            "ce_step2_final": float(final_report["ce_step2"]),
        },
        "runtime_sec": total_runtime_sec,
        "tau": tau_np.tolist(),
        "objective_history": [
            {
                "step": int(step_list[i]),
                "J": float(objective_history[i]),
                "ce_step1": float(ce1_history[i]),
                "ce_step2": float(ce2_history[i]),
                "grad_norm": None
                if np.isnan(grad_norm_history[i])
                else float(grad_norm_history[i]),
            }
            for i in range(len(step_list))
        ],
        "top_tokens": top_tokens,
    }

    save_json(run_dir / "config.json", config_payload)
    save_json(run_dir / "metrics.json", metrics_payload)

    print(f"[done] runtime={total_runtime_sec:.2f}s")
    print(f"[done] J_init={j_init:.6f} J_final={j_final:.6f} improvement={abs_improvement:.6f}")
    print(f"[done] wrote artifacts to {run_dir}")


if __name__ == "__main__":
    run()
