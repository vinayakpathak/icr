#!/usr/bin/env python3
"""
Sweep small Bernoulli transformer configs and compare to Bayes-optimal loss.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from bernoulli_transformer import BernoulliTransformer  # noqa: E402


@dataclass
class ModelConfig:
    n_layers: int
    d_model: int
    n_heads: int
    d_mlp: int

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_mlp": self.d_mlp,
        }


def pick_heads(d_model: int) -> int:
    if d_model >= 256:
        return 4
    if d_model >= 128:
        return 2
    return 1


def parse_grid(grid: Optional[str], mlp_mult: int) -> List[ModelConfig]:
    if not grid:
        base = [
            (1, 32),
            (1, 64),
            (1, 128),
            (2, 64),
            (2, 128),
            (2, 256),
            (4, 128),
            (4, 256),
        ]
    else:
        base = []
        for entry in grid.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            entry = entry.replace("x", ",")
            parts = [p.strip() for p in entry.split(",")]
            if len(parts) != 2:
                raise ValueError(f"Bad grid entry: {entry}")
            base.append((int(parts[0]), int(parts[1])))

    configs = []
    for n_layers, d_model in base:
        n_heads = pick_heads(d_model)
        d_mlp = d_model * mlp_mult
        configs.append(ModelConfig(n_layers, d_model, n_heads, d_mlp))
    return configs


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def sample_batch(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    gen: torch.Generator,
) -> torch.Tensor:
    p = torch.rand(batch_size, 1, device=device, generator=gen)
    return (torch.rand(batch_size, seq_len, device=device, generator=gen) < p).long()


def bayes_optimal_loss(
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
    seed: int,
) -> float:
    gen = make_generator(seed, device)
    total_loss = 0.0
    total_tokens = 0
    for _ in range(eval_batches):
        seqs = sample_batch(batch_size, seq_len, device, gen).float()
        ones_cum = seqs.cumsum(dim=1)[:, :-1]
        n = torch.arange(1, seq_len, device=device).unsqueeze(0).float()
        p_hat = (1.0 + ones_cum) / (2.0 + n)
        p_hat = p_hat.clamp(1e-6, 1.0 - 1e-6)
        targets = seqs[:, 1:]
        loss = -(targets * torch.log(p_hat) + (1.0 - targets) * torch.log(1.0 - p_hat))
        total_loss += loss.sum().item()
        total_tokens += targets.numel()
    return total_loss / total_tokens


def evaluate_loss(
    model: BernoulliTransformer,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
    seed: int,
) -> float:
    gen = make_generator(seed, device)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(eval_batches):
            seqs = sample_batch(batch_size, seq_len, device, gen)
            inputs = seqs[:, :-1]
            targets = seqs[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += targets.numel()
    return total_loss / total_tokens


def train_one(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    learning_rate: float,
    warmup_steps: int,
    grad_clip: Optional[float],
    device: torch.device,
    seed: int,
    loss_window: int,
    print_every: int,
) -> tuple[BernoulliTransformer, float]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    model = BernoulliTransformer(
        max_seq_len=None,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_mlp=config.d_mlp,
        use_prenorm=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    gen = make_generator(seed + 999, device)
    loss_hist = collections.deque(maxlen=loss_window)

    def get_lr(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return learning_rate * (step / warmup_steps)
        return learning_rate

    model.train()
    for step in range(num_steps):
        seqs = sample_batch(batch_size, seq_len, device, gen)
        inputs = seqs[:, :-1]
        targets = seqs[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        loss_hist.append(loss.item())
        if print_every and (step + 1) % print_every == 0:
            avg = sum(loss_hist) / len(loss_hist)
            print(f"  step {step+1}/{num_steps} | loss={avg:.4f} | lr={lr:.6f}")

    avg_loss = sum(loss_hist) / len(loss_hist)
    return model, avg_loss


def format_config(config: ModelConfig) -> str:
    return f"L{config.n_layers}-D{config.d_model}-H{config.n_heads}-M{config.d_mlp}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bernoulli transformer architecture sweep")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mlp-mult", type=int, default=1)
    parser.add_argument("--grid", type=str, default=None, help="Format: 'layers,d_model;layers,d_model'")
    parser.add_argument("--eval-batches", type=int, default=25)
    parser.add_argument("--loss-window", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gap-ratio", type=float, default=0.02)
    parser.add_argument("--log-path", type=str, default="logs/bernoulli_arch_sweep.jsonl")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)

    configs = parse_grid(args.grid, args.mlp_mult)
    eval_seed = args.seed + 1337
    bayes_loss = bayes_optimal_loss(
        args.batch_size,
        args.seq_len,
        args.eval_batches,
        device,
        eval_seed,
    )
    print(f"Device: {device}")
    print(f"Bayes-optimal eval loss: {bayes_loss:.6f} (seq_len={args.seq_len})")

    results = []
    for idx, config in enumerate(configs):
        run_seed = args.seed + (idx + 1) * 1000
        print(f"\n[{idx+1}/{len(configs)}] Training {format_config(config)}")
        start = time.time()
        model, train_loss = train_one(
            config=config,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            grad_clip=args.grad_clip,
            device=device,
            seed=run_seed,
            loss_window=args.loss_window,
            print_every=args.print_every,
        )
        eval_loss = evaluate_loss(
            model=model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            eval_batches=args.eval_batches,
            device=device,
            seed=eval_seed,
        )
        params = sum(p.numel() for p in model.parameters())
        elapsed = time.time() - start
        gap = eval_loss - bayes_loss
        gap_ratio = gap / bayes_loss
        row = {
            "config": config.to_dict(),
            "params": params,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "bayes_loss": bayes_loss,
            "gap": gap,
            "gap_ratio": gap_ratio,
            "seconds": elapsed,
        }
        results.append(row)
        with open(args.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"  params={params:,} | train_loss={train_loss:.4f} | "
            f"eval_loss={eval_loss:.4f} | gap={gap:.4f} ({gap_ratio*100:.2f}%) | "
            f"time={elapsed:.1f}s"
        )

    viable = [r for r in results if r["gap_ratio"] <= args.gap_ratio]
    viable.sort(key=lambda r: r["params"])

    print("\n=== Summary ===")
    results.sort(key=lambda r: r["eval_loss"])
    for r in results:
        cfg = r["config"]
        print(
            f"L{cfg['n_layers']}-D{cfg['d_model']}-H{cfg['n_heads']}-M{cfg['d_mlp']} | "
            f"params={r['params']:,} | eval_loss={r['eval_loss']:.4f} | "
            f"gap={r['gap']:.4f} ({r['gap_ratio']*100:.2f}%)"
        )

    if viable:
        best = viable[0]
        cfg = best["config"]
        print(
            "\nSmallest within gap ratio "
            f"{args.gap_ratio*100:.1f}%: "
            f"L{cfg['n_layers']}-D{cfg['d_model']} (params={best['params']:,})"
        )
    else:
        print(
            "\nNo configs within gap ratio threshold. "
            "Consider more steps or a wider/deeper sweep."
        )


if __name__ == "__main__":
    main()
