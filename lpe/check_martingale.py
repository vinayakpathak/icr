#!/usr/bin/env python3
import argparse
import math
from typing import List

import numpy as np
import torch

from bernoulli_transformer import BernoulliTransformer


def load_model(checkpoint: str, device: str) -> BernoulliTransformer:
    model = BernoulliTransformer(
        max_seq_len=None,
        d_model=16,
        n_layers=1,
        n_heads=1,
        d_mlp=16,
        use_prenorm=True,
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def next_prob_one_batch(model: BernoulliTransformer, seqs: torch.Tensor) -> np.ndarray:
    if seqs.numel() == 0:
        raise ValueError("seqs must be non-empty")
    device = next(model.parameters()).device
    seqs = seqs.to(device)
    with torch.no_grad():
        logits = model(seqs)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        return probs[:, 1].detach().cpu().numpy()


def parse_int_list(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def format_seq(bits: torch.Tensor, max_len: int = 64) -> str:
    as_list = bits.tolist()
    prefix = "".join(str(int(b)) for b in as_list[:max_len])
    if len(as_list) <= max_len:
        return prefix
    ones = int(sum(as_list))
    return f"{prefix}...(len={len(as_list)}, ones={ones})"


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check martingale property of next-bit predictions.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bernoulli_transformer_L1_D16_seq1024.pt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on.",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="64,256,1024",
        help="Comma-separated sequence lengths to test.",
    )
    parser.add_argument(
        "--num-per-length",
        type=int,
        default=25,
        help="Number of random strings per length.",
    )
    parser.add_argument(
        "--data-p",
        type=float,
        default=0.5,
        help="Bernoulli(p) used to sample input strings.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for |p2 - p1| to count as close.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=0,
        help="Show details for the first K samples per length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Global torch/numpy seed.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    lengths = parse_int_list(args.lengths)
    if not lengths:
        raise ValueError("No lengths provided.")

    model = load_model(args.checkpoint, args.device)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
    print(f"Data Bernoulli p: {args.data_p}")
    print("Note: p1 = P(y_{n+1}=1|s), p2 = E_{x~Bern(p1)}[P(y_{n+2}=1|s,x)].")

    rng = np.random.default_rng(args.seed)

    for n in lengths:
        data = rng.random((args.num_per_length, n)) < args.data_p
        seqs = torch.from_numpy(data.astype(np.int64))
        p1 = next_prob_one_batch(model, seqs)

        ones = torch.ones((args.num_per_length, 1), dtype=torch.long)
        zeros = torch.zeros((args.num_per_length, 1), dtype=torch.long)
        seqs1 = torch.cat([seqs, ones], dim=1)
        seqs0 = torch.cat([seqs, zeros], dim=1)
        p2_1 = next_prob_one_batch(model, seqs1)
        p2_0 = next_prob_one_batch(model, seqs0)
        p2 = p1 * p2_1 + (1.0 - p1) * p2_0

        diff = p2 - p1
        abs_diff = np.abs(diff)
        close_frac = float(np.mean(abs_diff <= args.tolerance))

        print(f"\nLength {n} | samples={args.num_per_length}")
        print(f"  p1: {summarize(p1)}")
        print(f"  p2: {summarize(p2)}")
        print(f"  diff (p2 - p1): {summarize(diff)}")
        print(f"  mean_abs_diff: {float(abs_diff.mean())}")
        print(f"  max_abs_diff: {float(abs_diff.max())}")
        print(f"  max_abs_diff_bits: {float(abs_diff.max() / math.log(2))}")
        print(f"  close_frac(|diff| <= {args.tolerance}): {close_frac}")

        if args.show_samples > 0:
            max_show = min(args.show_samples, args.num_per_length)
            for i in range(max_show):
                s = format_seq(seqs[i])
                print(
                    f"  s={s} | p1={p1[i]:.6f} | p2={p2[i]:.6f} | "
                    f"p2_1={p2_1[i]:.6f} | p2_0={p2_0[i]:.6f} | diff={diff[i]:.6f}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
