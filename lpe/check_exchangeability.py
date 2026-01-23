#!/usr/bin/env python3
import argparse
import math
from typing import List, Tuple

import numpy as np
import torch

from lpe.bernoulli_transformer import BernoulliTransformer


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


def sequence_logprob(model: BernoulliTransformer, seq: torch.Tensor) -> float:
    if seq.numel() < 2:
        return 0.0
    device = next(model.parameters()).device
    seq = seq.to(device)
    inputs = seq[:-1].unsqueeze(0)
    targets = seq[1:].unsqueeze(0)
    with torch.no_grad():
        logits = model(inputs)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.double().sum().item()


def nextbit_prob(model: BernoulliTransformer, seq: torch.Tensor) -> float:
    device = next(model.parameters()).device
    if seq.dim() == 1:
        seq = seq.unsqueeze(0)
    seq = seq.to(device)
    with torch.no_grad():
        logits = model.predict_next_logits(seq)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1].detach().cpu().numpy()[0]


def summarize_logps(logps: np.ndarray, n_tokens: int, base_logp: float | None = None):
    per_token = logps / max(n_tokens - 1, 1)
    summary = {
        "count": len(logps),
        "logp_mean": float(logps.mean()),
        "logp_std": float(logps.std(ddof=1)) if len(logps) > 1 else 0.0,
        "logp_min": float(logps.min()),
        "logp_max": float(logps.max()),
        "logp_range": float(logps.max() - logps.min()),
        "per_token_mean": float(per_token.mean()),
        "per_token_std": float(per_token.std(ddof=1)) if len(per_token) > 1 else 0.0,
        "per_token_min": float(per_token.min()),
        "per_token_max": float(per_token.max()),
    }
    if base_logp is not None:
        diffs = logps - base_logp
        summary.update(
            {
                "base_logp": float(base_logp),
                "max_abs_diff": float(np.max(np.abs(diffs))),
                "max_abs_diff_bits": float(np.max(np.abs(diffs)) / math.log(2)),
            }
        )
    return summary


def summarize_probs(probs: np.ndarray, base_prob: float | None = None):
    summary = {
        "count": len(probs),
        "prob_mean": float(probs.mean()),
        "prob_std": float(probs.std(ddof=1)) if len(probs) > 1 else 0.0,
        "prob_min": float(probs.min()),
        "prob_max": float(probs.max()),
        "prob_range": float(probs.max() - probs.min()),
    }
    if base_prob is not None:
        diffs = probs - base_prob
        summary.update(
            {
                "base_prob": float(base_prob),
                "max_abs_diff": float(np.max(np.abs(diffs))),
            }
        )
    return summary


def perm_test(
    model: BernoulliTransformer, n: int, num_perms: int, seed: int
) -> Tuple[torch.Tensor, dict]:
    rng = np.random.default_rng(seed)
    seq = torch.from_numpy(rng.integers(0, 2, size=n, dtype=np.int64))
    base_logp = sequence_logprob(model, seq)
    logps = [base_logp]
    for _ in range(num_perms):
        perm = seq[torch.randperm(seq.numel())]
        logps.append(sequence_logprob(model, perm))
    logps = np.array(logps, dtype=np.float64)
    return seq, summarize_logps(logps, n, base_logp=base_logp)


def perm_test_nextbit(
    model: BernoulliTransformer, n: int, num_perms: int, seed: int
) -> Tuple[torch.Tensor, dict]:
    rng = np.random.default_rng(seed)
    seq = torch.from_numpy(rng.integers(0, 2, size=n, dtype=np.int64))
    base_prob = nextbit_prob(model, seq)
    probs = [base_prob]
    for _ in range(num_perms):
        perm = seq[torch.randperm(seq.numel())]
        probs.append(nextbit_prob(model, perm))
    probs = np.array(probs, dtype=np.float64)
    return seq, summarize_probs(probs, base_prob=base_prob)


def perm_test_nextbit_multi(
    model: BernoulliTransformer,
    n: int,
    num_perms: int,
    num_strings: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    string_summaries = []
    for _ in range(num_strings):
        seq = torch.from_numpy(rng.integers(0, 2, size=n, dtype=np.int64))
        base_prob = nextbit_prob(model, seq)
        probs = [base_prob]
        for _ in range(num_perms):
            perm = seq[torch.randperm(seq.numel())]
            probs.append(nextbit_prob(model, perm))
        probs = np.array(probs, dtype=np.float64)
        summary = summarize_probs(probs, base_prob=base_prob)
        string_summaries.append(summary)

    metrics = [
        "prob_mean",
        "prob_std",
        "prob_min",
        "prob_max",
        "prob_range",
        "base_prob",
        "max_abs_diff",
    ]
    aggregated = {}
    for key in metrics:
        values = np.array([s[key] for s in string_summaries], dtype=np.float64)
        aggregated[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "num_strings": num_strings,
        "num_perms": num_perms,
        "aggregated": aggregated,
    }


def exhaustive_small_n(model: BernoulliTransformer, n: int) -> None:
    all_logps = []
    all_counts = []
    for x in range(1 << n):
        bits = [(x >> i) & 1 for i in range(n)]
        seq = torch.tensor(bits, dtype=torch.long)
        all_logps.append(sequence_logprob(model, seq))
        all_counts.append(sum(bits))
    all_logps = np.array(all_logps, dtype=np.float64)
    all_counts = np.array(all_counts, dtype=np.int64)
    print("\nExhaustive check for small n")
    print(f"n={n}, total sequences={len(all_logps)}")
    for k in range(n + 1):
        mask = all_counts == k
        if not np.any(mask):
            continue
        logps = all_logps[mask]
        summary = summarize_logps(logps, n)
        print(
            f"  ones={k}, count={summary['count']}, logp_std={summary['logp_std']}, "
            f"logp_range={summary['logp_range']}, per_token_std={summary['per_token_std']}"
        )


def parse_int_list(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check exchangeability of BernoulliTransformer.")
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
        "--perm-lengths",
        type=str,
        default="256,1024",
        help="Comma-separated sequence lengths for permutation tests.",
    )
    parser.add_argument(
        "--perm-counts",
        type=str,
        default="200,50",
        help="Comma-separated permutation counts for each length.",
    )
    parser.add_argument(
        "--perm-seeds",
        type=str,
        default="1,2",
        help="Comma-separated RNG seeds for each length.",
    )
    parser.add_argument(
        "--nextbit-lengths",
        type=str,
        default="",
        help="Comma-separated lengths for next-bit permutation tests (empty to skip).",
    )
    parser.add_argument(
        "--nextbit-counts",
        type=str,
        default="",
        help="Comma-separated permutation counts for next-bit tests.",
    )
    parser.add_argument(
        "--nextbit-seeds",
        type=str,
        default="",
        help="Comma-separated RNG seeds for next-bit tests.",
    )
    parser.add_argument(
        "--nextbit-num-strings",
        type=int,
        default=10,
        help="Number of base strings per length for next-bit tests.",
    )
    parser.add_argument(
        "--small-n",
        type=int,
        default=12,
        help="Exhaustive small-n check size (set to 0 to skip).",
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

    perm_lengths = parse_int_list(args.perm_lengths)
    perm_counts = parse_int_list(args.perm_counts)
    perm_seeds = parse_int_list(args.perm_seeds)
    if not (len(perm_lengths) == len(perm_counts) == len(perm_seeds)):
        raise ValueError("perm-lengths, perm-counts, and perm-seeds must have matching lengths.")

    nextbit_lengths = parse_int_list(args.nextbit_lengths)
    if nextbit_lengths:
        nextbit_counts = parse_int_list(args.nextbit_counts)
        nextbit_seeds = parse_int_list(args.nextbit_seeds)
        if not (len(nextbit_lengths) == len(nextbit_counts) == len(nextbit_seeds)):
            raise ValueError(
                "nextbit-lengths, nextbit-counts, and nextbit-seeds must have matching lengths."
            )

    print(f"Checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
    print("Note: logprobs exclude the first token (no BOS token).")

    for n, num_perms, seed in zip(perm_lengths, perm_counts, perm_seeds):
        seq, summary = perm_test(model, n, num_perms, seed=seed)
        ones = int(seq.sum().item())
        zeros = n - ones
        print("\nPermutation test")
        print(f"n={n}, perms={num_perms}, ones={ones}, zeros={zeros}")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    if nextbit_lengths:
        for n, num_perms, seed in zip(nextbit_lengths, nextbit_counts, nextbit_seeds):
            results = perm_test_nextbit_multi(
                model=model,
                n=n,
                num_perms=num_perms,
                num_strings=args.nextbit_num_strings,
                seed=seed,
            )
            print("\nNext-bit permutation test (multiple base strings)")
            print(
                f"n={n}, base_strings={results['num_strings']}, perms_per_string={results['num_perms']}"
            )
            for metric, stats in results["aggregated"].items():
                print(
                    f"  {metric}: mean={stats['mean']}, std={stats['std']}, "
                    f"min={stats['min']}, max={stats['max']}"
                )

    if args.small_n > 0:
        exhaustive_small_n(model, args.small_n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
