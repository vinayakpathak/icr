#!/usr/bin/env python3
"""Overnight trainer for k=2 aiming for <=1% Bayes gap."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AttemptConfig:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    d_mlp: int
    batch_size: int
    grad_accum_steps: int
    num_steps: int
    learning_rate: float
    warmup_steps: int
    min_lr: float
    weight_decay: float
    seed: int
    init_from_best_same_arch: bool = False


ATTEMPTS: List[AttemptConfig] = [
    AttemptConfig(
        name="a01_l8d256_seed123_scout",
        n_layers=8,
        d_model=256,
        n_heads=8,
        d_mlp=1024,
        batch_size=8,
        grad_accum_steps=8,
        num_steps=6000,
        learning_rate=2.5e-4,
        warmup_steps=500,
        min_lr=2e-6,
        weight_decay=1e-3,
        seed=123,
    ),
    AttemptConfig(
        name="a02_l8d256_seed123_ft",
        n_layers=8,
        d_model=256,
        n_heads=8,
        d_mlp=1024,
        batch_size=8,
        grad_accum_steps=8,
        num_steps=14000,
        learning_rate=9e-5,
        warmup_steps=700,
        min_lr=8e-7,
        weight_decay=1e-3,
        seed=123,
        init_from_best_same_arch=True,
    ),
    AttemptConfig(
        name="a03_l8d256_seed777_scout",
        n_layers=8,
        d_model=256,
        n_heads=8,
        d_mlp=1024,
        batch_size=8,
        grad_accum_steps=8,
        num_steps=6000,
        learning_rate=2.5e-4,
        warmup_steps=500,
        min_lr=2e-6,
        weight_decay=1e-3,
        seed=777,
    ),
    AttemptConfig(
        name="a04_l8d256_seed777_ft",
        n_layers=8,
        d_model=256,
        n_heads=8,
        d_mlp=1024,
        batch_size=8,
        grad_accum_steps=8,
        num_steps=14000,
        learning_rate=9e-5,
        warmup_steps=700,
        min_lr=8e-7,
        weight_decay=1e-3,
        seed=777,
        init_from_best_same_arch=True,
    ),
    AttemptConfig(
        name="a05_l10d320_seed123_scout",
        n_layers=10,
        d_model=320,
        n_heads=10,
        d_mlp=1280,
        batch_size=4,
        grad_accum_steps=12,
        num_steps=8000,
        learning_rate=2.0e-4,
        warmup_steps=800,
        min_lr=1e-6,
        weight_decay=1e-3,
        seed=123,
    ),
    AttemptConfig(
        name="a06_l10d320_seed123_ft",
        n_layers=10,
        d_model=320,
        n_heads=10,
        d_mlp=1280,
        batch_size=4,
        grad_accum_steps=12,
        num_steps=18000,
        learning_rate=7e-5,
        warmup_steps=900,
        min_lr=5e-7,
        weight_decay=1e-3,
        seed=123,
        init_from_best_same_arch=True,
    ),
    AttemptConfig(
        name="a07_l10d320_seed777_scout",
        n_layers=10,
        d_model=320,
        n_heads=10,
        d_mlp=1280,
        batch_size=4,
        grad_accum_steps=12,
        num_steps=8000,
        learning_rate=2.0e-4,
        warmup_steps=800,
        min_lr=1e-6,
        weight_decay=1e-3,
        seed=777,
    ),
    AttemptConfig(
        name="a08_l10d320_seed777_ft",
        n_layers=10,
        d_model=320,
        n_heads=10,
        d_mlp=1280,
        batch_size=4,
        grad_accum_steps=12,
        num_steps=18000,
        learning_rate=7e-5,
        warmup_steps=900,
        min_lr=5e-7,
        weight_decay=1e-3,
        seed=777,
        init_from_best_same_arch=True,
    ),
    AttemptConfig(
        name="a09_l12d384_seed123_scout",
        n_layers=12,
        d_model=384,
        n_heads=12,
        d_mlp=1536,
        batch_size=3,
        grad_accum_steps=12,
        num_steps=9000,
        learning_rate=1.5e-4,
        warmup_steps=900,
        min_lr=1e-6,
        weight_decay=1e-3,
        seed=123,
    ),
    AttemptConfig(
        name="a10_l12d384_seed123_ft",
        n_layers=12,
        d_model=384,
        n_heads=12,
        d_mlp=1536,
        batch_size=3,
        grad_accum_steps=12,
        num_steps=20000,
        learning_rate=6e-5,
        warmup_steps=1000,
        min_lr=5e-7,
        weight_decay=1e-3,
        seed=123,
        init_from_best_same_arch=True,
    ),
    AttemptConfig(
        name="a11_l12d384_seed2027_scout",
        n_layers=12,
        d_model=384,
        n_heads=12,
        d_mlp=1536,
        batch_size=3,
        grad_accum_steps=12,
        num_steps=9000,
        learning_rate=1.5e-4,
        warmup_steps=900,
        min_lr=1e-6,
        weight_decay=1e-3,
        seed=2027,
    ),
    AttemptConfig(
        name="a12_l12d384_seed2027_ft",
        n_layers=12,
        d_model=384,
        n_heads=12,
        d_mlp=1536,
        batch_size=3,
        grad_accum_steps=12,
        num_steps=20000,
        learning_rate=6e-5,
        warmup_steps=1000,
        min_lr=5e-7,
        weight_decay=1e-3,
        seed=2027,
        init_from_best_same_arch=True,
    ),
]


def run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def load_summary(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markov_script = repo_root / "lpe" / "markov_k_transformer.py"

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = repo_root / "artifacts" / f"markov_k2_overnight_{run_stamp}"
    ckpt_root = repo_root / "checkpoints" / f"markov_k2_overnight_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    manifest_path = run_root / "overnight_manifest.json"
    manifest: Dict[str, object] = {
        "timestamp": time.time(),
        "run_stamp": run_stamp,
        "k": 2,
        "target_gap_pct": 1.0,
        "attempt_budget": len(ATTEMPTS),
        "status": "running",
        "attempts": [],
        "successful_attempt": None,
        "best_attempt_so_far": None,
        "best_gap_pct_so_far": None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    success_record: Optional[Dict[str, object]] = None
    success = False
    best_attempt_so_far: Optional[Dict[str, object]] = None
    best_gap_so_far = float("inf")
    best_ckpt_by_arch: Dict[str, str] = {}
    best_gap_by_arch: Dict[str, float] = {}

    for idx, cfg in enumerate(ATTEMPTS, start=1):
        out_dir = run_root / f"attempt_{idx:02d}_{cfg.name}"
        attempt_ckpt_root = ckpt_root / f"attempt_{idx:02d}_{cfg.name}"
        summary_path = out_dir / "k2" / "summary.json"
        log_path = out_dir / "train.log"
        arch_key = f"L{cfg.n_layers}_D{cfg.d_model}_H{cfg.n_heads}_M{cfg.d_mlp}"
        init_checkpoint: Optional[str] = None
        if cfg.init_from_best_same_arch:
            init_checkpoint = best_ckpt_by_arch.get(arch_key)

        cmd = [
            sys.executable,
            "-u",
            str(markov_script),
            "--k-list",
            "2",
            "--device",
            "cuda",
            "--seed",
            str(cfg.seed),
            "--use-positional-encoding",
            "--step1-only",
            "--no-report",
            "--out-dir",
            str(out_dir),
            "--checkpoint-root",
            str(attempt_ckpt_root),
            "--rollout-length-mult",
            "1000",
            "--max-seq-len",
            "4096",
            "--batch-size",
            str(cfg.batch_size),
            "--grad-accum-steps",
            str(cfg.grad_accum_steps),
            "--n-layers",
            str(cfg.n_layers),
            "--d-model",
            str(cfg.d_model),
            "--n-heads",
            str(cfg.n_heads),
            "--d-mlp",
            str(cfg.d_mlp),
            "--num-steps",
            str(cfg.num_steps),
            "--learning-rate",
            str(cfg.learning_rate),
            "--weight-decay",
            str(cfg.weight_decay),
            "--lr-schedule",
            "cosine",
            "--min-lr",
            str(cfg.min_lr),
            "--warmup-steps",
            str(cfg.warmup_steps),
            "--target-gap-pct",
            "0.95",
            "--min-steps-before-stop",
            "2000",
            "--require-gap-pct",
            "1.0",
            "--grad-clip",
            "0.8",
            "--print-every",
            "25",
            "--eval-every",
            "50",
            "--eval-batches",
            "8",
            "--eval-batch-size",
            "2",
        ]
        if init_checkpoint:
            cmd.extend(["--init-checkpoint", init_checkpoint])

        print(
            f"[overnight-k2] attempt {idx}/{len(ATTEMPTS)} start name={cfg.name} seed={cfg.seed} "
            f"arch={arch_key} init_checkpoint={init_checkpoint or 'none'}",
            flush=True,
        )

        t0 = time.time()
        rc = run_cmd(cmd=cmd, cwd=repo_root, log_path=log_path)
        elapsed_sec = float(time.time() - t0)

        summary = load_summary(summary_path)
        gap_pct = float("inf")
        quality_gate_passed = False
        checkpoint_path = ""
        training_curve_plot = ""
        history_csv = ""
        train_stats: Dict[str, object] = {}
        if summary is not None:
            gap_pct = float(summary.get("step1_gap_pct", float("inf")))
            quality_gate_passed = bool(summary.get("quality_gate_passed", False))
            checkpoint_path = str(summary.get("checkpoint_path", ""))
            training_curve_plot = str(summary.get("training_curve_plot", ""))
            history_csv = str(summary.get("training_history_csv", ""))
            if isinstance(summary.get("train_stats", {}), dict):
                train_stats = dict(summary.get("train_stats", {}))

        rec: Dict[str, object] = {
            "attempt_index": idx,
            "config": asdict(cfg),
            "rc": rc,
            "elapsed_sec": elapsed_sec,
            "out_dir": str(out_dir),
            "checkpoint_root": str(attempt_ckpt_root),
            "summary_path": str(summary_path),
            "step1_gap_pct": gap_pct,
            "quality_gate_passed": quality_gate_passed,
            "checkpoint_path": checkpoint_path,
            "seed": cfg.seed,
            "arch_key": arch_key,
            "init_checkpoint_used": init_checkpoint or "",
            "training_curve_plot": training_curve_plot,
            "training_history_csv": history_csv,
            "train_stats": train_stats,
        }
        manifest["attempts"].append(rec)

        if checkpoint_path and gap_pct < best_gap_by_arch.get(arch_key, float("inf")):
            best_ckpt_by_arch[arch_key] = checkpoint_path
            best_gap_by_arch[arch_key] = gap_pct

        if gap_pct < best_gap_so_far:
            best_gap_so_far = gap_pct
            best_attempt_so_far = rec
            manifest["best_attempt_so_far"] = best_attempt_so_far
            manifest["best_gap_pct_so_far"] = best_gap_so_far

        print(
            f"[overnight-k2] attempt {idx}/{len(ATTEMPTS)} done rc={rc} gap_pct={gap_pct:.6f} "
            f"quality_gate_passed={quality_gate_passed}",
            flush=True,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if rc == 0 and quality_gate_passed and gap_pct <= 1.0:
            success = True
            success_record = rec
            manifest["successful_attempt"] = success_record
            manifest["status"] = "completed_success"
            manifest["finished_at"] = time.time()
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(manifest, indent=2))
            return

    manifest["successful_attempt"] = success_record
    manifest["status"] = "completed_without_success" if not success else "completed_success"
    manifest["finished_at"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
