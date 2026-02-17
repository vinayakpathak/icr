#!/usr/bin/env python3
"""Overnight trainer for k=4,5,6 with automatic R2 upload on success."""

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
    num_steps: int
    learning_rate: float


ATTEMPTS: Dict[int, List[AttemptConfig]] = {
    4: [
        AttemptConfig("k4_l8d192_s3500", n_layers=8, d_model=192, n_heads=8, d_mlp=768, batch_size=1, num_steps=3500, learning_rate=3e-4),
        AttemptConfig("k4_l8d192_s6000", n_layers=8, d_model=192, n_heads=8, d_mlp=768, batch_size=1, num_steps=6000, learning_rate=2e-4),
    ],
    5: [
        AttemptConfig("k5_l6d160_s3500", n_layers=6, d_model=160, n_heads=8, d_mlp=640, batch_size=1, num_steps=3500, learning_rate=3e-4),
        AttemptConfig("k5_l6d160_s5500", n_layers=6, d_model=160, n_heads=8, d_mlp=640, batch_size=1, num_steps=5500, learning_rate=2e-4),
    ],
    6: [
        AttemptConfig("k6_l4d128_s3000", n_layers=4, d_model=128, n_heads=8, d_mlp=512, batch_size=1, num_steps=3000, learning_rate=3e-4),
        AttemptConfig("k6_l4d128_s5000", n_layers=4, d_model=128, n_heads=8, d_mlp=512, batch_size=1, num_steps=5000, learning_rate=2e-4),
    ],
}


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


def upload_checkpoint(repo_root: Path, checkpoint_dir: Path, prefix: str, log_path: Path) -> int:
    cmd = [
        sys.executable,
        "upload_checkpoints_to_r2.py",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--s3-prefix",
        prefix,
        "--skip-existing",
    ]
    return run_cmd(cmd=cmd, cwd=repo_root, log_path=log_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    markov_script = repo_root / "lpe" / "markov_k_transformer.py"

    run_root = repo_root / "artifacts" / "markov_k456_overnight"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "overnight_manifest.json"

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    manifest: Dict[str, object] = {
        "timestamp": time.time(),
        "run_stamp": run_stamp,
        "status": "running",
        "k_results": {},
    }

    for k in [4, 5, 6]:
        k_records: List[Dict[str, object]] = []
        success = False
        success_record: Optional[Dict[str, object]] = None

        for idx, cfg in enumerate(ATTEMPTS[k], start=1):
            out_dir = run_root / f"k{k}" / f"attempt_{idx:02d}_{cfg.name}"
            ckpt_root = repo_root / "checkpoints" / "markov_k456_overnight" / f"k{k}" / f"attempt_{idx:02d}_{cfg.name}"
            log_path = out_dir / "train.log"

            cmd = [
                sys.executable,
                str(markov_script),
                "--k-list",
                str(k),
                "--use-positional-encoding",
                "--step1-only",
                "--no-report",
                "--out-dir",
                str(out_dir),
                "--checkpoint-root",
                str(ckpt_root),
                "--batch-size",
                str(cfg.batch_size),
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
                "--warmup-steps",
                "1000",
                "--target-gap-pct",
                "2.8",
                "--min-steps-before-stop",
                "600",
                "--require-gap-pct",
                "3.0",
                "--print-every",
                "50",
                "--eval-every",
                "100",
                "--eval-batches",
                "4",
                "--eval-batch-size",
                "2",
            ]

            t0 = time.time()
            rc = run_cmd(cmd=cmd, cwd=repo_root, log_path=log_path)
            elapsed_sec = float(time.time() - t0)

            summary_path = out_dir / f"k{k}" / "summary.json"
            summary = load_summary(summary_path)
            gap_pct = float("inf")
            quality_gate_passed = False
            checkpoint_path = ""
            training_curve_plot = ""
            history_csv = ""
            if summary is not None:
                gap_pct = float(summary.get("step1_gap_pct", float("inf")))
                quality_gate_passed = bool(summary.get("quality_gate_passed", False))
                checkpoint_path = str(summary.get("checkpoint_path", ""))
                training_curve_plot = str(summary.get("training_curve_plot", ""))
                history_csv = str(summary.get("training_history_csv", ""))

            rec: Dict[str, object] = {
                "k": k,
                "attempt_index": idx,
                "config": asdict(cfg),
                "rc": rc,
                "elapsed_sec": elapsed_sec,
                "out_dir": str(out_dir),
                "checkpoint_root": str(ckpt_root),
                "summary_path": str(summary_path),
                "step1_gap_pct": gap_pct,
                "quality_gate_passed": quality_gate_passed,
                "checkpoint_path": checkpoint_path,
                "training_curve_plot": training_curve_plot,
                "training_history_csv": history_csv,
                "r2_upload_rc": None,
                "r2_prefix": None,
            }

            if rc == 0 and quality_gate_passed and gap_pct <= 3.0:
                prefix = f"checkpoints/markov_k456_overnight/{run_stamp}/k{k}/attempt_{idx:02d}_{cfg.name}"
                upload_log = out_dir / "r2_upload.log"
                upload_rc = upload_checkpoint(
                    repo_root=repo_root,
                    checkpoint_dir=ckpt_root,
                    prefix=prefix,
                    log_path=upload_log,
                )
                rec["r2_upload_rc"] = upload_rc
                rec["r2_prefix"] = prefix if upload_rc == 0 else None
                success = True
                success_record = rec
                k_records.append(rec)
                break

            k_records.append(rec)

            manifest["k_results"][str(k)] = {
                "success": success,
                "attempts": k_records,
                "successful_attempt": success_record,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        manifest["k_results"][str(k)] = {
            "success": success,
            "attempts": k_records,
            "successful_attempt": success_record,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_success = all(bool(manifest["k_results"][str(k)]["success"]) for k in [4, 5, 6])
    manifest["status"] = "completed_all" if all_success else "completed_with_failures"
    manifest["finished_at"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

