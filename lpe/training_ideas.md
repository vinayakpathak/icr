# Markov k=4,5,6 Training Ideas (Resume Notes)

Last updated: 2026-02-17 (UTC)

## Scope and constraints

- Experiment family: `lpe/markov_k_transformer.py`
- Target ks: `k=4,5,6`
- Hard requirement from project notes: **train sequence length must equal inference rollout length** (`100 * 2^k`).
- We used positional encoding for these runs.
- Quality gate for continuing to Step 2/3: Step-1 gap (model NLL vs Bayes NLL) must be <= `3%`.
- Per request: **do not use Bayes-teacher loss** (keep standard next-token CE training).

---

## Where the latest run lives

- Run stamp: `overnightD_025644`
- Artifacts root: `artifacts/markov_k456_overnight_overnightD_025644`
- Checkpoints root: `checkpoints/markov_k456_overnight_overnightD_025644`
- Launcher script: `lpe/run_markov_k456_overnight.sh`

Attempt summaries:
- `artifacts/markov_k456_overnight_overnightD_025644/k4/*/k4/summary.json`
- `artifacts/markov_k456_overnight_overnightD_025644/k5/*/k5/summary.json`
- `artifacts/markov_k456_overnight_overnightD_025644/k6/*/k6/summary.json`

Training curves/history are already produced per attempt:
- `.../training_history.csv`
- `.../figures/step1_training_curve.png`

---

## Current status (best seen in latest overnight run)

### k=4
- Best final gap: **13.15%** (`attempt05_l8d192_s12000`)
- Best observed eval gap during training: **9.72%** at step `11800`
- Did not pass `<=3%` gate

### k=5
- Best final gap: **23.47%** (`attempt02_l6d160_s5500`)
- Best observed eval gap during training: **18.51%** (`attempt05_l8d192_s9000`)
- Did not pass `<=3%` gate

### k=6
- Best final gap: **27.80%** (`attempt05_l6d128_s7000`)
- Best observed eval gap during training: **25.56%** (`attempt02_l4d128_s5000`)
- Did not pass `<=3%` gate

Because Step 1 never passed, Step 2/3 were skipped in these attempts and R2 upload was not triggered.

---

## Likely reasons training stalls at higher k

1. Optimization noise is high.
- Overnight runs used tiny per-step batch (often `batch_size=1`) and low effective batch.
- This makes gradient variance high for long sequences and difficult posterior-tracking behavior.

2. Evaluation is noisy, so checkpoint/gate signal is unstable.
- Current run settings used small eval (`eval_batches=4`, `eval_batch_size=2` in overnight script).
- Final reported gap can be much worse than best seen mid-training.

3. Capacity/memory tradeoff at long sequence lengths is tight.
- For `k=6`, sequence length is `6400`.
- Full causal attention at this length constrains feasible model size and batch size on one GPU.

4. Coverage of k-states is uneven in finite sequences.
- Even with length scaling, worst-covered states can be sparse, which makes learning state-specific Bayesian updates harder.

---

## Suggested plan (without Bayes-teacher)

## Phase A: Make optimization and gating reliable first

1. Increase effective batch size.
- Keep `batch_size` small if needed for memory, but raise `--grad-accum-steps` aggressively (e.g. 16 to 32).
- This is the highest-priority change.

2. Use a less noisy eval for checkpointing and gating.
- During training: increase to something like `--eval-batches 32` (or 64 if affordable).
- For final gate decision: run an additional large eval pass (e.g. 128 batches) on the selected checkpoint.

3. Select checkpoints by robust metric.
- Keep saving `best.pt` by best eval gap, but only from sufficiently large eval.
- Avoid relying on tiny-eval final-step gap.

4. Tune LR schedule for long runs.
- For long runs, lower base LR (`1e-4` to `2e-4`) and longer warmup (`2k-4k` steps).
- Keep grad clip at `1.0`.

## Phase B: Get more usable capacity per GPU memory

1. Enable mixed precision (`bf16`) for training/eval.
2. Add gradient checkpointing in transformer blocks.
3. Use PyTorch SDPA/Flash attention path if available.

Goal: unlock larger/deeper models and/or higher effective batch at `k=5,6`.

## Phase C: Search strategy improvements

1. Replace fixed 5-attempt ladder with adaptive ladder:
- Start medium model + high grad accumulation.
- If gap plateaus, increase depth first, then width.
- Extend steps only if best-gap trend is still improving.

2. Spend compute where signal exists.
- For `k=4`, there was clear movement to ~10%.
- For `k=6`, several runs plateaued early; prioritize stabilization and capacity before just adding steps.

---

## Concrete resume checklist

1. Keep the sequence-length rule unchanged:
- `train_seq_len = rollout_len = 100 * 2^k`

2. First retry settings (recommended baseline):
- `--batch-size 1`
- `--grad-accum-steps 16` (or 32 if stable)
- `--learning-rate 1e-4` (try `2e-4` second)
- `--warmup-steps 3000`
- `--eval-every 200`
- `--eval-batches 32`
- `--eval-batch-size 2` (increase only if memory allows)
- Keep `--use-positional-encoding`

3. Gate logic:
- Keep early-stop target (e.g. `--target-gap-pct 2.8`) but
- decide pass/fail only after a larger final eval.

4. If still stuck:
- add bf16 + gradient checkpointing before trying much larger models.

---

## Notes about current script behavior

- `lpe/run_markov_k456_overnight.sh` currently runs `--step1-only` attempts with a fixed ladder per k.
- It only uploads to R2 after passing quality gate (`<=3%`).
- In this run no attempt passed, so there are no upload logs/artifacts from this stamp.

---

## Suggested code changes when resuming

1. In `lpe/markov_k_transformer.py`:
- Add optional `--amp-bf16` training path.
- Add optional gradient checkpointing flag.
- Add separate `--final-eval-batches` and use it for gate/final summary.

2. In `lpe/run_markov_k456_overnight.sh`:
- Increase eval robustness defaults.
- Increase grad accumulation.
- Make ladder adaptive (promote/demote model size based on recent best gap slope).

---

## Minimum success criterion before doing Step 2/3

- For each k in `{4,5,6}`:
  - robust final Step-1 gap <= `3%` against Bayes NLL
  - stable across at least 2 independent large-eval checks (same checkpoint)

Only then run posterior/LPE stages and upload finalized checkpoints.

