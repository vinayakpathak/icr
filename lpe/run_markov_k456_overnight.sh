#!/usr/bin/env bash
set -euo pipefail

source venv/bin/activate

RUN_STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
ART_ROOT="artifacts/markov_k456_overnight_${RUN_STAMP}"
CKPT_ROOT="checkpoints/markov_k456_overnight_${RUN_STAMP}"
mkdir -p "$ART_ROOT"

echo "Run stamp: $RUN_STAMP"
echo "Artifacts: $ART_ROOT"
echo "Checkpoints: $CKPT_ROOT"

check_summary_pass() {
  local summary_path="$1"
  python - "$summary_path" <<'PY'
import json
import math
import sys

path = sys.argv[1]
try:
    s = json.loads(open(path, "r", encoding="utf-8").read())
except Exception:
    sys.exit(1)
gap = float(s.get("step1_gap_pct", float("inf")))
ok = bool(s.get("quality_gate_passed", False)) and math.isfinite(gap) and gap <= 3.0
sys.exit(0 if ok else 1)
PY
}

run_attempt() {
  local k="$1"
  local attempt_name="$2"
  local n_layers="$3"
  local d_model="$4"
  local n_heads="$5"
  local d_mlp="$6"
  local batch_size="$7"
  local num_steps="$8"
  local lr="$9"
  local grad_accum_steps="${10}"

  local out_dir="${ART_ROOT}/k${k}/${attempt_name}"
  local ckpt_dir="${CKPT_ROOT}/k${k}/${attempt_name}"
  local summary_path="${out_dir}/k${k}/summary.json"
  local train_log="${out_dir}/train.log"
  local upload_log="${out_dir}/upload.log"

  mkdir -p "$out_dir" "$ckpt_dir"

  echo
  echo "============================================================"
  echo "k=${k} attempt=${attempt_name}"
  echo "model=L${n_layers}_D${d_model}_H${n_heads}_M${d_mlp} batch=${batch_size} steps=${num_steps} lr=${lr} grad_accum=${grad_accum_steps}"
  echo "============================================================"

  set +e
  python lpe/markov_k_transformer.py \
    --k-list "${k}" \
    --use-positional-encoding \
    --step1-only \
    --no-report \
    --out-dir "${out_dir}" \
    --checkpoint-root "${ckpt_dir}" \
    --batch-size "${batch_size}" \
    --grad-accum-steps "${grad_accum_steps}" \
    --n-layers "${n_layers}" \
    --d-model "${d_model}" \
    --n-heads "${n_heads}" \
    --d-mlp "${d_mlp}" \
    --num-steps "${num_steps}" \
    --learning-rate "${lr}" \
    --warmup-steps 1000 \
    --target-gap-pct 2.8 \
    --min-steps-before-stop 600 \
    --require-gap-pct 3.0 \
    --print-every 50 \
    --eval-every 100 \
    --eval-batches 4 \
    --eval-batch-size 2 \
    > "${train_log}" 2>&1
  local rc=$?
  set -e

  echo "train rc=${rc} log=${train_log}"

  if [[ ! -f "${summary_path}" ]]; then
    echo "summary missing: ${summary_path}"
    return 1
  fi

  if ! check_summary_pass "${summary_path}"; then
    echo "quality gate not met for k=${k} attempt=${attempt_name}"
    return 1
  fi

  local r2_prefix="checkpoints/markov_k456_overnight/${RUN_STAMP}/k${k}/${attempt_name}"
  echo "quality gate passed; uploading to R2 prefix=${r2_prefix}"
  set +e
  python upload_checkpoints_to_r2.py \
    --checkpoint-dir "${ckpt_dir}" \
    --s3-prefix "${r2_prefix}" \
    --skip-existing \
    > "${upload_log}" 2>&1
  local upload_rc=$?
  set -e
  echo "upload rc=${upload_rc} log=${upload_log}"
  if [[ "${upload_rc}" -ne 0 ]]; then
    return 1
  fi

  return 0
}

run_k_attempts() {
  local k="$1"
  shift
  local ok=0
  while [[ "$#" -gt 0 ]]; do
    local attempt_name="$1"; shift
    local n_layers="$1"; shift
    local d_model="$1"; shift
    local n_heads="$1"; shift
    local d_mlp="$1"; shift
    local batch_size="$1"; shift
    local num_steps="$1"; shift
    local lr="$1"; shift
    local grad_accum="$1"; shift

    if run_attempt "$k" "$attempt_name" "$n_layers" "$d_model" "$n_heads" "$d_mlp" "$batch_size" "$num_steps" "$lr" "$grad_accum"; then
      ok=1
      break
    fi
  done

  if [[ "$ok" -eq 1 ]]; then
    echo "k=${k} succeeded (uploaded)."
  else
    echo "k=${k} failed all configured attempts."
  fi
}

# Attempts per k (name n_layers d_model n_heads d_mlp batch steps lr grad_accum)
# We intentionally vary size/depth/optimization and allow multiple fallbacks.
run_k_attempts 4 \
  attempt01_l8d192_s3500 8 192 8 768 1 3500 3e-4 1 \
  attempt02_l8d192_s6000 8 192 8 768 1 6000 2e-4 1 \
  attempt03_l6d128_s9000 6 128 8 512 1 9000 2e-4 2 \
  attempt04_l10d224_s6000 10 224 8 896 1 6000 1.5e-4 1 \
  attempt05_l8d192_s12000 8 192 8 768 1 12000 1e-4 2

run_k_attempts 5 \
  attempt01_l6d160_s3500 6 160 8 640 1 3500 3e-4 1 \
  attempt02_l6d160_s5500 6 160 8 640 1 5500 2e-4 1 \
  attempt03_l8d192_s5000 8 192 8 768 1 5000 1.5e-4 1 \
  attempt04_l6d128_s9000 6 128 8 512 1 9000 2e-4 2 \
  attempt05_l8d192_s9000 8 192 8 768 1 9000 1e-4 2

run_k_attempts 6 \
  attempt01_l4d128_s3000 4 128 8 512 1 3000 3e-4 1 \
  attempt02_l4d128_s5000 4 128 8 512 1 5000 2e-4 1 \
  attempt03_l6d160_s4500 6 160 8 640 1 4500 1.5e-4 1 \
  attempt04_l4d96_s10000 4 96 8 384 1 10000 2e-4 2 \
  attempt05_l6d128_s7000 6 128 8 512 1 7000 1e-4 2

echo
echo "Overnight run completed."
echo "Artifacts root: ${ART_ROOT}"
echo "Checkpoints root: ${CKPT_ROOT}"
