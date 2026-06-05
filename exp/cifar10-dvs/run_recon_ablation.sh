#!/usr/bin/env bash
# ==========================================================================
# Layer-wise Membrane-Aware Reconstruction Ablation (CIFAR10-DVS)
# ==========================================================================
# Runs a broad single-GPU PTQ ablation using temporary configs, so the base
# config is not modified. The default queue is intentionally long; use
# MAX_SECONDS=86400 to keep starting new runs for roughly one day.
#
# Usage:
#   GPU=0 MAX_SECONDS=86400 ./exp/cifar10-dvs/run_recon_ablation.sh
#   GROUP=mem GPU=0 ./exp/cifar10-dvs/run_recon_ablation.sh
#   DEFAULT_ADAROUND_ITERS=20000 GPU=0 ./exp/cifar10-dvs/run_recon_ablation.sh
#
# Groups:
#   base      reference runs
#   mem       recon_mem_lam sweep
#   batches   calibration-batch sweep
#   lr        Adam learning-rate sweep
#   shift     max integer-code movement sweep
#   reg       code-shift regularization sweep
#   scale     membrane scale bridge sweep
#   bits      weight/membrane bit-width sweep
#   long      higher-budget combined configs
#
# Failed runs are recorded in the summary and do not stop later runs.
# ==========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# shellcheck disable=SC1090
if ! source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  && ! source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; then
  echo "Error: cannot find conda initialization script" >&2
  exit 1
fi
if ! conda activate spikeTransformer; then
  echo "Error: cannot activate conda env: spikeTransformer" >&2
  exit 1
fi

BASE_CONFIG="${BASE_CONFIG:-exp/cifar10-dvs/config.yml}"
GROUP="${GROUP:-all}"
GPU_OVERRIDE="${GPU:-0}"
MAX_SECONDS="${MAX_SECONDS:-0}"
MAX_RUNS="${MAX_RUNS:-0}"

# CIFAR10-DVS has a small eval set, but 20k AdaRound iters can still make each
# run expensive. Use 5k by default to cover more ablations in a one-day queue;
# the "long" group below restores 20k for selected promising settings.
DEFAULT_ADAROUND_ITERS="${DEFAULT_ADAROUND_ITERS:-5000}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Error: base config not found: ${BASE_CONFIG}" >&2
  exit 1
fi

# Format: "tag|group|key=val|key=val|..."
# The list is ordered from broad, cheaper probes to longer combined runs.
EXPS=(
  "base-it5k|base"

  "mem0.00|mem|recon_mem_lam=0.0"
  "mem0.05|mem|recon_mem_lam=0.05"
  "mem0.10|mem|recon_mem_lam=0.10"
  "mem0.25|mem|recon_mem_lam=0.25"
  "mem0.50|mem|recon_mem_lam=0.50"
  "mem1.00|mem|recon_mem_lam=1.00"

  "b2|batches|recon_num_batches=2"
  "b4|batches|recon_num_batches=4"
  "b8|batches|recon_num_batches=8"
  "b16|batches|recon_num_batches=16"
  "b32|batches|recon_num_batches=32"

  "lr5e-4|lr|adaround_lr=0.0005"
  "lr1e-3|lr|adaround_lr=0.001"
  "lr3e-3|lr|adaround_lr=0.003"
  "lr1e-2|lr|adaround_lr=0.01"

  "shift1|shift|recon_max_code_shift=1.0"
  "shift2|shift|recon_max_code_shift=2.0"
  "shift4|shift|recon_max_code_shift=4.0"

  "reg0|reg|recon_reg_lam_scale=0.0"
  "reg1e-5|reg|recon_reg_lam_scale=0.00001"
  "reg1e-4|reg|recon_reg_lam_scale=0.0001"
  "reg1e-3|reg|recon_reg_lam_scale=0.001"

  "scale-weight|scale|scale_bridge=weight"
  "scale-pot|scale|scale_bridge=pot"
  "scale-observer|scale|scale_bridge=observer"
  "scale-unify|scale|scale_bridge=unify"

  "bits-w4m4|bits|weight_bit=4|mem_bit=4|first_mem_bit=4"
  "bits-w3m4|bits|weight_bit=3|mem_bit=4|first_mem_bit=4"
  "bits-w2m4|bits|weight_bit=2|mem_bit=4|first_mem_bit=4"
  "bits-w4m3|bits|weight_bit=4|mem_bit=3|first_mem_bit=3"
  "bits-w3m3|bits|weight_bit=3|mem_bit=3|first_mem_bit=3"

  "long-base-it20k|long|adaround_iters=20000"
  "long-b16-mem010-reg1e5-lr1e3|long|adaround_iters=20000|recon_num_batches=16|recon_mem_lam=0.10|recon_reg_lam_scale=0.00001|adaround_lr=0.001"
  "long-b16-mem025-reg1e5-lr1e3|long|adaround_iters=20000|recon_num_batches=16|recon_mem_lam=0.25|recon_reg_lam_scale=0.00001|adaround_lr=0.001"
  "long-b16-mem050-reg1e5-lr1e3|long|adaround_iters=20000|recon_num_batches=16|recon_mem_lam=0.50|recon_reg_lam_scale=0.00001|adaround_lr=0.001"
  "long-unify-b16-mem010|long|adaround_iters=20000|scale_bridge=unify|recon_num_batches=16|recon_mem_lam=0.10|recon_reg_lam_scale=0.00001|adaround_lr=0.001"
)

OUT_ROOT="${REPO_ROOT}/output/ptq/cifar10-dvs"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY="${OUT_ROOT}/recon_ablation_${RUN_STAMP}.csv"
LOG_DIR="${OUT_ROOT}/recon_ablation_logs_${RUN_STAMP}"
START_EPOCH="$(date +%s)"
RUN_COUNT=0
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

echo "tag,group,status,exit_code,weight_bit,mem_bit,first_mem_bit,scale_bridge,recon_mem_lam,recon_num_batches,adaround_iters,adaround_lr,recon_max_code_shift,recon_reg_lam_scale,recon_grad_clip,recon_early_stop_patience,orig_acc1,folded_acc1,quant_acc1,drop,output_dir" \
  > "${SUMMARY}"

echo "Repo:      ${REPO_ROOT}"
echo "Base:      ${BASE_CONFIG}"
echo "Group:     ${GROUP}"
echo "GPU:       ${GPU_OVERRIDE}"
echo "Max sec:   ${MAX_SECONDS}"
echo "Max runs:  ${MAX_RUNS}"
echo "Summary:   ${SUMMARY}"
echo "Logs:      ${LOG_DIR}"
echo "----------------------------------------"

set_config_value() {
  local file="$1"
  local key="$2"
  local val="$3"
  local tmp_file
  tmp_file="$(mktemp "${TMPDIR:-/tmp}/ptq_cfg_update.XXXXXX.yml")"
  awk -v k="${key}" -v v="${val}" '
    BEGIN { done = 0 }
    index($0, k ":") == 1 {
      print k ": " v
      done = 1
      next
    }
    { print }
    END {
      if (!done) {
        print ""
        print k ": " v
      }
    }
  ' "${file}" > "${tmp_file}" && mv "${tmp_file}" "${file}"
}

get_config_value() {
  local file="$1"
  local key="$2"
  awk -F':' -v k="${key}" '$1 == k {gsub(/^[ \t]+|[ \t]+$/, "", $2); split($2, a, /[ \t#]/); print a[1]; exit}' "${file}"
}

extract_metric() {
  local file="$1"
  local pattern="$2"
  local field="$3"
  if [[ ! -f "${file}" ]]; then
    printf "-"
    return
  fi
  local val
  if [[ "${field}" == "NF" ]]; then
    val="$(grep "${pattern}" "${file}" | tail -n1 | awk '{print $NF}' | tr -d '%' || true)"
  else
    val="$(grep "${pattern}" "${file}" | tail -n1 | awk -v f="${field}" '{print $f}' | tr -d '%' || true)"
  fi
  [[ -n "${val}" ]] && printf "%s" "${val}" || printf "-"
}

time_budget_reached() {
  if [[ "${MAX_SECONDS}" == "0" ]]; then
    return 1
  fi
  local now elapsed
  now="$(date +%s)"
  elapsed=$((now - START_EPOCH))
  [[ "${elapsed}" -ge "${MAX_SECONDS}" ]]
}

for spec in "${EXPS[@]}"; do
  if time_budget_reached; then
    echo ""
    echo "Time budget reached before starting next run; stopping queue."
    break
  fi
  if [[ "${MAX_RUNS}" != "0" && "${RUN_COUNT}" -ge "${MAX_RUNS}" ]]; then
    echo ""
    echo "Max run count reached; stopping queue."
    break
  fi

  IFS='|' read -r -a parts <<<"${spec}"
  tag="${parts[0]// /}"
  group="${parts[1]// /}"
  overrides=("${parts[@]:2}")

  if [[ "${GROUP}" != "all" && "${GROUP}" != "${group}" ]]; then
    continue
  fi

  RUN_COUNT=$((RUN_COUNT + 1))
  echo ""
  echo "========== [${group}] ${tag} =========="
  if [[ ${#overrides[@]} -gt 0 ]]; then
    echo "  overrides: ${overrides[*]}"
  else
    echo "  overrides: default ablation baseline"
  fi

  tmp="$(mktemp "${TMPDIR:-/tmp}/ptq_cifar10dvs_recon_${tag}.XXXXXX.yml")"
  cp "${BASE_CONFIG}" "${tmp}"

  set_config_value "${tmp}" "gpu" "${GPU_OVERRIDE}"
  set_config_value "${tmp}" "output_dir" "output/ptq/cifar10-dvs"
  set_config_value "${tmp}" "adaround_iters" "${DEFAULT_ADAROUND_ITERS}"
  set_config_value "${tmp}" "recon_num_batches" "8"
  set_config_value "${tmp}" "recon_mem_lam" "0.25"
  set_config_value "${tmp}" "recon_reg_lam_scale" "0.0001"
  set_config_value "${tmp}" "recon_max_code_shift" "2.0"
  set_config_value "${tmp}" "recon_min_iters" "50"
  set_config_value "${tmp}" "recon_early_stop_patience" "0"
  set_config_value "${tmp}" "recon_improve_eps" "0.000001"
  set_config_value "${tmp}" "recon_grad_clip" "1.0"
  set_config_value "${tmp}" "recon_log_interval" "100"

  for kv in "${overrides[@]}"; do
    key="${kv%%=*}"
    val="${kv#*=}"
    key="${key// /}"
    val="${val// /}"
    set_config_value "${tmp}" "${key}" "${val}"
  done

  run_log="${LOG_DIR}/${tag}.log"
  python ptq/main.py --config "${tmp}" 2>&1 | tee "${run_log}"
  exit_code="${PIPESTATUS[0]}"

  last_dir="$(grep '^Output:' "${run_log}" | tail -n1 | awk '{print $2}' || true)"
  if [[ -z "${last_dir}" || ! -f "${last_dir}/ptq.log" ]]; then
    last_dir="$(ls -1dt "${OUT_ROOT}"/*-sdt-w*-cifar10dvs 2>/dev/null | head -n1 || true)"
  fi

  if [[ -n "${last_dir}" && -f "${last_dir}/ptq.log" ]]; then
    orig="$(extract_metric "${last_dir}/ptq.log" "Original  Acc@1:" "NF")"
    folded="$(extract_metric "${last_dir}/ptq.log" "After BN  Acc@1:" "4")"
    quant="$(extract_metric "${last_dir}/ptq.log" "Quantized Acc@1:" "NF")"
    drop="$(extract_metric "${last_dir}/ptq.log" "Accuracy drop:" "NF")"
  else
    last_dir="-"
    orig="-"
    folded="-"
    quant="-"
    drop="-"
  fi

  status="ok"
  if [[ "${exit_code}" -ne 0 || "${quant}" == "-" ]]; then
    status="failed"
    echo "  WARN: ${tag} failed or produced no final accuracy (exit=${exit_code}); continuing." >&2
  fi

  weight_bit="$(get_config_value "${tmp}" "weight_bit")"
  mem_bit="$(get_config_value "${tmp}" "mem_bit")"
  first_mem_bit="$(get_config_value "${tmp}" "first_mem_bit")"
  scale_bridge="$(get_config_value "${tmp}" "scale_bridge")"
  mem_lam="$(get_config_value "${tmp}" "recon_mem_lam")"
  recon_batches="$(get_config_value "${tmp}" "recon_num_batches")"
  iters="$(get_config_value "${tmp}" "adaround_iters")"
  lr="$(get_config_value "${tmp}" "adaround_lr")"
  shift="$(get_config_value "${tmp}" "recon_max_code_shift")"
  reg="$(get_config_value "${tmp}" "recon_reg_lam_scale")"
  clip="$(get_config_value "${tmp}" "recon_grad_clip")"
  pat="$(get_config_value "${tmp}" "recon_early_stop_patience")"

  echo "${tag},${group},${status},${exit_code},${weight_bit},${mem_bit},${first_mem_bit},${scale_bridge},${mem_lam},${recon_batches},${iters},${lr},${shift},${reg},${clip},${pat},${orig},${folded},${quant},${drop},${last_dir}" \
    >> "${SUMMARY}"
  echo "  -> status=${status}, quant=${quant}%, drop=${drop}%, output=${last_dir}"

  rm -f "${tmp}"
done

echo ""
echo "========== DONE =========="
echo "Summary saved to: ${SUMMARY}"
column -t -s ',' "${SUMMARY}" | head -80 || cat "${SUMMARY}"
