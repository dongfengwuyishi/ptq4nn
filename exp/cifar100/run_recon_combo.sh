#!/usr/bin/env bash
# ==========================================================================
# Reconstruction combo sweep (CIFAR100)
# ==========================================================================
# Combine the best one-factor ablation settings without editing the base
# config. Each run writes its own temporary config and appends a summary CSV.
#
# Usage:
#   GPU=1 ./exp/cifar100/run_recon_combo.sh
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate spikeTransformer

BASE_CONFIG="${BASE_CONFIG:-exp/cifar100/config.yml}"
GPU_OVERRIDE="${GPU:-1}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Error: base config not found: ${BASE_CONFIG}" >&2
  exit 1
fi

# Format: "tag|key=val|key=val|..."
# Chosen from the strongest CIFAR100 one-factor results:
# mem0.05, lr1e-3, b4/b2, reg0/reg1e-3.
EXPS=(
  "mem005_lr1e3_b4_reg0|recon_mem_lam=0.05|adaround_lr=0.001|recon_num_batches=4|recon_reg_lam_scale=0.0"
  "mem005_lr1e3_b4_reg1e3|recon_mem_lam=0.05|adaround_lr=0.001|recon_num_batches=4|recon_reg_lam_scale=0.001"
  "mem005_lr1e3_b2_reg0|recon_mem_lam=0.05|adaround_lr=0.001|recon_num_batches=2|recon_reg_lam_scale=0.0"
  "mem005_lr1e3_b8_reg0|recon_mem_lam=0.05|adaround_lr=0.001|recon_num_batches=8|recon_reg_lam_scale=0.0"
  "mem005_lr3e3_b4_reg0|recon_mem_lam=0.05|adaround_lr=0.003|recon_num_batches=4|recon_reg_lam_scale=0.0"
  "mem025_lr1e3_b4_reg0|recon_mem_lam=0.25|adaround_lr=0.001|recon_num_batches=4|recon_reg_lam_scale=0.0"
)

OUT_ROOT="${REPO_ROOT}/output/ptq/cifar100"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY="${OUT_ROOT}/recon_combo_${RUN_STAMP}.csv"
LOG_DIR="${OUT_ROOT}/recon_combo_logs_${RUN_STAMP}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

echo "tag,recon_mem_lam,recon_num_batches,adaround_iters,adaround_lr,recon_max_code_shift,recon_reg_lam_scale,recon_grad_clip,recon_early_stop_patience,orig_acc1,folded_acc1,quant_acc1,drop,output_dir" \
  > "${SUMMARY}"

echo "Repo:    ${REPO_ROOT}"
echo "Base:    ${BASE_CONFIG}"
echo "GPU:     ${GPU_OVERRIDE}"
echo "Summary: ${SUMMARY}"
echo "Logs:    ${LOG_DIR}"
echo "----------------------------------------"

set_config_value() {
  local file="$1"
  local key="$2"
  local val="$3"
  if grep -qE "^${key}:" "${file}"; then
    sed -i "s/^${key}:.*/${key}: ${val}/" "${file}"
  else
    printf "\n%s: %s\n" "${key}" "${val}" >> "${file}"
  fi
}

get_config_value() {
  local file="$1"
  local key="$2"
  awk -F':' -v k="${key}" '$1 == k {gsub(/^[ \t]+|[ \t]+$/, "", $2); split($2, a, /[ \t#]/); print a[1]; exit}' "${file}"
}

for spec in "${EXPS[@]}"; do
  IFS='|' read -r -a parts <<<"${spec}"
  tag="${parts[0]// /}"
  overrides=("${parts[@]:1}")

  echo ""
  echo "========== ${tag} =========="
  echo "  overrides: ${overrides[*]}"

  tmp="$(mktemp "${TMPDIR:-/tmp}/ptq_cifar100_combo_${tag}.XXXXXX.yml")"
  cp "${BASE_CONFIG}" "${tmp}"

  set_config_value "${tmp}" "gpu" "${GPU_OVERRIDE}"

  for kv in "${overrides[@]}"; do
    key="${kv%%=*}"
    val="${kv#*=}"
    key="${key// /}"
    val="${val// /}"
    set_config_value "${tmp}" "${key}" "${val}"
  done

  run_log="${LOG_DIR}/${tag}.log"
  python ptq/main.py --config "${tmp}" 2>&1 | tee "${run_log}"

  last_dir="$(grep '^Output:' "${run_log}" | tail -n1 | awk '{print $2}')"
  if [[ -z "${last_dir}" || ! -f "${last_dir}/ptq.log" ]]; then
    last_dir="$(ls -1dt "${OUT_ROOT}"/*-sdt-w*-cifar100 2>/dev/null | head -n1 || true)"
  fi

  if [[ -z "${last_dir}" || ! -f "${last_dir}/ptq.log" ]]; then
    echo "  WARN: cannot locate output dir for ${tag}" >&2
    rm -f "${tmp}"
    continue
  fi

  orig=$(grep "Original  Acc@1:" "${last_dir}/ptq.log" | tail -n1 | awk '{print $NF}' | tr -d '%')
  folded=$(grep "After BN  Acc@1:" "${last_dir}/ptq.log" | tail -n1 | awk '{print $4}' | tr -d '%')
  quant=$(grep "Quantized Acc@1:" "${last_dir}/ptq.log" | tail -n1 | awk '{print $NF}' | tr -d '%')
  drop=$(grep "Accuracy drop:" "${last_dir}/ptq.log" | tail -n1 | awk '{print $NF}' | tr -d '%')
  [[ -z "${folded}" ]] && folded="-"

  mem_lam=$(get_config_value "${tmp}" "recon_mem_lam")
  recon_batches=$(get_config_value "${tmp}" "recon_num_batches")
  iters=$(get_config_value "${tmp}" "adaround_iters")
  lr=$(get_config_value "${tmp}" "adaround_lr")
  shift=$(get_config_value "${tmp}" "recon_max_code_shift")
  reg=$(get_config_value "${tmp}" "recon_reg_lam_scale")
  clip=$(get_config_value "${tmp}" "recon_grad_clip")
  pat=$(get_config_value "${tmp}" "recon_early_stop_patience")

  echo "${tag},${mem_lam},${recon_batches},${iters},${lr},${shift},${reg},${clip},${pat},${orig},${folded},${quant},${drop},${last_dir}" \
    >> "${SUMMARY}"
  echo "  -> quant=${quant}%, drop=${drop}%, output=${last_dir}"

  rm -f "${tmp}"
done

echo ""
echo "========== DONE =========="
echo "Summary saved to: ${SUMMARY}"
column -t -s ',' "${SUMMARY}" || cat "${SUMMARY}"
