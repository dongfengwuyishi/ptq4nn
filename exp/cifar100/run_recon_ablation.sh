#!/usr/bin/env bash
# ==========================================================================
# Layer-wise Membrane-Aware Reconstruction Ablation (CIFAR100)
# ==========================================================================
# Runs one-factor sweeps for the reconstruction hyperparameters that most
# affect accuracy. Each run uses a temporary config, so the base config is not
# modified.
#
# Usage:
#   ./exp/cifar100/run_recon_ablation.sh
#   GROUP=mem   ./exp/cifar100/run_recon_ablation.sh
#   GPU=1       ./exp/cifar100/run_recon_ablation.sh
#
# Groups:
#   mem       recon_mem_lam sweep, including 0.0 output-only baseline
#   batches   calibration-batch sweep
#   iters     optimization-iteration sweep
#   lr        Adam learning-rate sweep
#   shift     max integer-code movement sweep
#   reg       code-shift regularization sweep
#   control   small controls for grad clip / early stop
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
GROUP="${GROUP:-all}"
GPU_OVERRIDE="${GPU:-1}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Error: base config not found: ${BASE_CONFIG}" >&2
  exit 1
fi

# Format: "tag|group|key=val|key=val|..."
# Keep this as one-factor ablation around the base config to make the result
# interpretable and avoid a large Cartesian product.
EXPS=(
  "base|base"

  # Most important: membrane reconstruction weight.
  "mem0.00|mem|recon_mem_lam=0.0"
  "mem0.05|mem|recon_mem_lam=0.05"
  "mem0.10|mem|recon_mem_lam=0.10"
  "mem0.25|mem|recon_mem_lam=0.25"
  "mem0.50|mem|recon_mem_lam=0.50"
  "mem1.00|mem|recon_mem_lam=1.00"

  # Calibration samples used by layer-wise reconstruction.
  "b2|batches|recon_num_batches=2"
  "b4|batches|recon_num_batches=4"
  "b8|batches|recon_num_batches=8"
  "b16|batches|recon_num_batches=16"

  # Optimizer budget and step size.
  "it200|iters|adaround_iters=200"
  "it500|iters|adaround_iters=500"
  "it1000|iters|adaround_iters=1000"
  "it2000|iters|adaround_iters=2000"

  "lr1e-3|lr|adaround_lr=0.001"
  "lr3e-3|lr|adaround_lr=0.003"
  "lr1e-2|lr|adaround_lr=0.01"

  # Stability controls.
  "shift1|shift|recon_max_code_shift=1.0"
  "shift2|shift|recon_max_code_shift=2.0"
  "shift4|shift|recon_max_code_shift=4.0"

  "reg0|reg|recon_reg_lam_scale=0.0"
  "reg1e-5|reg|recon_reg_lam_scale=0.00001"
  "reg1e-4|reg|recon_reg_lam_scale=0.0001"
  "reg1e-3|reg|recon_reg_lam_scale=0.001"

  "noclip|control|recon_grad_clip=0.0"
  "pat20|control|recon_early_stop_patience=20"
  "pat80|control|recon_early_stop_patience=80"
)

OUT_ROOT="${REPO_ROOT}/output/ptq/cifar100"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY="${OUT_ROOT}/recon_ablation_${RUN_STAMP}.csv"
LOG_DIR="${OUT_ROOT}/recon_ablation_logs_${RUN_STAMP}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

echo "tag,group,recon_mem_lam,recon_num_batches,adaround_iters,adaround_lr,recon_max_code_shift,recon_reg_lam_scale,recon_grad_clip,recon_early_stop_patience,orig_acc1,folded_acc1,quant_acc1,drop,output_dir" \
  > "${SUMMARY}"

echo "Repo:    ${REPO_ROOT}"
echo "Base:    ${BASE_CONFIG}"
echo "Group:   ${GROUP}"
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
  group="${parts[1]// /}"
  overrides=("${parts[@]:2}")

  if [[ "${GROUP}" != "all" && "${GROUP}" != "${group}" ]]; then
    continue
  fi

  echo ""
  echo "========== [${group}] ${tag} =========="
  if [[ ${#overrides[@]} -gt 0 ]]; then
    echo "  overrides: ${overrides[*]}"
  else
    echo "  overrides: (base config)"
  fi

  tmp="$(mktemp "${TMPDIR:-/tmp}/ptq_cifar100_recon_${tag}.XXXXXX.yml")"
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

  echo "${tag},${group},${mem_lam},${recon_batches},${iters},${lr},${shift},${reg},${clip},${pat},${orig},${folded},${quant},${drop},${last_dir}" \
    >> "${SUMMARY}"
  echo "  -> quant=${quant}%, drop=${drop}%, output=${last_dir}"

  rm -f "${tmp}"
done

echo ""
echo "========== DONE =========="
echo "Summary saved to: ${SUMMARY}"
column -t -s ',' "${SUMMARY}" | head -80 || cat "${SUMMARY}"
