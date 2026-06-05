#!/usr/bin/env bash
# ==========================================================================
# PTQ4SDT paper ablation runner
# ==========================================================================
# Runs the paper-facing ablation matrix with temporary configs only.
#
# Usage:
#   DATASET=cifar10  GPU=0 ./exp/run_paper_ablation.sh
#   DATASET=cifar100 GPU=0 ./exp/run_paper_ablation.sh
#   DATASET=imagenet GPU=2 ./exp/run_paper_ablation.sh
#
# Groups:
#   core   none / idea1 only / idea2 only / both
#   scale  pot and observer scale controls
#   bits   weight and membrane bit-width controls
#   all    all groups above
# ==========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "Cannot find conda.sh" >&2
  exit 1
fi
conda activate spikeTransformer

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="${DATASET:-cifar10}"
GROUP="${GROUP:-all}"
GPU_OVERRIDE="${GPU:-0}"

case "${DATASET}" in
  cifar10)
    BASE_CONFIG="${BASE_CONFIG:-exp/cifar10/config.yml}"
    OUT_ROOT="${REPO_ROOT}/output/ptq/cifar10"
    DEFAULT_MEM_LAM="0.25"
    DEFAULT_RECON_BATCHES="8"
    DEFAULT_ADAROUND_ITERS="500"
    DEFAULT_ADAROUND_LR="0.003"
    DEFAULT_REG="0.00001"
    BEST_NOTE="CIFAR10 prior best: reg1e-5, quant=94.35"
    ;;
  cifar100)
    BASE_CONFIG="${BASE_CONFIG:-exp/cifar100/config.yml}"
    OUT_ROOT="${REPO_ROOT}/output/ptq/cifar100"
    DEFAULT_MEM_LAM="0.05"
    DEFAULT_RECON_BATCHES="8"
    DEFAULT_ADAROUND_ITERS="2000"
    DEFAULT_ADAROUND_LR="0.003"
    DEFAULT_REG="0.0001"
    BEST_NOTE="CIFAR100 prior best: mem0.05, quant=75.25"
    ;;
  imagenet|imageNet|ImageNet)
    DATASET="imagenet"
    BASE_CONFIG="${BASE_CONFIG:-exp/imageNet/config.yml}"
    OUT_ROOT="${REPO_ROOT}/output/ptq/imagenet"
    # Prior ImageNet top result was idea1-only (mem0.00, 74.30).
    # For idea2/both, use the strongest nonzero membrane setting (mem0.50).
    DEFAULT_MEM_LAM="0.50"
    DEFAULT_RECON_BATCHES="8"
    DEFAULT_ADAROUND_ITERS="1000"
    DEFAULT_ADAROUND_LR="0.003"
    DEFAULT_REG="0.0001"
    BEST_NOTE="ImageNet prior best overall: mem0.00, quant=74.30; best nonzero membrane: mem0.50, quant=74.27"
    ;;
  *)
    echo "Unknown DATASET=${DATASET}; expected cifar10, cifar100, or imagenet" >&2
    exit 1
    ;;
esac

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Base config not found: ${BASE_CONFIG}" >&2
  exit 1
fi

set_config_value() {
  local file="$1"
  local key="$2"
  local val="$3"
  local escaped
  escaped="${val//&/\\&}"
  if grep -qE "^${key}:" "${file}"; then
    sed -i -E "s|^(${key}:).*|\\1 ${escaped}|" "${file}"
  else
    printf "\n%s: %s\n" "${key}" "${val}" >> "${file}"
  fi
}

get_config_value() {
  local file="$1"
  local key="$2"
  awk -F':' -v k="${key}" '
    $1 == k {
      gsub(/^[ \t]+|[ \t]+$/, "", $2)
      split($2, a, /[ \t#]/)
      print a[1]
      exit
    }
  ' "${file}"
}

extract_acc1() {
  local file="$1"
  local pattern="$2"
  grep -E "${pattern}" "${file}" 2>/dev/null \
    | tail -n1 \
    | sed -nE 's/.*Acc@1:[[:space:]]*([0-9.]+)%.*/\1/p'
}

extract_drop() {
  local file="$1"
  grep "Accuracy drop:" "${file}" 2>/dev/null \
    | tail -n1 \
    | sed -nE 's/.*Accuracy drop:[[:space:]]*([0-9.]+)%.*/\1/p'
}

make_tmp_config() {
  local tmp="$1"
  cp "${BASE_CONFIG}" "${tmp}"

  set_config_value "${tmp}" "gpu" "${GPU_OVERRIDE}"
  set_config_value "${tmp}" "output_dir" "${OUT_ROOT}"

  set_config_value "${tmp}" "fake_quant" "adaround"
  set_config_value "${tmp}" "observer" "mse"
  set_config_value "${tmp}" "mem_observer" "mse"

  set_config_value "${tmp}" "weight_bit" "4"
  set_config_value "${tmp}" "first_layer_bit" "16"
  set_config_value "${tmp}" "last_layer_bit" "8"
  set_config_value "${tmp}" "mem_bit" "4"
  set_config_value "${tmp}" "first_mem_bit" "4"

  set_config_value "${tmp}" "adaround_iters" "${DEFAULT_ADAROUND_ITERS}"
  set_config_value "${tmp}" "adaround_lr" "${DEFAULT_ADAROUND_LR}"
  set_config_value "${tmp}" "recon_num_batches" "${DEFAULT_RECON_BATCHES}"
  set_config_value "${tmp}" "recon_mem_lam" "${DEFAULT_MEM_LAM}"
  set_config_value "${tmp}" "recon_reg_lam_scale" "${DEFAULT_REG}"
  set_config_value "${tmp}" "recon_max_code_shift" "2.0"
  set_config_value "${tmp}" "recon_min_iters" "50"
  set_config_value "${tmp}" "recon_early_stop_patience" "0"
  set_config_value "${tmp}" "recon_improve_eps" "0.000001"
  set_config_value "${tmp}" "recon_grad_clip" "1.0"
  set_config_value "${tmp}" "recon_log_interval" "100"
  set_config_value "${tmp}" "scale_bridge" "unify"
}

# Format: tag|group|key=value|key=value|...
EXPS=(
  "none_w4m4|core|scale_bridge=weight|recon_mem_lam=0.0"
  "idea1_unify_w4m4|core|scale_bridge=unify|recon_mem_lam=0.0"
  "idea2_mem_w4m4|core|scale_bridge=weight|recon_mem_lam=${DEFAULT_MEM_LAM}"
  "both_unify_mem_w4m4|core|scale_bridge=unify|recon_mem_lam=${DEFAULT_MEM_LAM}"

  "scale_pot_mem|scale|scale_bridge=pot|recon_mem_lam=${DEFAULT_MEM_LAM}"
  "scale_observer_mem|scale|scale_bridge=observer|recon_mem_lam=${DEFAULT_MEM_LAM}"

  "bits_w8m4|bits|scale_bridge=unify|recon_mem_lam=${DEFAULT_MEM_LAM}|weight_bit=8|first_layer_bit=16|last_layer_bit=8|mem_bit=4|first_mem_bit=4"
  "bits_w3m4|bits|scale_bridge=unify|recon_mem_lam=${DEFAULT_MEM_LAM}|weight_bit=3|first_layer_bit=16|last_layer_bit=8|mem_bit=4|first_mem_bit=4"
  "bits_w2m4|bits|scale_bridge=unify|recon_mem_lam=${DEFAULT_MEM_LAM}|weight_bit=2|first_layer_bit=16|last_layer_bit=8|mem_bit=4|first_mem_bit=4"
  "bits_w4m8|bits|scale_bridge=unify|recon_mem_lam=${DEFAULT_MEM_LAM}|weight_bit=4|first_layer_bit=16|last_layer_bit=8|mem_bit=8|first_mem_bit=8"
  "bits_w4m3|bits|scale_bridge=unify|recon_mem_lam=${DEFAULT_MEM_LAM}|weight_bit=4|first_layer_bit=16|last_layer_bit=8|mem_bit=3|first_mem_bit=3"
)

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY="${OUT_ROOT}/paper_ablation_${RUN_STAMP}.csv"
LOG_DIR="${OUT_ROOT}/paper_ablation_logs_${RUN_STAMP}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

echo "tag,group,status,exit_code,dataset,weight_bit,first_layer_bit,last_layer_bit,mem_bit,first_mem_bit,scale_bridge,recon_mem_lam,recon_num_batches,adaround_iters,adaround_lr,recon_max_code_shift,recon_reg_lam_scale,recon_grad_clip,recon_early_stop_patience,orig_acc1,folded_acc1,quant_acc1,drop,seconds,output_dir" \
  > "${SUMMARY}"

echo "Repo:    ${REPO_ROOT}"
echo "Dataset: ${DATASET}"
echo "Base:    ${BASE_CONFIG}"
echo "Group:   ${GROUP}"
echo "GPU:     ${GPU_OVERRIDE}"
echo "Summary: ${SUMMARY}"
echo "Logs:    ${LOG_DIR}"
echo "Note:    ${BEST_NOTE}"
echo "----------------------------------------"

for spec in "${EXPS[@]}"; do
  IFS='|' read -r -a parts <<<"${spec}"
  tag="${parts[0]// /}"
  exp_group="${parts[1]// /}"
  overrides=("${parts[@]:2}")

  if [[ "${GROUP}" != "all" && "${GROUP}" != "${exp_group}" ]]; then
    continue
  fi

  echo ""
  echo "========== [${DATASET}/${exp_group}] ${tag} =========="
  echo "  overrides: ${overrides[*]}"

  tmp="$(mktemp "${TMPDIR:-/tmp}/ptq4sdt_${DATASET}_${tag}.XXXXXX.yml")"
  make_tmp_config "${tmp}"

  for kv in "${overrides[@]}"; do
    key="${kv%%=*}"
    val="${kv#*=}"
    key="${key// /}"
    val="${val// /}"
    set_config_value "${tmp}" "${key}" "${val}"
  done

  run_log="${LOG_DIR}/${tag}.log"
  start_s="$(date +%s)"
  python ptq/main.py --config "${tmp}" 2>&1 | tee "${run_log}"
  exit_code="${PIPESTATUS[0]}"
  end_s="$(date +%s)"
  seconds="$((end_s - start_s))"

  last_dir="$(grep '^Output:' "${run_log}" | tail -n1 | awk '{print $2}' || true)"
  if [[ -z "${last_dir}" || ! -f "${last_dir}/ptq.log" ]]; then
    ds_short="$(get_config_value "${tmp}" "dataset" | sed 's|torch/||; s|-||g')"
    last_dir="$(ls -1dt "${OUT_ROOT}"/*-sdt-w*-"${ds_short}" 2>/dev/null | head -n1 || true)"
  fi

  ptq_log=""
  if [[ -n "${last_dir}" && -f "${last_dir}/ptq.log" ]]; then
    ptq_log="${last_dir}/ptq.log"
  fi

  if [[ -n "${ptq_log}" ]]; then
    orig="$(extract_acc1 "${ptq_log}" "Original[[:space:]]+Acc@1:")"
    folded="$(extract_acc1 "${ptq_log}" "After (fold )?BN[[:space:]]+Acc@1:")"
    quant="$(extract_acc1 "${ptq_log}" "Quantized Acc@1:")"
    drop="$(extract_drop "${ptq_log}")"
  else
    last_dir="-"
    orig="-"
    folded="-"
    quant="-"
    drop="-"
  fi

  status="ok"
  if [[ "${exit_code}" -ne 0 || -z "${quant}" || "${quant}" == "-" ]]; then
    status="failed"
    [[ -z "${quant}" ]] && quant="-"
    [[ -z "${drop}" ]] && drop="-"
    echo "  WARN: ${tag} failed or produced no final accuracy (exit=${exit_code}); continuing." >&2
  fi
  [[ -z "${orig}" ]] && orig="-"
  [[ -z "${folded}" ]] && folded="-"

  weight_bit="$(get_config_value "${tmp}" "weight_bit")"
  first_layer_bit="$(get_config_value "${tmp}" "first_layer_bit")"
  last_layer_bit="$(get_config_value "${tmp}" "last_layer_bit")"
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

  echo "${tag},${exp_group},${status},${exit_code},${DATASET},${weight_bit},${first_layer_bit},${last_layer_bit},${mem_bit},${first_mem_bit},${scale_bridge},${mem_lam},${recon_batches},${iters},${lr},${shift},${reg},${clip},${pat},${orig},${folded},${quant},${drop},${seconds},${last_dir}" \
    >> "${SUMMARY}"
  echo "  -> status=${status}, quant=${quant}%, drop=${drop}%, seconds=${seconds}, output=${last_dir}"

  rm -f "${tmp}"
done

echo ""
echo "========== DONE =========="
echo "Summary saved to: ${SUMMARY}"
column -t -s ',' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
