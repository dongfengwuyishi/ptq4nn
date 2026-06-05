#!/usr/bin/env bash
# 在单卡上按顺序跑多组 PTQ，仅改写 scale_bridge（其余与 config.yml 一致）。
# 用法（在仓库根或本目录执行均可）:
#   ./exp/cifar10/run_scale_bridge_sweep.sh
#   SCALE_BRIDGES="weight unify" ./exp/cifar10/run_scale_bridge_sweep.sh
# GPU 仍由 config.yml 里的 gpu 字段决定（ptq/main.py 会设置 CUDA_VISIBLE_DEVICES）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# 与 ptq_run.sh 一致，按需改成本机 conda 路径
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate spikeTransformer

CONFIG="${CONFIG:-exp/cifar10/config.yml}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Error: config not found: ${CONFIG}"
  exit 1
fi

# 空格分隔；合法值见 ptq/quantize.py: weight | pot | observer | unify
DEFAULT_BRIDGES="unify pot observer weight"
SCALE_BRIDGES="${SCALE_BRIDGES:-${DEFAULT_BRIDGES}}"

echo "Repo:    ${REPO_ROOT}"
echo "Base:    ${CONFIG}"
echo "Bridges: ${SCALE_BRIDGES}"
echo "----------------------------------------"

for bridge in ${SCALE_BRIDGES}; do
  echo ""
  echo "========== scale_bridge=${bridge} =========="
  tmp="$(mktemp "${TMPDIR:-/tmp}/ptq_cifar10_${bridge}.XXXXXX.yml")"
  sed "s/^scale_bridge:.*/scale_bridge: ${bridge}/" "${CONFIG}" > "${tmp}"
  python ptq/main.py --config "${tmp}"
  rm -f "${tmp}"
done

echo ""
echo "All scale_bridge runs finished."
