#!/usr/bin/env bash
# 在已配置 CEED-VLA + CUDA 的容器内跑 perf_jacobi.py
# 用法（宿主机）:
#   chmod +x vla-scripts/extern/run_benchmark_ceedvla_vs_af_jacobi_docker.sh
#   MODEL_PATH="/home/daiyuntao/jetson-containers/data/models/huggingface/models--chenpyyy--openvla-ac/..." \
#   UNNORM_KEY="libero_10_no_noops" \
#   ./vla-scripts/extern/run_benchmark_ceedvla_vs_af_jacobi_docker.sh --runs 5 --warmup 2
#
# 环境变量:
#   CONTAINER   默认 actionflow-container
#   AF_ROOT     容器内 Action-Flow 根目录，默认 /home/daiyuntao/Action-Flow
#   CEED_ROOT   容器内 CEED-VLA 根目录，默认 /home/daiyuntao/CEED-VLA
#   MODEL_PATH  默认与 memory/ceedvla_libero.md 中 libero_10 checkpoint 一致（可按需覆盖）

set -euo pipefail

CONTAINER="${CONTAINER:-actionflow-container}"
AF_ROOT="${AF_ROOT:-/home/daiyuntao/Action-Flow}"
CEED_ROOT="${CEED_ROOT:-/home/daiyuntao/CEED-VLA}"
MODEL_PATH="${MODEL_PATH:-/home/daiyuntao/jetson-containers/data/models/huggingface/models--chenpyyy--openvla-ac/openvla-7b+libero_10_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug-ac3-80000-ckpt}"
UNNORM_KEY="${UNNORM_KEY:-libero_10_no_noops}"
ACTION_CHUNK="${ACTION_CHUNK:-3}"

if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "错误: 找不到容器 ${CONTAINER}。请先创建/命名容器或设置 CONTAINER=..."
  exit 1
fi

docker start "${CONTAINER}" >/dev/null 2>&1 || true

# perf 脚本内部用 namespace 只挂 CEED-VLA/prismatic/extern/hf；此处以 Action-Flow 为主即可
export PYTHONPATH="${AF_ROOT}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ -n "${UNNORM_KEY}" ]]; then
  EXTRA_ARGS+=(--unnorm-key "${UNNORM_KEY}")
fi

exec docker exec \
  -w "${AF_ROOT}" \
  -e PYTHONPATH="${PYTHONPATH}" \
  -e CEED_VLA_ROOT="${CEED_ROOT}" \
  "${CONTAINER}" \
  python vla-scripts/extern/perf_jacobi.py \
    --model-path "${MODEL_PATH}" \
    --backend both \
    --action-chunk "${ACTION_CHUNK}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
