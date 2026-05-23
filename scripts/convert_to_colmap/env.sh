# FastGS EuRoC 流程路径变量 — 修改本文件后: source scripts/env.sh

export FASTGS_ROOT="/home/jackson/opensource/FastGS"
export EUROC_ROOT="/home/jackson/data/euroc/recon/euroc_VID_Arealm_20260417160341422"
export EUROC_NAME="$(basename "${EUROC_ROOT}")"
export COLMAP_DATA="${FASTGS_ROOT}/data/${EUROC_NAME}_cam0"
export MODEL_OUTPUT="${FASTGS_ROOT}/output/${EUROC_NAME}"
export TRAJECTORY_FILE="${EUROC_ROOT}/CameraTrajectory.txt"
export DEPTH_MIN=0.3
export DEPTH_MAX=5.0
export CUDA_VISIBLE_DEVICES=0
