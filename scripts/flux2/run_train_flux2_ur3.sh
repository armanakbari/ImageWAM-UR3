#!/usr/bin/env bash
# Fine-tune FLUX.2-klein ImageWAM on the EmbodyX/UR3 (organize_pi05) real dataset.
# Mirrors scripts/flux2/run_train_flux2_klein_imagewam.sh but for the UR3 LeRobot v2.1 dataset.
#
# Required env (set in .env.local or the shell):
#   UR3_DATA_ROOT   path to the LeRobot root (the dir containing data/ meta/ videos/)
#   FLUX2_SRC, FLUX2_MODEL_PATH, FLUX2_AE_MODEL_PATH   (see README Model Preparation)
# Optional:
#   GPU_PER_NODE (default 8), FLUX2_VARIANT (4b|9b, default 4b),
#   PRECOMPUTE_QWEN3_CACHE (default true), REBUILD_ACTION_INIT (default false)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}"      # 4b | 9b
ZERO_STAGE="${ZERO_STAGE:-1}"
PRECOMPUTE_QWEN3_CACHE="${PRECOMPUTE_QWEN3_CACHE:-true}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"

imagewam_require_env UR3_DATA_ROOT
imagewam_require_env FLUX2_SRC
imagewam_require_env FLUX2_AE_MODEL_PATH

case "${FLUX2_VARIANT}" in
  4b)
    MODEL_CONFIG="configs/model/imagewam_flux2_klein_4b_base.yaml"
    TASK_NAME="ur3_flux2_klein_4b_base_imagewam"
    FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
    FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
    ;;
  9b)
    MODEL_CONFIG="configs/model/imagewam_flux2_klein_9b_base.yaml"
    TASK_NAME="ur3_flux2_klein_9b_base_imagewam"   # create this task analogously if you use 9B
    FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-8B}"
    FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-klein-base-9B/flux-2-klein-base-9b.safetensors}"
    ZERO_STAGE=2
    ;;
  *) echo "Invalid FLUX2_VARIANT=${FLUX2_VARIANT}; expected 4b or 9b" >&2; exit 1 ;;
esac
export FLUX2_MODEL_PATH FLUX2_QWEN3_MODEL_SPEC ZERO_STAGE

ACTION_DIM=14
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${UR3_DATA_ROOT}/flux2_qwen3_cache_${FLUX2_VARIANT}}"
DATASET_OVERRIDES=(
  "data.train.dataset_dirs=[${UR3_DATA_ROOT}]"
  "data.val.dataset_dirs=[${UR3_DATA_ROOT}]"
  "data.train.qwen_text_cache_dir=${QWEN_CACHE_DIR}"
  "data.val.qwen_text_cache_dir=${QWEN_CACHE_DIR}"
  "data.train.qwen_context_len=128"
  "data.val.qwen_context_len=128"
  "data.train.qwen_text_cache_format=qwen3_flux2"
  "data.val.qwen_text_cache_format=qwen3_flux2"
  "model.qwen_context_len=128"
  "model.proprio_dim=14"
)

ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_flux2_${FLUX2_VARIANT}_ur3_init.pt}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

imagewam_print_config TASK_NAME FLUX2_VARIANT UR3_DATA_ROOT FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH QWEN_CACHE_DIR ACTION_INIT

# 1) Copy-initialize the Action DiT from the FLUX.2 image DiT (auto, once).
if [ "${REBUILD_ACTION_INIT:-false}" = "true" ] || [ ! -f "${ACTION_INIT}" ]; then
  imagewam_run imagewam_python scripts/flux2/preprocess_action_dit_flux2.py \
    --model-config "${MODEL_CONFIG}" \
    --flux2-src-path "${FLUX2_SRC}" \
    --flux2-model-path "${FLUX2_MODEL_PATH}" \
    --variant "klein-base-${FLUX2_VARIANT}" \
    --action-dim "${ACTION_DIM}" \
    --output "${ACTION_INIT}" \
    --apply-alpha-scaling true
fi

# 2) Precompute the Qwen3 text-embedding cache (required: load_text_encoder=false).
if [ "${PRECOMPUTE_QWEN3_CACHE}" = "true" ]; then
  imagewam_run torchrun --standalone --nproc_per_node="${GPU_PER_NODE}" \
    scripts/flux2/precompute_flux2_qwen3_embeds.py \
    task="${TASK_NAME}" \
    qwen_cache_batch_size="${QWEN_CACHE_BATCH_SIZE:-16}" \
    qwen_cache_save_workers="${QWEN_CACHE_SAVE_WORKERS:-4}" \
    qwen_cache_overwrite="${QWEN_CACHE_OVERWRITE:-false}" \
    model.variant="klein-base-${FLUX2_VARIANT}" \
    model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    flux2_qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    "${DATASET_OVERRIDES[@]}"
fi

# 3) Train.
COMMON_OVERRIDES=(
  "model.flux2_model_path=${FLUX2_MODEL_PATH}"
  "model.ae_model_path=${FLUX2_AE_MODEL_PATH}"
  "model.qwen3_model_spec=${FLUX2_QWEN3_MODEL_SPEC}"
  "model.action_dit_pretrained_path=${ACTION_INIT}"
)

TASK="${TASK_NAME}" imagewam_run bash scripts/flux2/train_flux2_klein_imagewam.sh "${GPU_PER_NODE}" \
  "${DATASET_OVERRIDES[@]}" \
  "${COMMON_OVERRIDES[@]}" \
  "$@"
