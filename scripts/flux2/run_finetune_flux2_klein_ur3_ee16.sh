#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the released 16D InternData-A1 checkpoint on the real-world dual-arm
# UR3 data (blue_basket + mug, trained jointly). UR3 remains 14D; the task config
# pads dimensions 7 and 15 inside the processor, exactly like the RoboTwin ee16 run.
#
# Required env (set in .env.local or the shell):
#   PRETRAIN_CHECKPOINT  path to ImageWAM-FLUX.2-4B-InternData-A1-EE/model.pt
#   FLUX2_SRC, FLUX2_MODEL_PATH, FLUX2_AE_MODEL_PATH   (see README Model Preparation)
# Optional:
#   GPU_PER_NODE (default 8), ZERO_STAGE (default 1),
#   PRECOMPUTE_QWEN3_CACHE (default false), QWEN_CACHE_DIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
ZERO_STAGE="${ZERO_STAGE:-1}"
PRECOMPUTE_QWEN3_CACHE="${PRECOMPUTE_QWEN3_CACHE:-false}"
FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"

imagewam_require_env PRETRAIN_CHECKPOINT
imagewam_require_env FLUX2_SRC
imagewam_require_env FLUX2_MODEL_PATH
imagewam_require_env FLUX2_AE_MODEL_PATH

# Which UR3 task config to run. The qwen3 cache is content-addressed by prompt hash,
# so all UR3 tasks can share one cache dir.
TASK_NAME="${TASK_NAME:-ur3_basket_mug_flux2_klein_4b_base_imagewam_ee16}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${REPO_ROOT}/data/ur3/flux2_qwen3_cache_4b}"

export FLUX2_QWEN3_MODEL_SPEC ZERO_STAGE
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_OVERRIDES=(
  "data.train.qwen_text_cache_dir=${QWEN_CACHE_DIR}"
)

MODEL_OVERRIDES=(
  "model.flux2_src_path=${FLUX2_SRC}"
  "model.flux2_model_path=${FLUX2_MODEL_PATH}"
  "model.ae_model_path=${FLUX2_AE_MODEL_PATH}"
  "model.qwen3_model_spec=${FLUX2_QWEN3_MODEL_SPEC}"
  "resume=${PRETRAIN_CHECKPOINT}"
)

imagewam_print_config \
  TASK_NAME PRETRAIN_CHECKPOINT QWEN_CACHE_DIR \
  FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH GPU_PER_NODE ZERO_STAGE

if [ "${PRECOMPUTE_QWEN3_CACHE}" = "true" ]; then
  # Prompts are sharded across ranks but every rank loads Qwen3-4B, and these two
  # datasets have exactly two distinct prompts — extra ranks only pay the model load.
  imagewam_run torchrun --standalone --nproc_per_node="${QWEN_CACHE_NPROC:-1}" \
    scripts/flux2/precompute_flux2_qwen3_embeds.py \
    task="${TASK_NAME}" \
    qwen_cache_batch_size="${QWEN_CACHE_BATCH_SIZE:-16}" \
    qwen_cache_save_workers="${QWEN_CACHE_SAVE_WORKERS:-4}" \
    qwen_cache_overwrite="${QWEN_CACHE_OVERWRITE:-false}" \
    model.variant=klein-base-4b \
    flux2_qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    "${DATASET_OVERRIDES[@]}" \
    "${MODEL_OVERRIDES[@]}"
elif [ "${PRECOMPUTE_QWEN3_CACHE}" != "false" ]; then
  echo "Invalid PRECOMPUTE_QWEN3_CACHE=${PRECOMPUTE_QWEN3_CACHE}; expected true or false" >&2
  exit 1
fi

TASK="${TASK_NAME}" imagewam_run bash scripts/flux2/train_flux2_klein_imagewam.sh "${GPU_PER_NODE}" \
  "${DATASET_OVERRIDES[@]}" \
  "${MODEL_OVERRIDES[@]}" \
  "$@"
