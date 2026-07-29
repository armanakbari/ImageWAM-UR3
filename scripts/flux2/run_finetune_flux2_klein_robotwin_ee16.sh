#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the released 16D InternData-A1 checkpoint on RoboTwin. RoboTwin
# remains 14D; the task config pads dimensions 7 and 15 inside the processor.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
ZERO_STAGE="${ZERO_STAGE:-1}"
USE_CLEAN_ROBOTWIN="${USE_CLEAN_ROBOTWIN:-true}"
PRECOMPUTE_QWEN3_CACHE="${PRECOMPUTE_QWEN3_CACHE:-false}"
FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"

imagewam_require_env PRETRAIN_CHECKPOINT
imagewam_require_env ROBOTWIN_ROOT
imagewam_require_env FLUX2_SRC
imagewam_require_env FLUX2_MODEL_PATH
imagewam_require_env FLUX2_AE_MODEL_PATH

case "${USE_CLEAN_ROBOTWIN}" in
  true) TASK_NAME="robotwin_flux2_klein_4b_base_clean_imagewam_ee16" ;;
  false) TASK_NAME="robotwin_flux2_klein_4b_base_imagewam_ee16" ;;
  *)
    echo "Invalid USE_CLEAN_ROBOTWIN=${USE_CLEAN_ROBOTWIN}; expected true or false" >&2
    exit 1
    ;;
esac

QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${ROBOTWIN_ROOT}/flux2_qwen3_cache_4b}"
NONIDLE_FILTER_PATH="${NONIDLE_FILTER_PATH:-${ROBOTWIN_ROOT}/nonidle_ranges.json}"

export FLUX2_QWEN3_MODEL_SPEC ZERO_STAGE
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_OVERRIDES=(
  "data.train.dataset_dirs=[${ROBOTWIN_ROOT}]"
  "data.val.dataset_dirs=[${ROBOTWIN_ROOT}]"
  "data.train.nonidle_filter_path=${NONIDLE_FILTER_PATH}"
  "data.val.nonidle_filter_path=${NONIDLE_FILTER_PATH}"
  "data.train.qwen_text_cache_dir=${QWEN_CACHE_DIR}"
  "data.val.qwen_text_cache_dir=${QWEN_CACHE_DIR}"
  "data.train.qwen_context_len=128"
  "data.val.qwen_context_len=128"
  "data.train.qwen_text_cache_format=qwen3_flux2"
  "data.val.qwen_text_cache_format=qwen3_flux2"
)

MODEL_OVERRIDES=(
  "model.flux2_src_path=${FLUX2_SRC}"
  "model.flux2_model_path=${FLUX2_MODEL_PATH}"
  "model.ae_model_path=${FLUX2_AE_MODEL_PATH}"
  "model.qwen3_model_spec=${FLUX2_QWEN3_MODEL_SPEC}"
  "model.qwen_context_len=128"
  "model.load_text_encoder=false"
  "model.proprio_dim=16"
  "model.action_dit_pretrained_path=null"
  "resume=${PRETRAIN_CHECKPOINT}"
)

imagewam_print_config \
  TASK_NAME PRETRAIN_CHECKPOINT ROBOTWIN_ROOT QWEN_CACHE_DIR NONIDLE_FILTER_PATH \
  FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH GPU_PER_NODE ZERO_STAGE

if [ "${PRECOMPUTE_QWEN3_CACHE}" = "true" ]; then
  imagewam_run torchrun --standalone --nproc_per_node="${GPU_PER_NODE}" \
    scripts/flux2/precompute_flux2_qwen3_embeds.py \
    task="${TASK_NAME}" \
    qwen_cache_batch_size="${QWEN_CACHE_BATCH_SIZE:-16}" \
    qwen_cache_save_workers="${QWEN_CACHE_SAVE_WORKERS:-4}" \
    qwen_cache_overwrite="${QWEN_CACHE_OVERWRITE:-false}" \
    model.variant=klein-base-4b \
    model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
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
