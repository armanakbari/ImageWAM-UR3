#!/usr/bin/env bash
# Shared helpers for ImageWAM release scripts.
# Source this file from bash entrypoints after `set -euo pipefail`.

imagewam_init() {
  local default_root="$1"
  REPO_ROOT="${REPO_ROOT:-$(cd "${default_root}" && pwd)}"
  export REPO_ROOT
  cd "${REPO_ROOT}"

  if [ -f "${REPO_ROOT}/.env.local" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env.local"
    set +a
  fi

  imagewam_setup_runtime_libs
}

# Make the repo venv usable without the caller having activated it, and put the
# CUDA 11 NPP libs on the loader path. torchcodec's CUDA build needs
# libnppicc.so.11 (pip package nvidia-npp-cu11, not a torch dependency); without
# it torchcodec fails to import and LeRobot silently falls back to a pyav
# decoder that leaks GBs per batch and OOM-kills the dataloader workers.
imagewam_setup_runtime_libs() {
  if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    case ":${PATH}:" in
      *":${REPO_ROOT}/.venv/bin:"*) ;;
      *) export PATH="${REPO_ROOT}/.venv/bin:${PATH}" ;;
    esac
    export PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
  fi

  local npp_lib
  for npp_lib in "${REPO_ROOT}"/.venv/lib/python*/site-packages/nvidia/npp/lib; do
    if [ -d "${npp_lib}" ]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":${npp_lib}:"*) ;;
        *) export LD_LIBRARY_PATH="${npp_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
      esac
    fi
  done
}

imagewam_require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: ${name}" >&2
    echo "Set it in the shell or in ${REPO_ROOT}/.env.local" >&2
    exit 2
  fi
}

imagewam_print_config() {
  if [ "${IMAGEWAM_QUIET:-false}" = "true" ]; then
    return 0
  fi
  local name
  for name in "$@"; do
    printf '[config] %s=%s\n' "${name}" "${!name:-<unset>}"
  done
}

imagewam_run() {
  if [ "${DRY_RUN:-false}" = "true" ]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

imagewam_activate_env() {
  local env_file="${1:-}"
  if [ -n "${env_file}" ]; then
    # shellcheck disable=SC1090
    source "${env_file}"
  fi
}

imagewam_python() {
  "${PYTHON_BIN:-python}" "$@"
}

imagewam_ckpt_from_exp() {
  if [ -z "${CKPT_PATH:-}" ]; then
    imagewam_require_env EXP_PATH
    imagewam_require_env EVAL_TRAIN_STEP
    CKPT_PATH="${EXP_PATH}/checkpoints/weights/step_${EVAL_TRAIN_STEP}.pt"
    export CKPT_PATH
  fi
  if [ -z "${DATASET_STATS_PATH:-}" ] && [ -n "${EXP_PATH:-}" ]; then
    DATASET_STATS_PATH="${EXP_PATH}/dataset_stats.json"
    export DATASET_STATS_PATH
  fi
}

imagewam_prepare_eval_ckpt() {
  if [ -n "${LOCAL_CKPT_ROOT:-}" ]; then
    imagewam_require_env CKPT_PATH
    if [ "${DRY_RUN:-false}" = "true" ]; then
      echo "[dry-run] skipping local checkpoint copy"
      return 0
    fi
    local task_name="${TASK:-eval}"
    local run_name="$(basename "$(dirname "$(dirname "$(dirname "${CKPT_PATH}")")")")"
    local local_path="${LOCAL_CKPT_ROOT}/runs/${task_name}/${run_name}/checkpoints/weights/$(basename "${CKPT_PATH}")"
    mkdir -p "$(dirname "${local_path}")"
    if [ ! -f "${local_path}" ] || [ "${CKPT_PATH}" -nt "${local_path}" ]; then
      echo "Copying checkpoint to local disk: ${local_path}"
      cp "${CKPT_PATH}" "${local_path}.tmp"
      mv "${local_path}.tmp" "${local_path}"
    else
      echo "Using existing local checkpoint: ${local_path}"
    fi
    CKPT_PATH="${local_path}"
    export CKPT_PATH
  fi
}
