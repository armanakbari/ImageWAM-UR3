# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ImageWAM is the training/evaluation codebase for the paper *ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?* It builds a **World Action Model (WAM)** that repurposes a pretrained **image-editing** model (instead of a video generator) as the visual backbone for robot action prediction. A flow-matching **action expert** is attached to the editing branch; the action expert conditions on the per-layer **key-value caches** produced while the editing backbone denoises a *single future endpoint frame*. At inference the edited image is never decoded — only the editing-branch KV cache is harvested and fed to the action expert. It is evaluated on **LIBERO**, **LIBERO-plus**, **RoboTwin 2.0**, and real-world dual-arm tasks.

The codebase is a generalization of FastWAM (which used a Wan2.2 *video* DiT). FastWAM lives on here only as a comparison baseline; the `src/` package is `imagewam`. There are three editing backbones, in recommended order: **FLUX.2 [klein] 4B/9B** (strongest), **OmniGen2**, **Ovis-U1** (smallest, 1.1B DiT).

There is no test suite, linter, or CI.

## Environment

Uses **`uv`** (not conda). Tested on CUDA 11.8 / Python 3.11 / PyTorch 2.7.1.

```bash
uv sync --python 3.11 --extra shared   # installs the `imagewam` package + shared deps
source .venv/bin/activate
cp .env.example .env.local             # shell entrypoints auto-source .env.local
```

`--extra shared` installs only common deps. Each backbone needs its own external source repo (cloned under `third_party/`) and **a different `transformers` version** — they are mutually exclusive in one venv, so switch before running a given backbone:

- **FLUX.2**: `transformers==4.56.1`; clone `black-forest-labs/flux2` → `third_party/flux2` (pinned commit), set `FLUX2_SRC`, `FLUX2_MODEL_PATH`, `FLUX2_AE_MODEL_PATH`, `FLUX2_QWEN3_MODEL_SPEC`.
- **OmniGen2**: `transformers==4.51.3`; clone `yuyangalin/OmniGen2` → `third_party/OmniGen2`, set `OMNIGEN2_SRC`, `OMNIGEN2_MODEL_PATH`, `QWEN_MODEL_PATH`.
- **Ovis-U1**: code vendored under `third_party/ovis_u1_hf`, set `OVIS_U1_MODEL_PATH` (default HF id `AIDC-AI/Ovis-U1-3B`).

Backbone scripts prepend the relevant `third_party/<repo>/src` to `PYTHONPATH` at launch. Common env vars: `DATA_ROOT` (per-dataset), `ROBOTWIN_ROOT`, `MODEL_ROOT` (default `./checkpoints`), `OUTPUT_ROOT` (default `./runs`). See `README.md` and `docs/dependencies.md` for the full setup.

## Required preprocessing (before training)

The `run_train_*` wrappers handle these automatically when the env flags are set, but understand what they do:

1. **ActionDiT init weights** — `scripts/<backbone>/preprocess_action_dit_*.py` copies/interpolates the editing DiT weights into an action-head initialization `.pt` (referenced by `model.action_dit_pretrained_path`). The wrapper builds it at `checkpoints/action_dit_<backbone>_<variant>_<task>_init.pt` if missing; force with `REBUILD_ACTION_INIT=true`.
2. **Text-embedding cache** — configs set `load_text_encoder: false`, so prompt embeddings must be precomputed. FLUX.2 uses Qwen3 (`precompute_flux2_qwen3_embeds.py`, format `qwen3_flux2`); OmniGen2 uses Qwen2.5-VL (`precompute_qwen_embeds.py`). Enable via `PRECOMPUTE_QWEN3_CACHE=true` / `PRECOMPUTE_QWEN_CACHE=true`; the model receives `context`/`context_mask` at train time, never raw text.
3. **RoboTwin no-op filter** — `scripts/data/precompute_noops_lerobot.sh` writes `${ROBOTWIN_ROOT}/nonidle_ranges.json` (referenced by `nonidle_filter_path`).

## Common commands

Training/eval is a **layered shell system**. You normally invoke the top env-var-driven wrapper; it resolves config/paths and calls the inner launcher, which calls `train_zero1.sh`/`train_zero2.sh` → `accelerate launch` → `scripts/train.py`.

```bash
# Train FLUX.2 ImageWAM (env vars select backbone/dataset; first arg to inner script = nproc).
GPU_PER_NODE=8 TASK_TYPE=libero   FLUX2_VARIANT=4b PRECOMPUTE_QWEN3_CACHE=true bash scripts/flux2/run_train_flux2_klein_imagewam.sh
GPU_PER_NODE=8 TASK_TYPE=robotwin FLUX2_VARIANT=4b PRECOMPUTE_QWEN3_CACHE=true bash scripts/flux2/run_train_flux2_klein_imagewam.sh
GPU_PER_NODE=8 TASK_TYPE=libero   PRECOMPUTE_QWEN_CACHE=true  bash scripts/omnigen2/run_train_imagewam.sh
GPU_PER_NODE=8 TASK_TYPE=libero   bash scripts/ovis_u1/run_train_ovis_u1_imagewam.sh

# Train directly (single process, debugging). First arg = nproc_per_node; rest = Hydra overrides.
bash scripts/train_zero1.sh 8 task=libero_flux2_klein_4b_base_imagewam
python scripts/train.py task=libero_omnigen2_imagewam output_dir=./runs/dbg   # bypass accelerate entirely

# Evaluate — by explicit checkpoint...
CKPT_PATH=/path/model.pt DATASET_STATS_PATH=/path/dataset_stats.json \
  NUM_GPUS=8 FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero.sh
# ...or derive from a run dir + step:
EXP_PATH=./runs/<task>/<run_id> EVAL_TRAIN_STEP=10000 NUM_GPUS=8 bash scripts/flux2/run_eval_flux2_robotwin.sh
```

Eval wrappers exist per backbone × benchmark under `scripts/{flux2,omnigen2,ovis_u1}/run_eval_*.sh` (LIBERO, LIBERO-plus, RoboTwin). Useful knobs: `ZERO_STAGE=1|2`, `USE_CLEAN_ROBOTWIN=true`, `QWEN_CACHE_DIR`, `ACTION_INIT`, `DRY_RUN=true` (print commands), `IMAGEWAM_QUIET=true`. `scripts/common.sh` provides the `imagewam_*` shell helpers all wrappers source.

**Run dir convention**: `train_zero1.sh` derives `runs/<task>/<RUN_ID>` (RUN_ID = timestamp; for multi-node `NNODES>1` it syncs RUN_ID across machines via a `torch.distributed.TCPStore`). Eval managers parse `.../runs/<task>/<date_dir>/...` to tag results, so keep that layout. FLUX.2 9B uses ZeRO-2 (VRAM); others use ZeRO-1.

## Configuration system (Hydra + OmegaConf)

Everything is config-driven; follow `_target_` keys to the class/factory that a config builds (`hydra.utils.instantiate`).

- **`configs/train.yaml`** — root; `data`/`model`/`task` default to `null` and are set by the chosen task.
- **`configs/task/*.yaml`** — the entrypoint you select (`task=...`). Overrides `/data` and `/model`, sets bs/epochs/lr/intervals. Naming encodes backbone+benchmark, e.g. `libero_flux2_klein_4b_base_imagewam`, `robotwin_omnigen2_imagewam`, `libero_ovis_u1_imagewam`.
- **`configs/model/*.yaml`** — `_target_` points at a `create_imagewam*` factory in `src/imagewam/runtime.py`. Defines the editing-backbone wiring, the `action_dit_config` (smaller `hidden_dim` than the editing DiT, larger `attn_head_dim` for cross-expert attention), independent `video_scheduler`/`action_scheduler`, and `loss.lambda_{video,action}`. Optional per-backbone LoRA config.
- **`configs/data/*.yaml`** — build `RobotVideoDataset` + `ImageWAMProcessor`; define `shape_meta` (camera/action/state keys+dims), `num_frames`, `action_video_freq_ratio`, `concat_multi_camera`, normalization.
- **`configs/sim_libero*.yaml` / `configs/sim_robotwin.yaml`** — eval-manager configs (`EVALUATION.*`, `MULTIRUN.*`); compose `train` + a task override.

Custom OmegaConf resolvers are registered by `register_default_resolvers()` (called in every entrypoint, `src/imagewam/utils/config_resolvers.py`): `${oc.load:...}`, `${eval:'...'}` (arbitrary Python in configs), `${sum_shapes:...}`, `${max_action_dim:...}`, `${oc.env:...}`.

## Architecture

### Model (`src/imagewam/models/backbones/`)
- **`imagewam.py` — `ImageWAM`**: the core `nn.Module`. Wraps an editing **video expert**, an **ActionDiT** action expert, a **`MoT`**, a VAE, and (optionally) a text encoder. Built via backbone-specific classmethods: `from_flux2_klein_pretrained`, `from_omnigen2_pretrained`, `from_ovis_u1_pretrained`, `from_wan22_pretrained`. Exposes `training_loss` / `infer`.
- **`mot.py` — `MoT` (Mixture-of-Transformers)**: joint self-attention over four token types — language-context, visual-condition, visual-prediction, action. Action tokens attend one-way to the others; noisy tokens attend only to clean context. **`model.dit` is aliased to the `MoT`** — the trainer and freeze logic operate on `model.dit`. `mot_checkpoint_mixed_attn` toggles gradient checkpointing on cross-expert attention.
- **Per-backbone video experts**: `flux2_video_expert.py`, `omnigen2_video_expert.py`, `ovis_u1_video_expert.py` (and `wan22.py`/`wan_video_dit.py` for the FastWAM-style video baseline). Matching action experts: `action_dit_flux2.py`, `action_dit_omnigen2.py`, plus `action_dit.py` and others.
- **WAM variants** (subclass chain matters): base `ImageWAM` (editing-cache conditioning) → `ImageWAMJoint` → `ImageWAMIDM` → `ImageWAMCacheIDM` (IDM-style; actions condition on first frame + cached future video latents) and `ImageWAMNoiseIDM`. Each has a `create_imagewam_*` factory in `runtime.py` and a corresponding `configs/model/*.yaml`. `imagewam_cache_idm.py` is the FastWAM-style co-training baseline (video tokens used only during training, removed at inference).
- **`schedulers/scheduler_continuous.py`**: continuous flow-matching scheduler with separate train/infer `shift`. Video/image and action denoising have independent schedulers and timesteps (τ for the editing cache, s for action flow).

### Training (`src/imagewam/trainer.py` — `Wan22Trainer`)
HF `accelerate` + DeepSpeed (ZeRO configs in `scripts/ds_configs/`, accelerate configs in `scripts/accelerate_configs/`).
- **Only the DiT/`MoT` (+ optional `proprio_encoder`) is trainable** — `_apply_dit_only_train_mode` freezes everything else (VAE, frozen VLM/understanding components, text encoder) before the optimizer/ZeRO is built.
- Step-based loop (`max_steps`, derived from dataset length × epochs if unset), warmup-cosine schedule (~5% warmup), AdamW betas (0.9, 0.95), lr `1e-4`, grad clip 1.0.
- **Resume semantics**: `resume=<dir>` restores full ZeRO/training state (`load_training_state`); `resume=<file.pt>` loads weights only (detected by `_looks_like_weights_checkpoint`, applied before `accelerate.prepare`).
- `evaluate()` runs a rollout: logs train loss, PSNR/SSIM, denormalized action L1/L2, and saves a stitched mp4.
- **First run for a new task**: set `pretrained_norm_stats: null` in the data config; a `dataset_stats.json` is written to the run dir — point `pretrained_norm_stats`/`DATASET_STATS_PATH` at it for subsequent runs and eval.

### Data (`src/imagewam/datasets/lerobot/`)
LeRobot-format datasets. `robot_video_dataset.py` (`RobotVideoDataset`) yields video clips + actions + proprio + **cached text embeddings**. `processors/imagewam_processor.py` (`ImageWAMProcessor`) handles normalization (`utils/normalizer.py`), action/state merging (`transforms/action_state_merger.py`), and delta-action masking (`delta_action_dim_mask` — e.g. eef poses delta, gripper absolute). `action_video_freq_ratio` ties action horizon to video frames (4 → 32 actions / 9 video frames). Eval denormalization goes through the val dataset's processor.

### Evaluation (`experiments/`)
Two-tier: a **manager** (`run_{libero,robotwin}_manager.py`, Hydra) shells out to a **parallel runner** that launches **single-task workers** (`eval_*_single.py`) across GPUs. LIBERO workers run in **separate processes** that activate `LIBERO_WORKER_ENV_SOURCE` (a venv `activate` path). `experiments/libero/action_ensembler.py` provides temporal action ensembling (`EVALUATION.use_action_ensembler`). RoboTwin requires symlinking the policy: `ln -sfn "$(pwd)/experiments/robotwin/imagewam_policy" third_party/RoboTwin/policy/imagewam_policy`; the entrypoint loaded by RoboTwin is `experiments/robotwin/imagewam_policy/deploy_policy.py`. RoboTwin eval defaults `EVALUATION.skip_get_obs_within_replan=true` for speed (set `SKIP_GET_OBS_WITHIN_REPLAN=false` to render full videos). `summarize_results.py` aggregates.

`third_party/RoboTwin/` is a vendored copy of the RoboTwin platform (`README.vendor.md`); treat it as external code.

## Conventions / gotchas
- **One backbone per venv**: FLUX.2 (`transformers==4.56.1`) and OmniGen2 (`transformers==4.51.3`) require different versions — switch before running the other.
- Video tensors are `[B, 3, T, H, W]` in `(-1, 1)`. Multi-camera frames are tiled before VAE encoding: LIBERO concatenates 2 views horizontally to `224×448`; RoboTwin tiles 3 views (wrist-horizontal + vertical) to `288×256`. Layout is set by `concat_multi_camera` and the dataset config.
- `load_text_encoder: false` means the text/Qwen embedding cache **must exist first** (see preprocessing). The cache format differs per backbone (`qwen_text_cache_format`).
- Editing-backbone DiT + action expert are the only trainable parts; the VLM/understanding modules stay frozen (this decoupling is a deliberate design choice — see the paper's "why not unified U+G models" ablation).
- Action chunk length and future horizon are typically 16; LIBERO `action_dim=7`, RoboTwin `action_dim=14` (and `proprio_dim=14`).
- Release checkpoints (`yuyangalin/ImageWAM-FLUX.2-{4B,9B}-{LIBERO,RoboTwin}`) each contain `model.pt`, `dataset_stats.json`, and the training config (`train_config.yaml`).
