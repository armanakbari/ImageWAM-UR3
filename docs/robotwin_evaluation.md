# RoboTwin 2.0 Evaluation Guide

How to set up the RoboTwin 2.0 simulator and evaluate an ImageWAM FLUX.2 checkpoint on it.

Written while reproducing the released `yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin` checkpoint on the
randomized benchmark. Everything below is verified on this machine (8× NVIDIA L40S, Ubuntu 22.04,
CUDA driver 580.x, Python 3.11).

**Reproduced result: 93.24% overall success on all 50 tasks, randomized phase, 50 episodes/task
(~3 h on 7 GPUs).**

---

## Table of contents

- [0. TL;DR](#0-tldr)
- [1. Environment setup](#1-environment-setup)
- [2. Model weights](#2-model-weights)
- [3. RoboTwin simulator setup](#3-robotwin-simulator-setup)
- [4. Running the evaluation](#4-running-the-evaluation)
- [5. Reading the results](#5-reading-the-results)
- [6. What the mp4 files show](#6-what-the-mp4-files-show)
- [7. Troubleshooting](#7-troubleshooting)

---

## 0. TL;DR

If the environment is already set up, a full randomized eval is:

```bash
cd /path/to/ImageWAM

export PYTHON_BIN="$(pwd)/.venv/bin/python"      # scripts default to bare `python` — see §7
export CUDA_HOME=/usr/local/cuda-11.7            # for warp/curobo runtime
export CKPT_PATH="$(pwd)/checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt"
export DATASET_STATS_PATH="$(pwd)/checkpoints/imagewam_release/robotwin/flux2_klein_4b/dataset_stats.json"

NUM_GPUS=7 MAX_TASKS_PER_GPU=1 FLUX2_VARIANT=4b PHASES="[random]" EVAL_NUM_EPISODES=50 \
  bash scripts/flux2/run_eval_flux2_robotwin.sh \
  "MULTIRUN.gpu_ids=[0,1,2,3,5,6,7]"
```

Add `DRY_RUN=true` to print the resolved command without running it.

---

## 1. Environment setup

### 1.1 Base venv

```bash
uv sync --python 3.11 --extra shared
uv pip install "transformers==4.56.1"     # FLUX.2 requires this exact version
cp .env.example .env.local
```

> **Note — `transformers` is backbone-specific.** FLUX.2 needs `4.56.1`, OmniGen2 needs `4.51.3`.
> They are mutually exclusive in one venv.

### 1.2 `.env.local`

The shell entrypoints auto-source `.env.local`, and **it overrides anything you export first**
(it is sourced with `set -a` inside `imagewam_init`). So paths must be correct *in the file*:

```bash
MODEL_ROOT=/abs/path/to/ImageWAM/checkpoints
OUTPUT_ROOT=./runs
FLUX2_SRC=/abs/path/to/ImageWAM/third_party/flux2
FLUX2_MODEL_PATH=/abs/path/to/ImageWAM/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors
FLUX2_AE_MODEL_PATH=/abs/path/to/ImageWAM/checkpoints/flux2/FLUX.2-dev/ae.safetensors
FLUX2_QWEN3_MODEL_SPEC=Qwen/Qwen3-4B
```

Verify every path resolves before running anything:

```bash
for v in FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH; do
  p=$(grep "^$v=" .env.local | cut -d= -f2-); [ -e "$p" ] && echo "$v OK" || echo "$v MISSING: $p"
done
```

### 1.3 FLUX.2 source

```bash
git clone https://github.com/black-forest-labs/flux2 third_party/flux2
git -C third_party/flux2 checkout 50fe5162777813d869182b139e83b10743caef15
```

---

## 2. Model weights

### 2.1 Base FLUX.2 weights

```bash
bash scripts/flux2/prepare_flux2_files.sh
```

This downloads:
- `FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors` (public)
- `FLUX.2-dev/ae.safetensors` (**gated** — see below)
- optionally the 9B variant

### 2.2 If you cannot access the gated `FLUX.2-dev` autoencoder

`black-forest-labs/FLUX.2-dev` is gated. If your HF account is not approved, `ae.safetensors`
will silently fail to download (leaving only an empty `.cache/`), and the model build will fail.

**Option A (preferred):** request access at https://huggingface.co/black-forest-labs/FLUX.2-dev,
then re-run `prepare_flux2_files.sh`.

**Option B (no gated access):** the *same* VAE ships inside the public klein repo, but in
diffusers key-naming, which `flux2.autoencoder.AutoEncoder` cannot load (it uses `strict=True`).
Convert it to flux-native naming:

```bash
hf download black-forest-labs/FLUX.2-klein-base-4B \
  vae/diffusion_pytorch_model.safetensors vae/config.json \
  --local-dir checkpoints/flux2/FLUX.2-klein-base-4B
# then run a diffusers -> flux-native key remap producing checkpoints/flux2/FLUX.2-dev/ae.safetensors
```

The remap is the standard LDM↔diffusers VAE mapping:

| diffusers | flux-native | note |
|---|---|---|
| `quant_conv.*` | `encoder.quant_conv.*` | top-level → nested |
| `post_quant_conv.*` | `decoder.post_quant_conv.*` | top-level → nested |
| `encoder.down_blocks.{i}.resnets.{j}` | `encoder.down.{i}.block.{j}` | **not** reversed |
| `decoder.up_blocks.{d}.resnets.{j}` | `decoder.up.{3-d}.block.{j}` | **reversed!** |
| `*.downsamplers.0.conv` / `*.upsamplers.0.conv` | `*.downsample.conv` / `*.upsample.conv` | |
| `*.conv_shortcut` | `*.nin_shortcut` | |
| `mid_block.resnets.0/1` | `mid.block_1/block_2` | |
| `mid_block.attentions.0.group_norm` | `mid.attn_1.norm` | |
| `mid_block.attentions.0.to_q/k/v` | `mid.attn_1.q/k/v` | reshape Linear `[C,C]` → Conv `[C,C,1,1]` |
| `mid_block.attentions.0.to_out.0` | `mid.attn_1.proj_out` | same reshape |
| `*.conv_norm_out` | `*.norm_out` | |
| `bn.*` | `bn.*` | copy as-is |

**Always verify a conversion** by round-tripping a real image through the resulting AE
(`ae.decode(ae.encode(x))`). Correct weights give ≳30 dB PSNR; a wrong mapping gives <15 dB.
Our converted AE scored **34.9 dB** and produced results matching the paper.

### 2.3 Release checkpoint

```bash
mkdir -p checkpoints/imagewam_release/robotwin/flux2_klein_4b
hf download yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin --repo-type model \
  --local-dir checkpoints/imagewam_release/robotwin/flux2_klein_4b
```

Contains `model.pt` (9 GB, step 69900), `dataset_stats.json`, `config.yaml`.

> `config.yaml` references `fastwam.*` targets — that is only provenance from the older codebase.
> Evaluation uses this repo's configs plus the checkpoint's weights.

### 2.4 Qwen3 text encoder

Eval sets `model.load_text_encoder=true`, so the text encoder runs live. Pre-download it once so
that N parallel workers don't race to fetch it:

```bash
hf download Qwen/Qwen3-4B --repo-type model
```

---

## 3. RoboTwin simulator setup

> ⚠️ `third_party/RoboTwin/script/_install_uv.sh` references a `requirements_mod.txt` that **does
> not exist** in the vendored copy, so it fails immediately. Install manually as below.

### 3.1 Python packages

```bash
uv pip install --python .venv/bin/python \
  "sapien==3.0.0b1" "mplib==0.2.1" "toppra==0.6.0" "transforms3d==0.4.1" \
  "gymnasium==0.29.1" "trimesh==4.4.3" "open3d==0.18.0" "pyyaml" \
  "warp-lang==1.12.0" "setuptools<81"
```

Notes:
- **`mplib` must be `0.2.1`, not `0.1.1`.** A code comment in `envs/robot/planner.py` says 0.1.1,
  but 0.1.1 has no `mplib.sapien_utils` (→ `ModuleNotFoundError: No module named
  'mplib.sapien_utils'`). RoboTwin 2.0 needs 0.2.1.
- `setuptools<81` is required — SAPIEN 3 beta still imports `pkg_resources`.
- **`pytorch3d` is not needed.** It is only used for point-cloud FPS behind a `try/except`, and the
  task configs set `pointcloud: false`. You will see a harmless `missing pytorch3d` log line.
- `pynput` is listed upstream but is unused by the vendored code; skip it (it fails on headless
  machines with an X-display error).
- `open3d` **is** required — it is imported eagerly by `envs/utils/__init__.py`.

### 3.2 System packages (Vulkan)

SAPIEN renders with **GPU Vulkan ray tracing**. Needs (usually already present):

```bash
apt install libvulkan1 mesa-vulkan-drivers vulkan-tools   # requires root
vulkaninfo --summary | grep deviceName   # must list your NVIDIA GPU
```

> `MUJOCO_GL=osmesa` / `PYOPENGL_PLATFORM=osmesa` set by the eval wrapper are for **LIBERO/MuJoCo
> only** and have no effect on SAPIEN. There is no CPU fallback for SAPIEN's RT renderer.

### 3.3 Source patches

Two upstream patches (from `_install_uv.sh`), applied to the installed packages:

```bash
SAPIEN_LOC=$(.venv/bin/python -c "import sapien,os;print(os.path.dirname(sapien.__file__))")
MPLIB_LOC=$(.venv/bin/python -c "import mplib,os;print(os.path.dirname(mplib.__file__))")

# 1. utf-8 encoding when reading urdf/srdf
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SAPIEN_LOC/wrapper/urdf_loader.py"

# 2. don't fail screw-planning on collision
sed -i -E 's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
  "$MPLIB_LOC/planner.py"
```

Re-apply patch #2 if you ever reinstall/upgrade `mplib`.

### 3.4 curobo (required — cannot be skipped)

The default `aloha-agilex` embodiment sets `planner: "curobo"`, and `envs/robot/robot.py`
`set_planner()` instantiates `CuroboPlanner` **unconditionally**. RoboTwin also validates every
seed with a curobo-planned expert demo before running the policy. Without curobo you get an
`ImportError` at import time, or (if stubbed) an infinite seed-retry loop.

There is no prebuilt wheel — it compiles CUDA kernels:

```bash
cd third_party/RoboTwin/envs
git clone https://github.com/NVlabs/curobo.git
cd curobo && git fetch --tags && git checkout v0.7.8

export CUDA_HOME=/usr/local/cuda-11.7      # must be CUDA 11.x to match torch cu118
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6+PTX"      # see note below
MAX_JOBS=32 CMAKE_BUILD_PARALLEL_LEVEL=32 \
  uv pip install --python /path/to/ImageWAM/.venv/bin/python -e . --no-build-isolation

# verify
.venv/bin/python -c "from curobo.curobolib import lbfgs_step_cu, geom_cu; print('curobo OK')"
```

**Why `TORCH_CUDA_ARCH_LIST="8.6+PTX"`:** torch here is built for CUDA 11.8, so you need an 11.x
`nvcc` (12.x errors out on major-version mismatch). But nvcc 11.7 only supports up to `compute_87`,
while L40S/Ada is `sm_89`. Compiling for `8.6` works — an 8.6 cubin runs on an 8.9 device (same
major arch) — and `+PTX` embeds PTX so the driver can JIT for 8.9. If you have real CUDA **11.8**,
you can use `8.9` directly and drop the override. `nvcc` 11.7 vs torch 11.8 is only a *minor*
mismatch → warning, not an error.

### 3.5 Assets

```bash
cd third_party/RoboTwin/assets && python _download.py     # ~15 GB from TianxingChen/RoboTwin2.0
unzip -q embodiments.zip && unzip -q objects.zip && unzip -q background_texture.zip
rm -f *.zip && cd ..
python script/update_embodiment_config_path.py            # bakes ABSOLUTE paths into curobo*.yml
```

> **Re-run `update_embodiment_config_path.py` whenever the repo moves** — it writes absolute paths
> into `assets/embodiments/*/curobo*.yml`.

### 3.6 Policy symlink

```bash
ln -sfn "$(pwd)/experiments/robotwin/imagewam_policy" \
        "$(pwd)/third_party/RoboTwin/policy/imagewam_policy"
```

(`eval_robotwin_single.py` also creates this automatically.)

### 3.7 Verify the setup

```bash
cd third_party/RoboTwin
CUDA_VISIBLE_DEVICES=0 /path/to/ImageWAM/.venv/bin/python script/test_render.py
# expect: "Render Well"
```

This is the exact gate `eval_policy.py` runs first — it `exit()`s on any render failure.

---

## 4. Running the evaluation

### 4.1 Required environment

```bash
export PYTHON_BIN="$(pwd)/.venv/bin/python"   # REQUIRED — see §7
export CUDA_HOME=/usr/local/cuda-11.7         # warp/curobo runtime
export CKPT_PATH=/abs/path/to/model.pt
export DATASET_STATS_PATH=/abs/path/to/dataset_stats.json
```

Or derive both from a training run instead:

```bash
export EXP_PATH=./runs/<task>/<run_id>
export EVAL_TRAIN_STEP=10000
# → CKPT_PATH=${EXP_PATH}/checkpoints/weights/step_${EVAL_TRAIN_STEP}.pt
# → DATASET_STATS_PATH=${EXP_PATH}/dataset_stats.json
```

### 4.2 Smoke test first (strongly recommended)

Before committing many GPU-hours, validate the whole pipeline on one task:

```bash
NUM_GPUS=1 FLUX2_VARIANT=4b PHASES="[random]" \
  bash scripts/flux2/run_eval_flux2_robotwin.sh \
  EVALUATION.task_name=place_empty_cup EVALUATION.eval_num_episodes=2
```

Takes ~2 min. Expect `Render Well`, `missing_keys=0 unexpected_keys=0`, then a success rate.

### 4.3 Full randomized eval (all 50 tasks)

```bash
NUM_GPUS=7 MAX_TASKS_PER_GPU=1 FLUX2_VARIANT=4b PHASES="[random]" EVAL_NUM_EPISODES=50 \
  bash scripts/flux2/run_eval_flux2_robotwin.sh \
  "MULTIRUN.gpu_ids=[0,1,2,3,5,6,7]"
```

- **~3 h on 7× L40S** for 50 tasks × 50 episodes.
- `MULTIRUN.gpu_ids` lets you skip busy GPUs (here GPU 4 was in use).
- **`MAX_TASKS_PER_GPU=1`**: each worker uses **~23 GB** (≈16 GB model + ≈7 GB curobo planner
  subprocess). Two workers on a 46 GB card will OOM. The script's default of `3` is too high for
  46 GB cards.

### 4.4 Key knobs

| Env var | Default | Meaning |
|---|---|---|
| `PHASES` | `[clean,random]` | `random` → `demo_randomized`; `clean` → `demo_clean`. Use `"[random]"` for randomized only. |
| `EVAL_NUM_EPISODES` | `50` | Episodes per task. RoboTwin leaderboard convention is 100. |
| `NUM_GPUS` | `8` | Number of GPUs. |
| `MAX_TASKS_PER_GPU` | `3` | **Set to 1** on 46 GB GPUs. |
| `FLUX2_VARIANT` | `4b` | `4b` or `9b`. |
| `SKIP_GET_OBS_WITHIN_REPLAN` | `true` | `false` → full-motion videos (slower). See §6. |
| `ACTION_HORIZON` / `REPLAN_STEPS` | `16` / `16` | Action chunk length / replan interval. |
| `DRY_RUN` | `false` | Print the command instead of running it. |
| `MULTIRUN.gpu_ids` | `null` | Hydra override, e.g. `"[0,1,2,3,5,6,7]"`. |

Single task / subset:

```bash
... EVALUATION.task_name=place_empty_cup           # one task
... EVALUATION.task_name=place_empty_cup,click_bell # comma-separated subset
```

"Full RoboTwin" = the 50 tasks listed in `third_party/RoboTwin/task_config/_eval_step_limit.yml`
(that file also sets each task's step limit, 400–1700).

### 4.5 Clean phase / both phases

```bash
PHASES="[clean]"        # clean only
PHASES="[clean,random]" # both (default)
```

> The wrapper hardcodes a swap of `_imagewam` → `_clean_imagewam` in the task config name. This is
> **unrelated to the clean/random phase** — it only selects a different *training* episode subset
> (`episode_index_filter`). `shape_meta` is identical and eval normalization comes from
> `DATASET_STATS_PATH`, so it does not affect eval results.

---

## 5. Reading the results

```
evaluate_results/robotwin/model/<timestamp>/
├── summary.csv                     # per-task + __overall__ row
├── summary.json                    # same, structured
├── eval_<task>_<ts>.log            # per-task log
└── <task>/
    ├── _result_random.txt          # success rate for the random phase
    ├── _result_clean.txt           # (if the clean phase was run)
    └── episode<N>_randomized-<bool>_success-<bool>.mp4
```

`summary.csv`:

```csv
task_name,clean_success_rate,random_success_rate
adjust_bottle,,1.0
...
__overall__,,0.9324
```

Aggregate across runs with `experiments/robotwin/summarize_results.py`.

### Reference numbers (this repo, released 4B checkpoint)

50 tasks × 50 episodes, randomized: **overall 93.24%**.
Lowest: `move_stapler_pad` 0.58, `place_can_basket` 0.68, `turn_switch` 0.70.
16 tasks at 1.00.

---

## 6. What the mp4 files show

One mp4 per episode (50 tasks × 50 episodes = 2500 files), named:

```
episode<N>_randomized-<true|false>_success-<true|false>.mp4
```

so you can filter failures directly:

```bash
find evaluate_results/robotwin/model/<ts> -name "*success-false.mp4"
```

### Layout

**640×480, 10 fps, H.264.** Each frame is a 2×2 composite of the robot's own cameras
(`_get_eval_video_frame`, `envs/_base_task.py:41`):

```
┌─────────────────────┬─────────────────────┐
│  head camera        │                     │
│  (320×240)          │   black padding     │
├─────────────────────┼─────────────────────┤
│  left wrist camera  │  right wrist camera │
│  (320×240)          │  (320×240)          │
└─────────────────────┴─────────────────────┘
```

The black quadrant is not a bug: the head camera (320 px wide) is zero-padded to match the width of
the two wrist views side-by-side (640 px).

These are the same three camera streams the policy consumes — the policy just receives them resized
and tiled differently (`compact_288x256`).

### Timing / length

- One frame is written **per policy action step** (`take_action`, `_base_task.py:1506`), at 10 fps.
- Recording stops at **success** or at the task's **step limit**
  (`_eval_step_limit.yml`, e.g. `place_empty_cup: 500`, `beat_block_hammer: 400`).
- So **video length tells you the outcome**: a short video = quick success; a video exactly
  `step_lim` frames long = timeout failure.

### ⚠️ Videos are choppy by default (only every 16th frame is real)

With the default `SKIP_GET_OBS_WITHIN_REPLAN=true`, the environment only calls `get_obs()` when the
policy requests a new observation — i.e. once per replan (every `REPLAN_STEPS=16` actions). The
video writer reads the cached `self.now_obs`, so **the same frame is written 16× and then jumps**.

Measured on a real episode (mean abs pixel diff between consecutive frames):

```
 96-> 97: 0.032     ... ~0.01 (H.264 noise only, frames are duplicates)
111->112: 0.001
112->113: 7.455     <-- the replan boundary: the only real update
113->114: 0.007
```

So a 137-frame video contains only ~8 distinct observations. This is fine for checking *what
happened* (did it grasp? did it drop?), but it is **not** a smooth motion recording.

To get true per-step video (slower, renders every step):

```bash
SKIP_GET_OBS_WITHIN_REPLAN=false ... bash scripts/flux2/run_eval_flux2_robotwin.sh
```

To disable video entirely, set `eval_video_log: false` in
`third_party/RoboTwin/task_config/demo_randomized.yml`.

`ffmpeg` must be on `PATH` — the sim shells out to it to encode.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` although the venv has it | `source .venv/bin/activate` may be silently overridden by a conda hook. **Always call `.venv/bin/python` explicitly**, and set `PYTHON_BIN=$(pwd)/.venv/bin/python` for the scripts (they default to bare `python`; the manager passes `sys.executable` to children, so this propagates). |
| `.venv/bin/python: No such file or directory` | uv upgraded its managed interpreter and orphaned the symlink. Repoint it and fix `pyvenv.cfg`: `ln -sfn ~/.local/share/uv/python/cpython-3.11.<new>-linux-x86_64-gnu/bin/python3.11 .venv/bin/python`. Site-packages survive. |
| `GatedRepoError: 403 ... FLUX.2-dev` | Gated AE. See §2.2. |
| `ModuleNotFoundError: No module named 'mplib.sapien_utils'` | Wrong mplib. Install `mplib==0.2.1` and re-apply the planner patch (§3.3). |
| `ImportError: cannot import name 'CuroboPlanner'` | curobo not installed/built. See §3.4 — it is required. |
| `nvcc fatal: Unsupported gpu architecture 'compute_89'` | nvcc 11.7 can't target Ada. Build curobo with `TORCH_CUDA_ARCH_LIST="8.6+PTX"`. |
| `The detected CUDA version mismatches the version used to compile PyTorch` | Using CUDA 12.x nvcc with a cu118 torch. Set `CUDA_HOME=/usr/local/cuda-11.7`. |
| `Render Error` then exit | SAPIEN Vulkan failure. Check `vulkaninfo --summary` lists the NVIDIA GPU and that `/usr/share/vulkan/icd.d/nvidia_icd.json` exists. osmesa does **not** help SAPIEN. |
| CUDA OOM with several workers per GPU | ~23 GB/worker. Use `MAX_TASKS_PER_GPU=1` on 46 GB cards. |
| Model paths not found despite exporting them | `.env.local` is sourced **after** your exports and overrides them. Fix the file (§1.2). |
| Eval hangs, never produces episodes | Usually curobo failing inside `expert_check` — every seed fails and the loop retries forever. Check the per-task log. |
| `missing pytorch3d` in logs | Harmless — not needed for RGB eval (§3.1). |
