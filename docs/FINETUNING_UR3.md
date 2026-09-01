# Fine-tuning ImageWAM on real-world UR3 data

End-to-end procedure for fine-tuning **ImageWAM (FLUX.2 [klein] base 4B)** from the released
InternData-A1 pretrain onto new EmbodyX UR3 tasks. Written from four completed runs; the
gotchas section is the part that will actually save you time.

---

## 0. Does anything live outside `ImageWAM/`?

**Only the datasets.** `ImageWAM/data/ur3/*_lerobot` are symlinks into `../data/ur3/`
(the FastWAM checkout), so the two repos share one copy of the bytes:

```
ImageWAM/data/ur3/blue_basket_lerobot -> /…/FastWAM/data/ur3/blue_basket_lerobot
ImageWAM/data/ur3/drawer_lerobot      -> /…/FastWAM/data/ur3/drawer_lerobot
ImageWAM/data/ur3/stacking_cubes_lerobot, mug_lerobot, ethernet_2.0_lerobot   (same pattern)
```

Nothing else reaches outside. Model weights, the FLUX.2 source, the venv, the text cache, and
all run outputs are inside `ImageWAM/`. The one incidental external path is the shared
HuggingFace cache (`~/.cache/huggingface`, ~7.6 GB for Qwen3-4B), which is only touched when
precomputing text embeddings.

**On a fresh machine you need nothing from the parent folder** — download the data straight
into `ImageWAM/data/ur3/` from
[`armanakbari4/ur3-3task-lerobot`](https://huggingface.co/datasets/armanakbari4/ur3-3task-lerobot)
and skip the symlinks entirely.

---

## 1. One-time setup

**a. Environment** — `uv sync --python 3.11 --extra shared`, then `transformers==4.56.1` for
the FLUX.2 backbone, and clone `black-forest-labs/flux2` into `third_party/flux2`.

**b. `nvidia-npp-cu11` — do not skip this.**
```bash
uv pip install --python .venv/bin/python nvidia-npp-cu11
```
`scripts/common.sh:imagewam_setup_runtime_libs` (called from `imagewam_init`) puts it on
`LD_LIBRARY_PATH` for every wrapper. See §5.1 for why this is load-bearing.

**c. Base weights** into `checkpoints/`:
- `flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors`
- `flux2/FLUX.2-dev/ae.safetensors`

**d. The pretrain** (9.0 GB):
```bash
hf download yuyangalin/ImageWAM-FLUX.2-4B-InternData-A1-EE --repo-type model \
  --local-dir checkpoints/imagewam_release/interndata_pretrained/flux2_klein_4b
```

**e. `.env.local`** — `FLUX2_SRC`, `FLUX2_MODEL_PATH`, `FLUX2_AE_MODEL_PATH`,
`FLUX2_QWEN3_MODEL_SPEC=Qwen/Qwen3-4B`, `PRETRAIN_CHECKPOINT` (the `model.pt` from step d).

---

## 2. Adding a new task

UR3 data always has the same shape — 3 cameras at 240×320, 14-dim joint action/state — so a
new task is two config files and a symlink, no code.

**a.** Put the LeRobot dir at `data/ur3/<name>_lerobot` (or symlink it).

**b. Data config** — copy an existing one and change only `dataset_dirs`:
```bash
sed 's|/mug_lerobot|/<name>_lerobot|' \
  configs/data/ur3_basket_mug_ee16.yaml > configs/data/ur3_<name>_ee16.yaml
```
For a multi-task run, list several dirs under `dataset_dirs:` — the datasets are concatenated
and each keeps its own instruction.

**c. Task config** — copy `configs/task/ur3_ethernet_flux2_klein_4b_base_imagewam_ee16.yaml`,
point `override /data` at your new data config, set `max_steps`, and rename the wandb entry.

### Why `ee16`

The pretrain uses a **16D** action/state space. UR3's 14D vector is *zero-padded* at dims 7
and 15 via `transforms.canonical.RobotWin14ToCanonicalEE16` and flagged in
`action_dim_is_pad` — nothing is converted to end-effector poses. UR3's layout
(`[L_arm(6)|L_grip|R_arm(6)|R_grip]`) is identical to RoboTwin's, so the transform applies
unchanged. **At inference you must un-pad back to 14D**: `concat(x[...,0:7], x[...,8:15])`.

---

## 3. Launching

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_PER_NODE=4 PRECOMPUTE_QWEN3_CACHE=true \
  TASK_NAME=ur3_<name>_flux2_klein_4b_base_imagewam_ee16 \
  nohup bash scripts/flux2/run_finetune_flux2_klein_ur3_ee16.sh wandb.enabled=false \
  > logs/ur3_<name>_train.log 2>&1 &
```

`PRECOMPUTE_QWEN3_CACHE=true` on the first run only — the cache is content-addressed by prompt
hash, so all UR3 tasks share `data/ur3/flux2_qwen3_cache_4b` and new prompts just get appended.

### Keep global batch at 192

Every run so far uses **global batch 192**, which is what makes them comparable. When the GPU
count changes you must re-derive `batch_size × gpus × accum = 192`:

| GPUs | `batch_size` | `gradient_accumulation_steps` | measured speed | peak mem/GPU |
|---|---|---|---|---|
| 2 | 12 | 8 | 0.14 step/s | **94–95 GB** (at the ceiling) |
| 4 | 12 | 4 | 0.22 step/s | 77–80 GB |
| 6 | 8 | 4 | 0.27 step/s | 59–61 GB |

192 does not divide evenly across 5 GPUs at any sane per-GPU batch — use 4 or 6.
**Avoid the 2-GPU configuration if you can**: it has under 1 GB of headroom and has OOM-ed.

### Recipe (upstream's fine-tune-from-pretrain values)

lr **2.5e-5** cosine, 5% warmup · AdamW(0.9, 0.95) · wd 0.01 · grad-clip 1.0 · bf16 ·
DeepSpeed **ZeRO-1** · `num_frames 17`, `action_video_freq_ratio 1` → 16-step action horizon ·
`endpoint_frames_only: true` · 3 cams tiled to **288×256** `compact_288x256` · z-score norm.

(Use 5e-5 only when starting from base FLUX.2 rather than the pretrain.)

---

## 4. Checkpoints

`save_every: 1000` → one 9.04 GB weights file per 1000 steps.

**`save_optimizer_state: false`** (added in `trainer.py`, default `true` elsewhere) skips the
DeepSpeed ZeRO state. That state is **~55 GB per save** against a 9 GB weights file — see §5.2.
The trade-off: no optimizer resume. Restarting means weights-only from a `.pt` with a fresh
optimizer and LR schedule, or starting over.

**Checkpoints exist only at multiples of `save_every`, but evals run every `eval_every` (500).**
So the best-scoring *eval* may have no corresponding checkpoint — don't promise someone a
`step_007500.pt` that was never written.

---

## 5. Gotchas that cost real time

### 5.1 torchcodec / AV1 / `libnppicc.so.11`
`torchcodec 0.5+cu118` needs the CUDA-11 NPP libs, which torch does **not** pull in. Without
them `import torchcodec` fails, LeRobot silently falls back to a pyav decoder that leaks GBs
per batch, and dataloader workers get OOM-killed mid-run. Most UR3 datasets are genuinely AV1
(`blue_basket` is the exception — it was re-encoded to h264 but its `info.json` still says
`av1`). Fix is §1b. Verify with:
```bash
python -c "from torchcodec.decoders import VideoDecoder; print('ok')"
```

### 5.2 NFS stalls from large checkpoint writes
Two runs hung **silently and indefinitely** mid-training — no error, no NCCL timeout, GPUs at
0% util with memory resident. The ranks were in uninterruptible-D state on wait-channel
`rpc_wait_bit_killable`, i.e. blocked on NFS I/O, never inside a collective (which is why no
watchdog fired). The run dir is on an NFS mount and each save wrote ~55 GB of ZeRO state.
Setting `save_optimizer_state: false` cut per-save writes by ~86% and no stall has recurred.

If a run goes quiet, check D-state before assuming a dataloader or NCCL problem:
```bash
for p in $(pgrep -f "train.py.*<task>"); do ps -o stat=,wchan:24= -p $p; done
```

### 5.3 Other jobs stealing GPUs
Five consecutive launches died because another job claimed the GPUs during our ~90-second model
load. On this cluster a `scripts._common.gpu_wait --need-gb 85` poller re-grabs any GPU that
frees, within ~60 s, and respawns when killed. **Verify the GPUs are still yours ~45 s after
launch**, not just before it:
```bash
sleep 45; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```
Ranks sitting at ~9 GB instead of climbing means you lost the race.

### 5.4 The eval metrics cannot rank checkpoints
`trainer.evaluate()` seeds its RNG with `global_step`, so **every eval scores a different random
clip set**. At the default `eval_num_samples: 1` per rank the numbers are pure noise — set
**`eval_num_samples: 8`** (32 clips). Even then, per-eval `sd ≈ 0.003–0.007` on an `action_l1`
of ~0.03 exceeds the gap between adjacent checkpoints. Use them to confirm training is healthy;
**select checkpoints by real-robot success rate.**

---

## 6. What four runs showed

| run | tasks | frames | steps | `action_l1` trend | shape |
|---|---|---|---|---|---|
| basket+mug | 2 | 53,918 | 13,800 | flat | plateaus ~2k |
| ethernet | 1 | 59,796 | 7,000 | −0.0027/1k, **t = −4.26** | still improving at 7k |
| drawer+basket | 2 | 62,635 | 7,000 | −0.0011/1k, t = −1.91 | plateaus ~2.5k |
| basket+drawer+stack | 3 | 99,677 | 10,000 | −0.00053/1k, t = −2.14 | plateaus ~2k |

Train `loss_action` always falls hard (up to 21×); held-out `action_l1` mostly does not. All
three **multi-task** runs plateau early while the one **single-task** run kept improving —
suggestive, but task count and task difficulty are confounded, so treat it as a hypothesis.

Practical consequence: **~7,000 steps is enough** for these dataset sizes. The 10k run's last
3,000 steps bought nothing measurable.

---

## 7. Publishing

Upload weights-only `.pt` files plus **`dataset_stats.json`** (inference is broken without it —
the stats are per-dataset, not the pretrain's) and `config.yaml` for provenance. Existing repos:
`armanakbari4/imagewam-ur3` (private), `-ethernet`, `-3task` (public), and the dataset at
`armanakbari4/ur3-3task-lerobot`.

State in the model card that the checkpoints are **statistically indistinguishable** on offline
metrics, so nobody downloads the nominal best assuming it is genuinely best.
