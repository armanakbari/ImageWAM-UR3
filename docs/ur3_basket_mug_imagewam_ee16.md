# Real-World ImageWAM: UR3 (blue_basket + mug) fine-tuned from the InternData-A1 pretrain

Counterpart to `../../RealWorld_FastWAM.md`. Same robot, same two datasets, same
joint-training setup — but the world model is an **image-editing** DiT
(FLUX.2 [klein] base 4B) instead of a **video** DiT (Wan2.2-TI2V-5B), so the two
runs are directly comparable on the real robot.

---

## 1. Goal & summary

Fine-tune the upstream ImageWAM checkpoint **pretrained on InternData-A1**
(`yuyangalin/ImageWAM-FLUX.2-4B-InternData-A1-EE`, announced 2026-07-30) on our two
real UR3 tasks, following upstream's own fine-tune-from-pretrain recipe, to compare
against the Fast-WAM UR3 policy.

- **Model:** base `ImageWAM` variant (`create_imagewam_flux2_klein`), FLUX.2 klein-base-4B
  editing DiT + ActionDiT action expert, editing-cache conditioning on a single future
  **endpoint** frame.
- **Init from:** `yuyangalin/ImageWAM-FLUX.2-4B-InternData-A1-EE/model.pt` (step 60,000).
  Loads into this repo's config with **missing_keys=0, unexpected_keys=0**.
- **Data:** `data/ur3/{blue_basket_lerobot,mug_lerobot}` — the same two LeRobot datasets
  Fast-WAM was fine-tuned on, trained **jointly**.
- **Why it matters:** upstream reports the InternData-A1 pretrain gives **+40.9 SR** over
  ImageWAM-without-pretrain on RoboTwin Clean2Random (59.0% vs 18.1%), and **+11.5** over
  π-0.5. Our earlier real-world ImageWAM runs (`imagewam_realworld_finetune_report.md`)
  started from **base FLUX.2**, i.e. without that pretrain.

---

## 2. Data

| Task | Episodes | Frames | fps | Instruction |
|---|---|---|---|---|
| `blue_basket_lerobot` | 100 | 29,653 | 15 | "put the medicine then the measuring tape inside the blue basket" |
| `mug_lerobot` | 100 | 24,265 | 15 | "hang the mug on the mug holder" |

- **Cameras (3):** `camera_top`, `camera_wrist_left`, `camera_wrist_right`, each **240×320** RGB.
- **Action & proprio:** 14-dim absolute joint space, `[L_arm(6) | L_grip | R_arm(6) | R_grip]`.
- LeRobot v2.1. Symlinked into `data/ur3/` from the FastWAM checkout so both codebases
  read the same bytes.
- **Stale codec metadata:** `meta/info.json` declares `video.codec: av1`, but the files are
  actually **h264** — they were re-encoded (`RealWorld_FastWAM.md` §10) without updating the
  metadata. `ffprobe` and torchcodec both report h264 on every episode checked. Decoding is
  driven by the container, not `info.json`, so this is harmless in practice — but don't
  chase an AV1 decoder when debugging.

### 14D → 16D (`ee16`)

The released pretrain is in a **16D** action/state space. Following upstream's RoboTwin
`ee16` handling, UR3 stays semantically 14D and is only **padded** — dims 7 and 15 are
zero-filled and flagged in `action_dim_is_pad` / `state_dim_is_pad`. Nothing is converted
into end-effector poses. UR3's 14D layout is identical to RoboTwin's (two 7D arm+gripper
groups), so `transforms.canonical.RobotWin14ToCanonicalEE16` applies unchanged.

---

## 3. Camera pipeline (what the model actually sees)

Three cameras tiled into one **288×256** `compact_288x256` mosaic
(`concat_multi_camera: "robotwin"`):

| Camera | Resized to (H×W) | Position |
|---|---|---|
| `camera_top` | 192×256 | full top |
| `camera_wrist_left` | 96×128 | bottom-left |
| `camera_wrist_right` | 96×128 | bottom-right |

Order is fixed `[top, left, right]`; pixels normalized to (−1, 1).
**Note this differs from Fast-WAM's 384×320 `legacy` mosaic** — replicate the
288×256 layout at ImageWAM inference time.

---

## 4. Config

Two configs were added.

**`configs/data/ur3_basket_mug_ee16.yaml`**
```yaml
dataset_dirs: [${oc.env:UR3_ROOT,./data/ur3}/blue_basket_lerobot,
               ${oc.env:UR3_ROOT,./data/ur3}/mug_lerobot]
images: [camera_top, camera_wrist_left, camera_wrist_right]   # raw [3,240,320]
action/state: raw 14 -> 16 via RobotWin14ToCanonicalEE16
num_frames: 17
action_video_freq_ratio: 1        # 16-step action horizon
video_size: [288, 256]
robotwin_camera_layout: compact_288x256
endpoint_frames_only: true
norm: z-score, ConcatLeftAlign, no delta-action masking (absolute joints)
pretrained_norm_stats: null       # first run writes dataset_stats.json to the run dir
```

**`configs/task/ur3_basket_mug_flux2_klein_4b_base_imagewam_ee16.yaml`** — upstream's
fine-tune-from-pretrain hyperparameters (`robotwin_flux2_klein_4b_base_clean_imagewam_ee16`):

| Param | Value | Note |
|---|---|---|
| lr | **2.5e-5** | upstream's warm-start LR (vs 5e-5 from base FLUX.2) |
| schedule | cosine, 5% warmup | AdamW(0.9, 0.95), wd 1e-2, grad-clip 1.0, bf16 |
| batch_size | 12/GPU | grad_accum 4 |
| num_epochs | 50 | |
| save_every | **1000** | tightened from upstream's 2500 — see §7 |
| eval_every | 500 | |
| `qwen_context_len` | 128 | must match the pretrain |
| `proprio_dim` / `action_dim` | 16 | ee16 |
| ZeRO | stage 1 | DeepSpeed + HF accelerate |

`resume=<file.pt>` loads **weights only** (fresh optimizer/scheduler/step).

---

## 5. Prerequisites

**a. Pretrain checkpoint** (9.0 GB):
```bash
huggingface-cli download yuyangalin/ImageWAM-FLUX.2-4B-InternData-A1-EE \
  --repo-type model \
  --local-dir checkpoints/imagewam_release/interndata_pretrained/flux2_klein_4b
```
Its `config.yaml` matches this repo's `configs/model/imagewam_flux2_klein_4b_base.yaml`
exactly once `proprio_dim=16`, `action_dim=16`, `qwen_context_len=128` are set — which the
task config does.

**b. Base FLUX.2 weights**: `flux-2-klein-base-4b.safetensors` + FLUX.2 `ae.safetensors`
(`FLUX2_MODEL_PATH`, `FLUX2_AE_MODEL_PATH`), plus the `third_party/flux2` source
(`FLUX2_SRC`).

**c. Qwen3 text-embedding cache.** `load_text_encoder: false`, so the two task prompts must
be pre-embedded (`PRECOMPUTE_QWEN3_CACHE=true` does this automatically on first launch) →
`data/ur3/flux2_qwen3_cache_4b/`.

**d. Norm stats.** First launch computes `dataset_stats.json` into the run dir.

---

## 6. Environment (critical gotcha)

Same class of bug as `RealWorld_FastWAM.md` §6, now **fixed** rather than avoided.

`ImageWAM/.venv` ships `torchcodec 0.5+cu118`, whose CUDA build needs the CUDA 11 NPP
libraries. Those are **not** a torch dependency, so `import torchcodec` failed with
`libnppicc.so.11: cannot open shared object file` and LeRobot silently fell back to a pyav
decoder that leaks ~GBs per batch and OOM-kills the dataloader workers.

Fix:
```bash
uv pip install --python .venv/bin/python nvidia-npp-cu11
```
and `scripts/common.sh:imagewam_setup_runtime_libs` (called from `imagewam_init`, so every
wrapper gets it) puts `…/site-packages/nvidia/npp/lib` on `LD_LIBRARY_PATH` and the repo
venv on `PATH`. torchcodec now decodes the UR3 h264 videos directly, with flat memory.

---

## 7. Launch

```bash
cd /home/user1/workspace/arman/FastWAM/ImageWAM
CUDA_VISIBLE_DEVICES=4,5,6,7 GPU_PER_NODE=4 PRECOMPUTE_QWEN3_CACHE=true \
  bash scripts/flux2/run_finetune_flux2_klein_ur3_ee16.sh num_workers=8 wandb.enabled=false
```

- Effective batch = 12 × 4 GPUs × 4 accum = **192**.
- Dataset → **276 steps/epoch**, 50 epochs = **13,800 steps** (690 warmup), ≈14 h on 4×H100.
- `num_workers` is **per rank**. The config sets 12 (upstream's 24 assumes an 8-GPU box);
  this run overrode it to 8 because another job holds ~525 GB of the box's 755 GB.
- `wandb.enabled=false`: the upstream default project `imagewam` is not writable by this
  account (403). Set `wandb.project`/entity to your own to re-enable.

**Overfitting note.** 200 real episodes is small; as with Fast-WAM the last checkpoint is
not necessarily the best. `save_every=1000` gives 14 checkpoints — select by real-robot
success rate, not by final loss. See §8 for what the offline metrics did and did not show.

**Read `infer_psnr`/`infer_ssim`/`action_l1` as noise unless `eval_num_samples` is large.**
`trainer.evaluate()` seeds its RNG with `global_step + process_index`, so **every eval draws
a different random set of val clips**. At the default `eval_num_samples: 1` per rank (4 clips
on 4 GPUs) consecutive evals are not comparable — the 2026-08-12 run swung ssim 0.925 → 0.843
between steps 500 and 1000 while `val_loss` *fell* 0.279 → 0.114. The task config now sets
`eval_num_samples: 8` (32 clips). Trust `loss`/`loss_action`/`val_loss` for trend.

**Disk.** Weights are 9.0 GB each (13 × 9 GB ≈ 117 GB) plus a ~55 GB ZeRO state dir
(`keep_latest_state_only: true`, so it does not accumulate). Budget ~175 GB per run.

---

## 8. Result (run `2026-08-12_15-20-35`)

Ran the full **13,800 steps / 50 epochs** cleanly in **14h34m** on 4×H100, 0.27 step/s.
**14 checkpoints** (step 1000 … 13000, plus 13800), 9.04 GB each; 177 GB for the run.

| | step 500 | step 2000 | step 5000 | step 8000 | step 13800 |
|---|---|---|---|---|---|
| train `loss_action` | 0.0834 | 0.0218 | 0.0136 | 0.0116 | ~0.003 |
| held-out `action_l1` | 0.0267 | 0.0224 | 0.0160 | 0.0148 | ~0.022 |
| `infer_ssim` | 0.9254 | 0.8854 | 0.9029 | 0.8686 | ~0.87 |

**How to read this:** train `loss_action` fell ~28× (0.083 → 0.003), i.e. the model fits the
training trajectories very tightly. But **held-out `action_l1` is flat across the whole run**
— by thirds, 0.0252 → 0.0283 → 0.0259 — and `infer_ssim` is flat noise (mean 0.855, sd 0.037).
`val_loss` swings between 0.09 and 1.22 and its by-thirds trend (0.261 → 0.549 → 0.306) is
noise, not overfitting: at 4 clips per eval plus a random diffusion timestep, single evals
cannot be compared.

So: **held-out action error plateaus by ~step 2000 and neither improves nor collapses after.**
Nothing in the offline metrics separates the later checkpoints. Lowest `action_l1` evals were
steps 8000 (0.0148), 5000 (0.0160), 12000 (0.0162), 9500 (0.0174), 4000 (0.0184) — but that
ranking is within noise, so do not treat it as a selection.

**Recommended real-robot sweep:** steps **2000, 4000, 8000, 13800** — one from the plateau
knee, one early-plateau, one at the best-observed `action_l1`, one final. That mirrors how the
Fast-WAM checkpoint was chosen (2500/3500/4000) and is the only measurement that discriminates.

## 9. Deployment

Same client/server shape as the earlier ImageWAM real-world deployment (WebSocket +
msgpack, RoboTwin `WorldActionRobotWinPolicy` obs→action pipeline). Two things must change
relative to that deployment:

1. **Mosaic** is `compact_288x256` (already the case) — but note Fast-WAM's is 384×320, so
   the two policies are **not** drop-in swappable on the same client.
2. **Actions come back 16D** and must be un-padded to 14D (drop dims 7 and 15) before
   `servoJ`. `EVALUATION.apply_action_transform_backward` (added upstream with ee16) drives
   this; make sure the real-robot server does the same `_from_canonical` step.

**Carry-over issue to re-check:** the earlier organize_pi05 ImageWAM deployment produced
strongly **back-loaded** 16-step chunks (~56% of the motion in steps 8–15), so executing
only the first 8 under-travelled by ~44% and the gripper never triggered. Before blaming
the policy, log the per-chunk travel and either execute more of the chunk or lower
`replan_steps` accordingly.

---

## 10. File map

```
configs/data/ur3_basket_mug_ee16.yaml                              # data (both tasks, ee16 pad)
configs/task/ur3_basket_mug_flux2_klein_4b_base_imagewam_ee16.yaml # task (upstream FT recipe)
scripts/flux2/run_finetune_flux2_klein_ur3_ee16.sh                 # launch wrapper
scripts/common.sh                                                  # imagewam_setup_runtime_libs
data/ur3/{blue_basket_lerobot,mug_lerobot}                         # symlinks to the FastWAM copies
data/ur3/flux2_qwen3_cache_4b/                                     # precomputed prompt embeddings
checkpoints/imagewam_release/interndata_pretrained/flux2_klein_4b/ # InternData-A1 pretrain
runs/ur3_basket_mug_flux2_klein_4b_base_imagewam_ee16/<ts>/        # weights/ + state/ + dataset_stats.json
logs/ur3_ee16_train.log                                            # training log
```
