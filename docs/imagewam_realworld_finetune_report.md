# ImageWAM real-world fine-tuning — task settings, config & deployment notes

Real-world fine-tuning of **FLUX.2-Klein-4B ImageWAM** on a bimanual **dual-arm UR3**
(EmbodyX). We fine-tuned two single-task models and deployed via a client/server
pipeline. Sharing settings + the one deployment issue we're unsure about.

## 1. Task settings / datasets

Two single-task LeRobot datasets, same embodiment (dual-arm UR3, 3 cameras
[head + 2 wrists], 14-dim **joint-space** action & state `[L_arm(6) | L_grip | R_arm(6) | R_grip]`,
binary grippers 0=open/1=closed, 15 fps):

| dataset | task | episodes | frames | cam res |
|---|---|---|---|---|
| `EmbodyX/UR3 organize_pi05` | "sort the tools into their matching containers" | 78 | 118,211 | 480×640 |
| `EmbodyX/old_new_scan_lerobot` | "scan barcode" | 96 | 45,298 | 240×320 |

All 3 cameras are tiled into a **288×256 `compact_288x256` robotwin mosaic**
(head 256×192 on top; left+right wrist 128×96 below). Every episode starts from
the identical home pose.

## 2. Model / recipe

- Backbone: **FLUX.2 [klein] base 4B** editing DiT; base `ImageWAM` variant
  (editing-cache conditioning, single future **endpoint** frame). Text encoder:
  **Qwen3-4B** (frozen; embeddings encoded live at inference / precomputed for train).
- Action expert (ActionDiT): `hidden_dim=1024, num_heads=24, attn_head_dim=128,
  num_layers_double=5, num_layers_single=20, max_action_horizon=64`.
- **Full fine-tune** of the editing DiT + action expert + proprio_encoder; VAE +
  text encoder frozen. **LoRA off.** `mot_checkpoint_mixed_attn=true`.
- Loss: flow-matching, `lambda_video=0.5, lambda_action=1.0`; video & action
  schedulers `train_shift=infer_shift=5.0`, `num_train_timesteps=1000`.
- Data: `num_frames=17`, `action_video_freq_ratio=1` → **16-step action horizon**,
  `endpoint_frames_only=true`, **z-score** normalization, `ConcatLeftAlign` merger,
  no delta-action masking (absolute joints).

## 3. Training config (the `scan` run)

- **From base FLUX.2** (not warm-started from the other task's ckpt).
- lr `5e-5`, cosine (~5% warmup), AdamW(0.9,0.95), weight_decay `0.01`,
  grad_clip `1.0`, **bf16**, DeepSpeed **ZeRO-1**.
- batch_size **12/GPU × 5 GPUs × grad_accum 4 = 240 global**, **50 epochs = 9,300 steps**.
- `pretrained_norm_stats=null` first run → `dataset_stats.json` written to the run dir.
- Eval every 500 steps, `num_inference_steps=10` (action flow).
- Preproc handled automatically: action-DiT ini`t (interp from FLUX DiT, action_dim=14),
  Qwen3 text-embedding cache for the task prompt.

Prompt actually fed to the text encoder (train == inference):
`"A video recorded from a robot's point of view executing the following instruction: scan barcode"`

## 4. Results (curves attached separately)

- Train `loss` → ~0.11, **`loss_action` → ~0.003** (from ~2.0), `loss_video` ~0.10.
- Eval: `infer_ssim` ≈ **0.84**, **`action_l1` ≈ 0.03**, **`action_l2` ≈ 0.003–0.006**
  (denormalized joint units). Predicted "edited" future frames are visually coherent.

So on-distribution the models fit tightly (low action error, healthy image SSIM).

## 5. Deployment & the issue we'd like advice on

Deployment: a WebSocket + msgpack client/server. Server wraps the RoboTwin
`WorldActionRobotWinPolicy` obs→action pipeline (mosaic build → `infer_action`
→ denorm); client sends 3 raw-RGB cameras + measured 14-dim joint state + prompt,
receives a denormalized `[16,14]` chunk, executes it via RTDE `servoJ` at 15 Hz,
re-infers (receding horizon). Steady inference ≈ **0.23 s** / 16-step chunk.

**Symptom (organize model, real robot): weak performance — arms move slowly and
grippers never close, so grasps never happen.** We verified the input side is clean:
- deployment mosaic ≈ training mosaic (camera layout, RGB order, viewpoints all match);
- predicted edited frame is coherent;
- proprio == measured qpos, start pose == training start pose exactly, gripper
  convention (0/1) correct, actions in valid joint range.

**What we found:** the predicted 16-step action chunks are strongly **back-loaded** —
`action[0] ≈ current pose` and motion ramps up, with **~56% of a chunk's motion in
steps 8–15**. With `replan/execute = 8` (the RoboTwin `deploy_policy` default), we
only ever execute the slow first half, then re-infer a fresh (again slow-starting)
chunk → the arm systematically **under-travels (~44% of intended motion)**, never
reaches the object, so the gripper never triggers. Total travel ≈ 3.2 rad vs ≈ 5.6 rad
in the ground-truth demo over the same step count.

**Questions for you:**
1. Recommended **real-world receding-horizon setting** — execute the full 16-step
   chunk before re-inferring, a fraction, or temporal ensembling? (RoboTwin
   `deploy_policy` uses `replan_steps=8`.)
2. Recommended **action `num_inference_steps`** for deployment (we use 10) and any
   **action guidance** setting? Could too-few denoise steps bias actions toward the
   prior (small motion)?
3. Anything about the back-loaded chunk profile — expected, or a sign of a
   train/inference mismatch we should fix?
4. For a small single-task set (~80–100 demos), recommended epochs / LR / whether to
   warm-start from a related-task ImageWAM checkpoint vs from base.

## 6. Environment note (non-standard, in case relevant)

GPUs are NVIDIA **B300 (Blackwell, sm_103)**; the pinned `torch 2.7.1+cu118` has no
kernels for them, so we run `torch 2.8.0` (cu129 index) + matching torchvision/
torchcodec (+ FFmpeg 7 for torchcodec). Everything else per the repo.
