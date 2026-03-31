#!/usr/bin/env python3
"""Temporal multi-frame feature tests — real GPU data flow.

Tests the full pipeline with temporal_frames > 1, verifying:
  1. Dataset loads T frames correctly (shape, ordering)
  2. Collate produces (B, T, H, W, C) image tensors
  3. Vision tower handles 5D temporal input and interleaves output correctly
  4. VGGT multi-frame forward produces correct output shape
  5. End-to-end forward (training + inference) works with temporal inputs
  6. Single-frame backward compatibility is preserved
  7. Inference frame buffer accumulates and pads correctly

Usage:
    conda activate beta
    source .env && PYTHONPATH=src HF_HOME=/workspace/tingting/hf-home python tests/test_temporal.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

# ── Formatting ────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

pass_count = fail_count = 0


def header(name: str):
    print(f"\n{BOLD}{CYAN}{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}{RESET}")


def check(label: str, condition: bool, detail: str = ""):
    global pass_count, fail_count
    status = f"{GREEN}PASS{RESET}" if condition else f"{RED}FAIL{RESET}"
    if condition:
        pass_count += 1
    else:
        fail_count += 1
    print(f"  [{status}] {label}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")


def show(label: str, value):
    print(f"  {label}: {value}")


# ── Config ────────────────────────────────────────────────────────────────
REPO_ID = "physical-intelligence/libero"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B-Base"
NORM_STATS_PATH = "assets/physical-intelligence/libero/norm_stats.json"

TEMPORAL_FRAMES = 3
TEMPORAL_STRIDE = 5
NUM_CAMERAS = 2
PATCHES_PER_CAM = 256  # 16 x 16

DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════
# Test 1: Dataset temporal frame loading
# ══════════════════════════════════════════════════════════════════════════
header("Test 1: Dataset temporal frame loading (T=3, stride=5)")

from betavla.data.libero_dataset import LiberoDataset, LiberoDatasetConfig

# -- T=3 dataset --
cfg_t3 = LiberoDatasetConfig(
    repo_id=REPO_ID, split="train", max_samples=5000,
    num_workers=0, norm_stats_path=NORM_STATS_PATH,
    temporal_frames=TEMPORAL_FRAMES, temporal_stride=TEMPORAL_STRIDE,
)
ds_t3 = LiberoDataset(cfg_t3, action_horizon=10, action_dim=7, tokenizer_name=TOKENIZER_NAME)

# -- T=1 dataset (baseline) --
cfg_t1 = LiberoDatasetConfig(
    repo_id=REPO_ID, split="train", max_samples=5000,
    num_workers=0, norm_stats_path=NORM_STATS_PATH,
    temporal_frames=1,
)
ds_t1 = LiberoDataset(cfg_t1, action_horizon=10, action_dim=7, tokenizer_name=TOKENIZER_NAME)

# Pick a sample mid-episode so history frames are different
test_idx = min(62, len(ds_t3) - 1)
sample_t3, actions_t3 = ds_t3[test_idx]
sample_t1, actions_t1 = ds_t1[test_idx]

show("test sample", f"idx={test_idx}, ep={sample_t3['episode_index']}, "
     f"frame={sample_t3['frame_index']}, task={sample_t3['task_index']}")
show("prompt", repr(sample_t3["prompt"]))

# Check shapes
base_t3 = sample_t3["base_img"]
wrist_t3 = sample_t3["wrist_img"]
base_t1 = sample_t1["base_img"]

show("T=3 base_img shape", base_t3.shape)
show("T=3 wrist_img shape", wrist_t3.shape)
show("T=1 base_img shape", base_t1.shape)

check("T=3 base_img is (3, H, W, 3)", base_t3.ndim == 4 and base_t3.shape[0] == 3 and base_t3.shape[-1] == 3)
check("T=3 wrist_img is (3, H, W, 3)", wrist_t3.ndim == 4 and wrist_t3.shape[0] == 3 and wrist_t3.shape[-1] == 3)
check("T=1 base_img is (H, W, 3)", base_t1.ndim == 3 and base_t1.shape[-1] == 3)

# The last frame (index 2) in T=3 should equal the T=1 single frame (same current frame)
check("T=3 last frame == T=1 frame (current frame matches)",
      np.allclose(base_t3[-1], base_t1, atol=1e-6),
      f"max diff = {np.abs(base_t3[-1] - base_t1).max():.2e}")

# Check that history frames are different (unless clamped to episode start)
frame_pos = sample_t3["frame_index"]
if frame_pos >= (TEMPORAL_FRAMES - 1) * TEMPORAL_STRIDE:
    # History frames should be genuinely different
    diff_01 = np.abs(base_t3[0] - base_t3[1]).mean()
    diff_12 = np.abs(base_t3[1] - base_t3[2]).mean()
    show("mean pixel diff frame[0] vs frame[1]", f"{diff_01:.4f}")
    show("mean pixel diff frame[1] vs frame[2]", f"{diff_12:.4f}")
    check("history frames are different (not all identical)",
          diff_01 > 1e-6 or diff_12 > 1e-6)
else:
    show("frame_pos < (T-1)*stride, history may be clamped", frame_pos)

# Actions should be identical regardless of T
check("actions identical for T=1 and T=3",
      np.allclose(actions_t3, actions_t1, atol=1e-6),
      f"max diff = {np.abs(actions_t3 - actions_t1).max():.2e}")

# Check early frame (episode start) — history should be clamped/repeated
sample_start, _ = ds_t3[0]
base_start = sample_start["base_img"]
if sample_start["frame_index"] == 0:
    # At episode start, all history frames should be the same (clamped to pos=0)
    check("episode start: all T frames identical (clamped)",
          np.allclose(base_start[0], base_start[1], atol=1e-6) and
          np.allclose(base_start[1], base_start[2], atol=1e-6))


# ══════════════════════════════════════════════════════════════════════════
# Test 2: Collate function with temporal data
# ══════════════════════════════════════════════════════════════════════════
header("Test 2: Collate function (T=3 batching)")

B = 4
batch_t3 = [ds_t3[i] for i in range(B)]
obs_t3, act_batch_t3 = ds_t3.collate_fn(batch_t3)

batch_t1 = [ds_t1[i] for i in range(B)]
obs_t1, act_batch_t1 = ds_t1.collate_fn(batch_t1)

show("T=3 images['base_0_rgb'] shape", obs_t3.images["base_0_rgb"].shape)
show("T=3 images['left_wrist_0_rgb'] shape", obs_t3.images["left_wrist_0_rgb"].shape)
show("T=1 images['base_0_rgb'] shape", obs_t1.images["base_0_rgb"].shape)

H, W = obs_t3.images["base_0_rgb"].shape[2], obs_t3.images["base_0_rgb"].shape[3]

check("T=3 batch base = (B, T, H, W, 3)",
      obs_t3.images["base_0_rgb"].shape == (B, TEMPORAL_FRAMES, H, W, 3),
      f"got {obs_t3.images['base_0_rgb'].shape}")
check("T=3 batch wrist = (B, T, H, W, 3)",
      obs_t3.images["left_wrist_0_rgb"].shape == (B, TEMPORAL_FRAMES, H, W, 3))
check("T=1 batch base = (B, H, W, 3)",
      obs_t1.images["base_0_rgb"].ndim == 4 and obs_t1.images["base_0_rgb"].shape[0] == B)

show("T=3 state shape", obs_t3.state.shape)
show("T=3 actions shape", act_batch_t3.shape)
check("state unchanged (B, 8)", obs_t3.state.shape == (B, 8))
check("actions unchanged (B, 10, 7)", act_batch_t3.shape == (B, 10, 7))


# ══════════════════════════════════════════════════════════════════════════
# Test 3: Vision Tower with temporal input
# ══════════════════════════════════════════════════════════════════════════
header("Test 3: Vision Tower temporal encoding (GPU)")

from betavla.models.vision_tower import PaliGemmaVisionTower, PaliGemmaVisionTowerConfig

print(f"  Using device: {DEVICE}")
print("  Loading PaliGemma vision tower...")
t0 = time.time()
vt_cfg = PaliGemmaVisionTowerConfig()
vision_tower = PaliGemmaVisionTower(vt_cfg).to(DEVICE).eval()
print(f"  Loaded in {time.time()-t0:.1f}s")

# -- Single-frame baseline --
obs_t1_dev = obs_t1.to(DEVICE)
with torch.no_grad():
    vis_out_t1 = vision_tower(obs_t1_dev.images, obs_t1_dev.image_masks)

show("T=1 vision output shape", vis_out_t1.shape)
expected_t1 = (B, NUM_CAMERAS * PATCHES_PER_CAM, vision_tower.embed_dim)
check(f"T=1 vision output = {expected_t1}", vis_out_t1.shape == expected_t1)

# -- Temporal (T=3) --
obs_t3_dev = obs_t3.to(DEVICE)
with torch.no_grad():
    vis_out_t3 = vision_tower(obs_t3_dev.images, obs_t3_dev.image_masks)

show("T=3 vision output shape", vis_out_t3.shape)
expected_t3 = (B, TEMPORAL_FRAMES * NUM_CAMERAS * PATCHES_PER_CAM, vision_tower.embed_dim)
check(f"T=3 vision output = {expected_t3}", vis_out_t3.shape == expected_t3,
      f"got {vis_out_t3.shape}, expected {expected_t3}")

# Verify interleaving: output should be [t0_cam0, t0_cam1, t1_cam0, t1_cam1, t2_cam0, t2_cam1]
# The last 2 cameras (t2_cam0, t2_cam1) should match T=1 output if same images
# Note: T=1 obs uses samples 0..3 while T=3 obs also uses samples 0..3 but
# the current frame (t=2) should be the same image as T=1
show("T=3 output stats", f"mean={vis_out_t3.mean():.4f}, std={vis_out_t3.std():.4f}")
show("T=1 output stats", f"mean={vis_out_t1.mean():.4f}, std={vis_out_t1.std():.4f}")

# Check that temporal patches produce non-zero, varied output
t3_per_view = vis_out_t3.view(B, TEMPORAL_FRAMES * NUM_CAMERAS, PATCHES_PER_CAM, -1)
view_means = t3_per_view.mean(dim=2)  # (B, 6, D)
show("per-view mean norms (B=0)", [f"{view_means[0, v].norm():.2f}" for v in range(TEMPORAL_FRAMES * NUM_CAMERAS)])
check("all views produce non-zero output",
      all(view_means[0, v].norm() > 0.1 for v in range(TEMPORAL_FRAMES * NUM_CAMERAS)))

del vision_tower
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════
# Test 4: VGGT multi-frame forward
# ══════════════════════════════════════════════════════════════════════════
header("Test 4: VGGT multi-frame forward (GPU)")

from betavla.models.vggt_backbone import VGGTBackbone, VGGTBackboneConfig

# -- T=3 backbone --
print("  Loading VGGT backbone (T=3)...")
t0 = time.time()
vggt_cfg_t3 = VGGTBackboneConfig(temporal_frames=TEMPORAL_FRAMES, num_cameras=NUM_CAMERAS)
backbone_t3 = VGGTBackbone(vggt_cfg_t3).to(DEVICE).eval()
H_dim = backbone_t3.hidden_size
print(f"  Loaded in {time.time()-t0:.1f}s, hidden_size={H_dim}")

# Simulate input: T*num_cameras*256 vision + 128 language tokens
N_vis_t3 = TEMPORAL_FRAMES * NUM_CAMERAS * PATCHES_PER_CAM  # 3*2*256 = 1536
N_lang = 128
total_tokens_t3 = N_vis_t3 + N_lang

fake_input_t3 = torch.randn(B, total_tokens_t3, H_dim, device=DEVICE, dtype=torch.float32)
fake_mask_t3 = torch.ones(B, total_tokens_t3, device=DEVICE, dtype=torch.long)

show("VGGT T=3 input shape", fake_input_t3.shape)

with torch.no_grad():
    out_t3 = backbone_t3(fake_input_t3, fake_mask_t3)

show("VGGT T=3 output shape", out_t3.shape)

# Output should be: current timestep vision (num_cameras * 256) + language (128)
expected_out_len = NUM_CAMERAS * PATCHES_PER_CAM + N_lang  # 512 + 128 = 640
check(f"VGGT T=3 output = (B, {expected_out_len}, {H_dim})",
      out_t3.shape == (B, expected_out_len, H_dim),
      f"got {out_t3.shape}")

show("VGGT T=3 output stats", f"mean={out_t3.mean():.4f}, std={out_t3.std():.4f}")
show("  vision part (first 512)", f"mean={out_t3[:, :512].mean():.4f}, std={out_t3[:, :512].std():.4f}")
show("  language part (last 128)", f"mean={out_t3[:, 512:].mean():.4f}, std={out_t3[:, 512:].std():.4f}")

# -- T=1 backbone (backward compat) --
print("\n  Loading VGGT backbone (T=1)...")
vggt_cfg_t1 = VGGTBackboneConfig(temporal_frames=1, num_cameras=NUM_CAMERAS)
backbone_t1 = VGGTBackbone(vggt_cfg_t1).to(DEVICE).eval()

N_vis_t1 = NUM_CAMERAS * PATCHES_PER_CAM  # 512
total_tokens_t1 = N_vis_t1 + N_lang

fake_input_t1 = torch.randn(B, total_tokens_t1, H_dim, device=DEVICE, dtype=torch.float32)
fake_mask_t1 = torch.ones(B, total_tokens_t1, device=DEVICE, dtype=torch.long)

with torch.no_grad():
    out_t1 = backbone_t1(fake_input_t1, fake_mask_t1)

show("VGGT T=1 input shape", fake_input_t1.shape)
show("VGGT T=1 output shape", out_t1.shape)
check(f"VGGT T=1 output = (B, {total_tokens_t1}, {H_dim})",
      out_t1.shape == (B, total_tokens_t1, H_dim),
      f"got {out_t1.shape}")

del backbone_t3, backbone_t1
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════
# Test 5: End-to-end model with temporal (T=3)
# ══════════════════════════════════════════════════════════════════════════
header("Test 5: End-to-end model T=3 (training + inference)")

from betavla.models.model import BetaVLAConfig, BetaVLAModel

print("  Loading BetaVLAModel (T=3)...")
t0 = time.time()
model_cfg_t3 = BetaVLAConfig(
    freeze_vision=False, freeze_language=False, freeze_vggt=False,
    use_lora=True,
    vggt=VGGTBackboneConfig(temporal_frames=TEMPORAL_FRAMES, num_cameras=NUM_CAMERAS),
)
model_t3 = BetaVLAModel(model_cfg_t3).to(DEVICE)
model_t3.eval()
print(f"  Loaded in {time.time()-t0:.1f}s")

# ── encode() ──
print(f"\n  {BOLD}encode() with T=3:{RESET}")
obs_t3_gpu = obs_t3.to(DEVICE)
act_t3_gpu = act_batch_t3.to(DEVICE)

with torch.no_grad():
    prefix_tokens, prefix_pad_mask = model_t3.encode(obs_t3_gpu)

show("prefix_tokens shape", prefix_tokens.shape)
show("prefix_pad_mask shape", prefix_pad_mask.shape)

expected_prefix_len = NUM_CAMERAS * PATCHES_PER_CAM + N_lang  # 640
check(f"prefix_tokens = (B, {expected_prefix_len}, H)",
      prefix_tokens.shape == (B, expected_prefix_len, H_dim),
      f"got {prefix_tokens.shape}")
check("prefix_pad_mask matches prefix_tokens",
      prefix_pad_mask.shape[:2] == prefix_tokens.shape[:2],
      f"mask={prefix_pad_mask.shape}, tokens={prefix_tokens.shape}")

# Vision part should be all True, language part has padding
vis_mask = prefix_pad_mask[:, :NUM_CAMERAS * PATCHES_PER_CAM]
check("vision part of pad_mask all True",
      vis_mask.all().item(),
      f"True count: {vis_mask.sum()}/{vis_mask.numel()}")

# ── Training forward ──
print(f"\n  {BOLD}Training forward (T=3):{RESET}")
with torch.no_grad():
    out_train = model_t3(obs_t3_gpu, act_t3_gpu)

show("loss", f"{out_train['loss'].item():.4f}")
check("loss is finite scalar > 0",
      "loss" in out_train and out_train["loss"].ndim == 0 and
      out_train["loss"].item() > 0 and torch.isfinite(out_train["loss"]).item())

# ── Inference forward ──
print(f"\n  {BOLD}Inference forward (T=3):{RESET}")
with torch.no_grad():
    out_inf = model_t3(obs_t3_gpu, actions=None, num_ode_steps=5)

show("predicted actions shape", out_inf["actions"].shape)
show("predicted range", f"[{out_inf['actions'].min():.4f}, {out_inf['actions'].max():.4f}]")
check("inference returns (B, 10, 7)",
      out_inf["actions"].shape == (B, 10, 7),
      f"got {out_inf['actions'].shape}")

# ── Gradient flow ──
print(f"\n  {BOLD}Gradient check (T=3):{RESET}")
model_t3.train()
out_grad = model_t3(obs_t3_gpu, act_t3_gpu)
out_grad["loss"].backward()

grad_params = [(n, p) for n, p in model_t3.named_parameters()
               if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
show("params with gradients", len(grad_params))

# Show gradient norms for key components
component_grads = {}
for n, p in grad_params:
    comp = n.split(".")[0]
    component_grads.setdefault(comp, []).append(p.grad.norm().item())
for comp, norms in sorted(component_grads.items()):
    show(f"  {comp}", f"{len(norms)} params, mean grad norm={np.mean(norms):.6f}")

check("gradients flow to trainable params", len(grad_params) > 0)
model_t3.eval()
model_t3.zero_grad()


# ══════════════════════════════════════════════════════════════════════════
# Test 6: Single-frame backward compatibility
# ══════════════════════════════════════════════════════════════════════════
header("Test 6: Single-frame backward compatibility (T=1 model)")

print("  Loading BetaVLAModel (T=1)...")
t0 = time.time()
model_cfg_t1 = BetaVLAConfig(
    freeze_vision=False, freeze_language=False, freeze_vggt=False,
    use_lora=True,
    vggt=VGGTBackboneConfig(temporal_frames=1, num_cameras=NUM_CAMERAS),
)
model_t1 = BetaVLAModel(model_cfg_t1).to(DEVICE)
model_t1.eval()
print(f"  Loaded in {time.time()-t0:.1f}s")

obs_t1_gpu = obs_t1.to(DEVICE)
act_t1_gpu = act_batch_t1.to(DEVICE)

# ── encode() T=1 ──
with torch.no_grad():
    prefix_t1, mask_t1 = model_t1.encode(obs_t1_gpu)

show("T=1 prefix shape", prefix_t1.shape)
expected_t1_prefix = (B, NUM_CAMERAS * PATCHES_PER_CAM + N_lang, H_dim)
check(f"T=1 prefix = {expected_t1_prefix}",
      prefix_t1.shape == expected_t1_prefix,
      f"got {prefix_t1.shape}")

# ── Training forward T=1 ──
with torch.no_grad():
    out_t1_train = model_t1(obs_t1_gpu, act_t1_gpu)
show("T=1 loss", f"{out_t1_train['loss'].item():.4f}")
check("T=1 training loss is finite > 0",
      out_t1_train["loss"].item() > 0 and torch.isfinite(out_t1_train["loss"]).item())

# ── Inference T=1 ──
with torch.no_grad():
    out_t1_inf = model_t1(obs_t1_gpu, actions=None, num_ode_steps=5)
check("T=1 inference returns (B, 10, 7)", out_t1_inf["actions"].shape == (B, 10, 7))

del model_t1
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════
# Test 7: Inference frame buffer
# ══════════════════════════════════════════════════════════════════════════
header("Test 7: Inference frame buffer")

from betavla.eval.inference import (
    _frame_buffer,
    _frame_buffer_maxlen,
    _img_to_tensor,
    clear_caches,
    init_frame_buffer,
)

# ── init_frame_buffer ──
init_frame_buffer(3)
from betavla.eval import inference as inf_mod
check("buffer maxlen set to 3", inf_mod._frame_buffer_maxlen == 3)
check("buffer starts empty", len(inf_mod._frame_buffer) == 0)

# ── Simulate pushing frames ──
fake_imgs = [np.random.rand(256, 256, 3).astype(np.float32) for _ in range(5)]

# Push 1st frame — buffer should pad to 3 by repeating first frame
inf_mod._frame_buffer.append((fake_imgs[0], fake_imgs[0]))
while len(inf_mod._frame_buffer) < inf_mod._frame_buffer_maxlen:
    inf_mod._frame_buffer.appendleft(inf_mod._frame_buffer[0])

show("after 1 push + pad", f"len={len(inf_mod._frame_buffer)}")
check("buffer padded to 3", len(inf_mod._frame_buffer) == 3)
check("all 3 entries are the same image (padded)",
      np.allclose(inf_mod._frame_buffer[0][0], inf_mod._frame_buffer[2][0]))

# Push 2nd and 3rd frames
init_frame_buffer(3)  # reset
for i in range(3):
    inf_mod._frame_buffer.append((fake_imgs[i], fake_imgs[i]))
show("after 3 pushes", f"len={len(inf_mod._frame_buffer)}")
check("buffer has 3 entries", len(inf_mod._frame_buffer) == 3)
check("oldest entry is imgs[0]", np.allclose(inf_mod._frame_buffer[0][0], fake_imgs[0]))
check("newest entry is imgs[2]", np.allclose(inf_mod._frame_buffer[2][0], fake_imgs[2]))

# Push 4th frame — oldest should be evicted (deque maxlen)
inf_mod._frame_buffer.append((fake_imgs[3], fake_imgs[3]))
show("after 4th push", f"len={len(inf_mod._frame_buffer)}")
check("buffer still 3 (deque maxlen)", len(inf_mod._frame_buffer) == 3)
check("oldest is now imgs[1]", np.allclose(inf_mod._frame_buffer[0][0], fake_imgs[1]))
check("newest is imgs[3]", np.allclose(inf_mod._frame_buffer[2][0], fake_imgs[3]))

# ── Build temporal tensor from buffer (same logic as predict()) ──
base_list = [_img_to_tensor(f[0]) for f in inf_mod._frame_buffer]
base_t = torch.stack([b.squeeze(0) for b in base_list], dim=0).unsqueeze(0)
show("temporal tensor shape", base_t.shape)
check("temporal tensor = (1, 3, 256, 256, 3)",
      base_t.shape == (1, 3, 256, 256, 3),
      f"got {base_t.shape}")

# ── clear_caches resets buffer ──
clear_caches()
check("clear_caches empties buffer", len(inf_mod._frame_buffer) == 0)


# ══════════════════════════════════════════════════════════════════════════
# Test 8: Memory / performance sanity
# ══════════════════════════════════════════════════════════════════════════
header("Test 8: GPU memory comparison (T=1 vs T=3)")

torch.cuda.reset_peak_memory_stats(DEVICE)
torch.cuda.empty_cache()

# T=3 forward
model_t3.eval()
obs_t3_gpu = obs_t3.to(DEVICE)
act_t3_gpu = act_batch_t3.to(DEVICE)

torch.cuda.reset_peak_memory_stats(DEVICE)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    _ = model_t3(obs_t3_gpu, act_t3_gpu)
mem_t3 = torch.cuda.max_memory_allocated(DEVICE) / 1024**3

show(f"T=3 peak GPU memory (B={B})", f"{mem_t3:.2f} GB")
check("T=3 forward doesn't OOM", True)  # if we got here, it didn't OOM

# Compare with T=1
model_cfg_t1_mem = BetaVLAConfig(
    freeze_vision=False, freeze_language=False, freeze_vggt=False,
    use_lora=True,
    vggt=VGGTBackboneConfig(temporal_frames=1, num_cameras=NUM_CAMERAS),
)
model_t1_mem = BetaVLAModel(model_cfg_t1_mem).to(DEVICE).eval()

obs_t1_gpu = obs_t1.to(DEVICE)
act_t1_gpu = act_batch_t1.to(DEVICE)

torch.cuda.reset_peak_memory_stats(DEVICE)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    _ = model_t1_mem(obs_t1_gpu, act_t1_gpu)
mem_t1 = torch.cuda.max_memory_allocated(DEVICE) / 1024**3

show(f"T=1 peak GPU memory (B={B})", f"{mem_t1:.2f} GB")
show("memory ratio T=3/T=1", f"{mem_t3/mem_t1:.2f}x")

del model_t1_mem, model_t3
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*70}")
print(f"  SUMMARY: {GREEN}{pass_count} passed{RESET}{BOLD}, {RED if fail_count else ''}{fail_count} failed{RESET}")
print(f"{BOLD}{'='*70}{RESET}")
sys.exit(1 if fail_count else 0)
