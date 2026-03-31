#!/usr/bin/env python3
"""Beta-VLA full pipeline diagnostic — prints key values at each layer.

Usage:
    conda activate beta
    source .env && PYTHONPATH=src HF_HOME=/workspace/tingting/hf-home python tests/mytest.py
"""
from __future__ import annotations

import glob
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────
REPO_ID = "physical-intelligence/libero"
TOKENIZER_NAME = "Qwen/Qwen3-0.6B-Base"
NORM_STATS_PATH = "assets/physical-intelligence/libero/norm_stats.json"
SNAP_GLOB = "/workspace/tingting/hf-home/hub/datasets--physical-intelligence--libero/snapshots/*/data/chunk-*/episode_*.parquet"

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


# # ══════════════════════════════════════════════════════════════════════════
# # Layer 1: Task Descriptions
# # ══════════════════════════════════════════════════════════════════════════
# header("Layer 1: Task Descriptions (task_index -> prompt)")

from betavla.data.libero_dataset import _load_task_descriptions

tasks = _load_task_descriptions(REPO_ID)
# check("Loaded 40 tasks", len(tasks) == 40, f"got {len(tasks)}")

# show("Sample tasks", "")
# for suite, start in [("libero_10", 0), ("libero_goal", 10), ("libero_object", 20), ("libero_spatial", 30)]:
#     print(f"    {BOLD}{suite}{RESET}:")
#     for i in range(start, start + 3):
#         print(f"      [{i:2d}] {tasks.get(i, 'MISSING')}")
#     print(f"      ...")

# goal_unique = len(set(tasks[i] for i in range(10, 20)))
# check("Goal tasks (10-19) all unique", goal_unique == 10, f"{goal_unique}/10 unique")

# # ── Parquet-level verification ──
# files = sorted(glob.glob(SNAP_GLOB))
# if files:
#     schema = pq.read_schema(files[0])
#     text_cols = {"prompt", "task", "instruction", "language_instruction"} & set(schema.names)
#     check("Parquet has NO text column (bug we fixed)", len(text_cols) == 0,
#           f"columns: {schema.names}")
#     check("Parquet has task_index column", "task_index" in schema.names)


# ══════════════════════════════════════════════════════════════════════════
# Layer 2: Dataset.__getitem__
# ══════════════════════════════════════════════════════════════════════════
header("Layer 2: Dataset.__getitem__")

try:
    from betavla.data.libero_dataset import LiberoDataset, LiberoDatasetConfig

    cfg = LiberoDatasetConfig(
        repo_id=REPO_ID, split="train", max_samples=5000,
        num_workers=0, norm_stats_path=NORM_STATS_PATH,
    )
    ds = LiberoDataset(cfg, action_horizon=10, action_dim=7, tokenizer_name=TOKENIZER_NAME)

    sample, actions = ds[0]
    print("task_index")
    print(sample["task_index"])
    print("episode_index")
    print(sample["episode_index"])
    print("frame_index")
    print(sample["frame_index"])



    # ── All keys in sample ──
    # print(f"\n  {BOLD}sample keys:{RESET}")
    # for key, data in sample.items():
    #     if isinstance(data, np.ndarray):
    #         print(f"    {key:>15s}: ndarray shape={data.shape}, dtype={data.dtype}, range=[{data.min():.3f}, {data.max():.3f}]")
    #     elif isinstance(data, str):
    #         print(f"    {key:>15s}: str = {data!r}")
    #     else:
    #         print(f"    {key:>15s}: {type(data).__name__} = {data}")
    # print()


    # # ── Prompt ──
    show("prompt", repr(sample["prompt"]))

    # # ── Images ──
    # img = sample["base_img"]
    # show("base_img shape", f"{img.shape}, dtype={img.dtype}")
    # show("base_img range", f"[{img.min():.3f}, {img.max():.3f}]")
    # show("base_img sample pixel [10,20]", img[10, 20, :])

    # wrist = sample["wrist_img"]
    # show("wrist_img shape", f"{wrist.shape}, dtype={wrist.dtype}")
    # show("wrist_img range", f"[{wrist.min():.3f}, {wrist.max():.3f}]")

    # # ── State (8D) ──
    # st = sample["state"]
    # show("state shape", f"{st.shape}, dtype={st.dtype}")
    # state_labels = ["eef_x", "eef_y", "eef_z", "rot_ax", "rot_ay", "rot_az", "grip_L", "grip_R"]
    # for i, (label, val) in enumerate(zip(state_labels, st)):
    #     print(f"    state[{i}] {label:>6s} = {val:+.4f}")

    # ── Normalization details ──
    # from betavla.data.normalize import load_norm_stats, normalize_quantile, unnormalize_quantile
    # norm = load_norm_stats(NORM_STATS_PATH)

    # print(f"\n  {BOLD}Normalization formula:{RESET}")
    # print(f"    normalized = (raw - q01) / (q99 - q01) * 2.0 - 1.0")
    # print(f"    raw        = (normalized + 1.0) / 2.0 * (q99 - q01) + q01\n")

    # # ── State: raw vs normalized ──
    # st = sample["state"]  # already normalized
    # st_raw = unnormalize_quantile(st, norm["state"])
    # show("state shape", f"{st.shape}")

    # state_labels = ["eef_x", "eef_y", "eef_z", "rot_ax", "rot_ay", "rot_az", "grip_L", "grip_R"]
    # print(f"    {'dim':>6s}  {'q01':>10s} {'q99':>10s} {'raw':>10s} {'normalized':>10s}")
    # print(f"    {'------':>6s}  {'----------':>10s} {'----------':>10s} {'----------':>10s} {'----------':>10s}")
    # for i, label in enumerate(state_labels):
    #     q01 = norm["state"].q01[i]
    #     q99 = norm["state"].q99[i]
    #     print(f"    {label:>6s}  {q01:+10.4f} {q99:+10.4f} {st_raw[i]:+10.4f} {st[i]:+10.4f}")

    # # ── Actions: raw vs normalized (10 steps × 7D) ──
    # act_normed = actions  # already normalized
    # act_raw = unnormalize_quantile(act_normed, norm["action"])

    # act_labels = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]

    # print(f"\n  {BOLD}Action norm stats (q01 / q99):{RESET}")
    # print(f"    {'dim':>4s}  {'q01':>10s} {'q99':>10s}")
    # print(f"    {'----':>4s}  {'----------':>10s} {'----------':>10s}")
    # for i, label in enumerate(act_labels):
    #     print(f"    {label:>4s}  {norm['action'].q01[i]:+10.6f} {norm['action'].q99[i]:+10.6f}")

    # print(f"\n  {BOLD}Action chunk — RAW (before normalization):{RESET}")
    # print(f"    {'step':>4s}  {'dx':>9s} {'dy':>9s} {'dz':>9s} {'drx':>9s} {'dry':>9s} {'drz':>9s} {'grip':>5s}")
    # print(f"    {'----':>4s}  {'---------':>9s} {'---------':>9s} {'---------':>9s} {'---------':>9s} {'---------':>9s} {'---------':>9s} {'-----':>5s}")
    # for t in range(act_raw.shape[0]):
    #     vals = " ".join(f"{act_raw[t, d]:+9.5f}" for d in range(6))
    #     print(f"    [{t:2d}]   {vals} {act_raw[t, 6]:+5.1f}")

    # print(f"\n  {BOLD}Action chunk — NORMALIZED (after normalization, what model sees):{RESET}")
    # print(f"    {'step':>4s}  {'dx':>9s} {'dy':>9s} {'dz':>9s} {'drx':>9s} {'dry':>9s} {'drz':>9s} {'grip':>9s}")
    # print(f"    {'----':>4s}  {'---------':>9s} {'---------':>9s} {'---------':>9s} {'---------':>9s} {'---------':>9s} {'---------':>9s} {'---------':>9s}")
    # for t in range(act_normed.shape[0]):
    #     vals = " ".join(f"{act_normed[t, d]:+9.5f}" for d in range(7))
    #     print(f"    [{t:2d}]   {vals}")

    # # ── Verify round-trip ──
    # recovered = unnormalize_quantile(normalize_quantile(act_raw, norm["action"]), norm["action"])
    # max_err = np.abs(recovered - act_raw).max()
    # check("normalize round-trip error < 1e-5", max_err < 1e-5, f"max error = {max_err:.2e}")

    # # ── Checks ──
    # check("prompt is non-empty", len(sample["prompt"]) > 0)
    # check("base_img = (256,256,3) float32", img.shape == (256, 256, 3) and img.dtype == np.float32)
    # check("base_img in [0,1]", 0.0 <= img.min() and img.max() <= 1.0)
    # check("state = (8,) float32", st.shape == (8,) and st.dtype == np.float32)
    # check("actions = (10,7) float32", actions.shape == (10, 7) and actions.dtype == np.float32)

    # # ── Multiple prompts ──
    # prompts = {ds[i][0]["prompt"] for i in range(min(len(ds), 500))}
    # show("unique prompts in first 500 samples", len(prompts))
    # check("Multiple unique prompts exist", len(prompts) > 1)
except Exception as e:
    print(f"  {RED}SKIPPED: {e}{RESET}")


# ══════════════════════════════════════════════════════════════════════════
# Layer 3: collate_fn
# ══════════════════════════════════════════════════════════════════════════
# header("Layer 3: collate_fn (batching + tokenization)")

# try:
#     B = min(len(ds), 4)
#     batch = [ds[i] for i in range(B)]
#     obs, act_batch = ds.collate_fn(batch)

#     # ── Batch-level shapes ──
#     show("batch_size", B)
#     show("images['base_0_rgb']", f"{obs.images['base_0_rgb'].shape}, dtype={obs.images['base_0_rgb'].dtype}")
#     show("images['left_wrist_0_rgb']", f"{obs.images['left_wrist_0_rgb'].shape}")
#     show("state", f"{obs.state.shape}, dtype={obs.state.dtype}")
#     show("tokenized_prompt", f"{obs.tokenized_prompt.shape}, dtype={obs.tokenized_prompt.dtype}")
#     show("tokenized_prompt_mask", f"{obs.tokenized_prompt_mask.shape}, dtype={obs.tokenized_prompt_mask.dtype}")
#     show("actions", f"{act_batch.shape}, dtype={act_batch.dtype}")

#     # ── Per-sample details ──
#     real_tokens = obs.tokenized_prompt_mask.sum(dim=1)
#     tokenizer_tmp = ds.tokenizer

#     for b in range(B):
#         raw_sample = batch[b][0]
#         print(f"\n  {BOLD}── Sample {b} ──{RESET}")
#         print(f"    episode={raw_sample.get('episode_index','?')}, frame={raw_sample.get('frame_index','?')}, task={raw_sample.get('task_index','?')}")
#         print(f"    prompt: {raw_sample['prompt']!r}")

#         # Token IDs (first 20 real tokens)
#         n_real = int(real_tokens[b].item())
#         ids = obs.tokenized_prompt[b, :n_real].tolist()
#         decoded = tokenizer_tmp.decode(ids)
#         print(f"    tokens: {n_real} real / {obs.tokenized_prompt.shape[1]} total")
#         print(f"    token_ids[:20]: {ids[:20]}")
#         print(f"    decoded: {decoded!r}")

#         # Image stats
#         base_img = obs.images["base_0_rgb"][b]
#         wrist_img = obs.images["left_wrist_0_rgb"][b]
#         print(f"    base_img:  range=[{base_img.min():.3f}, {base_img.max():.3f}], mean={base_img.mean():.3f}")
#         print(f"    wrist_img: range=[{wrist_img.min():.3f}, {wrist_img.max():.3f}], mean={wrist_img.mean():.3f}")

#         # State: normalized vs raw
#         from betavla.data.normalize import load_norm_stats, unnormalize_quantile
#         norm = load_norm_stats(NORM_STATS_PATH)

#         st = obs.state[b].numpy()
#         st_raw = unnormalize_quantile(st, norm["state"])
#         state_labels = ["eef_x", "eef_y", "eef_z", "rot_ax", "rot_ay", "rot_az", "grip_L", "grip_R"]
#         print(f"    state (normalized → raw):")
#         for i, label in enumerate(state_labels):
#             print(f"      {label:>6s}: {st[i]:+8.4f} → {st_raw[i]:+10.6f}")

#         # Actions: normalized vs raw (first 3 steps + last step)
#         act = act_batch[b].numpy()
#         act_raw = unnormalize_quantile(act, norm["action"])
#         act_labels = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
#         print(f"    actions ({act.shape[0]} steps, normalized → raw):")
#         print(f"      {'step':>4s}  {'':>3s} {'dx':>9s} {'dy':>9s} {'dz':>9s} {'drx':>9s} {'dry':>9s} {'drz':>9s} {'grip':>6s}")
#         steps_to_show = list(range(min(3, act.shape[0]))) + [act.shape[0] - 1]
#         for t in dict.fromkeys(steps_to_show):
#             norm_vals = " ".join(f"{act[t, d]:+9.4f}" for d in range(6))
#             raw_vals = " ".join(f"{act_raw[t, d]:+9.5f}" for d in range(6))
#             print(f"      [{t:2d}] norm {norm_vals} {act[t,6]:+6.2f}")
#             print(f"           raw  {raw_vals} {act_raw[t,6]:+6.1f}")
#         if act.shape[0] > 4:
#             print(f"      ... ({act.shape[0] - 4} steps omitted)")

#     # ── Checks ──
#     print()
#     check("batch images = (B, 256, 256, 3)", obs.images["base_0_rgb"].shape == (B, 256, 256, 3))
#     check("token IDs = (B, 128) long", obs.tokenized_prompt.shape == (B, 128) and obs.tokenized_prompt.dtype == torch.long)
#     check("mask has real tokens (>3)", (real_tokens > 3).all().item())
#     check("actions = (B, 10, 7)", act_batch.shape == (B, 10, 7))
# except Exception as e:
#     print(f"  {RED}SKIPPED: {e}{RESET}")


# ══════════════════════════════════════════════════════════════════════════
# Layer 4: Vision Tower
# ══════════════════════════════════════════════════════════════════════════
# header("Layer 4: Vision Tower (PaliGemma)")

from betavla.models.model import BetaVLAConfig, BetaVLAModel

DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {DEVICE}")

print("  Loading BetaVLAModel (this downloads PaliGemma, Qwen, VGGT)...")
t0 = time.time()
model_cfg = BetaVLAConfig(freeze_vision=False, freeze_language=False, freeze_vggt=False, use_lora=True)
model = BetaVLAModel(model_cfg).to(DEVICE)
model.eval()
print(f"  Model loaded in {time.time()-t0:.1f}s")

show("vision_tower.embed_dim", model.vision_tower.embed_dim)
show("language_encoder.hidden_size", model.language_encoder.hidden_size)
show("vggt_backbone.hidden_size", model.vggt_backbone.hidden_size)

# # Use real images from dataset if available, otherwise random
# try:
#     real_sample = ds[0][0]
#     base_img = torch.from_numpy(real_sample["base_img"]).unsqueeze(0)   # (1, 256, 256, 3)
#     wrist_img = torch.from_numpy(real_sample["wrist_img"]).unsqueeze(0)
#     show("using real images", f"base={base_img.shape}, wrist={wrist_img.shape}")
#     print(f"  (note: 256x256 input will be resized to 224x224 inside vision tower)")
# except Exception:
#     base_img = torch.rand(1, 256, 256, 3)
#     wrist_img = torch.rand(1, 256, 256, 3)
#     show("using random images (dataset not loaded)", f"{base_img.shape}")

# # Single camera
# img1 = {"base_0_rgb": base_img}
# mask1 = {"base_0_rgb": torch.ones(1, dtype=torch.bool)}
# with torch.no_grad():
#     vis1 = model.vision_tower(img1, mask1)
# print("vis1 last patch")
# print(vis1[0, 255, :])

# show("1 camera output", f"{vis1.shape}, range=[{vis1.min():.3f}, {vis1.max():.3f}], mean={vis1.mean():.3f}")
# check("1 camera = (1, 256, D)", vis1.shape[0] == 1 and vis1.shape[1] == 256)

# # Two cameras
# img2 = {"base_0_rgb": base_img, "left_wrist_0_rgb": wrist_img}
# mask2 = {"base_0_rgb": torch.ones(1, dtype=torch.bool), "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool)}
# with torch.no_grad():
#     vis2 = model.vision_tower(img2, mask2)
# print("vis2")
# print(vis2)
# show("2 cameras output", f"{vis2.shape}")
# show("  cam1 (base) patches", f"mean={vis2[0, :256].mean():.3f}, std={vis2[0, :256].std():.3f}")
# show("  cam2 (wrist) patches", f"mean={vis2[0, 256:].mean():.3f}, std={vis2[0, 256:].std():.3f}")
# check("2 cameras = (1, 512, D)", vis2.shape[0] == 1 and vis2.shape[1] == 512)


# # ══════════════════════════════════════════════════════════════════════════
# # Layer 5: Language Encoder
# # ══════════════════════════════════════════════════════════════════════════
# header("Layer 5: Language Encoder (Qwen)")

# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # Use real prompts from dataset samples — find two different ones
# prompt_a = ds[0][0]["prompt"]
# task_a = ds[0][0]["task_index"]
# prompt_b, task_b = prompt_a, task_a
# for i in range(1, len(ds)):
#     s = ds[i][0]
#     if s["prompt"] != prompt_a:
#         prompt_b = s["prompt"]
#         task_b = s["task_index"]
#         break
# show("prompt A (from dataset)", f"task_index={task_a}, {prompt_a!r}")
# show("prompt B (from dataset)", f"task_index={task_b}, {prompt_b!r}")

# enc_a = tokenizer([prompt_a], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
# enc_b = tokenizer([prompt_b], truncation=True, padding="max_length", max_length=128, return_tensors="pt")

# show("tokens A", f"{enc_a['attention_mask'].sum().item()} real / 128 total")
# show("tokens B", f"{enc_b['attention_mask'].sum().item()} real / 128 total")
# show("token_ids A[:15]", enc_a["input_ids"][0, :15].tolist())
# show("token_ids B[:15]", enc_b["input_ids"][0, :15].tolist())

# with torch.no_grad():
#     emb_a = model.language_encoder(enc_a["input_ids"], enc_a["attention_mask"].bool())
#     emb_b = model.language_encoder(enc_b["input_ids"], enc_b["attention_mask"].bool())

# show("embedding shape", emb_a.shape)
# show("emb A stats", f"mean={emb_a.mean():.4f}, std={emb_a.std():.4f}")
# show("emb B stats", f"mean={emb_b.mean():.4f}, std={emb_b.std():.4f}")

# # Mean-pool real tokens, compare
# n_a = int(enc_a["attention_mask"].sum().item())
# n_b = int(enc_b["attention_mask"].sum().item())
# pool_a = emb_a[0, :n_a].mean(0)
# pool_b = emb_b[0, :n_b].mean(0)
# cos_sim = torch.nn.functional.cosine_similarity(pool_a.unsqueeze(0), pool_b.unsqueeze(0)).item()
# l2_dist = (pool_a - pool_b).norm().item()

# show("cosine similarity (A vs B)", f"{cos_sim:.4f}")
# show("L2 distance (A vs B)", f"{l2_dist:.4f}")
# check("Different prompts -> different embeddings (cos < 0.99)", cos_sim < 0.99)

# # Also compare two goal tasks with same scene — this is the hardest case
# print(f"\n  {BOLD}Goal suite pairwise cosine similarity (task 10-19, same scene):{RESET}")
# goal_pools = []
# for ti in range(10, 20):
#     enc = tokenizer([tasks[ti]], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
#     with torch.no_grad():
#         emb = model.language_encoder(enc["input_ids"], enc["attention_mask"].bool())
#     n = int(enc["attention_mask"].sum().item())
#     goal_pools.append(emb[0, :n].mean(0))

# for i in range(10):
#     sims = []
#     for j in range(10):
#         if i == j:
#             sims.append("  1.00")
#         else:
#             s = torch.nn.functional.cosine_similarity(goal_pools[i].unsqueeze(0), goal_pools[j].unsqueeze(0)).item()
#             sims.append(f"{s:+.2f}")
#     print(f"    task {10+i}: [{', '.join(sims)}]")


# # ══════════════════════════════════════════════════════════════════════════
# # Layer 6: VGGT Backbone — real data flow
# # ══════════════════════════════════════════════════════════════════════════
# header("Layer 6: VGGT Backbone (real data flow)")

H = model.vggt_backbone.hidden_size

# # ── Step 1: Get real sample from dataset ──
# real_sample, real_actions = ds[0]
# print(f"  prompt: {real_sample['prompt']!r}")

# # ── Step 2: Build a real ObservationBatch (same as collate_fn) ──
# from betavla.data.types import ObservationBatch

# obs, _ = ds.collate_fn([(real_sample, real_actions)])
# obs = obs.to(DEVICE)

# # ── Step 3: Vision Tower → 2 cameras → 512 patches ──
# with torch.no_grad():
#     vision_tokens = model.vision_tower(obs.images, obs.image_masks)
# show("vision_tower output", f"{vision_tokens.shape}")
# print(f"    前 256 个 patch = base camera (第三人称)")
# print(f"    后 256 个 patch = wrist camera (手腕)")
# show("  base patches mean", f"{vision_tokens[0, :256].mean():.4f}")
# show("  wrist patches mean", f"{vision_tokens[0, 256:].mean():.4f}")

# # ── Step 4: Language Encoder → 128 tokens ──
# with torch.no_grad():
#     text_tokens = model.language_encoder(obs.tokenized_prompt, obs.tokenized_prompt_mask)
# n_real = int(obs.tokenized_prompt_mask.sum().item())
# show("language_encoder output", f"{text_tokens.shape}")
# show("  real tokens (non-padding)", n_real)

# # ── Step 5: Project to VGGT hidden size ──
# with torch.no_grad():
#     vision_proj = model.vision_projector(vision_tokens)
#     text_proj = model.language_projector(text_tokens)
# show("vision projected", f"{vision_proj.shape} ({vision_tokens.shape[-1]}D -> {H}D)")
# show("language projected", f"{text_proj.shape} ({text_tokens.shape[-1]}D -> {H}D)")

# # ── Step 6: Concatenate → fused tokens ──
# fused = torch.cat([vision_proj, text_proj], dim=1)
# show("fused (vision + language)", f"{fused.shape}")
# print(f"    token [  0:255] = base camera patches")
# print(f"    token [256:511] = wrist camera patches")
# print(f"    token [512:639] = language tokens (128, incl padding)")

# # ── Step 7: Attention mask ──
# attn_mask = model._build_attn_mask(vision_proj, obs.tokenized_prompt_mask)
# show("attn_mask", f"{attn_mask.shape}, dtype={attn_mask.dtype}")
# show("  vision part (all 1)", f"sum={attn_mask[0, :512].sum().item()}/512")
# show("  language part", f"real={attn_mask[0, 512:].sum().item()}/128 (rest is padding=0)")

# # ── Step 8: Intercept real positional encoding from VGGT forward ──
# # Hook into the backbone to capture the actual `pos` tensor built during forward
# backbone = model.vggt_backbone.model
# _captured_pos = {}

# if hasattr(backbone, "vision_patch_hw"):
#     ph, pw = backbone.vision_patch_hw
#     show("patch grid", f"{ph}x{pw} = {ph*pw} patches per camera")

#     # Monkey-patch forward to capture `pos`
#     _orig_forward = backbone.forward

#     def _capturing_forward(tokens, attn_mask):
#         # Reproduce the pos construction (same code as vggt_backbone.py:82-110)
#         # tokens here is PRE-prepend; real forward prepends 5 special tokens first
#         B, T, _ = tokens.shape
#         P = T + backbone.patch_start_idx  # after prepend: 640 + 5 = 645
#         n_cam_patches = ph * pw
#         n_vision = backbone.num_cameras * n_cam_patches
#         n_special = backbone.patch_start_idx  # 5
#         n_other = P - n_special - n_vision

#         pos_parts = []
#         pos_parts.append(torch.zeros(B, n_special, 2, device=tokens.device, dtype=torch.long))
#         for cam_i in range(backbone.num_cameras):
#             cam_pos = backbone.position_getter(B, ph, pw, device=tokens.device)
#             cam_pos = cam_pos.clone()
#             cam_pos[..., 0] += cam_i * ph
#             pos_parts.append(cam_pos)
#         if n_other > 0:
#             lang_pos = backbone.position_getter(B, 1, n_other, device=tokens.device)
#             lang_pos = lang_pos.clone()
#             lang_pos[..., 0] += backbone.num_cameras * ph
#             pos_parts.append(lang_pos)
#         _captured_pos["pos"] = torch.cat(pos_parts, dim=1)
#         _captured_pos["n_special"] = n_special
#         _captured_pos["n_cam_patches"] = n_cam_patches
#         _captured_pos["n_other"] = n_other
#         return _orig_forward(tokens, attn_mask)

#     backbone.forward = _capturing_forward

# # ── Step 9: VGGT forward (with pos capture) ──
# with torch.no_grad():
#     vggt_out = model.vggt_backbone(fused, attn_mask)

# # Restore original forward
# if hasattr(backbone, "vision_patch_hw"):
#     backbone.forward = _orig_forward

# show("VGGT input", f"{fused.shape}")
# show("VGGT output", f"{vggt_out.shape}")
# show("  output mean", f"{vggt_out.mean():.4f}, std={vggt_out.std():.4f}")
# check("VGGT preserves token count", vggt_out.shape == fused.shape)

# # ── Step 10: Verify the REAL positional encoding ──
# if "pos" in _captured_pos:
#     pos = _captured_pos["pos"]
#     ns = _captured_pos["n_special"]
#     nc = _captured_pos["n_cam_patches"]
#     no = _captured_pos["n_other"]

#     print(f"\n  {BOLD}Real positional encoding from VGGT forward:{RESET}")
#     show("pos shape", f"{pos.shape}, dtype={pos.dtype}")
#     print(f"    Token layout: [special({ns}) | cam0({nc}) | cam1({nc}) | language({no})]")

#     # Special tokens
#     sp = pos[0, :ns]
#     show("special tokens pos", f"all (0,0)? {(sp == 0).all().item()}")

#     # Camera 0 (base)
#     c0 = pos[0, ns : ns + nc]
#     show("cam0 (base) y range", f"[{c0[:, 0].min().item()}, {c0[:, 0].max().item()}]")
#     show("cam0 (base) x range", f"[{c0[:, 1].min().item()}, {c0[:, 1].max().item()}]")

#     # Camera 1 (wrist)
#     c1 = pos[0, ns + nc : ns + 2 * nc]
#     show("cam1 (wrist) y range", f"[{c1[:, 0].min().item()}, {c1[:, 0].max().item()}]")
#     show("cam1 (wrist) x range", f"[{c1[:, 1].min().item()}, {c1[:, 1].max().item()}]")

#     # Language
#     if no > 0:
#         lang = pos[0, ns + 2 * nc:]
#         show("language y range", f"[{lang[:, 0].min().item()}, {lang[:, 0].max().item()}]")
#         show("language x range", f"[{lang[:, 1].min().item()}, {lang[:, 1].max().item()}]")

#     print(f"\n    Expected layout:")
#     print(f"      special:  y=0,       x=0")
#     print(f"      cam0:     y=[0,{ph-1}],   x=[0,{pw-1}]")
#     print(f"      cam1:     y=[{ph},{2*ph-1}],  x=[0,{pw-1}]")
#     print(f"      language:  y={2*ph},      x=[0,{no-1}]")

#     check("pos dtype is long", pos.dtype in (torch.long, torch.int64))
#     check("cam0 y has 16 unique values", len(c0[:, 0].unique()) == ph)
#     check("cam1 y has 16 unique values", len(c1[:, 0].unique()) == ph)
#     check("cam0/cam1 y don't overlap", c1[:, 0].min().item() > c0[:, 0].max().item())
#     check("language y below all cameras", lang[:, 0].min().item() > c1[:, 0].max().item() if no > 0 else True)
# check("VGGT preserves token count", vggt_out.shape == fused.shape)


# # ══════════════════════════════════════════════════════════════════════════
# # Layer 7: Action Head (real data from pipeline)
# # ══════════════════════════════════════════════════════════════════════════
# header("Layer 7: Action Head (Flow Matching, real data)")

# ah = model.action_head
# show("action_dim", ah.cfg.action_dim)
# show("action_horizon", ah.cfg.action_horizon)
# show("gripper_loss_weight", ah.cfg.gripper_loss_weight)

# # Use real prefix_tokens from model.encode + real state/actions
# # ds[0] = episode start (gripper open, not moving)
# # ds[62] = mid-episode (gripper closing, grasping)
# test_idx = 62 if len(ds) > 62 else 0
# real_s7, real_a7 = ds[test_idx]
# print(f"  using ds[{test_idx}], frame={real_s7.get('frame_index','?')}, prompt={real_s7['prompt']!r}")
# obs_7, act_7 = ds.collate_fn([(real_s7, real_a7)])
# obs_7 = obs_7.to(DEVICE)
# act_7 = act_7.to(DEVICE)

# with torch.no_grad():
#     prefix_tokens, prefix_pad_mask = model.encode(obs_7)

# show("prefix_tokens (from encode)", f"{prefix_tokens.shape}")
# show("prefix_pad_mask", f"{prefix_pad_mask.shape}, real={prefix_pad_mask.sum().item()}/{prefix_pad_mask.numel()}")
# show("state", f"{obs_7.state.shape}, range=[{obs_7.state.min():.3f}, {obs_7.state.max():.3f}]")
# show("actions (GT)", f"{act_7.shape}, range=[{act_7.min():.3f}, {act_7.max():.3f}]")

# # ── Training: flow matching loss ──
# print(f"\n  {BOLD}Training (flow matching loss):{RESET}")
# loss = ah.compute_loss(prefix_tokens, obs_7.state, act_7, prefix_pad_mask=prefix_pad_mask)
# show("loss", f"{loss.item():.4f}")
# check("loss is scalar > 0", loss.ndim == 0 and loss.item() > 0)

# details = ah.compute_loss(prefix_tokens, obs_7.state, act_7, prefix_pad_mask=prefix_pad_mask, return_details=True)
# show("unweighted_loss", f"{details['unweighted_loss'].item():.4f}")
# show("gripper_loss", f"{details['gripper_loss'].item():.4f}")
# show("weighted_loss", f"{details['loss'].item():.4f}")
# if ah.cfg.gripper_loss_weight != 1.0:
#     check("gripper weighting changes loss",
#           abs(details["loss"].item() - details["unweighted_loss"].item()) > 1e-6)

# # ── Inference: ODE sampling ──
# print(f"\n  {BOLD}Inference (ODE sampling):{RESET}")
# with torch.no_grad():
#     sampled = ah.sample(prefix_tokens, obs_7.state, num_steps=5, prefix_pad_mask=prefix_pad_mask)
# show("sampled shape", sampled.shape)
# show("sampled range", f"[{sampled.min():.4f}, {sampled.max():.4f}]")
# check("sample = (1, 10, 7)", sampled.shape == (1, 10, 7))

# # Compare sampled vs ground truth — normalized + raw
# from betavla.data.normalize import load_norm_stats, unnormalize_quantile
# norm = load_norm_stats(NORM_STATS_PATH)
# gt_raw = unnormalize_quantile(act_7[0].cpu().numpy(), norm["action"])
# pred_raw = unnormalize_quantile(sampled[0].cpu().numpy(), norm["action"])

# print(f"\n  {BOLD}Normalized (what model sees):{RESET}")
# print(f"    {'step':>4s}  {'':>5s} {'dx':>8s} {'dy':>8s} {'dz':>8s} {'drx':>8s} {'dry':>8s} {'drz':>8s} {'grip':>8s}")
# for t in range(sampled.shape[1]):
#     gt_vals = " ".join(f"{act_7[0, t, d]:+8.4f}" for d in range(7))
#     sa_vals = " ".join(f"{sampled[0, t, d]:+8.4f}" for d in range(7))
#     print(f"    [{t:2d}]   GT   {gt_vals}")
#     print(f"          pred {sa_vals}")

# print(f"\n  {BOLD}Raw (real physical values, after unnormalize):{RESET}")
# print(f"    {'step':>4s}  {'':>5s} {'dx':>9s} {'dy':>9s} {'dz':>9s} {'drx':>9s} {'dry':>9s} {'drz':>9s} {'grip':>5s}")
# for t in range(sampled.shape[1]):
#     gt_r = " ".join(f"{gt_raw[t, d]:+9.5f}" for d in range(6))
#     pr_r = " ".join(f"{pred_raw[t, d]:+9.5f}" for d in range(6))
#     print(f"    [{t:2d}]   GT   {gt_r} {gt_raw[t, 6]:+5.1f}")
#     print(f"          pred {pr_r} {pred_raw[t, 6]:+5.1f}")


# ══════════════════════════════════════════════════════════════════════════
# Layer 8: End-to-End (real data, full forward)
# ══════════════════════════════════════════════════════════════════════════
header("Layer 8: End-to-End (full forward, real data)")

from betavla.data.types import ObservationBatch

# ── Two real observations with different prompts ──
sample_a, actions_a = ds[0]
prompt_a_text = sample_a["prompt"]

sample_b, actions_b = None, None
for i in range(1, len(ds)):
    s, a = ds[i]
    if s["prompt"] != prompt_a_text:
        sample_b, actions_b = s, a
        break
if sample_b is None:
    sample_b, actions_b = ds[1]

obs_a, act_a = ds.collate_fn([(sample_a, actions_a)])
obs_b, act_b = ds.collate_fn([(sample_b, actions_b)])
obs_a = obs_a.to(DEVICE)
act_a = act_a.to(DEVICE)
obs_b = obs_b.to(DEVICE)
act_b = act_b.to(DEVICE)

print(f"  prompt A: {sample_a['prompt']!r}")
print(f"  prompt B: {sample_b['prompt']!r}")

# ── Training forward ──
print(f"\n  {BOLD}Training forward:{RESET}")
with torch.no_grad():
    out_a = model(obs_a, act_a)
show("loss A", f"{out_a['loss'].item():.4f}")
check("training forward returns loss", "loss" in out_a and out_a["loss"].ndim == 0)

# ── Inference forward ──
print(f"\n  {BOLD}Inference forward:{RESET}")
with torch.no_grad():
    inf_a = model(obs_a, actions=None, num_ode_steps=5)
show("predicted actions shape", inf_a["actions"].shape)
show("predicted range", f"[{inf_a['actions'].min():.4f}, {inf_a['actions'].max():.4f}]")
check("inference returns (1, 10, 7)", inf_a["actions"].shape == (1, 10, 7))

# ── Gradient check ──
print(f"\n  {BOLD}Gradient check:{RESET}")
model.train()
out_grad = model(obs_a, act_a)
out_grad["loss"].backward()
grad_params = [(n, p) for n, p in model.named_parameters()
               if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
show("params with gradients", len(grad_params))
if grad_params:
    for n, p in grad_params[:50000]:
        show(f"  {n}", f"grad norm={p.grad.norm():.4f}")
    if len(grad_params) > 50000:
        print(f"    ... and {len(grad_params) - 50000} more")
check("gradients flow to trainable params", len(grad_params) > 0)
model.eval()
model.zero_grad()

# ── Language sensitivity: same images, different prompts ──
print(f"\n  {BOLD}Language sensitivity (same images, different prompts):{RESET}")
obs_b_same_vis = ObservationBatch(
    images=obs_a.images,
    image_masks=obs_a.image_masks,
    state=obs_a.state,
    tokenized_prompt=obs_b.tokenized_prompt,
    tokenized_prompt_mask=obs_b.tokenized_prompt_mask,
)

with torch.no_grad():
    loss_pa = model(obs_a, act_a)["loss"].item()
    loss_pb = model(obs_b_same_vis, act_a)["loss"].item()

show("loss with prompt A", f"{loss_pa:.6f}")
show("loss with prompt B (same images)", f"{loss_pb:.6f}")
show("loss difference", f"{abs(loss_pa - loss_pb):.6f}")
check("different prompts -> different loss", loss_pa != loss_pb)


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*70}")
print(f"  SUMMARY: {GREEN}{pass_count} passed{RESET}{BOLD}, {RED if fail_count else ''}{fail_count} failed{RESET}")
print(f"{BOLD}{'='*70}{RESET}")
sys.exit(1 if fail_count else 0)
