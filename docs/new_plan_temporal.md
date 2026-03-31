# Phase 3: VGGT 多帧时序输入改造

## Context

Beta-VLA 当前只用单帧（2 个相机视角作为 token 拼接），VGGT aggregator 的 frame/global attention 没有发挥跨帧推理能力。改造目标：支持 T 帧时序输入，使模型获得运动感知和隐式 3D 理解，同时保留单帧模式的开关。

## 方案选择：A（隐式 3D）

只改造 aggregator 支持多帧，不接 DPT/Camera head。VGGT aggregator 内部表示已隐式编码 3D 信息。

---

## 关键发现

### 预训练权重加载（确认正确）
- `VGGT.from_pretrained("facebook/VGGT-1B")` 加载完整预训练模型
- 当前代码只提取 aggregator 组件（frame_blocks, global_blocks, rope 等），权重是预训练的
- LoRA 微调可以适配 PaliGemma 特征分布（与原始 DINOv2 不同）

### camera_token 问题
- 原始 VGGT: `camera_token` 形状 `(1, 2, 1, C)` — index=0 给第一帧, index=1 给其余帧
- 当前代码: `agg.camera_token[:, 0:1, :, :]` 只取了 index=0，丢弃了 index=1
- **多帧模式需要保留两个 index**

### 当前输出维度
- 当前输出: `(B, T, C)` 其中 C = embed_dim = 1024
- 原始 VGGT aggregator 输出: `output_list` 每层 `(B, S, P, 2C=2048)` (frame+global concat)
- 当前代码**没有做 frame+global concat**，也没有保存中间层

---

## 实现计划

### 1. Config 开关 — `vggt_backbone.py`

```python
@dataclass(frozen=True)
class VGGTBackboneConfig:
    model_name: str = "facebook/VGGT-1B"
    trust_remote_code: bool = True
    vision_patch_hw: tuple[int, int] = (16, 16)
    num_cameras: int = 2
    temporal_frames: int = 1           # 新增：1=单帧(现有行为), >1=多帧
```

`temporal_frames=1` 时所有行为与当前完全一致，不破坏已有 checkpoint 兼容性。

### 2. `_VGGTAggregatorBackbone.__init__` 改动

```python
# 保留完整的 camera_token 和 register_token (两个 index)
self.camera_token = agg.camera_token     # (1, 2, 1, C) — 不再截断
self.register_token = agg.register_token  # (1, 2, 4, C) — 不再截断
self.temporal_frames = temporal_frames
```

单帧兼容：当 temporal_frames=1 时，forward 中只使用 index=0（与当前行为一致）。

### 3. `_VGGTAggregatorBackbone.forward` 改动 — 核心

#### 输入变化
- 单帧: `tokens` 形状 `(B, N_vis+N_lang, C)` 其中 N_vis = num_cameras * 256
- 多帧: `tokens` 形状 `(B, T*num_cameras*256 + N_lang, C)` — vision tower 对 T*num_cameras 帧分别编码后拼接

#### Forward 逻辑（temporal_frames > 1 路径）

```python
S = self.temporal_frames * self.num_cameras  # e.g., 3 * 2 = 6
P_per_view = ph * pw  # 256 patches per camera view
N_lang = total_tokens - S * P_per_view  # language tokens

# 1. 分离 vision tokens 和 language tokens
vision_tokens = x[:, :S * P_per_view, :]  # (B, S*256, C)
lang_tokens = x[:, S * P_per_view:, :]     # (B, N_lang, C)

# 2. Reshape vision 为 (B, S, P_per_view, C)
vision_tokens = vision_tokens.view(B, S, P_per_view, C)

# 3. 准备 camera + register tokens (使用原始 VGGT 的 slice_expand_and_flatten 逻辑)
#    index=0 给第一帧, index=1 给其余 S-1 帧
cam = slice_expand_camera(self.camera_token, B, S)    # (B, S, 1, C)
reg = slice_expand_camera(self.register_token, B, S)  # (B, S, 4, C)

# 4. 每帧 token 序列: [camera(1), register(4), patches(256)] = 261 per frame
per_frame = torch.cat([cam, reg, vision_tokens], dim=2)  # (B, S, 261, C)
P = 261  # tokens per frame

# 5. Language tokens 作为额外 "帧" 或附加到每帧
#    方案: 附加到 global attention 的序列末尾

# 6. Alternating attention loop
for block_num in range(aa_block_num):
    for attn_type in aa_order:
        if attn_type == "frame":
            # 每帧内部 attention: (B*S, P, C)
            tokens = per_frame.view(B * S, P, C)
            tokens = frame_blocks[idx](tokens, pos_frame)
            per_frame = tokens.view(B, S, P, C)
        else:
            # 跨帧 attention: (B, S*P + N_lang, C)
            flat = per_frame.view(B, S * P, C)
            global_tokens = torch.cat([flat, lang_tokens], dim=1)
            global_tokens = global_blocks[idx](global_tokens, pos_global)
            # 分离回去
            per_frame = global_tokens[:, :S*P, :].view(B, S, P, C)
            lang_tokens = global_tokens[:, S*P:, :]

# 7. 输出: 只取当前帧（最后一个时间步）的 patch tokens
current_frame_idx = (self.temporal_frames - 1) * self.num_cameras  # 当前时间步的第一个相机
current_patches = per_frame[:, current_frame_idx:, patch_start_idx:, :]  # (B, num_cameras, 256, C)
current_patches = current_patches.reshape(B, self.num_cameras * P_per_view, C)
output = torch.cat([current_patches, lang_tokens], dim=1)  # (B, 512+N_lang, C)
return output
```

#### 单帧路径 (temporal_frames=1)

保持当前代码不变，确保完全向后兼容。

### 4. RoPE 2D 位置编码更新

```python
# 多帧模式下的位置编码布局:
# 每个 (time_step, camera) 组合是一个 "view"
# view 之间通过 y-offset 分隔

for t in range(temporal_frames):
    for cam_i in range(num_cameras):
        view_idx = t * num_cameras + cam_i
        cam_pos = position_getter(B * S, ph, pw, device)
        # y-offset: 每个 view 偏移 ph 行
        cam_pos[..., 0] += view_idx * ph
        pos_parts.append(cam_pos)

# Language tokens: y-offset 在所有 view 下方
lang_pos[..., 0] += S * ph
```

### 5. Vision Tower 改动 — `vision_tower.py`

- 输入: images 从 `{cam_key: (B, H, W, C)}` 变为 `{cam_key: (B, T, H, W, C)}`
- PaliGemma 对每帧独立编码: reshape to `(B*T, H, W, C)` → encode → reshape back
- 输出: `(B, T * num_cameras * 256, D)` 按 [t0_cam0, t0_cam1, t1_cam0, t1_cam1, ...] 排列

单帧兼容: 当 T=1 时，输出形状与当前完全一致 `(B, 512, D)`。

### 6. Data Pipeline 改动 — `libero_dataset.py`

新增 config:
```python
temporal_frames: int = 1      # 1 = 当前行为
temporal_stride: int = 5      # 帧间隔 (匹配 replan_steps)
```

`__getitem__` 改动:
- 当前帧位置 `pos`，额外加载 `pos - k*stride` (k=1,...,T-1)，clamp 到 episode 起始
- 返回 `{cam_key: (T, H, W, C)}` per camera

### 7. Inference 改动 — `inference.py`

- 维护一个 frame buffer (deque, maxlen=T)
- 每步 push 当前 base_img + wrist_img
- replan 时将完整 T 帧传入 model

### 8. Model / Config 改动 — `model.py`, `config.py`

- `BetaVLAConfig` 透传 `temporal_frames` 到 `VGGTBackboneConfig`
- `configs/libero_vggt.yaml` 新增:
  ```yaml
  model:
    vggt:
      temporal_frames: 3
  data:
    temporal_frames: 3
    temporal_stride: 5
  ```
- 单帧 config 保持 `temporal_frames: 1`

---

## 关键文件改动清单

| 文件 | 改动 |
|------|------|
| `src/betavla/models/vggt_backbone.py` | 核心: 保留完整 camera/register token, 多帧 forward 路径, RoPE 更新 |
| `src/betavla/models/vision_tower.py` | 支持 (B, T, H, W, C) 输入 |
| `src/betavla/models/model.py` | 透传 temporal_frames config |
| `src/betavla/data/libero_dataset.py` | 时序帧加载 |
| `src/betavla/data/types.py` | ObservationBatch 适配 |
| `src/betavla/eval/inference.py` | 帧缓冲区 |
| `src/betavla/training/config.py` | 新增 temporal config 字段 |
| `configs/libero_vggt.yaml` | 新增 temporal_frames, temporal_stride |

## 向后兼容性

- `temporal_frames=1` 时所有代码路径与当前完全一致
- 已训练的 checkpoint 可以在 temporal_frames=1 下正常加载
- 新 checkpoint 训练时只需改 config

## 验证方案

1. **单元测试**: temporal_frames=1 时输出与改动前完全一致 (数值 match)
2. **Smoke test**: temporal_frames=3 时 forward pass 不 OOM, 输出形状正确
3. **训练**: 在 libero 上训练 50K steps, 监控 loss 收敛
4. **评估**: 对比 temporal_frames=1 vs 3 在 libero_spatial + libero_10 上的准确率


Update Todos

Add temporal_frames config to VGGTBackboneConfig and BetaVLAConfig

Rewrite _VGGTAggregatorBackbone: preserve full camera/register tokens, add multi-frame forward path

Update vision_tower.py to support (B, T, H, W, C) temporal images

Update data types (ObservationBatch) for temporal images

Update libero_dataset.py for temporal frame loading

Update model.py to wire temporal_frames through encode()

Update inference.py with frame buffer for temporal inference

Update training config.py and libero_vggt.yaml

Smoke test: verify single-frame backward compat and multi-frame forward pass
