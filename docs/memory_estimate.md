# Beta-VLA 显存估算

## 模型参数量（约 2B）

| 模块 | 参数量 | 说明 |
|------|--------|------|
| DINOv2 ViT-L | ~304M | vit_large_patch14_reg4_dinov2 |
| SigLIP ViT-SO400M | ~400M | vit_so400m_patch14_siglip_224 |
| Qwen 0.6B | ~600M | Qwen3-0.6B-Base |
| VGGT-1B aggregator | ~700M | frame_blocks + global_blocks |
| Projectors + Action head | ~10M | 可忽略 |
| **总计** | **~2B** | |

## 单卡显存（FP32，DDP 每卡完整模型副本）

| 项目 | 计算 | 显存 |
|------|------|------|
| 模型权重 | 2B × 4 bytes | **8 GB** |
| 梯度 | 2B × 4 bytes | **8 GB** |
| AdamW 优化器 | 2B × 2 states × 4 bytes | **16 GB** |
| **训练状态小计** | | **32 GB** |
| 激活值 (batch=8, seq≈640) | VGGT 48 层 + Vision 51 层 + LLM 28 层 | **~12–18 GB** |
| **总计** | | **~44–50 GB** |

L20 单卡 46GB → batch=8 时易 OOM。

## 建议

1. **batch_size=2 或 4**：每卡 2–4 样本
2. **启用 gradient_checkpointing**：用计算换显存，激活值可降约 60%
3. **混合精度 (AMP)**：FP16/BF16 可减半模型+梯度+优化器显存
