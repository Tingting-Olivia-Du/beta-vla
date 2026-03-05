# Beta-VLA 代码关系说明（方案2：完全自包含）

该文档描述当前 `beta-vla` 与参考项目的关系。  
状态：**self-contained**（训练与数据加载不再 import `openpi`）。

## 当前结论

- 运行时不再依赖本地 `../openpi` 路径。
- 训练入口与数据加载均在 `betavla.*` 命名空间内部。
- `openvla-oft` 与 `openpi` 仅作为“设计参考”，不是代码运行依赖。

## 代码结构与来源

### 1) 模型主干（自定义实现）

- `src/betavla/models/vision_tower.py`
  - 参考 `openvla-oft` 的 Dino+SigLIP 思路
  - 实现为轻量 `timm` 双塔特征提取
- `src/betavla/models/language_encoder.py`
  - 新增：Qwen3-0.6B 包装
- `src/betavla/models/vggt_backbone.py`
  - 新增：VGGT-1B 包装
- `src/betavla/models/action_head.py`
  - 参考 openpi 的 flow-matching 思想，重写为当前模型输入接口
- `src/betavla/models/beta_vla_model.py`
  - 新增：完整组网（vision/language projector + VGGT + action head）与 LoRA 注入

### 2) 数据与训练（完全内置）

- `src/betavla/data/libero_loader.py`
  - 新增：LIBERO 数据集加载、tokenize、batch collate
  - 不再调用 `openpi.training.data_loader`
- `src/betavla/training/config_beta_vla.py`
  - 使用 `runtime + data + model` 配置
  - checkpoint 路径：`checkpoints/<exp_name>/`
- `src/betavla/training/train_beta_vla.py`
  - 自定义训练循环（DDP、checkpoint、best model）
  - 不再 import `openpi.*`

### 3) 脚本（完全内置）

- `scripts/train_beta_vla.sh`
  - conda `beta` 环境启动
  - 训练前依赖预检查（torch/transformers/datasets/timm/betavla）
  - 默认将 HF 缓存写入 `/data/hf-home`，避免 `/root` 爆盘
- `scripts/smoke_test_forward.sh`
  - 直接走 `betavla` 内置 loader + model
- `scripts/setup_env.sh`
  - 只安装 `beta-vla` 本仓依赖

## 与参考代码的关系（保留项）

- 保留的“思想”：
  - openvla-oft：双视觉塔设计（DINOv2 + SigLIP）
  - openpi：flow-matching 动作学习思路、训练脚本组织方式（DDP/checkpoint）
- 未直接保留的“运行依赖”：
  - openpi 的 config/data_loader/transforms 代码路径
  - openvla-oft 的 prismatic 训练框架

## 说明

- 若未来要进一步对齐 openpi 的数据变换细节，可在 `betavla/data/` 内继续扩展，但不需要再引入 `openpi` 作为运行时依赖。
