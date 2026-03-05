# Beta-VLA 调试指南

## 重要：运行前需先激活 conda

```bash
conda activate beta
cd /umd-datapool/tingting/beta-vla
GPU_IDS="0" bash scripts/train_beta_vla.sh
```

脚本已改为使用 `conda activate` + 直接运行 `python`，不再用 `conda run`（conda run 可能丢失环境变量、缓冲输出）。

## 1. `tmp_jz50m5vwandb-artifacts` 目录来源

**来源**: **Weights & Biases (wandb)** 库在运行时会创建临时目录，用于暂存要上传的 artifacts。

- 格式: `tmp_<随机字符串>wandb-artifacts`（如 `tmp_jz50m5vwandb-artifacts`）
- 原因: 当系统 `/tmp` 权限不足或 wandb 回退到当前工作目录时，会在项目目录下创建
- 处理: 已在 `train_beta_vla.sh` 中设置 `WANDB_DIR=/tmp/wandb` 和 `TMPDIR=/tmp`，避免在项目目录生成

## 2. 训练脚本卡住

脚本在 `CUDA_VISIBLE_DEVICES=0,1,2,3` 后卡住，可能原因：

| 阶段 | 可能原因 | 解决方法 |
|------|----------|----------|
| **wandb.init()** | 连接 wandb 服务器超时/网络问题 | `export WANDB_MODE=offline` 或 config 中 `wandb_enabled: false` |
| **conda run 输出缓冲** | Python 输出被缓冲，看起来像卡住 | 已添加 `python -u` 和 `PYTHONUNBUFFERED=1` |
| **load_dataset** | 从 HuggingFace 下载 physical-intelligence/libero | 确保网络畅通，或提前 `huggingface-cli download` |
| **模型加载** | DINO/SigLIP/Qwen/VGGT 首次下载 | 同上，或检查 HF_HOME/TRANSFORMERS_CACHE |

## 3. 快速测试

```bash
# 禁用 wandb 测试（排除 wandb 卡住）
WANDB_MODE=disabled CONDA_ENV_NAME=beta bash scripts/train_beta_vla.sh

# 或修改 config: wandb_enabled: false
```

## 4. 单 GPU 调试（更快定位问题）

```bash
# 修改 train_beta_vla.sh 或传参 --gpus 0
GPU_IDS="0" CONDA_ENV_NAME=beta bash scripts/train_beta_vla.sh
```

## 5. 查看卡在哪一步

已添加 `[beta-vla]` 日志，观察最后一条：

- `train_beta_vla module loading...` → Python 已启动，若没有则卡在 conda/python 启动
- `setup_ddp done` → 卡在 wandb
- `initializing wandb...` → 卡在 wandb 连接
- `wandb done, building dataloader...` → 卡在 HF datasets 下载
- `dataloader ready, loading model...` → 卡在模型加载

## 6. 分步调试脚本

```bash
conda activate beta
cd /umd-datapool/tingting/beta-vla
bash scripts/debug_train.sh
```

依次执行：torch → wandb → datasets → libero → tokenizer → model，可精确定位卡住阶段。
