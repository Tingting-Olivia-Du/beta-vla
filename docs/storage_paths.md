# Beta-VLA 存储路径说明

**手动运行 Python 测试前**，请先 `source scripts/env_umd_datapool.sh`，否则 HF/timm 会下载到 `/root/.cache`（可能爆满）。

## 脚本用到的目录

| 用途 | 环境变量 | 默认路径 | 可能爆满？ |
|------|----------|----------|------------|
| HF 模型缓存 | HF_HOME | /umd-datapool/tingting/hf-home | 否（126T） |
| HF Hub 缓存 | HF_HUB_CACHE | (HF_HOME/hub) | 同上 |
| Transformers 缓存 | TRANSFORMERS_CACHE | .../hf-home/transformers | 同上 |
| HF Datasets 缓存 | HF_DATASETS_CACHE | .../hf-home/datasets | 同上 |
| wandb 日志 | WANDB_DIR | /umd-datapool/tingting/wandb | 否 |
| 临时文件 | TMPDIR | /umd-datapool/tingting/.tmp | 否 |
| 训练 checkpoint | checkpoint_base_dir | ./checkpoints（相对） | 否 |
| wandb run 目录 | (项目内) | beta-vla/wandb/ | 否 |

## 卡在 model.safetensors 下载的可能原因

1. **NFS 写入慢**：umd-datapool 是 NFS，5GB 写入可能很慢（8.55MB/s 时约 10 分钟）
2. **网络慢**：从 HuggingFace 拉取大文件
3. **磁盘满**：若 HF_HOME 指向本地且满，会失败

## 全部放到 umd-datapool

train_beta_vla.sh 已配置 HF/WANDB/TMP 到 umd-datapool。checkpoint 用相对路径 `./checkpoints`，运行目录在 beta-vla 时也会落在 umd-datapool。

## 若下载仍卡住

可尝试用**本地 TMPDIR** 加速下载（先下到本地再拷到 NFS）：

```bash
# 仅下载阶段用本地 temp
export TMPDIR=/data/tingting-tmp
export HF_HOME=/umd-datapool/tingting/hf-home
bash scripts/train_beta_vla.sh --gpus 0
```

或预先下载 VGGT 到 HF 缓存：

```bash
export HF_HOME=/umd-datapool/tingting/hf-home
python -c "from vggt.models.vggt import VGGT; VGGT.from_pretrained('facebook/VGGT-1B')"
```
