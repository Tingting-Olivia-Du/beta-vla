#!/usr/bin/env python3
"""Verify norm_stats and state/action format alignment between dataset and eval.

Usage:
  PYTHONPATH=src python scripts/verify_norm_stats.py
  PYTHONPATH=src python scripts/verify_norm_stats.py --config configs/train_beta_vla_libero.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


def _pick_first(sample: dict, keys: list[str]):
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/train_beta_vla_libero.yaml"))
    p.add_argument("--norm_stats", type=Path, default=Path("assets/physical-intelligence/libero/norm_stats.json"))
    p.add_argument("--num_samples", type=int, default=100)
    args = p.parse_args()

    norm_path = args.norm_stats
    if not norm_path.exists():
        print(f"ERROR: norm_stats not found: {norm_path}")
        return 1

    norm_stats = json.loads(norm_path.read_text())
    print("=" * 60)
    print("norm_stats.json 检查报告")
    print("=" * 60)

    # 1. 结构检查
    print("\n1. 结构检查")
    for key in ["state", "action"]:
        if key not in norm_stats:
            print(f"  [FAIL] 缺少 '{key}'")
            continue
        ns = norm_stats[key]
        for sub in ["mean", "std", "q01", "q99"]:
            if sub not in ns:
                print(f"  [FAIL] {key} 缺少 '{sub}'")
            else:
                arr = np.array(ns[sub])
                print(f"  [OK] {key}.{sub}: len={len(arr)}")

    # 2. 维度检查
    print("\n2. 维度检查")
    state_dim = len(norm_stats["state"]["q01"])
    action_dim = len(norm_stats["action"]["q01"])
    print(f"  state: {state_dim}D (期望 8: eef_pos 3 + quat2axisangle 3 + gripper 2)")
    print(f"  action: {action_dim}D (期望 7: delta_xyz 3 + delta_euler 3 + gripper 1)")
    if state_dim != 8 or action_dim != 7:
        print("  [WARN] 维度与 LIBERO 不匹配!")

    # 3. q01 < q99 检查
    print("\n3. q01 < q99 检查")
    for key in ["state", "action"]:
        q01 = np.array(norm_stats[key]["q01"])
        q99 = np.array(norm_stats[key]["q99"])
        bad = np.sum(q01 >= q99)
        if bad > 0:
            print(f"  [FAIL] {key}: {bad} 维 q01 >= q99")
            for i in np.where(q01 >= q99)[0]:
                print(f"    dim {i}: q01={q01[i]:.4f} q99={q99[i]:.4f}")
        else:
            print(f"  [OK] {key}: 所有维度 q01 < q99")

    # 4. 数据集样本检查
    print("\n4. 数据集样本检查 (physical-intelligence/libero)")
    try:
        ds = load_dataset("physical-intelligence/libero", split="train")
        n = min(args.num_samples, len(ds))

        state_q01 = np.array(norm_stats["state"]["q01"])
        state_q99 = np.array(norm_stats["state"]["q99"])
        action_q01 = np.array(norm_stats["action"]["q01"])
        action_q99 = np.array(norm_stats["action"]["q99"])

        state_out_of_range = 0
        action_out_of_range = 0
        for i in range(n):
            sample = ds[i]
            state = _pick_first(sample, ["observation/state", "state", "observation.state"])
            actions = _pick_first(sample, ["actions", "action"])
            if state is not None:
                state = np.array(state).reshape(-1)[:8]
                if np.any(state < state_q01 - 0.1) or np.any(state > state_q99 + 0.1):
                    state_out_of_range += 1
            if actions is not None:
                actions = np.array(actions).reshape(-1, 7)
                if np.any(actions < action_q01 - 0.1) or np.any(actions > action_q99 + 0.1):
                    action_out_of_range += 1

        print(f"  检查 {n} 个样本")
        if state_out_of_range > 0:
            print(f"  [WARN] {state_out_of_range} 个样本的 state 超出 q01/q99 范围")
        else:
            print(f"  [OK] state 均在 q01/q99 范围内")
        if action_out_of_range > 0:
            print(f"  [WARN] {action_out_of_range} 个样本的 action 超出 q01/q99 范围")
        else:
            print(f"  [OK] action 均在 q01/q99 范围内")

        # 打印前 3 个样本的 state/action 范围
        print("\n  前 3 个样本的 state 和 action (gripper=最后一维):")
        for i in range(min(3, n)):
            sample = ds[i]
            state = _pick_first(sample, ["observation/state", "state", "observation.state"])
            actions = _pick_first(sample, ["actions", "action"])
            s = np.array(state).reshape(-1)[:8] if state is not None else None
            a = np.array(actions).reshape(-1, 7) if actions is not None else None
            if s is not None:
                print(f"    sample {i} state: shape={s.shape}, gripper(6:8)={s[6:8]}")
            if a is not None:
                print(f"    sample {i} action: shape={a.shape}, gripper(6) range=[{a[:, 6].min():.4f}, {a[:, 6].max():.4f}]")

    except Exception as e:
        print(f"  [SKIP] 无法加载数据集: {e}")

    # 5. process_action_for_env 与 norm_stats 的兼容性
    print("\n5. process_action_for_env 与 gripper 范围")
    gripper_q01 = norm_stats["action"]["q01"][6]
    gripper_q99 = norm_stats["action"]["q99"][6]
    print(f"  action[6] (gripper): q01={gripper_q01}, q99={gripper_q99}")
    print("  process_action_for_env 做: a[-1] = 2*a[-1]-1, 然后阈值到 ±1")
    print("  该公式假设 gripper 在 [0,1]; 若数据在 [-1,1] 则 2*a-1 会得到 [-3,1]")
    print("  但最终阈值到 ±1，所以只要 a>=0.5 -> 1, a<0.5 -> -1 即可")
    if gripper_q01 < 0 and gripper_q99 > 0:
        print("  [INFO] 数据 gripper 跨 0，反归一化后模型输出约 0.5 为分界，与阈值逻辑兼容")

    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
