# norm_stats 检查报告

## 检查结果摘要

| 项目 | 状态 | 说明 |
|------|------|------|
| 结构 | ✅ OK | state/action 均有 mean, std, q01, q99 |
| 维度 | ✅ OK | state 8D, action 7D |
| q01 < q99 | ✅ OK | 所有维度有效 |
| state 范围 | ✅ OK | 50 样本均在 q01/q99 内 |
| action 范围 | ⚠️ WARN | 50 样本中有 10 个 action 超出 q01/q99 |
| checkpoint 一致性 | ✅ OK | assets 与 best/norm_stats.json 完全一致 |

---

## 潜在问题

### 1. 部分 action 超出 q01/q99

约 20% 的样本的 action 超出预计算的 q01/q99 范围。可能原因：

- **compute_norm_stats 用了 max_samples**：若用 `--max_samples 5000` 等，统计可能未覆盖全量数据
- **正常波动**：q01/q99 为 1% 和 99% 分位，约 2% 数据可超出；20% 偏高，需确认

**建议**：用全量数据重新跑 `compute_norm_stats.py`（不加 `--max_samples`）

### 2. State 格式一致性

- **训练数据**：`observation/state` 或 `state`，8D
- **Eval**：`prepare_state(obs)` = eef_pos(3) + quat2axisangle(3) + gripper_qpos(2)

physical-intelligence/libero 源自 openvla/modified_libero_rlds，state 应为相同格式。若不一致会导致 state 归一化错误。

### 3. Gripper 处理

- **norm_stats**：action[6] 的 q01=-1, q99≈1，数据为 [-1, 1]
- **process_action_for_env**：`2*a-1` 后阈值到 ±1，隐含假设 a 在 [0,1]
- 反归一化后 a 在 [-1, 1]，2*a-1 会得到 [-3, 1]，但阈值后仍为 ±1，逻辑上可接受

---

## 验证命令

```bash
# 运行验证脚本
PYTHONPATH=src python scripts/verify_norm_stats.py

# 用全量数据重新计算 norm_stats（若怀疑统计不全）
uv run python scripts/compute_norm_stats.py --config configs/train_beta_vla_libero.yaml
# 不加 --max_samples，会遍历全量数据
```

---

## 结论

norm_stats 结构正确，与 checkpoint 一致。主要风险是 **部分 action 超出 q01/q99**，建议用全量数据重新计算并对比。
