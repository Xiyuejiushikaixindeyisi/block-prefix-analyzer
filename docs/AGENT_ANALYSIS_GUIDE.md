# Agent 数据集分析指南

本文档说明：对于 Agent 类请求数据集，当前 project 能完成哪些数据分析、各分析的输入要求、以及完整的运行命令。

---

## 数据处理流程

### 输入 CSV 格式

Agent CSV 的原始字段（列顺序可通过 CLI 参数配置）：

| 列 | 字段 | 说明 |
|----|------|------|
| 0 | `request_id` | **会话标识符**（同一 Agent 任务的多次调用共享此 ID） |
| 1 | `user_id` | 租户 / 用户标识 |
| 2 | `input_prompt` | 原始 prompt 文本（平均约 50K 字符） |
| 3 | `timestamp` | 请求到达时间（Unix float 或 ISO-8601） |

### 转换为 JSONL

```bash
python scripts/convert_agent_csv_to_jsonl.py \
    --input  data/internal/<model>/raw/<file>.csv \
    --output data/internal/<model>/requests.jsonl \
    --col-chat-id 0 --col-user-id 1 --col-raw-prompt 2 --col-timestamp 3 \
    --has-header --encoding utf-8-sig
```

### 输出 JSONL 字段

| 字段 | 说明 |
|------|------|
| `chat_id` | 原始 `request_id`（会话标识符） |
| `request_id` | `{chat_id}_{turn_index}`（每轮唯一 ID） |
| `user_id` | 租户 / 用户标识 |
| `timestamp` | 请求时间 |
| `raw_prompt` | 原始 prompt 文本 |
| `turn_index` | 会话内轮次（0-based，按 timestamp 排序） |

---

## 可运行的数据分析

### ✅ 无需额外配置（直接可用）

以下分析通过 `business_loader` 加载 Agent JSONL，逻辑与普通单轮 MaaS 数据完全一致。

#### F4 — KV Cache 命中率随时间变化

```bash
python scripts/generate_f4_business.py configs/maas/<model>/f4_prefix.yaml
python scripts/generate_f4_business.py configs/maas/<model>/f4_reusable.yaml
```

- 展示 prefix cache 命中率在时间窗口内的变化趋势
- 对 Agent 数据：每一轮 turn 作为独立请求参与统计

#### F13 — KV Cache 复用时间 CDF（所有请求）

```bash
python scripts/generate_f13_business.py configs/maas/<model>/f13_prefix.yaml
python scripts/generate_f13_business.py configs/maas/<model>/f13_reusable.yaml
```

- 展示任意两个请求之间发生块复用的时间间隔分布

#### reuse_rank — 块复用 Pareto 分布

```bash
python scripts/generate_reuse_rank_business.py configs/maas/<model>/reuse_rank.yaml
```

- 展示哪些 prompt 块被复用最多次（长尾分布）

#### E1 — 每用户理想 prefix 命中率

```bash
python scripts/generate_user_hit_rate.py configs/maas/<model>/e1_user_hit_rate.yaml
```

- 按 `user_id` 聚合，统计每个用户的 prefix cache 命中率

#### E1B — 复用集中度（Lorenz 曲线）

```bash
python scripts/generate_skewness.py configs/maas/<model>/e1b_skewness.yaml
```

- 展示命中贡献和请求量在用户间的集中程度

#### E5 — 高复用块还原为原始文本

```bash
python scripts/generate_e5_block_text.py configs/maas/<model>/e5_block_text.yaml
```

- 将最高频复用的块序列解码为可读的 prompt 片段，揭示哪些内容被反复使用

---

### ✅ Agent 专属分析（利用会话结构）

以下分析利用 `chat_id` 和 `turn_index` 字段重建会话结构。

#### F9 — Agent 会话轮次分布 CDF

```bash
python scripts/generate_f9.py configs/maas/<model>/f9_agent.yaml
```

- X 轴：一个 session 内的 request 数量（轮次数）
- Y 轴：累计占比（CDF）
- 揭示 Agent 任务的会话深度分布：多数任务几轮内完成，少数任务轮次极深

**Config 示例：**
```yaml
input_file: data/internal/<model>/requests.jsonl
output_dir: outputs/maas/<model>/f9_agent
trace_name: <model>
```

#### F10 — 每用户会话轮次均值与标准差分布

```bash
python scripts/generate_f10.py configs/maas/<model>/f10_agent.yaml
```

- 图 A：每用户平均轮次（Mean Turns）分布 + Lorenz 累计曲线
- 图 B：每用户轮次标准差（Std Dev）分布 + Lorenz 累计曲线
- 揭示用户行为异质性：高频 Agent 用户 vs. 低频用户的贡献集中程度

**Config 示例：**
```yaml
input_file: data/internal/<model>/requests.jsonl
output_dir: outputs/maas/<model>/f10_agent
trace_name: <model>
```

#### F14 — 多轮 Agent 请求的 KV Cache 复用时间 CDF

```bash
python scripts/generate_f14_agent.py configs/maas/<model>/f14_agent.yaml
```

- 仅统计 follow-up turns（turn_index > 0）作为事件来源
- 主图：backward-looking CDF，展示后续轮与历史请求之间的块复用时间间隔
- 插图（inset）：forward-looking，展示后续轮的块被未来请求复用的比例
- 与 F13 的区别：仅关注会话内的后续轮，能单独量化 Agent 多轮上下文的 KV 复用效率

**Config 示例：**
```yaml
input_file: data/internal/<model>/requests.jsonl
output_dir: outputs/maas/<model>/f14_agent
trace_name: <model>
block_size: 128
x_axis_max_minutes: 60
```

---

## 不适用于 Agent 数据的分析

| 分析 | 原因 |
|------|------|
| `top_ngrams` / `benchmark_index` | 可运行但意义有限——Agent prompt 长度极长（50K 字符），ngram 频次低，块数量大，benchmark 结果受数据规模主导 |
| `generate_f14.py`（TraceA 版） | 依赖 `hash_ids` 字段 + `parent_chat_id`，Agent JSONL 不具备，请使用 `generate_f14_agent.py` |

---

## 技术说明

### 自动索引选择

Agent 数据平均 prompt 长度 ~50K 字符，`block_size=128` 时平均 **~390 blocks/request**，远超阈值 256。
`replay()` 会自动选用 `RadixTrieIndex`，无需手动配置。

### 内存估算

| 数据规模 | 估算内存 |
|---------|---------|
| 10K 请求 × 390 blocks | ~30M blocks，约 500 MB RAM |
| 50K 请求 × 390 blocks | ~195M blocks，约 2.5 GB RAM |

实验室机器内存充足，可直接运行全量数据。如需按时间窗口分片，在 YAML 中通过时间戳过滤后传入 loader 即可。

### turn_index 与 block 分析的关系

`turn_index` 仅用于 F9 / F10 / F14 的会话结构分析，不影响 block 哈希计算。
F4 / F13 / E1 / E1B / E5 / reuse_rank 将每一轮 turn 视为独立请求，与单轮 MaaS 数据处理方式完全一致。
