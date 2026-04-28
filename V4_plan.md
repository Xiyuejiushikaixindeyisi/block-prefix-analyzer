# V4 实验计划：KV Block 生命周期与保活边界研究

本阶段从离线理论分析延伸至在线 API 实测，三个模块相互验证，
最终回答核心问题：**当前硬件配置下，KV block 的实际保活能力是否足以覆盖生产环境中的复用需求？**

---

## 背景与动机

V3 阶段完成了对 6 个生产模型的 prefix cache 命中率静态分析，得到 F13（reuse time CDF）和 F14（多轮复用 CDF）。
V4 在此基础上新增两个维度：

- **空间维度**：两次复用之间插入了多少 unique KV block（reuse_distance_blocks），决定有限容量 LRU 能否保住该 block
- **实测验证**：通过裸机部署的专属 API，在真实和受控两种场景下测量 vLLM 的实际 KV block 行为

---

## 前置准备

### P1：available_cache_blocks 估算

模块一的阈值线、模块三的解读均依赖此参数。估算公式：

```
available_cache_blocks =
    (GPU总显存GB × gpu_memory_utilization - 模型权重显存GB)
    ÷ (block_size × num_layers × 2 × num_kv_heads × head_dim × dtype_bytes)
    × num_GPUs
```

对于 Qwen3-27B 910B4×8卡（参考值，需用实际部署参数替换）：

| 参数 | 参考值 |
|---|---|
| GPU总显存 | 64 GB × 8 = 512 GB |
| 模型权重显存（BF16） | ~54 GB |
| gpu_memory_utilization | 0.90 |
| block_size（Ascend prefix caching 强制）| 128 tokens |
| num_layers | 62 |
| num_kv_heads | 8 |
| head_dim | 128 |
| dtype_bytes（BF16）| 2 |

```
available ≈ (512×0.90 - 54) / (128 × 62 × 2 × 8 × 128 × 2 / 1e9) × 8
```

**实际值应在部署后通过 `/metrics` 中 `gpu_cache_usage_perc` 的分母反推，或查 vLLM 启动日志中的 `num_gpu_blocks`。**

### P2：指标可用性探查

部署模型后，执行以下命令确认 `/metrics` 实际暴露的指标：

```bash
curl -s http://<YOUR_ENDPOINT>/metrics | grep -E "^vllm:" | sed 's/{.*//' | sort -u
```

关键指标分级：

| 级别 | 指标 | 说明 |
|---|---|---|
| 必须可用 | `prefix_cache_hit_rate`、`gpu_cache_usage_perc`、`num_preemptions_total` | 标准 vLLM |
| 重要 | `num_blocks_stored_total`、`num_blocks_removed_total` | 版本依赖 |
| 高价值（Ascend） | `kv_block_idle_before_evict_seconds`、`kv_block_lifetime_seconds`、`kv_block_reuse_gap_seconds` | 若不可用则降级 |

### P3：数据特征预分析（在任何实验前执行）

从目标 CSV 前 200 行统计以下内容，用于指导实验参数设置：

- timestamp 分布：unique timestamp 数量，最大/最小/平均 batch size
- prompt 长度分布：min/p25/p50/p75/max（决定 block 占用量）
- 按 block_size=128 估算每条请求的 block 数：`prompt_length // 128`

**改进说明**：若前 200 行中 timestamp 全部唯一（每条请求 timestamp 不同），模块二退化为串行回放；若大量请求共享同一 timestamp，单批次并发很高。两种极端情况都需要在预分析阶段发现并调整实验设计。

---

## 模块一：理论 reuse_distance_blocks 分析（纯离线）

### 研究问题

在无限 cache 容量下，每次 prefix 复用发生时，距上次看见该 block 之间有多少 unique KV block 被插入？
若 `reuse_distance_blocks > available_cache_blocks`，则该 block 在 LRU 下大概率已被驱逐，理论上可复用的请求实际上也会 miss。

### 算法

```
回放数据集（时间序）
维护：
  last_seen_time[block_id] → 该 block 上次出现的时间戳
  global_block_log          → 按时间有序的 (time, set(block_ids)) 列表

对每条有 content_prefix_reuse_blocks > 0 的请求（时间 T2）：
  eligible_blocks = record.block_ids[:content_prefix_reuse_blocks]
  T1 = min(last_seen_time[b] for b in eligible_blocks)

  reuse_distance_blocks = |⋃ block_ids of all requests in interval (T1, T2)|
  reuse_time_s          = T2 - T1

  记录：(request_id, T1, T2, reuse_time_s, reuse_distance_blocks,
          prefix_len_blocks, content_prefix_reuse_blocks)
```

**注意**：`reuse_distance_blocks` 是 block 空间中的距离，与 LRU 驱逐直接对应；`reuse_time_s` 是时间距离（F13/F14 已有），两者结合才能完整描述"保活需求"。

### 输出图表

**图 1a：reuse_distance_blocks 降序分布**

```
X 轴：可复用 request 按 reuse_distance_blocks 降序排列（序号 1…N）
Y 轴：reuse_distance_blocks（对数刻度）
红色水平虚线：available_cache_blocks（用户指定）
红线以上区域 = LRU 下无法命中
标注：「LRU evicted: XX% of theoretically reusable requests (capacity=N blocks)」
```

**图 1b：reuse_distance_blocks 的 CDF**

```
X 轴：reuse_distance_blocks 值（0 → max）
Y 轴：CDF（累积比例）
垂直虚线：available_cache_blocks
标注：CDF 在阈值处的值 = LRU 下能命中的比例
```

**改进**：同时按请求类型分层（如 Agent vs TEXT，或按 prompt 长度 quartile），观察不同类型请求的 reuse_distance 分布差异——长 prompt 的 block 数多，单条请求就能大幅"刷新"cache。

### 新增文件

- `src/block_prefix_analyzer/analysis/reuse_distance.py`
- `src/block_prefix_analyzer/plotting/reuse_distance.py`
- `scripts/generate_reuse_distance.py`（YAML 驱动，新增 config 参数 `available_cache_blocks`）

### 与 F13/F14 的关系

| 分析 | 维度 | 回答的问题 |
|---|---|---|
| F13 reuse time CDF | 时间 | block 被复用时已经等了多久 |
| F14 multi-turn CDF | 时间 | 多轮对话的复用时延分布 |
| **模块一 reuse_distance** | **空间** | **block 被复用前有多少竞争者"挤占"了 cache** |

三者联合才能回答：**"需要多大的 cache，才能让 80% 的理论复用机会真正命中？"**

---

## 模块二：生产时序回放 × API 实测

### 研究问题

在真实生产请求的时序节奏下（同 timestamp = 同并发批次），vLLM 实际的 KV block 存活时间和命中率如何随时间演化？warm-up 需要多少批次？驱逐压力在哪个时间点首次出现？

### 数据准备

```
CSV 前 200 行
  → 预分析：统计 timestamp 分布、batch size 分布
  → replay_production_benchmark.py（--time-scale 0，不加延迟）
  → metric_timeseries.csv
```

**改进**：

1. **定义 warm-up 边界**：以 `gpu_cache_usage_perc` 首次超过 0.5 的 batch 作为 warm-up 结束点，warm-up 前的数据不纳入稳定期统计。
2. **降低 metrics scrape 频率**：若 batch 间隔极短（< 1s），改为每 5 批次 scrape 一次，避免 scrape 本身干扰结果。
3. **记录 scrape 耗时**：每次 scrape 记录实际耗时，发现异常时在 timeseries 中标记。

### 输出图表

**图 2a：prefix_cache_hit_rate 随 batch_idx 变化**

```
X 轴：batch_idx（或 experiment_elapsed_s）
Y 轴：prefix_cache_hit_rate（0~1）
特征标注：
  ① warm-up 结束点（gpu_cache_usage > 0.5）
  ② blocks_removed_delta 首次 > 0 的点（驱逐开始）
  ③ 命中率峰值和最终稳定值
```

**图 2b：block 存活时间对比（Ascend 可用时）**

```
X 轴：batch_idx
左 Y 轴：idle_before_evict_mean_s（实际保活时间，蓝色）
右 Y 轴：reuse_gap_mean_s（block 等待复用的时间，橙色）
关键线：若 reuse_gap > idle_before_evict → block 等不到复用就被驱逐
```

降级方案（无 Ascend 专有指标时）：

```
左 Y 轴：prefix_cache_hit_rate
右 Y 轴：blocks_removed_delta / blocks_stored_delta（驱逐压力比）
```

**图 2c：GPU cache 饱和度**

```
X 轴：batch_idx
Y 轴：gpu_cache_usage_perc
红色虚线：0.90 阈值（超过后强制驱逐）
```

### 新增文件

- `scripts/analyze_production_replay.py`（读取 metric_timeseries.csv，生成图 2a/2b/2c）

---

## 模块三：受控并发扫描 × API 实测

### 研究问题

固定使用真实生产 prompt，人工控制并发数（1/2/4/8/16），KV cache 命中率和 block 存活时间如何随并发压力变化？inflection point 在哪？

### 数据准备

```
CSV  →  sample_prompts_for_benchmark.py
      →  sampled_prompts.jsonl（100 条，4 quartile 均衡采样）
      →  run_kv_cache_benchmark.py（并发 1/2/4/8/16，各 3 轮取均值）
      →  run_results.csv
```

**关键改进**：

1. **固定 prompt 集合**：100 条 prompt 在所有并发级别下完全一致，确保比较公平。
2. **cache 预热**：每个并发级别正式测量前，先用并发=1 跑一轮相同 prompt 作为 warm-up，再开始计时采集。
3. **并发顺序**：从低到高（1→16），确保每个并发级别开始时 cache 处于"已预热"状态，测量的是稳态性能而非冷启动。
4. **preemption 保护**：若 `num_preemptions_delta > 0`，在结果中标记该并发级别已超出 `max_num_seqs` 上限，数据需谨慎解读。

### 输出图表

**图 3a：命中率 × block 存活时间双轴图（新增）**

```
X 轴：并发数（16 → 1，从高到低，右侧压力大）
左 Y 轴：prefix_cache_hit_rate（蓝色，折线+误差棒）
右 Y 轴：idle_before_evict_mean_s 或 block_lifetime_mean_s（橙色，折线+误差棒）
         若 Ascend 指标不可用，右 Y 轴改为 removed_over_stored（驱逐压力比）
特征标注：inflection point（红色虚线，命中率开始下跌处）
          F13 p80 reuse gap（水平参考线，右 Y 轴）
图例说明：「右轴 < F13 p80 reuse gap → blocks evicted before 80% of reuse events」
```

**图 3b：驱逐压力分解**

```
X 轴：并发数
Y 轴：blocks_stored_delta（绿色）和 blocks_removed_delta（红色），柱状图
      两者差值 = 净新增 cache block 量
```

**图 3c：延迟影响**（已在 analyze_benchmark_results.py 中，确认输出即可）

### 现有文件

- `scripts/run_kv_cache_benchmark.py`（已存在，使用 chat/completions 格式）
- `scripts/analyze_benchmark_results.py`（已存在，补充图 3a）

---

## 模块间联合分析

### 联合图：理论预测 vs 实测命中率

```
X 轴：并发数（或对应的 blocks_in_flight 估算）
左 Y 轴：
  曲线 A（模块一）：available_cache_blocks / median_reuse_distance_blocks
              = 理论上 LRU 可命中的比例上界
  曲线 B（模块三）：实测 prefix_cache_hit_rate
右 Y 轴：
  曲线 C（模块二）：真实时序回放的稳定期 prefix_cache_hit_rate（水平参考带）

如果 B ≈ A：LRU 近似成立，模块一的理论分析可信
如果 B > A：prompt 间存在高度相关性（Agent 多轮、系统 prompt 复用），
             LRU 模型低估了实际命中率
如果 B << A：可能存在 cache thrashing，或 available_cache_blocks 估算偏高
```

### 最终结论框架

| 问题 | 数据来源 | 结论形式 |
|---|---|---|
| 理论上多少复用会被 LRU 驱逐？| 模块一 | XX% of reusable requests have reuse_distance > N blocks |
| 生产节奏下实际命中率是多少？| 模块二 | 稳定期 hit_rate = XX%，warm-up = YY batches |
| 并发多高时命中率开始下跌？| 模块三 | inflection @ concurrency = Z |
| block 存活时间够用吗？| 模块二/三 | idle_before_evict vs F13 p80 reuse gap 对比 |
| 理论与实测是否一致？| 联合分析 | 误差 < 10% → 理论模型可用于预测新场景 |

---

## 执行顺序

```
阶段 0（立即可做，无需 API）
  └─ P3 数据特征预分析（200行timestamp/batch分布）
  └─ P1 available_cache_blocks 估算
  └─ 模块一：generate_reuse_distance.py

阶段 1（API 部署完成后）
  └─ P2 指标可用性探查（curl /metrics）
  └─ 模块二：replay_production_benchmark.py → analyze_production_replay.py
  └─ 模块三：run_kv_cache_benchmark.py → analyze_benchmark_results.py（补充图 3a）

阶段 2（数据回收后）
  └─ 联合分析：理论预测 vs 实测对比图
```

---

## 代码实施清单

| 状态 | 文件 | 对应模块 |
|---|---|---|
| 待建 | `src/block_prefix_analyzer/analysis/reuse_distance.py` | 模块一 |
| 待建 | `src/block_prefix_analyzer/plotting/reuse_distance.py` | 模块一 |
| 待建 | `scripts/generate_reuse_distance.py` | 模块一 |
| 待建 | `scripts/analyze_production_replay.py` | 模块二 |
| 已有 | `scripts/replay_production_benchmark.py` | 模块二 |
| 已有 | `scripts/run_kv_cache_benchmark.py` | 模块三 |
| 待补充图3a | `scripts/analyze_benchmark_results.py` | 模块三 |
| 待建 | `scripts/analyze_joint.py` | 联合分析 |
