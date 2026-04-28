# block_prefix_analyzer 能力概览

**版本**：v4.0（2026-04-28）
**定位**：KV cache prefix 复用分析工具，从离线理论分析到在线 API 实测，覆盖 KV block 生命周期研究的完整链路。

---

## 一、支持的输入数据格式

| 格式 | 数据来源 | 核心字段 | 说明 |
|---|---|---|---|
| **格式 A** TraceA 公开数据集 | Qwen 公开 trace | `chat_id`, `timestamp`, `hash_ids` | block hash 已预计算（block_size=16），直接加载 |
| **格式 B** 业务 JSONL | 生产系统 CSV → 转换 | `user_id`, `request_id`, `timestamp`, `raw_prompt` | 由 `convert_csv_to_jsonl.py` 从 CSV 生成；block hash 在加载时由 CharTokenizer + SimpleBlockBuilder 实时计算 |
| **格式 C** API 在线实验 | 运行中的 vLLM 实例 | vLLM `/metrics` Prometheus 端点 + 请求回放 | 不依赖离线数据文件；需要部署好的 vLLM 服务可访问 |

**格式 B 的数据准备命令**：

```bash
# 标准 CSV（列顺序：请求ID / 用户ID / 输入prompt / timestamp）
python scripts/convert_csv_to_jsonl.py \
    --input <file>.csv --output data/internal/<model>/requests.jsonl \
    --col-user-id 1 --col-request-id 0 --col-raw-prompt 2 --col-timestamp 3 \
    --has-header --encoding utf-8-sig

# 初始化标准分析配置（一次性）
python scripts/init_maas_configs.py <model_slug> "<展示名>"
```

---

## 二、离线分析能力（格式 A / B，无需 API）

---

### F4：请求流量与 KV Block 复用量的时间序列

**适用格式**：A（`generate_f4.py`）/ B（`generate_f4_business.py`）
**目的**：观察请求量与 KV block 复用量在一天内的波动规律，判断高峰时段的 cache 命中状况。

| 轴 | 定义 |
|---|---|
| 横轴 | 时间（小时，相对 trace 起始点）|
| 纵轴 | 每分钟 bin 的 total_blocks / hit_blocks，均已归一化（÷ bin 均值）|

**两条曲线**：Total（蓝）= 所有 block 请求量；Hit（橙）= 其中命中历史 cache 的 block 数。

**hit_metric 两种口径**：

| 值 | 含义 |
|---|---|
| `content_prefix_reuse` | 从请求开头连续命中的 block 数（= 无限容量 vLLM APC 等价命中） |
| `content_block_reuse` | 任意位置命中历史 block 的数（宽口径，非 vLLM APC 等价） |

**可得结论**：
- 高峰时段 Hit/Total 比值是否下降 → 反映并发压力下 cache 是否被"冲散"
- 昼夜周期是否显著 → 指导保活策略的时间窗口设置

---

### F9：会话轮数分布 CDF

**适用格式**：A / B（`generate_f9.py`）
**目的**：了解用户会话由几轮对话组成，判断多轮复用场景的规模。

| 轴 | 定义 |
|---|---|
| 横轴 | 会话轮数（单会话包含的请求数）|
| 纵轴 | CDF（累积占比，0~1）|

**可得结论**：
- p50/p90 会话轮数 → 典型多轮 session 长度
- 单轮会话占比 → 决定 F13（单轮）vs F14（多轮）的流量权重
- 若 p90 轮数很低 → 多轮复用场景稀少，cache 命中主要来自系统 prompt / hot prompt

---

### F10：用户平均/方差轮数分布（Lorenz 图）

**适用格式**：A / B（`generate_f10.py`）
**目的**：了解"重度多轮用户"和"轻度用户"的分布，识别高价值用户群体。

| 轴 | 定义 |
|---|---|
| 横轴 | 用户排序（按均值/标准差升序，序号 1…N）|
| 左纵轴 | 该用户均值/标准差轮数（归一化）|
| 右纵轴 | 累积占比（类 Lorenz 曲线）|

**可得结论**：
- 头部 X% 用户贡献了 Y% 的多轮请求 → Gini 系数 / 集中度
- 方差高的用户 → 行为不稳定，难以用静态策略预热 cache

---

### F13：单轮请求的 KV Block 复用时间 CDF

**适用格式**：A（`generate_f13.py` / `generate_f13_strict.py`）/ B（`generate_f13_business.py`）
**目的**：对于单轮请求（非多轮跟随），当一个 KV block 在当前请求中被复用时，距上次见到该 block 已过了多久？

| 轴 | 定义 |
|---|---|
| 横轴 | 复用时间间隔（分钟，截止 x_axis_max）|
| 纵轴（主图）| 复用事件的 CDF（0~1）|
| 纵轴（inset）| 各类型请求中有复用的比例（Text / File / Multimedia / Search）|

**event_definition 两种口径**：同 F4 的 `content_prefix_reuse` / `content_block_reuse`。

**可得结论**：
- **p50 / p80 / p95 reuse time** → vLLM block 保活窗口（`--kv-cache-evict-interval`）应设置多大才能覆盖 80% 的复用机会
- 若 p80 reuse time = 5 分钟 → block 需在空闲队列中存活 5 分钟才能让 80% 的请求受益
- 各类型请求的复用比例 → Text 类通常高（系统 prompt 共享），Multimedia 类低

---

### F14：多轮跟随请求的 KV Block 复用时间 CDF

**适用格式**：A（`generate_f14.py`）/ B Agent 数据（`generate_f14_agent.py`）
**目的**：对于多轮会话的跟随请求（非第一轮），复用前序轮次 KV block 时，等了多久？

| 轴 | 定义 |
|---|---|
| 横轴 | 多轮复用时间间隔（分钟）|
| 纵轴（主图）| CDF（跟随请求中有复用的比例）|
| inset | 跟随请求中"可被未来任意请求复用"的比例（前向可复用率）|

**hit_metric 两种口径**（Agent 数据）：

| 值 | 含义 |
|---|---|
| `content_prefix_reuse` | 从请求开头连续命中前序轮次的 block（= vLLM APC 等价） |
| `content_block_reuse` | 任意位置复用（宽口径）|

**可得结论**：
- 多轮请求的复用时间分布 vs 单轮（F13）→ 多轮通常更短（同用户连续交互）
- forward_reusable_count / total → 当前轮次的 KV block 在未来被复用的概率（= cache 预热价值）
- 若 forward_reusable > 60% → 当前请求的 KV block 值得长期保活

---

### E1：用户级别理想 prefix cache 命中率分布

**适用格式**：B（`generate_user_hit_rate.py`）
**目的**：对每个用户，计算其所有请求的 prefix reuse 命中率，观察用户间的分布差异。

| 轴 | 定义 |
|---|---|
| 横轴 | prefix cache 命中率（0~1，按 threshold 分桶）|
| 纵轴 | 用户数（频数）|

**同时输出 4 条曲线**（block_size = 16 / 32 / 64 / 128），直观展示 block 粒度对命中率的影响。

**可得结论**：
- 命中率分布是否双峰 → 存在"高命中用户群"（固定系统 prompt / 重复请求）和"低命中用户群"
- block_size 增大对命中率的影响 → 粒度越粗，尾 block 截断越多，命中率通常越低
- 高命中率用户比例 → 决定 prefix cache 投资回报率的用户覆盖面

---

### E1-B：KV Block 命中贡献的 Lorenz 偏度图

**适用格式**：B（`generate_skewness.py`）
**目的**：哪些用户贡献了绝大多数的 KV cache 命中？请求量和命中量的集中度是否一致？

| 轴 | 定义 |
|---|---|
| 横轴 | 用户排序（升序）|
| 左纵轴 | 该用户的归一化命中量 / 请求量 |
| 右纵轴 | 累积占比（Lorenz 曲线）|

**可得结论**：
- Gini 系数 → 命中集中度；若 Gini > 0.8，少数用户贡献了绝大多数命中
- 命中 Lorenz vs 请求量 Lorenz 的差异 → 若命中比请求更集中，说明高频用户有强复用
- Top-10% 用户覆盖 X% 命中 → 路由策略只需优先保障头部用户即可获得大部分收益

---

### E5：高频复用 Block Sequence 的原文还原

**适用格式**：B（`generate_e5_block_text.py`）
**目的**：找出最常被复用的 block 序列（n-gram），并还原为原始文本，理解 cache 中"最值钱"的内容是什么。

| 输出 | 定义 |
|---|---|
| rank | 按出现次数降序 |
| count | 该 n-gram 出现在多少条请求中 |
| pct | 占所有请求的比例 |
| text | 还原的原始字符文本（截断至 max_chars） |

**可得结论**：
- Top-1 n-gram 通常是系统 prompt → 验证系统 prompt 是否已被 cache 充分复用
- 若 Top n-gram 是特定业务模板 → 可作为"热 block"显式预热（proactive warming）
- n-gram 长度 × block_size = 覆盖的 token 数 → 估算单次命中节省的计算量

---

### reuse_rank：每条请求的 prefix 命中 block 数排行

**适用格式**：A / B（`generate_reuse_rank_business.py`）
**目的**：将所有请求按"可复用的 prefix block 数"降序排列，观察复用量的分布形态。

| 轴 | 定义 |
|---|---|
| 横轴 | 请求排名（1 = 命中最多，N = 命中最少）|
| 纵轴 | 该请求可复用的 prefix block 数 |

**分群输出**：单轮 vs 多轮跟随，形态通常不同（多轮命中量更集中在高值区）。

**可得结论**：
- 分布是否"长尾"→ 少数请求贡献了大量命中（典型重复请求 / 系统 prompt）
- 中位数命中 block 数 → 平均每条请求节省多少 prefill 计算
- 多轮 vs 单轮的命中量差异 → 多轮缓存价值是否显著高于单轮

---

### top_ngrams：最大公共 Block Sequence 频次统计

**适用格式**：A / B（`generate_top_ngrams_business.py`）
**目的**：找出所有请求中出现频率最高的最长公共 block 序列。

| 输出 | 定义 |
|---|---|
| rank | 按 count 降序 |
| blocks | block_id 序列 |
| count | 出现在多少条请求中 |
| pct | 占比 |

**可得结论**：
- Top n-gram 的 count/pct 高 → 存在强烈的系统 prompt 共享，cache 命中率本底高
- 多个 n-gram 的长度相近 → 不同用户群使用不同系统 prompt

---

### common_prefix：最长公共前缀的覆盖度分析

**适用格式**：B（`generate_common_prefix.py`）
**目的**：在所有请求中，找到被最多请求共享的连续前缀（从请求起始位置开始），分析其覆盖广度。

| 轴 | 定义 |
|---|---|
| 横轴 | 公共前缀的 block 位置（0 → max_blocks）|
| 纵轴 | 在该位置仍保持一致的请求比例（coverage_pct）|

**可得结论**：
- 曲线急剧下降的位置 → 公共前缀在此处结束，之后请求开始分化
- 若前 20 个 block（= 2560 token，block_size=128）覆盖率 > 80% → 系统 prompt 约 2560 token，cache 效果极佳
- min_count 阈值以上的最大前缀长度 → 可作为 proactive warming 的目标长度

---

### benchmark_index：TrieIndex vs RadixTrieIndex 性能对比

**适用格式**：B（`benchmark_index.py`）
**目的**：在实际数据上对比两种前缀索引的内存占用和速度，验证 RadixTrieIndex 在长上下文下的压缩优势。

| 输出 | 定义 |
|---|---|
| node_count | 索引中的节点数 |
| memory_mb | 内存占用（MB）|
| elapsed_s | 全量 replay 耗时（秒）|
| reuse_rate_match | 两种实现的命中率是否一致（正确性校验）|

**可得结论**：
- RadixTrieIndex 节点数 / TrieIndex 节点数 = 压缩比 → 长上下文下压缩比通常 > 10×
- 若两者 reuse_rate_match = True → 两种实现等价，RadixTrieIndex 可安全替换
- 速度差异 → 决定大规模数据集的分析可行性

---

## 三、在线实验能力（格式 C，需要 vLLM API）

---

### 模块二：生产时序回放（replay_production_benchmark.py）

**目的**：用真实生产 CSV 的前 N 条数据，按原始 timestamp 顺序回放请求（同 timestamp = 同批并发），在真实节奏下测量 vLLM 的 KV cache 行为。

**输入**：生产 CSV（列：请求ID / 用户ID / raw_prompt / timestamp）

**核心输出文件**：`metric_timeseries.csv`（每批次一行）

| 指标 | 轴/定义 |
|---|---|
| `prefix_cache_hit_rate` vs `batch_idx` | 命中率随时间演化；反映 warm-up 过程和驱逐压力出现时刻 |
| `idle_before_evict_mean_s`（Ascend 可用）| block 进入空闲队列到被驱逐的平均时间 = **实际保活窗口** |
| `reuse_gap_mean_s`（Ascend 可用）| block 被复用时已等待多久；若 > idle_before_evict → block 等不到复用就被驱逐 |
| `gpu_cache_usage_perc` vs `batch_idx` | cache 饱和度趋势；超过 0.9 开始强制驱逐 |
| `blocks_removed_delta` 首次 > 0 的 batch | 驱逐开始点；之前为纯 warm-up 阶段 |

**可得结论**：
- warm-up 需要多少批次才能达到稳定命中率
- 真实生产节奏下 cache 是否被填满，以及填满后命中率是否下跌
- 实际保活窗口（idle_before_evict）与 F13 p80 reuse time 的差距 → 若前者 < 后者，说明 block 等不到复用

---

### 模块三：受控并发扫描（run_kv_cache_benchmark.py + analyze_benchmark_results.py）

**目的**：固定使用从生产 CSV 分层采样的 100 条 prompt，人工控制并发数（1/2/4/8/16），找到 KV cache 命中率开始下跌的 inflection point。

**输入**：`sampled_prompts.jsonl`（由 `sample_prompts_for_benchmark.py` 从生产 CSV 生成）

**核心输出图**：

**图 3a：命中率 × block 存活时间双轴图**

| 轴 | 定义 |
|---|---|
| 横轴 | 并发数（从高到低：16 → 1）|
| 左纵轴 | `prefix_cache_hit_rate`（含误差棒，3 轮均值）|
| 右纵轴 | `idle_before_evict_mean_s` 或 `removed_over_stored`（驱逐压力比）|
| 参考线 | F13 p80 reuse gap（右轴水平虚线）；inflection point（红色垂直虚线）|

**图 3b：驱逐量分解**

| 轴 | 定义 |
|---|---|
| 横轴 | 并发数 |
| 纵轴 | blocks_stored_delta（绿）vs blocks_removed_delta（红）柱状图 |

**可得结论**：
- inflection point 并发数 → 当前硬件配置的 KV cache 保活能力边界
- 右轴 < F13 p80 reuse gap 的起始并发 → block 开始大量在复用前被驱逐的临界点
- 横向对比两个模型的 inflection point → 哪个模型的 cache 策略更有优化空间

---

## 四、分析链路总图

```
生产 CSV
  │
  ├─ convert_csv_to_jsonl.py ──► 业务 JSONL
  │                                │
  │              ┌─────────────────┼─────────────────────────────────┐
  │              │                 │                                 │
  │          F4  │  流量×命中时序  F9  会话分布    E1  用户命中率分布  │
  │          F13 │  单轮复用时间   F10 用户轮数    E1B 命中集中度     │
  │          F14 │  多轮复用时间   reuse_rank      E5  高频原文还原   │
  │              │                 top_ngrams      common_prefix      │
  │              │                 benchmark_index                    │
  │              └─────────────────────────────────────────────────┘
  │
  ├─ sample_prompts_for_benchmark.py ──► 采样 JSONL
  │                                         │
  │                            run_kv_cache_benchmark.py (模块三)
  │                                         │
  │                            analyze_benchmark_results.py
  │
  └─ 直接用原始 CSV (前200行)
                │
                replay_production_benchmark.py (模块二)
                │
                analyze_production_replay.py（待建）
```

---

## 五、已完成 vs 计划中

| 状态 | 分析模块 | 对应脚本 |
|---|---|---|
| ✅ 已完成 | F4、F9、F10、F13、F14 | generate_f4/f9/f10/f13/f14_*.py |
| ✅ 已完成 | E1、E1-B、E5 | generate_user_hit_rate/skewness/e5_block_text.py |
| ✅ 已完成 | reuse_rank、top_ngrams、common_prefix | generate_reuse_rank/top_ngrams/common_prefix_business.py |
| ✅ 已完成 | 模块二骨架 | replay_production_benchmark.py |
| ✅ 已完成 | 模块三骨架 | run_kv_cache_benchmark.py + analyze_benchmark_results.py |
| 📋 V4 计划 | reuse_distance（模块一）| generate_reuse_distance.py（待建）|
| 📋 V4 计划 | 模块二分析图 | analyze_production_replay.py（待建）|
| 📋 V4 计划 | 模块三图 3a 双轴 | analyze_benchmark_results.py（待补充）|
| 📋 V4 计划 | 联合分析 | analyze_joint.py（待建）|
