# block-prefix-analyzer

离线分析 LLM 请求 trace 中 **block 级前缀复用** 的工具。

当前状态：**V1.0 — 主链路完整，145 个测试全部通过**。

## 设计目标
- 给定按时间排序的请求流，回放并输出：
  - 每个请求可复用的最长 **block 前缀**
  - 在"无限容量前缀缓存"假设下的 **理想命中率**（prefix-aware）
  - 在"只要出现过即可复用"假设下的 **block 级可复用率**
  - 可选 token 级前缀命中、reuse time、block lifespan 等衍生指标
- 与 vLLM / vLLM-Ascend 的语义逐步对齐，但核心分析器不依赖在线框架
- 离线、确定性、可单元测试、模块可替换

## 版本里程碑
- **V1（当前）**：只处理已包含 `block_hash_ids` 的输入；顺序回放；简单 Trie；基础报表
- **V2**：接入 tokenizer / chat template / block hash 复现，与框架对齐
- **V3**：性能与规模（radix 压缩、磁盘索引、并行预处理 等）

## 目录结构
```
src/block_prefix_analyzer/
  types.py          # 规范化数据模型 RequestRecord
  replay.py         # 时间序回放引擎（先算指标、再插入）
  metrics.py        # 指标定义与计算
  index/
    base.py         # PrefixIndex 抽象协议
    trie.py         # 简单 Trie 实现
  io/
    jsonl_loader.py # JSONL trace 读取
  reports/
    summary.py      # 汇总统计
tests/              # 与 src 模块一一对应的测试文件
```

## 开发环境
- Windows + WSL
- Python ≥ 3.10

## 快速开始
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## 术语规范（V1）

本节定义分析器内部使用的规范术语，并标注与外部 trace 字段名的映射关系。
代码层名称（`code`）优先以模块实际字段名为准；外部 trace 字段名（`trace`）仅在数据加载层转换。

---

### 基础对象

| 规范术语 | 含义 | 代码层字段 / 类型 |
|---|---|---|
| **KV block** | KV 缓存的最小分配单元；一条 block hash ID 对应一个 KV block | `BlockId`（`int \| str`） |
| **hash_ids** | 外部 trace 中记录请求所占 block 序列的字段名 | 加载时映射为 `RequestRecord.block_ids` |
| **chat_id** | 外部 trace 中标识一次请求的唯一 ID | 加载时映射为 `RequestRecord.request_id` |
| **请求（request）** | 一次完整的推理请求；由 `RequestRecord` 表示 | `RequestRecord` dataclass |

---

### 请求分类（V2+，当前未实现）

以下概念在 V1 数据模型中**不存在**，字段暂未引入。

| 规范术语 | 含义 |
|---|---|
| **session** | 通过 `chat_id` / `parent_chat_id` 串联的多轮对话链 |
| **root request** | `parent_chat_id == -1` 的首轮请求 |
| **follow-up request** | `parent_chat_id != -1` 的续轮请求 |
| **request category** | 由 `"{type}-{turn}"` 构成的请求类型标签，用于分组分析 |

---

### 复用事件

| 规范术语 | 含义 | V1 实现状态 |
|---|---|---|
| **reuse event（复用事件）** | 某个 block hash 在更晚到达的请求中再次出现 | 由 `seen_blocks` 集合隐式追踪；不单独输出事件列表 |
| **可复用 block（reusable block）** | 该 block hash 在**任意更早请求**中出现过（最宽口径） | `PerRequestResult.reusable_block_count` |
| **前缀命中 block（prefix-hit block）** | 从请求 `block_ids[0]` 开始**连续**命中 Trie 的 block（第一次 miss 后截断） | `PerRequestResult.prefix_hit_blocks` |

> 两种口径的区别：`reusable_block_count ≥ prefix_hit_blocks`，见 `with_empty` fixture 中 `user` 请求的示例（前缀未命中但有 2 个可复用 block）。

---

### 聚合指标

| 规范术语 | 定义 | 代码字段名 | 口径 |
|---|---|---|---|
| **overall_block_reusable_ratio** | `Σ reusable_block_count / Σ total_blocks`（分母仅含非空请求） | `overall_block_level_reusable_ratio` | micro |
| **overall_prefix_hit_rate** | `Σ prefix_hit_blocks / Σ total_blocks`（分母仅含非空请求） | `overall_prefix_hit_rate` | micro |

> **注意**：规范术语 `overall_block_reusable_ratio` 在代码中对应字段名为 `overall_block_level_reusable_ratio`（多一个 `_level_`），含义相同。V2 重构时可统一命名。

---

### 时间衍生指标（V2+，当前未实现）

| 规范术语 | 定义 |
|---|---|
| **reuse_time** | `current.timestamp − last_seen.timestamp`（上次出现到本次被复用的时间差） |
| **KV block lifespan** | `last_reuse_ts − first_seen_ts`（block 从首次出现到最后一次被复用的时间跨度） |
| **next-turn frequency** | 给定一个 root request，发生 follow-up 的概率（需 session 概念支撑） |

---

### 回放顺序不变量

所有指标的计算都依赖以下严格顺序（禁止打乱，否则产生 self-hit）：

```
sort_records(timestamp, arrival_index)
  → for each record:
      prefix_hit  = index.longest_prefix_match(record.block_ids)   # query
      reusable    = count(bid in seen_blocks for bid in block_ids)  # query
      yield PerRequestResult(...)                                    # emit
      index.insert(record.block_ids)                                 # insert
      seen_blocks.update(record.block_ids)                           # update
```

---

---

## 论文复现与实验管理规范

本项目图表分为两条完全隔离的线，**任何情况下不得混用**：

| | 论文复现线（paper_repro） | 项目实验线（experiments） |
|---|---|---|
| 目标 | 复现论文 *KVCache in the Wild* 中的指定图表 | 项目自定义研究与分析 |
| 数据 | 只能使用公开数据 | 公开数据 + 内部数据均可 |
| 指标定义 | 优先使用论文原定义，适配版本须明确标注 | 可使用项目扩展指标 |
| 图名规则 | 必须带论文编号（F4 / F13 / F14 / F15） | 禁止使用论文编号 |
| 目录 | `outputs/paper_repro/` | `outputs/experiments/` |

---

### 当前论文复现范围

当前 `paper_repro` 线仅包含以下四张图：

| 图号 | 语义 | 主指标 | 复现状态 |
|---|---|---|---|
| **F4** | ideal overall hit ratio（全局 block 可复用率） | `overall_block_reusable_ratio` | V1 可计算（无 session 过滤） |
| **F13** | single-turn 场景 reuse time 分布 | `reuse_time`（last_seen 口径） | V2+（需 `parent_chat_id`） |
| **F14** | multi-turn / follow-up 场景 reuse time 分布 | `reuse_time`（last_seen 口径） | V2+（需 session 链） |
| **F15** | 不同 request category 的 reuse time 分布 | `reuse_time` by `"{type}-{turn}"` | V2+（需 `type` + `turn`） |

> F13 / F14 / F15 依赖 `parent_chat_id / type / turn` 字段，V1 尚不支持，需在 V2 引入 session 解析后实现。

---

### 输入数据格式（公开 TraceA）

论文复现统一使用公开数据仓库中的 TraceA 文件，JSONL 格式，每条记录形如：

```json
{
  "chat_id": 159,
  "parent_chat_id": 55,
  "timestamp": 61.114,
  "input_length": 521,
  "output_length": 132,
  "type": "text",
  "turn": 2,
  "hash_ids": [1089, 1090, 1091, 6326, 13148]
}
```

**字段到代码模型的映射**（加载层负责转换，内部模型不变）：

| trace 字段 | `RequestRecord` 字段 | V1 处理 |
|---|---|---|
| `chat_id` | `request_id` | 转为 `str` |
| `hash_ids` | `block_ids` | 原样保留为 `list[int]` |
| `timestamp` | `timestamp` | 原样保留 |
| `parent_chat_id` | `metadata["parent_chat_id"]` | V1 存入 metadata，不解析 |
| `type` | `metadata["type"]` | V1 存入 metadata，不解析 |
| `turn` | `metadata["turn"]` | V1 存入 metadata，不解析 |
| `input_length` | `token_count`（可选） | 可选映射 |

---

### "完整复现"与"公开数据适配"

公开 TraceA 是两小时窗口的匿名 trace；论文部分图依赖更长时间范围或跨天比较。
因此，图名须区分复现程度：

| 标签 | 含义 |
|---|---|
| `paper_reproduction` | 可在 TraceA 上直接完整复现 |
| `public_adaptation` | 仅能在公开数据窗口上做局部近似版本 |

当前四张图的预期标签：

```
F4_traceA_public          → paper_reproduction
F13_traceA_public         → paper_reproduction（V2 后）
F14_traceA_public         → paper_reproduction（V2 后）
F15_traceA_public_adaptation → public_adaptation（V2 后）
```

---

### 目录结构

```
project_root/
  configs/
    paper_repro/                        # 只放论文复现配置
      f4_traceA_public.yaml
      f13_traceA_public.yaml            # V2+
      f14_traceA_public.yaml            # V2+
      f15_traceA_public_adaptation.yaml # V2+
    experiments/                        # 只放项目实验配置（禁止用 F4/F13 等论文编号）
      ideal_prefix_hit_ratio_by_category.yaml
      reuse_time_definition_compare.yaml
      prefix_affinity_heatmap.yaml

  data/
    public/
      qwen_traceA_blksz_16.jsonl        # 公开 TraceA 数据
    internal/                           # 内部数据（不上传）

  outputs/
    paper_repro/                        # 只放论文复现结果
      f4_traceA_public/
        plot.png
        metrics.csv
        metadata.json
      f13_traceA_public/                # V2+
      f14_traceA_public/                # V2+
      f15_traceA_public_adaptation/     # V2+
    experiments/                        # 只放项目实验输出
      ideal_prefix_hit_ratio_by_category/
      reuse_time_definition_compare/
      prefix_affinity_heatmap/

  src/
    block_prefix_analyzer/
      types.py
      replay.py
      metrics.py
      index/
      io/
      reports/
      plotting/                         # 绘图模块（待建）
      analysis/                         # 统计分析模块（待建）

  tests/
```

---

### 命名规范

#### 论文复现图

格式：`F{figure_number}_{dataset}_{scope}[_{tag}]`

| 示例 | 说明 |
|---|---|
| `F4_traceA_public` | F4，TraceA 公开数据，完整复现 |
| `F13_traceA_public` | F13，TraceA 公开数据，完整复现 |
| `F15_traceA_public_adaptation` | F15，TraceA 公开数据，局部近似 |

**要求**：必须保留论文编号；必须写明数据集；局部近似必须加 `adaptation`。

**禁止**：`ideal_prefix_hit_rate`、`my_f4`、`paper_like_f15` 等模糊命名。

#### 项目实验图

格式：`{metric_or_question}_{grouping_or_scope}`

| 示例 | 说明 |
|---|---|
| `ideal_prefix_hit_ratio_by_category` | 按请求类别分组的前缀命中率 |
| `reuse_time_last_seen_vs_first_seen` | 两种 reuse time 定义对比 |
| `prefix_affinity_by_workload` | 不同 workload 的前缀亲和性 |

**要求**：直接表达图的语义；**禁止**使用 F4 / F13 / F14 / F15 等论文编号。

#### 配置文件

与图名保持一致，全部小写：

```
configs/paper_repro/f4_traceA_public.yaml
configs/experiments/ideal_prefix_hit_ratio_by_category.yaml
```

#### 输出目录内文件

```
outputs/paper_repro/f4_traceA_public/
  plot.png          # 主图
  metrics.csv       # 中间统计表
  metadata.json     # 配置快照、数据来源、指标定义
```

---

### 硬规则（不可违反）

> **凡是论文复现图，必须使用论文编号命名；凡是项目实验图，禁止使用论文编号命名。**

此规则保证：
- 看名字即可判断图的所属线路；
- 论文复现图与项目实验图永不混淆；
- review 时不会误判图的定位与数据来源。

---

### 指标口径对照

| 场景 | 使用指标 | 代码字段名 |
|---|---|---|
| F4 论文复现主指标 | `overall_block_reusable_ratio` | `overall_block_level_reusable_ratio`（`_level_` 为历史命名，含义相同） |
| 项目扩展指标 | `overall_prefix_hit_rate` | `overall_prefix_hit_rate` |
| 项目扩展指标 | `reuse_time_last_seen` | V2+ 实现 |
| 项目扩展指标 | `reuse_time_first_seen` | V2+ 实现 |

> 若需在论文复现线中输出扩展指标，必须明确标注为 `supplementary` / `debug` / `adaptation` / `alternative_definition`，**不能**与论文图主定义混名。

---

## 下一步
查看 [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) 了解分步实施计划。
详细设计约束参见 [PROJECT_SPEC_FOR_CLAUDE.md](./PROJECT_SPEC_FOR_CLAUDE.md) 与 [CLAUDE.md](./CLAUDE.md)。
