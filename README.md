# block-prefix-analyzer

离线分析 LLM 请求 trace 中 **block 级前缀复用** 的工具。

当前状态：**V1 完整 + V2-min 完整 + Phase 2 全部完成 + Phase 2.5 完成 — 670 个测试全部通过（10 个 xfail 为预期 pending）**。

---

## 新机器快速上手（Quick Start on a New Machine）

> 适用于 WSL / Linux / macOS，Python ≥ 3.10。合成数据已内置，clone 后无需任何外部数据即可运行。

```bash
# 1. 克隆仓库
git clone git@github.com:Xiyuejiushikaixindeyisi/block-prefix-analyzer.git
cd block-prefix-analyzer

# 2. 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖（核心库 + 绘图 + 测试工具）
pip install -e ".[dev,plots]"

# 4. 运行完整测试套件（670 passed，10 xfailed 为预期 pending）
pytest

# 5. 运行 quickstart 示例（加载合成数据 → replay → 打印指标）
python examples/quickstart.py

# 6. （可选）生成合成数据集的全套分析图
bash scripts/run_maas_analysis.sh phase2_business   # 需先有对应 YAML，见下方说明
# 或直接运行单项分析：
python scripts/generate_f4_business.py configs/phase2_business/f4_synthetic_prefix.yaml
```

**接入自有 MaaS 数据集**：见下方 [MaaS 系统数据接入指南](#maas-系统数据接入指南)。

---

## 设计目标
- 给定按时间排序的请求流，回放并输出：
  - 每个请求可复用的最长 **block 前缀**
  - 在"无限容量前缀缓存"假设下的 **理想命中率**（prefix-aware）
  - 在"只要出现过即可复用"假设下的 **block 级可复用率**
  - 可选 token 级前缀命中、reuse time、block lifespan 等衍生指标
- 与 vLLM / vLLM-Ascend 的语义逐步对齐，但核心分析器不依赖在线框架
- 离线、确定性、可单元测试、模块可替换

## 版本里程碑
- **V1（完成）**：只处理已包含 `block_hash_ids` 的输入；顺序回放；简单 Trie；基础报表
- **V2（部分完成）**：接入 tokenizer / chat template / block hash 复现，与框架对齐；见下方状态说明
- **V3**：性能与规模（radix 压缩、磁盘索引、并行预处理 等）

---

## 当前实现状态（V1 + V2）

> 详见 [`V2_READINESS.md`](./V2_READINESS.md) 获取完整的 validated scope 说明。

### V1 — 全部完成

- 时间序回放引擎（`replay.py`）：query → yield → insert，无 self-hit
- 两种 hit 口径：`content_prefix_reuse_blocks`（前缀连续命中）+ `content_reused_blocks_anywhere`（任意位置）
- `MetricsSummary` 聚合与报表输出
- TraceA 数据加载（`io/traceA_loader.py`）
- F4 双图复现（reusable + prefix-aware），已在 TraceA 上跑通

### V2 — 内部对齐已完成，框架对齐 pending

| 组件 | 状态 |
|---|---|
| V2-min 链路（normalize → render → tokenize → block build → V1） | **完成（内部一致）** |
| 三种 hit 口径（block reusable / prefix hit / token-level）| **已锁死并测试** |
| `reuse_time`（last_seen 口径）| **完成** |
| `lifespan`（last_reuse − first_seen）| **完成** |
| session/category helpers（F13–F15 前置）| **完成** |
| Qwen2 chat template Layer 1（渲染逻辑） | **VERIFIED**（纯 Python，无依赖） |
| Qwen2 tokenizer Layer 2（token IDs） | **PENDING**（需 `pip install transformers`） |
| vLLM block hash Layer 3（MurmurHash3 链式）| **PENDING**（需 `pip install mmh3`） |

### 两条分析路径的能力边界

#### 路径 A：TraceA replay 路径（**现在可用**）

TraceA 数据集自带预计算的 `hash_ids`（来自 vLLM 生产环境），加载时直接映射为 `block_ids`，**不经过 V2 chat template / tokenizer / block hash 生成链路**。

因此以下分析现在即可进行：

- F4 全局 block 复用率（已完成）
- F13 single-turn 场景 reuse_time 分布（依赖 session helper，已就绪）
- F14 multi-turn / follow-up 场景 reuse_time 分布（依赖 session helper，已就绪）
- F15 不同 request category 的 reuse_time 分布（依赖 category helper，已就绪）

#### 路径 B：raw request 完整对齐路径（**部分 pending**）

从原始消息（`messages: list[Message]`）出发，经 chat template → tokenizer → block builder 完整链路生成 `block_ids`，再进入 V1 replay。

当前状态：Layer 1（渲染）已验证；Layer 2（tokenizer）和 Layer 3（block hash）待真实框架接通后才能宣称"与 vLLM 完全对齐"。

**不要把路径 B 的 pending 状态与路径 A 的可用性混淆。**

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
| **可复用 block（reusable block）** | 该 block hash 在**任意更早请求**中出现过（最宽口径） | `PerRequestResult.content_reused_blocks_anywhere` |
| **前缀命中 block（prefix-hit block）** | 从请求 `block_ids[0]` 开始**连续**命中 Trie 的 block（第一次 miss 后截断） | `PerRequestResult.content_prefix_reuse_blocks` |

> 两种口径的区别：`content_reused_blocks_anywhere ≥ content_prefix_reuse_blocks`，见 `with_empty` fixture 中 `user` 请求的示例（前缀未命中但有 2 个可复用 block）。

---

### 聚合指标

| 规范术语 | 定义 | 代码字段名 | 口径 |
|---|---|---|---|
| **overall_block_reusable_ratio** | `Σ content_reused_blocks_anywhere / Σ total_blocks`（分母仅含非空请求） | `content_block_reuse_ratio` | micro |
| **content_prefix_reuse_rate** | `Σ content_prefix_reuse_blocks / Σ total_blocks`（分母仅含非空请求） | `content_prefix_reuse_rate` | micro |

> **注意**：规范术语 `overall_block_reusable_ratio` 在代码中对应字段名为 `content_block_reuse_ratio`（多一个 `_level_`），含义相同。V2 重构时可统一命名。

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
| F4 论文复现主指标 | `overall_block_reusable_ratio` | `content_block_reuse_ratio`（`_level_` 为历史命名，含义相同） |
| 项目扩展指标 | `content_prefix_reuse_rate` | `content_prefix_reuse_rate` |
| 项目扩展指标 | `reuse_time_last_seen` | V2+ 实现 |
| 项目扩展指标 | `reuse_time_first_seen` | V2+ 实现 |

> 若需在论文复现线中输出扩展指标，必须明确标注为 `supplementary` / `debug` / `adaptation` / `alternative_definition`，**不能**与论文图主定义混名。

---

---

## 业务数据集分析（Phase 2）

> 适用场景：数据集仅含已渲染的原始 prompt，无预计算 `hash_ids`。
> 加载器使用 `CharTokenizer`（1 字符 = 1 token）+ `SimpleBlockBuilder`（SHA-256 独立哈希）生成 block_ids，无需外部 tokenizer 或哈希库。
> **当前阶段的 block_ids 与实际 vLLM 部署的 block_ids 值不同，但命中率统计结论在同一数据集内部一致有效。**

---

### 数据集接入：三步走

#### Step 1 — 准备数据文件

**放置位置（推荐）**：

```
data/
  internal/              # 已加入 .gitignore，不上传到 git
    <dataset_name>.jsonl
```

将数据文件放到 `data/internal/`，例如：

```bash
cp /path/to/your/requests.jsonl data/internal/prod_requests.jsonl
```

**JSONL 格式**：每行一条 JSON 对象，字段如下：

| 字段 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `user_id` | str / int | **必填** | 租户 / 用户标识，用于按用户聚合命中率 |
| `request_id` | str | **必填** | 请求唯一标识，重复出现视为 Agent session（当前阶段过滤） |
| `timestamp` | float（秒）| **必填** | 请求到达时间，用于时序排序和 reuse_time 计算 |
| `raw_prompt` | str | **必填** | 完整渲染后的 prompt 原文（已经过 chat template 处理，直接入 block 分析） |

示例行：

```json
{"user_id": "tenant_42", "request_id": "req_001", "timestamp": 1700000000.0, "raw_prompt": "You are a helpful assistant.\n\nUser: How do I query Elasticsearch?\nAssistant:"}
```

**字段名重映射**（若源文件字段名不同）：

在 YAML config 中（或 Python 代码中）使用 `field_map` 参数：

```yaml
field_map:
  user_id: tenant_id        # 源文件用 "tenant_id" 表示 user_id
  request_id: chat_id       # 源文件用 "chat_id" 表示 request_id
  raw_prompt: prompt        # 源文件用 "prompt" 表示 raw_prompt
```

> 注意：当前 `_load_flat_yaml` 仅支持单层 key: value，`field_map` 需通过 Python 直接调用 `load_business_jsonl(field_map={...})` 传入，或等待 Phase 3 扩展 YAML 解析器。

---

#### Step 2 — 创建 YAML 配置文件

每个分析对应一个 YAML config。**直接复制合成数据示例并修改三个字段**：

```bash
cp configs/phase2_business/f4_synthetic_reusable.yaml \
   configs/phase2_business/f4_prod_reusable.yaml
```

打开新文件，修改以下三处：

```yaml
trace_name: prod_requests          # ← 改成你的数据集名（出现在图标题中）
input_file: data/internal/prod_requests.jsonl  # ← 改成你的数据文件路径
output_dir: outputs/phase2_business/f4_prod_reusable  # ← 改成你的输出目录
```

其余参数（`block_size`、`bin_size_seconds` 等）按需调整，默认值对多数场景适用。

**推荐 block_size**：

| block_size | 适用场景 |
|---|---|
| `128` | 主分析（vLLM-Ascend 默认，内存更友好，**优先使用**） |
| `64` | 中粒度补充 |
| `32` / `16` | 细粒度 / 与论文复现对比（内存消耗更大） |

---

#### Step 3 — 运行分析脚本

所有脚本统一入口：

```bash
python scripts/<script_name>.py configs/phase2_business/<your_config>.yaml
```

---

### 当前可产出的全部分析图

以下 9 项分析均已就绪，脚本和模块全部通过合成数据验证。

---

#### 分析 1 — F4：时间窗口 block 复用率（两张图）

**目的**：观察复用率随时间的整体趋势，判断系统是否存在"热身期"或周期性波动。

**两个口径**（建议同时运行）：

```bash
# 口径 A：最宽口径（任意位置复用，包括非前缀位置）
python scripts/generate_f4_business.py configs/phase2_business/f4_prod_reusable.yaml

# 口径 B：等价于无限容量 vLLM APC 命中率（从请求头部连续命中）
python scripts/generate_f4_business.py configs/phase2_business/f4_prod_prefix.yaml
```

对应 YAML 差异：`hit_metric: content_block_reuse`（口径 A）或 `hit_metric: content_prefix_reuse`（口径 B）。

**输出文件**（在 `output_dir/` 下）：

```
plot.png          # 折线图：每时间窗口的命中率（bar = 请求量，line = 命中率）
metrics.csv       # 每窗口的 hit_blocks, total_blocks, hit_rate
metadata.json     # 运行参数快照、全局汇总指标
```

**关键指标**（`metadata.json` 中）：
- `overall_hit_rate`：全局微平均命中率（`Σ hit / Σ total`）
- `macro_hit_rate`：全局宏平均（每请求命中率的均值）
- `total_requests`：参与分析的请求数

**解读**：`content_prefix_reuse` 命中率 = 无限容量 vLLM APC 的上界命中率。有限容量 LRU 实际命中率 ≤ 此值。

---

#### 分析 2 — F13：reuse_time CDF（单轮请求）

**目的**：分析 block 从首次出现到被再次复用的等待时间分布，评估 KV cache TTL 设置是否合理。

```bash
# 口径 A：任意位置复用
python scripts/generate_f13_business.py configs/phase2_business/f13_prod_reusable.yaml

# 口径 B：前缀连续复用（更接近 vLLM 实际行为）
python scripts/generate_f13_business.py configs/phase2_business/f13_prod_prefix.yaml
```

**输出文件**：

```
cdf.png           # reuse_time 的 CDF 曲线（X 轴：分钟，Y 轴：累积比例）
reuse_times.csv   # 每条 reuse event 的 reuse_time 值
metadata.json     # 总 event 数、P50/P90/P99 分位数
```

**关键指标**：P50、P90 reuse_time（分钟）——若 P90 < KV cache TTL，说明大部分复用能被当前 TTL 覆盖。

---

#### 分析 3 — reuse_rank：prefix 命中 block 数排名曲线

**目的**：按每个请求实际命中的 prefix block 数降序排列，直观显示命中分布的偏斜程度。

```bash
python scripts/generate_reuse_rank_business.py configs/phase2_business/reuse_rank_prod.yaml
```

**输出文件**：

```
reuse_rank.png    # 排名曲线图（X = 请求排名，Y = 命中 block 数）
reuse_rank.csv    # 每请求的排名、prefix_reuse_blocks、total_blocks
metadata.json
```

**解读**：曲线越陡峭，命中越集中在少数请求（高 skew）；曲线越平坦，命中均匀分布。

---

#### 分析 4 — E3 / top_ngrams：TOP-N 高频 block 序列

**目的**：找出所有请求中出现频率最高的连续 block 子序列，识别 system prompt、tool schema、固定模板等共享内容。

```bash
python scripts/generate_top_ngrams_business.py configs/phase2_business/top_ngrams_prod.yaml
```

**输出文件**：

```
top_ngrams_single_turn.txt  # 可读表格：rank, count, pct, 序列 block IDs
top_ngrams_single_turn.csv  # CSV 版本（rank, count, pct, length, blocks）
metadata.json
```

**解读**：Rank-1 出现次数占总请求比例高 → 该 block 序列对应强共享内容（如统一 system prompt）；单独对该内容做 warming 可显著提升命中率。

---

#### 分析 5 — E5：TOP-N 序列文本还原

**目的**：将 E3 找到的高频 block 序列反查为可读原始文本，确认具体是哪段 prompt 在被大量复用。

```bash
python scripts/generate_e5_block_text.py configs/phase2_business/e5_block_text_prod.yaml
```

**输出文件**：

```
top_ngrams_decoded.txt  # 每个 rank 的实际文本内容（截断到 max_chars 字符）
top_ngrams_decoded.csv  # rank, count, pct, length, truncated, text, block_ids
metadata.json
```

**解读**：直接读 `top_ngrams_decoded.txt` 即可看到高频复用的具体文本内容，不需要手动反查 block_id。`truncated=True` 表示文本超过 `max_chars`，完整内容在 CSV 的 `text` 列。

---

#### 分析 6 — E1：per-user 理想 prefix 命中率分布（4 条曲线）

**目的**：按用户维度拆分命中率，识别"高 cache 价值用户"和"冷启动用户"，评估不同 block_size 对命中率的影响。

```bash
python scripts/generate_user_hit_rate.py configs/phase2_business/e1_user_hit_rate_prod.yaml
```

YAML 中 `block_sizes: 16,32,64,128`（一次运行同时产出 4 条曲线）。

**输出文件**：

```
user_hit_rate.png         # 4 条 block_size 曲线叠加在同一图上（X = 用户排名，Y = 命中率）
user_hit_rate_bs16.csv    # block_size=16 的每用户统计（user_id, hit_rate, total_blocks, …）
user_hit_rate_bs32.csv
user_hit_rate_bs64.csv
user_hit_rate_bs128.csv
metadata.json             # 每个 block_size 的 micro/macro 命中率汇总
```

**YAML 参数说明**：
- `min_blocks_pct: 0.05`（默认）：过滤 `total_blocks` 低于 P5 分位的用户（长尾噪声），推荐保留
- `hit_rate_bar_threshold: 0.5`：子图 B 中，统计命中率 ≥ 此阈值的用户比例

**解读**：若 top-20% 用户的命中率明显高于其余用户，说明存在强共享前缀的"VIP 用户群"，针对该群体单独维护 warm cache 收益大。

---

#### 分析 7 — E1-B：用户维度 Lorenz 曲线（命中贡献 + 请求量，各一张图）

**目的**：量化 prefix cache 命中贡献的集中程度——是少数用户贡献了绝大多数命中（高 Gini），还是相对均匀分布。

```bash
python scripts/generate_skewness.py configs/phase2_business/e1b_skewness_prod.yaml
```

**输出文件**：

```
hit_contribution.png   # 图1：用户按命中 block 数降序排名的 Lorenz 曲线
request_volume.png     # 图2：用户按请求数降序排名的 Lorenz 曲线
hit_contribution.csv   # rank, value, value_norm, cumulative_fraction
request_volume.csv
metadata.json          # 两图的 Gini 系数 + top-10%/top-20% 用户占比
```

**关键指标**（`metadata.json` 中）：
- `hit_contribution.gini_coefficient`：命中贡献的 Gini 系数（0 = 完全均匀，1 = 完全集中）
- `request_volume.gini_coefficient`：请求量的 Gini 系数
- `top_10pct_users_fraction_of_hits`：top-10% 用户贡献了多少比例的总命中

**解读**：
- 命中 Gini >> 请求量 Gini → 存在内容特别 cache-friendly 的用户群（相同 system prompt / tool schema），针对这批用户做 warming 收益远超请求量比例
- 命中 Gini ≈ 请求量 Gini → cache 命中主要由请求量决定，无明显内容偏斜

---

### 完整执行示例

以下是接入真实数据集后，一次完整分析的推荐命令序列：

```bash
DATASET=prod_requests   # 替换为你的数据集名

# 1. 全局复用率趋势（必跑，最快，先看全貌）
python scripts/generate_f4_business.py     configs/phase2_business/f4_${DATASET}_reusable.yaml
python scripts/generate_f4_business.py     configs/phase2_business/f4_${DATASET}_prefix.yaml

# 2. reuse_time 分布（评估 TTL 设置）
python scripts/generate_f13_business.py    configs/phase2_business/f13_${DATASET}_prefix.yaml

# 3. 识别高频共享内容（E3 + E5 配合使用）
python scripts/generate_top_ngrams_business.py configs/phase2_business/top_ngrams_${DATASET}.yaml
python scripts/generate_e5_block_text.py   configs/phase2_business/e5_block_text_${DATASET}.yaml

# 4. 用户维度分析（E1 + E1-B）
python scripts/generate_user_hit_rate.py   configs/phase2_business/e1_user_hit_rate_${DATASET}.yaml
python scripts/generate_skewness.py        configs/phase2_business/e1b_skewness_${DATASET}.yaml

# 5. 命中排名曲线（辅助）
python scripts/generate_reuse_rank_business.py configs/phase2_business/reuse_rank_${DATASET}.yaml
```

所有输出写入 `outputs/phase2_business/`，与论文复现结果严格隔离。

---

### 注意事项

**1. block_id 与实际 vLLM 部署的差异**

当前加载器使用 `CharTokenizer`（1 字符 = 1 token）和独立 SHA-256 哈希，而非 vLLM 的链式 MurmurHash3。这意味着：

- block_id 的**数值**与实际 vLLM 部署不同
- 但同一数据集内部的**命中率统计结论**是一致有效的（等价条件见 `CLAUDE.md` Section 5a）
- `content_prefix_reuse_rate` = 无限容量 vLLM APC 命中率的上界

如需精确复现特定 vLLM 部署的 block_id，需等待 Path B 完整对齐（Layer 2/3 pending）。

**2. 大数据集处理**

当 `total_blocks > 30,000,000` 时，loader 会自动发出 `ResourceWarning`。建议：

```bash
# 按时间窗口切片（例如取最近 24 小时）
head -n 50000 data/internal/prod_requests.jsonl > data/internal/prod_requests_24h.jsonl
```

或在代码中按 `timestamp` 过滤后传入 loader。

**3. block_size 与 vLLM 部署对齐**

`block_size` 必须与实际 vLLM 部署的 `--block-size` 参数保持一致，否则命中率统计结果不具参考意义。**每次调用 loader 必须显式传入 block_size，无默认值**（防止静默使用错误参数）。

**4. 单模型假设**

KV cache 复用只在同一模型同一部署实例内有效。若数据集混合了多个模型的请求，需先按 `model_id` 拆分，对每个模型单独运行分析。

---

### 分析图汇总

| 图 | 脚本 | 产出文件 | 核心指标 |
|---|---|---|---|
| F4 复用率趋势（2 口径）| `generate_f4_business.py` | `plot.png`, `metrics.csv`, `metadata.json` | `overall_hit_rate`, `macro_hit_rate` |
| F13 reuse_time CDF | `generate_f13_business.py` | `cdf.png`, `reuse_times.csv`, `metadata.json` | P50/P90/P99 reuse_time |
| reuse_rank 排名曲线 | `generate_reuse_rank_business.py` | `reuse_rank.png`, `reuse_rank.csv` | 命中 block 数分布 |
| E3 top_ngrams 高频序列 | `generate_top_ngrams_business.py` | `*.txt`, `*.csv`, `metadata.json` | Rank-1 count + pct |
| E5 序列文本还原 | `generate_e5_block_text.py` | `top_ngrams_decoded.txt`, `*.csv` | 高频文本内容 |
| E1 per-user 命中率 | `generate_user_hit_rate.py` | `user_hit_rate.png`, `bs*.csv` | micro/macro hit rate per user |
| E1-B 命中贡献 Lorenz | `generate_skewness.py` | `hit_contribution.png`, `request_volume.png` | Gini 系数，top-10% 占比 |

所有结果写入 `outputs/phase2_business/<output_dir>/`，YAML `output_dir` 字段控制子目录名。

---

## Phase 2.5 — 性能优化（开发中）

**问题**：`TrieIndex` 每个节点 ≈ 116 字节，50K 请求 × 平均 256 blocks（32K token 上下文）= **1+ GB** 峰值内存。

**方案**：`RadixTrieIndex`（Patricia/Radix 压缩树）—— 把单孩子节点链压缩为带 `array.array('Q')` 标签的单条边，内存压缩 **~10–14×**。

**接口兼容**：满足 `PrefixIndex` 协议，`replay.py` / 所有分析脚本**零改动**。

**自动切换（已实现）**：`replay()` 的 `index_factory` 默认为 `None`（自动选择）：
- 平均块数 < 256 → 继续使用 `TrieIndex`（原有行为）
- 平均块数 ≥ 256（≈ bs=128 均值 32K tokens）→ 自动切换 `RadixTrieIndex`，防止内存爆炸

**Benchmark**：`python scripts/benchmark_index.py configs/phase2_business/benchmark_index_synthetic.yaml`  
输出 node_count 压缩比、峰值内存对比、命中率一致性（硬断言）。

详见 `PHASE2_PLAN.md` Section 6.3。

---

## MaaS 系统数据接入指南

> **每份 CSV 对应一个模型，必须分开分析**——不同模型的 KV cache 不可共享（CLAUDE.md §5a 单模型假设）。  
> `data/internal/` 已加入 `.gitignore`，不会上传到 git。

---

### 已支持的 7 个数据集（配置文件已就绪）

| 目录名 | 模型 | 上下文窗口 |
|---|---|---|
| `qwen_v3_32b_8k` | Qwen-V3-32B | 8K |
| `qwen_v3_32b_32k` | Qwen-V3-32B | 32K |
| `deepseek_v3_1_nothinking_8k` | DeepSeek-V3.1-Terminus-NoThinking | 8K |
| `deepseek_v3_1_nothinking_32k` | DeepSeek-V3.1-Terminus-NoThinking | 32K |
| `glm_4_7_hcmaas` | GLM-4.7-HCMAAS | — |
| `qwen_v3_5_27b_128k` | Qwen-V3.5-27B | 128K |
| `qwen_v3_5_27b_64k` | Qwen-V3.5-27B | 64K |

---

### CSV 文件格式

当前内部数据集 CSV 已直接包含四个字段，列顺序如下：

| 列索引 | 字段 | 说明 |
|---|---|---|
| 0 | `user_id` | 租户 / 用户标识 |
| 1 | `request_id` | 请求唯一 ID |
| 2 | `timestamp` | 到达时间（Unix 浮点或 ISO-8601） |
| 3 | `raw_prompt` | 已渲染的完整 prompt 文本 |

---

### 每个模型的完整流程（三步）

以 `qwen_v3_32b_8k` 为例，其余 6 个模型把目录名替换即可。

**第一步 — 放置 CSV 文件**

```
data/internal/
  qwen_v3_32b_8k/
    raw/
      <your_file>.csv    ← 把 CSV 放这里
```

**第二步 — 转换 CSV → JSONL**

```bash
python scripts/convert_csv_to_jsonl.py \
    --input  data/internal/qwen_v3_32b_8k/raw/<your_file>.csv \
    --output data/internal/qwen_v3_32b_8k/requests.jsonl \
    --col-user-id 0 --col-request-id 1 --col-timestamp 2 --col-raw-prompt 3
```

- 流式处理，500 MB 文件安全
- 每 10,000 行打印进度
- 若 CSV 有表头行，追加 `--has-header`

**第三步 — 运行所有 10 项分析**

```bash
bash scripts/run_maas_analysis.sh qwen_v3_32b_8k
```

输出全部写入 `outputs/maas/qwen_v3_32b_8k/`，各子目录见下表。

---

### 全部 7 个模型的命令汇总

```bash
# ---- Qwen-V3-32B-8K ----
python scripts/convert_csv_to_jsonl.py \
    --input  data/internal/qwen_v3_32b_8k/raw/<file>.csv \
    --output data/internal/qwen_v3_32b_8k/requests.jsonl \
    --col-user-id 0 --col-request-id 1 --col-timestamp 2 --col-raw-prompt 3
bash scripts/run_maas_analysis.sh qwen_v3_32b_8k

# ---- Qwen-V3-32B-32K ----
python scripts/convert_csv_to_jsonl.py \
    --input  data/internal/qwen_v3_32b_32k/raw/<file>.csv \
    --output data/internal/qwen_v3_32b_32k/requests.jsonl \
    --col-user-id 0 --col-request-id 1 --col-timestamp 2 --col-raw-prompt 3
bash scripts/run_maas_analysis.sh qwen_v3_32b_32k

# ---- DeepSeek-V3.1-Terminus-NoThinking-8K ----
python scripts/convert_csv_to_jsonl.py \
    --input  data/internal/deepseek_v3_1_nothinking_8k/raw/<file>.csv \
    --output data/internal/deepseek_v3_1_nothinking_8k/requests.jsonl \
    --col-user-id 0 --col-request-id 1 --col-timestamp 2 --col-raw-prompt 3
bash scripts/run_maas_analysis.sh deepseek_v3_1_nothinking_8k

# ---- DeepSeek-V3.1-Terminus-NoThinking-32K ----
python scripts/convert_csv_to_jsonl.py \
    --input  data/internal/deepseek_v3_1_nothinking_32k/raw/<file>.csv \
    --output data/internal/deepseek_v3_1_nothinking_32k/requests.jsonl \
    --col-user-id 0 --col-request-id 1 --col-timestamp 2 --col-raw-prompt 3
bash scripts/run_maas_analysis.sh deepseek_v3_1_nothinking_32k

# ---- GLM-4.7-HCMAAS ----
python scripts/convert_csv_to_jsonl.py \
    --input  data/internal/glm_4_7_hcmaas/raw/<file>.csv \
    --output data/internal/glm_4_7_hcmaas/requests.jsonl \
    --col-user-id 0 --col-request-id 1 --col-timestamp 2 --col-raw-prompt 3
bash scripts/run_maas_analysis.sh glm_4_7_hcmaas

# ---- Qwen-V3.5-27B-128K ----
python scripts/convert_csv_to_jsonl.py \
    --input  data/internal/qwen_v3_5_27b_128k/raw/<file>.csv \
    --output data/internal/qwen_v3_5_27b_128k/requests.jsonl \
    --col-user-id 0 --col-request-id 1 --col-timestamp 2 --col-raw-prompt 3
bash scripts/run_maas_analysis.sh qwen_v3_5_27b_128k

# ---- Qwen-V3.5-27B-64K ----
python scripts/convert_csv_to_jsonl.py \
    --input  data/internal/qwen_v3_5_27b_64k/raw/<file>.csv \
    --output data/internal/qwen_v3_5_27b_64k/requests.jsonl \
    --col-user-id 0 --col-request-id 1 --col-timestamp 2 --col-raw-prompt 3
bash scripts/run_maas_analysis.sh qwen_v3_5_27b_64k
```

---

### 每次运行产出的 10 项分析

| 步骤 | 脚本 | 输出目录 | 核心指标 |
|---|---|---|---|
| 1 | `generate_f4_business.py` | `f4_prefix/` | 理想前缀命中率时序（= 无限容量 vLLM APC） |
| 2 | `generate_f4_business.py` | `f4_reusable/` | 最宽口径复用率时序 |
| 3 | `generate_f13_business.py` | `f13_prefix/` | 前缀命中 reuse_time 分布 |
| 4 | `generate_f13_business.py` | `f13_reusable/` | 最宽口径 reuse_time 分布 |
| 5 | `generate_reuse_rank_business.py` | `reuse_rank/` | 请求级命中 block 数 Pareto |
| 6 | `generate_top_ngrams_business.py` | `top_ngrams/` | TOP-20 高频 block 序列 |
| 7 | `generate_user_hit_rate.py` | `e1_user_hit_rate/` | per-user 命中率（4 block_size 曲线） |
| 8 | `generate_skewness.py` | `e1b_skewness/` | 命中贡献 Lorenz + Gini 系数 |
| 9 | `generate_e5_block_text.py` | `e5_block_text/` | 高频序列还原为原始文本 |
| 10 | `benchmark_index.py` | `benchmark_index/` | TrieIndex vs RadixTrieIndex 对比 |

> **长上下文模型（128K/64K）注意**：replay 引擎会自动切换为 `RadixTrieIndex`（avg blocks ≥ 256 时）以防止内存爆炸，无需手动调整。

---

### 添加新模型（未来扩展）

```bash
# 生成 10 个 YAML 配置文件
python scripts/init_maas_configs.py <model_slug> "<Model Display Name>"

# 例：
python scripts/init_maas_configs.py llama3_70b_32k "Llama-3-70B-32K"
```

然后按上面三步流程操作。

---

### 目录结构总览

```
data/internal/
  <model_slug>/
    raw/               ← 放原始 CSV（gitignored）
    requests.jsonl     ← 转换后的 JSONL（自动生成，gitignored）

configs/maas/
  <model_slug>/        ← 10 个 YAML（已就绪，随代码提交）
    f4_prefix.yaml
    f4_reusable.yaml
    f13_prefix.yaml
    f13_reusable.yaml
    reuse_rank.yaml
    top_ngrams.yaml
    e1_user_hit_rate.yaml
    e1b_skewness.yaml
    e5_block_text.yaml
    benchmark_index.yaml

outputs/maas/
  <model_slug>/        ← 所有分析输出（自动创建，gitignored）
    f4_prefix/
    f4_reusable/
    ...
```

---

## 下一步
查看 [PHASE2_PLAN.md](./PHASE2_PLAN.md) 了解 Phase 2 开发任务列表与接口约定。  
查看 [EXPERIMENT_DESIGN.md](./EXPERIMENT_DESIGN.md) 了解完整实验设计方案。  
查看 [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) 了解 V1/V2 分步实施计划。  
详细设计约束参见 [PROJECT_SPEC_FOR_CLAUDE.md](./PROJECT_SPEC_FOR_CLAUDE.md) 与 [CLAUDE.md](./CLAUDE.md)。
