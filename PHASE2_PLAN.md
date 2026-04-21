# Phase 2 开发计划 — 业务数据集前缀缓存分析

本文档供 Claude Code 后续 session 直接参照。**编码前必须先读本文件确认当前完成状态**，
避免重复开发已有模块。

| 阶段 | 目标 | 状态 |
|---|---|---|
| **Phase 2**（当前）| 单轮 text 请求理想命中率分析；CharTokenizer + SHA-256 哈希 | **全部完成** |
| **Phase 2.5**（性能攻坚）| RadixTrieIndex，攻克长上下文 Trie 内存瓶颈 | ❌ 待开发 |
| **Phase 3**（多模型 + 容量 + 时序）| E6–E9；128K/64K 长上下文 | ❌ 待开发 |

---

## 0. 阅读顺序

1. `CLAUDE.md` — 工程原则、指标定义（必读，尤其 Section 5a 三指标区分）
2. `EXPERIMENT_DESIGN.md` — 所有实验完整设计规格
3. 本文件 — 实现状态与待办任务
4. `V2_READINESS.md` — V2 层 pending 说明

---

## 1. 数据集字段要求

| 字段 | 类型 | 用途 |
|---|---|---|
| `user_id` | str | 按用户聚合命中率（E1、E1-B） |
| `request_id` | str | 单轮识别；重复出现 = Agent session（当前阶段过滤） |
| `timestamp` | float（秒） | 时序排序 + reuse_time 计算 |
| `raw_prompt` | str | Path B 输入 → block_ids |
| `model_id` | str | Phase 3 分流；当前阶段数据集为单一模型时可选 |

字段名通过 `field_map` 参数重映射，语义不变。

**接入方式（三步）**：

1. 将数据文件放入 `data/internal/<dataset_name>.jsonl`（已加入 `.gitignore`）
2. 从 `configs/phase2_business/` 复制对应的 `*_synthetic.yaml`，改 `input_file`、`output_dir`、`trace_name` 三个字段
3. 运行 `python scripts/<script>.py configs/phase2_business/<your_config>.yaml`

详细接入指南见 `README.md` — "业务数据集分析（Phase 2）" 节。

---

## 2. Phase 2 完成状态总览

### 2.1 基础设施（✅ 全部完成）

| 模块 | 文件 | 实际接口 |
|---|---|---|
| 业务 JSONL 加载器 | `io/business_loader.py` | `load_business_jsonl(path, *, block_size, block_builder, field_map, include_debug_metadata, warn_memory_threshold, block_registry)` |
| 请求分类器 | `analysis/request_classifier.py` | `classify_requests(records, *, agent_keywords)` → `dict[str, Classification]`<br>`filter_single_turn(records)` → `list[RequestRecord]`<br>`classification_summary(records)` → `dict[str, int]` |
| 合成数据生成 | `scripts/make_synthetic_business.py` | 100 条记录，3 种复用模式，7200 秒窗口 |

**`business_loader.py` 关键行为**：
- `block_size` 与 `block_builder` 二选一（均缺失时抛 `ValueError`）
- `block_size` 无默认值——必须显式指定
- 内部使用 `_RawPromptTemplate`（raw passthrough，不添加 `<|user|>` 标记，避免人工前缀共享）
- `metadata["user_id"]` 通过 `arrival_index` 回填，不受排序影响
- `include_debug_metadata=True` 时写入 `v2_rendered_prompt`、`v2_token_count` 等调试字段
- 文件句柄用 `with` 块管理（防止 `ResourceWarning` 泄露）
- `block_registry: dict[int, str] | None`：可选参数，传入空 `{}` 后自动填充 `{block_id → text_slice}`，供 E5 文本还原使用

**`request_classifier.py` 关键行为**：
- 主判据：`request_id` 重复 → `"agent_likely"`
- 辅助：`metadata["v2_rendered_prompt"]` 含关键词 → `"agent_likely"`
- 关键词默认集：`tool_call, tool_result, opencode, claudecode, <tool_use>, <function_call>, <tool_response>`
- 诊断工具，**不是** loader 的强依赖

---

### 2.2 分析模块（✅ 全部完成）

| 模块 | 文件 | 实际接口 | 用途 |
|---|---|---|---|
| 时间窗口 block 复用 | `analysis/f4.py` | `compute_f4_series(results, hit_metric, bin_size_seconds)` → `F4Series` | F4 |
| reuse-time CDF | `analysis/f13.py` | `compute_f13_series(records, event_definition, x_axis_max_minutes, type_label_mapping)` → `F13Series` | F13 |
| prefix-reuse 排名 | `analysis/reuse_rank.py` | `build_reuse_rank_series(results, request_ids, label)` → `ReuseRankSeries` | reuse_rank |
| 高频 block 序列 | `analysis/top_ngrams.py` | `build_top_ngrams(records, request_ids, top_k, max_n, min_count)` → `list[NgramRow]` | E3 |
| 按用户聚合命中率 | `analysis/user_hit_rate.py` | 见下方详细接口 | E1、E1-B |
| block 序列文本还原 | `analysis/block_text_decoder.py` | `decode_ngram_rows(rows, block_registry, max_chars)` → `list[DecodedNgramRow]` | E5 |

**`user_hit_rate.py` 实际接口**：

```python
@dataclass
class UserHitStats:
    user_id: str
    total_blocks: int
    prefix_reuse_blocks: int    # Σ content_prefix_reuse_blocks
    request_count: int
    hit_rate: float             # prefix_reuse_blocks / total_blocks; 0.0 when total==0

@dataclass
class UserHitSeries:
    stats: list[UserHitStats]       # 过滤后，按 hit_rate 降序
    raw_stats: list[UserHitStats]   # 全部用户（E1-B 使用此字段）
    min_blocks_threshold: int       # 实际使用的 P-th 百分位截断值
    block_size: int

@dataclass
class SkewnessRow:
    rank: int
    value: int                      # hit_blocks 或 request_count（绝对值）
    value_norm: float               # value / max_value
    cumulative_fraction: float      # Σ value[1..rank] / Σ value[all]

def build_user_hit_series(
    results: Sequence[PerRequestResult],
    records: Sequence[RequestRecord],
    *,
    block_size: int,
    min_blocks_pct: float = 0.05,   # P5 长尾过滤；0.0 = 不过滤
) -> UserHitSeries: ...

def compute_hit_contribution_rows(stats: Sequence[UserHitStats]) -> list[SkewnessRow]: ...
def compute_request_volume_rows(stats: Sequence[UserHitStats]) -> list[SkewnessRow]: ...
def save_user_hit_csv(series: UserHitSeries, path: Path, *, filtered: bool = True) -> None: ...
def save_skewness_csv(rows: list[SkewnessRow], path: Path) -> None: ...
```

**长尾过滤逻辑**：`min_blocks_pct=0.05` → 取 `total_blocks` 分布的 P5 作为截断阈值，
排除 `total_blocks < threshold` 的用户进入 `stats`（`raw_stats` 保留全部，E1-B 偏斜图使用 `raw_stats`）。

---

### 2.3 绘图脚本（✅ 全部完成）

所有脚本均接受 YAML config 文件作为唯一参数（`python scripts/<script>.py <config.yaml>`）。

| 脚本 | 产出图 | YAML 必填键 | 可选键 |
|---|---|---|---|
| `generate_f4_business.py` | F4 时间窗口复用率 | `input_file, output_dir, hit_metric, block_size` | `bin_size_seconds=300, trace_name, note, figure_variant` |
| `generate_f13_business.py` | F13 reuse-time CDF | `input_file, output_dir, event_definition, block_size` | `x_axis_max_minutes=120, trace_name, note` |
| `generate_reuse_rank_business.py` | reuse_rank 排名曲线 | `input_file, output_dir, block_size` | `trace_name, note` |
| `generate_top_ngrams_business.py` | E3 TOP-N block 序列 | `input_file, output_dir, block_size` | `top_k=10, min_count=2, trace_name` |
| `generate_user_hit_rate.py` | E1 per-user 命中率 | `input_file, output_dir, block_sizes` | `min_blocks_pct=0.05, hit_rate_bar_threshold=0.5, trace_name, note` |
| `generate_skewness.py` | E1-B 图1+图2 | `input_file, output_dir, block_size` | `min_blocks_pct=0.0, trace_name, note` |
| `generate_e5_block_text.py` | E5 文本还原 | `input_file, output_dir, block_size` | `top_k=10, min_count=2, max_chars=300, trace_name, note` |

**`hit_metric` / `event_definition` 合法值**：`"content_block_reuse"` 或 `"content_prefix_reuse"`（全称，不支持缩写）。

**`block_sizes` 格式**：逗号分隔整数字符串，如 `16,32,64,128`。

---

### 2.4 YAML configs（✅ 全部完成）

```
configs/
  paper_repro/               # TraceA 论文复现（已有，不改动）
  phase2_business/           # 业务数据集配置（新增）
    f4_synthetic_reusable.yaml
    f4_synthetic_prefix.yaml
    f13_synthetic_reusable.yaml
    f13_synthetic_prefix.yaml
    reuse_rank_synthetic.yaml
    top_ngrams_synthetic.yaml
    e1_user_hit_rate_synthetic.yaml
    e1b_skewness_synthetic.yaml
    e5_block_text_synthetic.yaml
```

切换真实数据集只需修改 `input_file`、`output_dir`、`trace_name`，脚本无需改动。

---

### 2.5 合成数据验证（✅ 通过）

`data/synthetic/business_synthetic.jsonl`：100 条请求，38 个用户，7200 秒窗口。

| 分析 | 关键结果 | 含义 |
|---|---|---|
| F4（bs=128） | 整体命中率 69.3% | Pattern A 的 4-block 系统提示词贡献大量命中 |
| F13（bs=128） | 58/100 请求有 reuse 事件 | Pattern C（40 冷启动）拉低比例 |
| top_ngrams | Rank-1: 4 块出现 40 次 | 精确检测到植入的 SYSTEM_PROMPT |
| | Rank-2: 2 块出现 20 次 | 精确检测到 HOT_PROMPT |
| E1（bs=128） | micro hit rate 69.3%；27/38 用户 ≥50% | 4 条曲线（bs=16/32/64/128）生成正常 |
| E1-B | 命中 Gini=0.628；请求量 Gini=0.321 | 命中集中度远高于请求量集中度（Pattern A 用户内容特别 cache-friendly） |
| E5（bs=128） | Rank-1: 4 块出现 40 次（SYSTEM_PROMPT 前 512 chars，截断显示）；Rank-2: 2 块出现 20 次（HOT_PROMPT 全文）| 文本还原正确；top_ngrams 与 block_registry 联合输出符合预期 |

---

## 3. Phase 2 E5 实现（✅ 已完成）

### 3.1 E5：`analysis/block_text_decoder.py`

**目的**：将 `top_ngrams` 产出的 block_id 序列反查为可读原始文本，识别复用的 system prompt / tool schema 片段。

**实现路径**：
- `business_loader.py` 新增 `block_registry: dict[int, str] | None` 参数；传入空 `{}` 时，在构建 records 后逐 block 填充 `{block_id → raw_prompt[i*bs:(i+1)*bs]}`，first-seen-wins（不覆盖已有条目）
- `block_text_decoder.py` 提供 `decode_ngram_rows(rows, block_registry, max_chars)` → `list[DecodedNgramRow]`；缺失 block_id 用 `<MISSING:{id}>` 占位
- `generate_e5_block_text.py` 联合调用 loader + top_ngrams + decoder，输出 CSV + 文本表格 + metadata.json

**实际接口**：

```python
BlockRegistry = dict[int, str]   # block_id → text slice

@dataclass
class DecodedNgramRow:
    rank: int
    count: int
    pct: float
    length: int
    text: str          # 拼接后截断到 max_chars（末尾加 "…"）
    truncated: bool
    blocks: tuple[int, ...]

def decode_ngram_rows(
    rows: list[NgramRow],
    block_registry: BlockRegistry,
    max_chars: int = 300,
) -> list[DecodedNgramRow]: ...

def format_decoded_table(rows: list[DecodedNgramRow], title: str) -> str: ...
def save_decoded_csv(rows: list[DecodedNgramRow], path: Path) -> None: ...
```

**测试文件**：`tests/test_block_text_decoder.py`（16 个测试，全部通过）

---

## 4. 分析图全景表（当前状态）

> **状态符号**：✅ = 分析模块 + 绘图脚本均完成，可对真实数据直接运行；❌ = 待开发

| 图 ID | 内容 | 分析模块 | 绘图脚本 | 状态 | YAML config 示例 |
|---|---|---|---|---|---|
| **F4** | 时间窗口 block 复用率 | `analysis/f4.py` ✅ | `generate_f4_business.py` ✅ | ✅ | `f4_synthetic_{reusable,prefix}.yaml` |
| **F13** | 单轮 reuse-time CDF | `analysis/f13.py` ✅ | `generate_f13_business.py` ✅ | ✅ | `f13_synthetic_{reusable,prefix}.yaml` |
| **reuse_rank** | 前缀命中 block 数排名曲线 | `analysis/reuse_rank.py` ✅ | `generate_reuse_rank_business.py` ✅ | ✅ | `reuse_rank_synthetic.yaml` |
| **E3 / top_ngrams** | TOP-N 最长连续 block 序列 | `analysis/top_ngrams.py` ✅ | `generate_top_ngrams_business.py` ✅ | ✅ | `top_ngrams_synthetic.yaml` |
| **E1** | per-user 命中率（4 block_size 曲线） | `analysis/user_hit_rate.py` ✅ | `generate_user_hit_rate.py` ✅ | ✅ | `e1_user_hit_rate_synthetic.yaml` |
| **E1-B 图1** | 用户命中贡献 Lorenz 曲线 | `analysis/user_hit_rate.py` ✅ | `generate_skewness.py` ✅ | ✅ | `e1b_skewness_synthetic.yaml` |
| **E1-B 图2** | 用户请求量 Lorenz 曲线 | `analysis/user_hit_rate.py` ✅ | `generate_skewness.py` ✅（同上脚本） | ✅ | 同上 |
| **E5** | TOP-N block 序列还原为原始文本 | `analysis/block_text_decoder.py` ✅ | `generate_e5_block_text.py` ✅ | ✅ | `e5_block_text_synthetic.yaml` |

---

## 5. 开发顺序（已完成 / 待完成）

```
✅ Step 1  io/business_loader.py + 测试（50 个测试通过）
✅ Step 2  analysis/request_classifier.py + 测试（22 个测试通过）
✅ Step 3  scripts/make_synthetic_business.py（合成数据生成验证）
✅ Step 4  复用模块 → 新建专用 *_business.py 脚本
            generate_f4_business.py / generate_f13_business.py
            generate_reuse_rank_business.py / generate_top_ngrams_business.py
            configs/phase2_business/ 下对应 8 个 YAML 文件
✅ Step 5  analysis/user_hit_rate.py（UserHitStats / UserHitSeries / SkewnessRow）
✅ Step 6  generate_user_hit_rate.py（E1：4-block_size 叠加曲线）
✅ Step 7  generate_skewness.py（E1-B 图1 命中贡献 + 图2 请求量 Lorenz 曲线）
✅ Step 8  block_registry 写入（business_loader.py 新增 block_registry 参数）
✅ Step 9  analysis/block_text_decoder.py + 测试（16 个测试通过）
✅ Step 10 E5：generate_e5_block_text.py + configs/phase2_business/e5_block_text_synthetic.yaml

Phase 2.5（性能，独立分支）
✅ Step A  index/radix_trie.py（RadixTrieIndex 核心实现）
✅ Step B  index/trie.py 添加 node_count() 统计方法
✅ Step C  replay.py 自动选择逻辑（avg_blocks >= 256 时切换 RadixTrieIndex）
✅ Step D  tests/test_radix_trie.py（38 个测试，正确性 + 等价性 + 压缩比 + auto-select）
✅ Step E  scripts/benchmark_index.py + YAML config（内存/速度对比）
```

---

## 6. 性能约束与 Phase 2.5 设计

### 6.1 Trie 内存评估（实测数字）

每个 TrieIndex 节点 ≈ 116 字节（dict 空壳 64B + hash 槽 16B + key int 36B）。

| 场景 | block_size | 平均输入 | 请求数 | TrieIndex | RadixTrieIndex | 压缩比 |
|---|---|---|---|---|---|---|
| 8K token 业务请求（128 共享块）| 128 | 8K | 5 万 | **0.37 GB** | 0.04 GB | ~10× |
| 32K token 业务请求（128 共享块）| 128 | 32K | 5 万 | **1.11 GB** | 0.09 GB | ~12× |
| 128K long-context（128 共享块）| 128 | 128K | 2 万 | **2.17 GB** | 0.16 GB | ~14× |

### 6.2 Phase 2 处理策略

1. **按时间窗口切片**：若 `total_block_count > 30M`，改用 24 小时窗口 replay；metadata 注明 `window_hours=24`
2. **block_size 选择**：Phase 2 主分析用 `block_size=128`（内存更友好，vLLM-Ascend 默认）；`16/32/64` 作为细粒度补充
3. **平均输入 >32K token 的模型**：Phase 2.5 完成后解锁，由 `replay()` 自动切换 RadixTrieIndex

### 6.3 RadixTrieIndex 实现规格（Phase 2.5）

#### 核心数据结构

```
RadixTrieIndex
  └── _root: _RadixNode
         children: dict[int, (array.array('Q'), _RadixNode)]
                         ↑              ↑
                   首个 block_id    边标签（uint64 数组）+ 子节点
```

**`_RadixNode`**（`__slots__` 降低单对象开销）：
```python
class _RadixNode:
    __slots__ = ("children",)
    children: dict[int, tuple[array.array, "_RadixNode"]]
```

**边标签存储**：`array.array('Q', block_ids_slice)`（unsigned 64-bit integer，8 bytes/element）  
覆盖 TraceA 小整数（< 2¹⁶）和 SHA-256 截断 uint64 两种 block_id 类型。  
**仅支持 int block_id**；传入 str 时 `array.array` 构造自动抛 `TypeError`。  
**不设 `is_terminal` 标记**：`PrefixIndex` 协议只需前缀连续计数，不需要完整序列区分。

#### 两个核心算法

**`longest_prefix_match`**（O(L)，L = 命中 block 数）：

```
matched = 0, pos = 0
while pos < len(block_ids):
    first = block_ids[pos]
    if first not in node.children: break
    edge, child = node.children[first]
    k = common_prefix_length(edge, block_ids, pos)
    matched += k
    if k < len(edge): break      # 边部分匹配 → 停止
    pos += k; node = child        # 全边匹配 → 继续
return matched
```

**`insert`**（含边分裂，O(L)）：

```
while pos < len(block_ids):
    if first not in node.children:
        node.children[first] = (array('Q', block_ids[pos:]), new_leaf)
        return
    edge, child = node.children[first]
    k = common_prefix_length(edge, block_ids, pos)
    if k == len(edge):
        pos += k; node = child   # 全边匹配 → 继续深入
    else:
        split_node = _RadixNode()
        split_node.children[edge[k]] = (edge[k:], child)   # 旧后缀 → 原孩子
        if pos + k < n:
            split_node.children[block_ids[pos+k]] = (array('Q', block_ids[pos+k:]), new_leaf)
        node.children[first] = (edge[:k], split_node)      # 公共前缀 → split
        return
# 若 while 正常退出：所有 blocks 已被现有路径覆盖，无需操作
```

#### 公开接口

```python
class RadixTrieIndex:
    # PrefixIndex 协议（与 TrieIndex 完全等价）
    def longest_prefix_match(self, block_ids: Sequence[int]) -> int: ...
    def insert(self, block_ids: Sequence[int]) -> None: ...

    # Benchmark 统计（不在 PrefixIndex 协议内）
    def node_count(self) -> int:      # _RadixNode 数量（含 root）
    def edge_count(self) -> int:      # 边数量 = node_count - 1
    def edge_label_bytes(self) -> int # 所有 edge label 数组的总字节数
```

`TrieIndex` 同步添加 `node_count() -> int`（迭代 DFS，不用递归，避免深链栈溢出）。

#### replay.py 自动切换逻辑

```python
_RADIX_THRESHOLD_AVG_BLOCKS: int = 256  # 对应 bs=128 时 32K chars/tokens 的均值

def _auto_index_factory(records: list[RequestRecord]) -> type:
    if not records:
        return TrieIndex
    avg = sum(len(r.block_ids) for r in records) / len(records)
    return RadixTrieIndex if avg >= _RADIX_THRESHOLD_AVG_BLOCKS else TrieIndex

def replay(
    records: Iterable[RequestRecord],
    index_factory: IndexFactory | None = None,   # None → 自动选择
) -> Iterator[PerRequestResult]:
    sorted_records = sort_records(list(records))
    if index_factory is None:
        index_factory = _auto_index_factory(sorted_records)
    ...
```

**阈值说明**：256 blocks 对应 bs=128 时均值 32K chars（CharTokenizer：1 char = 1 token）。  
bs=16 时触发更早（均值 4K chars），但内存压缩同样有效，属于安全保守策略。

#### Benchmark 设计

**脚本**：`scripts/benchmark_index.py`  
**指标**：tracemalloc 峰值内存、replay 耗时、node_count、edge_label_bytes、命中率一致性（硬断言）  
**输出**：终端对比表格 + `metadata.json`  
**Config 驱动**：YAML 指定 `input_file`、`block_size`、`output_dir`

#### 文件清单

| 操作 | 文件 |
|---|---|
| 新建 | `src/block_prefix_analyzer/index/radix_trie.py` |
| 新建 | `tests/test_radix_trie.py` |
| 新建 | `scripts/benchmark_index.py` |
| 新建 | `configs/phase2_business/benchmark_index_synthetic.yaml` |
| 修改 | `src/block_prefix_analyzer/index/trie.py`（添加 `node_count()`）|
| 修改 | `src/block_prefix_analyzer/replay.py`（`index_factory=None` + 自动选择）|

---

## 7. 输出目录约定

```
outputs/
  paper_repro/        # TraceA 论文复现（禁止用业务数据结果覆盖）
  phase2_business/    # 业务数据集分析结果
    f4_*/
    f13_*/
    reuse_rank_*/
    top_ngrams_*/
    e1_user_hit_rate_*/
    e1b_skewness_*/
    e5_block_text_*/  # E5：文本还原（top_ngrams decoded text）
  benchmark_index_*/  # Phase 2.5 benchmark 输出
```

---

## 8. 测试状态

当前测试套件：**670 passed，10 xfailed**（xfail 为 Layer-2/3 pending 占位，符合预期）。

| 测试文件 | 覆盖模块 | 用例数 |
|---|---|---|
| `test_business_loader.py` | `io/business_loader.py` | 28 |
| `test_request_classifier.py` | `analysis/request_classifier.py` | 22 |
| `test_block_text_decoder.py` | `analysis/block_text_decoder.py` + loader registry | 16 |
| `test_radix_trie.py` | `index/radix_trie.py` + `trie.py` + `replay.py` auto-select | 38 |
| 其余已有测试 | V1 + V2 全模块 | 566 |

每个核心模块有独立测试文件，fixture 为手工构造最小值，不依赖真实数据集。
