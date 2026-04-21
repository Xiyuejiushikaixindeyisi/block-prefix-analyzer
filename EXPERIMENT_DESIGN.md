# Experiment Design — Business Dataset Prefix Cache Analysis

本文档针对"仅有原始 prompt 的业务数据集"，评估当前实现的完整性、实验设计的合理性，并给出完整的实验方案。

> **当前阶段（Phase 2）范围声明**：仅分析**单轮 text 生成式模型请求**。Agent 多轮场景（F14、E2、E4）暂缓见第 11 节；跨模型对比、容量估算、租户时序模式等（E6–E9）为 Phase 3，见第 12 节。

---

## 1. 数据集字段要求

### 1.1 必须提供的字段

| 字段 | 类型 | 用途 |
|---|---|---|
| `user_id` | str / int | 按用户聚合 prefix 命中率（E1 核心分组键） |
| `request_id` | str | 唯一标识单次请求；重复出现的 request_id = Agent session（当前阶段过滤掉） |
| `timestamp` | float (seconds) | 时序回放排序；reuse_time 计算 |
| `raw_prompt` | str | Path B 流水线输入：tokenize → block hash |
| `model_id` | str | 按模型分流 replay（保证同模型假设）；过滤 embedding 模型 |

### 1.2 建议提供的字段

| 字段 | 影响 |
|---|---|
| `input_token_count` | 提前估算 Trie 内存压力；无此字段则在 Path B 后统计 |
| `department_id` / `org_id` | 当前缺失；补充后解锁部门级分析（Phase 3） |
| `output_length` (token 数) | token 粒度 prefix 命中率；暂时不影响 block 级分析 |

### 1.3 不需要提供的字段

- `parent_id` / `chat_id`：当前阶段只处理单轮请求，无需 session 树结构
- `hash_ids`：Path B 流水线自动从 `raw_prompt` 计算

---

## 2. 系统约束与前提假设

| 约束 | 说明 |
|---|---|
| **同模型假设** | 所有请求来自同一模型的同一部署实例。KV cache 复用的物理有效性依赖此假设；不同模型的 K/V tensor 不可共享。**必须按 `model_id` 分流后再 replay，禁止跨模型混合回放**。 |
| **仅生成式模型** | Embedding 模型（bge-m3、QWen-V3-Embedding-0.6B 等）不走 autoregressive 推理，无 KV prefix cache，必须在 loader 阶段按白名单过滤掉。 |
| **仅 text 请求** | 当前系统只接受文本输入，无多模态。无需 extra_keys 中的多模态 identifier，ChainedBlockBuilder 不传入 mm_feature。 |
| **单轮为主** | 当前阶段仅处理同一 `request_id` 在数据集中只出现一次的请求（独立单轮对话）。 |

---

## 3. Path B 流水线依赖（新数据集必须）

**重要澄清**：F4、F13、block_freq、reuse_rank、top_ngrams、E3 等分析模块本身**不依赖** mmh3 或 transformers，它们只消费 `block_ids`（整数列表）。依赖来自**输入格式**：业务数据集只有 raw prompt，没有预计算的 `hash_ids`，必须先通过 Path B 将 raw prompt 转换成 `block_ids`，后续所有分析才能启动。一旦 `block_ids` 生成完毕，所有实验均可独立运行。

新数据集中没有预计算的 `hash_ids`，必须走 Path B：

```
raw_prompt
  → ChatTemplateAdapter (render to token string)
  → TokenizerAdapter    (text → token IDs)  ← 需 pip install transformers
  → ChainedBlockBuilder(block_size=N)        ← 需 pip install mmh3
  → block_ids (list[int])
```

**必须安装的依赖：**
```bash
pip install transformers  # Layer 2: 与部署 vLLM 使用相同的 tokenizer
pip install mmh3          # Layer 3: MurmurHash3，vLLM 链式哈希
```

**block_size 必须显式指定**，且与业务系统实际部署的 vLLM `block_size` 保持一致（常见值：16）。`build_block_records_from_raw_requests()` 现已强制要求此参数，缺失时抛 `ValueError`。

**NONE_HASH 说明**：`ChainedBlockBuilder(initial_hash=0)` 为分析占位值。对同一数据集内部的相对 prefix 命中率统计，固定种子已经足够准确；若需精确复现某 vLLM 部署的实际 block_id 值（如做跨系统 hash 对比），需从该部署获取其 NONE_HASH。

---

## 4. 请求人群分类

### 4.1 单轮文本请求（当前阶段处理对象）

**定义**：同一 `request_id` 在数据集中只出现一次，且 `raw_prompt` 不含 Agent 标识。

**实现**：新建 `analysis/request_classifier.py`。

**判断优先级**：
1. `request_id` 是否重复出现（主判据，可靠性高）：重复 → Agent session → **过滤掉**
2. `raw_prompt` 关键词检测（辅助，补充召回）：含 `tool_call` / `opencode` / `claudecode` 等 → 标记为疑似 Agent，二期再处理

### 4.2 Agent 多轮请求（当前阶段跳过）

Agent session 分析（request_id 重复出现）已识别但整体过滤，不参与本阶段实验。推迟原因见第 11 节。

---

## 5. 论文实验复现评估

| 实验 | 可行性 | 前提条件 | 适配说明 |
|---|---|---|---|
| **F4** block 频率分布 | ✅ 可行 | Path B 完成 | 无需适配 |
| **F13** 单轮 reuse_time CDF | ✅ 可行（需适配） | Path B 完成 | 无 parent_id，以全局时序重放代替 session 内重放；reuse_time 定义为"block 距上次在任意历史请求中出现的时间差"，语义与 TraceA F13 一致 |
| **F14** Agent reuse_time | ⏸ 暂缓 | — | Agent session 边界无法可靠确定，见第 11 节 |
| **block_freq** | ✅ 可行 | 同 F4 | 无需适配 |
| **prefix_hash_alignment** | ✅ **可行** | Path B 完成 | 目标是将 TOP-10 高频 block 序列**还原为原始 prompt 文本**，从而识别被大量复用的 system prompt 或 tool schema 片段。实现方式：Path B 运行时为每个 block 保存 `{block_id → decoded_text}`，TOP-10 分析完成后反向查找对应文本。无需 vLLM 部署侧数据，完全在本地数据集上实现（详见 E5）。 |
| **reuse_rank** | ✅ 可行 | Path B 完成 | 已有 `analysis/reuse_rank.py` 实现 |
| **top_ngrams** | ✅ 可行 | Path B 完成 | 已有 `analysis/top_ngrams.py` 实现 |

---

## 6. 新实验设计

### E1：单轮请求 — 按用户理想 prefix 命中率分布（4 种 block_size）

**研究问题**：不同用户从 shared KV prefix cache 中获益的程度是否高度异质？block_size 如何影响这种异质性？

**指标定义**：

```
per_user_ideal_hit_rate(u) =
    Σ content_prefix_reuse_blocks  (u 的所有单轮请求)
    ──────────────────────────────────────────────────
    Σ total_blocks                 (u 的所有单轮请求)
```

命中率分子/分母均来自**全局时序 replay**（所有用户请求按 timestamp 混合排序），即 shared KV cache 语义：当前请求的 prefix 可命中任何更早请求留下的缓存，不限于同一用户。

**前提**：loader 必须将 `user_id` 写入每条 `RequestRecord.metadata["user_id"]`，replay 结果通过 `result.request_id` 反查记录的 metadata 获取 user_id，再聚合。

**方法**：

1. 全局时序重放所有单轮文本请求
2. 按 `user_id` 聚合 `PerRequestResult`
3. 过滤长尾用户：**不宜硬编码阈值**，应先统计各用户的 `total_blocks` 分布，选取 P5 或 P10 作为截断点，保留统计上有意义的用户群体
4. 计算每用户 hit_rate，按降序排列
5. 对 `block_size ∈ {16, 32, 64, 128}` 各独立运行一次 Path B + replay

**图表规格**：

- 子图 a：4 条曲线叠加在同一坐标系
  - X 轴：用户序号（按 hit_rate 降序排列的 rank，1 = 命中率最高用户）
  - Y 轴：理想 prefix 命中率（0 ~ 1）
  - 每条曲线对应一种 block_size，图例标注 block_size
- 子图 b（可选）：各 block_size 下，hit_rate > 0.5 的用户占比（条形图）

**预期结论**：
- 命中率分布不均匀（少数高复用用户拉高整体命中率）
- block_size 越大，高命中率用户比例下降（更大块匹配更严格）

**实现状态**：需新建 `src/block_prefix_analyzer/analysis/user_hit_rate.py` 和对应 script

---

### E1-B：复用偏斜分析（Lorenz-curve 风格）

**研究问题**：系统整体的 KV prefix 命中贡献是否高度集中于少数用户？请求量是否同样偏斜？两者的偏斜模式是否一致？

这两张图揭示 shared KV cache 收益的"马太效应"：识别哪类用户是 cache 的主要受益者，并辨别"高命中率"与"高请求量"是否来自同一批用户。

#### 图 1：用户命中贡献累积分布

> 每个用户贡献的绝对命中 block 数在所有用户中的分布集中程度

**数据来源**：全局时序 replay 后，按 user_id 聚合 `Σ content_prefix_reuse_blocks`（绝对命中 block 数，非命中率）。

**坐标轴**：

| 轴 | 规格 |
|---|---|
| X 轴 | 用户序号（rank），按每用户命中 block 总数**从高到低**排序；1 = 命中贡献最多的用户 |
| Y Left（柱形） | 当前用户命中 block 数归一化值 = `hit_blocks(u) / max_hit_blocks`；Legend：**Per-user** |
| Y Right（折线） | 累积占比 = `Σ hit_blocks(rank 1..k) / Σ hit_blocks(all users)`；Legend：**Cumulation** |

**图例**：
- Per-user（柱）：单用户命中数（归一化）
- Cumulation（线）：累积占比（0→1）

**解读**：若折线在左侧 10% 处即达到 0.8，意味着最高命中贡献的 10% 用户贡献了系统 80% 的 prefix cache hit。

---

#### 图 2：用户请求量分布

> 请求量（或等价的 total_blocks）在用户间的分布集中程度

**数据来源**：全局请求池，按 user_id 聚合 `request_count` 或 `Σ total_blocks`。

**坐标轴**：

| 轴 | 规格 |
|---|---|
| X 轴 | 用户序号（rank），按每用户请求数**从高到低**排序；1 = 请求最多的用户 |
| Y Left（柱形） | 当前用户请求数归一化值 = `request_count(u) / max_request_count`；Legend：**Per-user** |
| Y Right（折线） | 累积占比 = `Σ request_count(rank 1..k) / Σ request_count(all users)`；Legend：**Cumulation** |

**图例**：
- Per-user（柱）：单用户请求数（归一化）
- Cumulation（线）：累积占比（0→1）

**解读**：若折线在左侧 5% 处达到 0.5，意味着最活跃 5% 的用户贡献了一半请求量。

---

#### 两图联合解读（分析目标）

| 比较维度 | 含义 |
|---|---|
| 图 1 偏斜 ≈ 图 2 偏斜 | 命中贡献集中主要来自重度用户的请求量优势，而非其请求内容特别"cache-friendly" |
| 图 1 远比图 2 偏斜 | 少数用户的**请求内容**对 prefix cache 特别友好（高重复系统 prompt / 固定模板），是优化重点 |
| 图 1 比图 2 更平均 | cache 收益分散，轻度用户也能获得不错命中率（系统 prompt 跨用户共享效果好） |

**实现状态**：`user_hit_rate.py` 需新增 `compute_skewness_stats(results, records)` 返回 per-user `(hit_blocks, request_count)` 对；绘图逻辑新建 `scripts/generate_skewness.py`，双 Y 轴用 `matplotlib` `twinx()`，柱形 + 折线叠加。

---

### E2（暂缓）：跨用户复用热度图

> 本实验在单轮场景下依然有意义，但需要扩展 replay 层以追踪 block provenance（`block_id → 最早写入该 block 的 user_id`），工程量较大，列为 P3。

**研究问题**：单轮请求中，哪些用户对之间发生了跨用户 KV 缓存共享？

- **X 轴**：source user_id（缓存的贡献方）
- **Y 轴**：target user_id（缓存的受益方）
- **格子颜色**：user_y 的请求命中 user_x 贡献的 block 的次数，颜色越深复用越多
- 对角线（自复用）排除或单独展示

**实现状态**：需在 replay 或 index 层新增 `ProvenanceIndex`，当前不支持。暂缓至 P3。

---

### E3：单轮请求 — TOP10 最长连续 block 组合（4 种 block_size）

**方法**：复用已有 `analysis/top_ngrams.py`，对 4 种 block_size 各跑一次。

**输出**：每种 block_size 输出一张表格（rank, count, pct, len, block_ids 前 8 个）

**分析意义**：识别跨用户的公共 hot prefix（如常用 system prompt 或固定 few-shot 模板），为 KV cache 预热策略提供依据。

**实现状态**：已有实现，只需增加 block_size 参数循环。

---

### E4（暂缓）：Agent 调用 — TOP10 连续 block 组合与 system prompt 设计含义

> Agent 场景下 per-call TOP-10 分析是 system prompt 设计的核心依据，但须先解决 Agent session 边界问题，见第 11 节。

---

### E5：prefix_hash_alignment — TOP-10 block 序列还原为原始文本

> **目的**：给 TOP-10 高频 block 序列配上可读的原始 prompt 文本，直接识别哪些 system prompt / tool schema 片段被大量复用。

**核心思路**：block_id 是 token 序列的链式 hash（不可逆），但 hash 的**正向过程可在 Path B 中同步记录**。只要在生成 block_ids 时顺便保存 `{block_id → 对应 token_ids → 对应原始文本}`，TOP-10 序列即可直接反查。

**实现步骤**：

1. **Path B 运行时建立 block registry**

   在 `ChainedBlockBuilder.build()` 中，每生成一个 `(block_id, token_slice)` 时同步写入一张全局注册表：

   ```python
   # block_registry: dict[int, str]  — block_id → decoded text
   block_registry[block_id] = tokenizer.decode(token_ids[i : i + block_size])
   ```

   因为 block_id 是链式 hash（相同 token 前缀 + 相同位置 → 相同 block_id），同一 block_id 的文本内容必然相同，只需存储一份。

2. **TOP-10 分析后反查**

   `top_ngrams.py` 产出 TOP-10 block 序列（`blocks: tuple[int, ...]`），对每个序列：
   - 逐 block_id 查 `block_registry`，拼接对应文本
   - 输出：rank, count, length, decoded_text（前 N 字符）

3. **输出格式**

   | Rank | Count | Pct(%) | Len | 还原文本（截断到 200 字符） |
   |---|---|---|---|---|
   | 1 | … | … | … | `You are a helpful assistant. You have access to the following tools: ...` |
   | 2 | … | … | … | `{"type": "function", "function": {"name": "bash", ...}}` |

**关键约束**：

- `block_registry` 中存储的是 **tokenizer decode 的结果**，而不是原始字符串切片，两者可能因 tokenization boundary 有轻微偏差（subword token 跨字符边界时），但对 system prompt 识别足够精确
- 此实验**不需要从 vLLM 获取任何数据**，也不需要与 vLLM 内部 block_id 做比对；只需保证本地 Path B 使用的 tokenizer 与生成该 prompt 时的 tokenizer 一致（同模型）
- 注意 token decode 的隐私问题：还原的文本会暴露原始 system prompt 内容，输出结果须作为内部分析数据处理，不对外发布

**实现状态**：需在 `ChainedBlockBuilder.build()` 或 Path B loader 中增加 registry 写入逻辑；新建 `analysis/block_text_decoder.py` 提供 `decode_ngram_rows(rows, block_registry, tokenizer)` 接口

---

## 7. 实验合理性评判

### 设计亮点

1. **per-user 粒度分析**是对 TraceA 论文的有益补充。TraceA 仅展示聚合统计（全局命中率），忽略了用户间的异质性。展示命中率的用户分布，可以回答"谁在占用前缀缓存？"。

2. **4 种 block_size 对比**是新颖贡献。TraceA 固定使用 block_size=16（与 Qwen 模型 vLLM 部署一致），本实验系统量化 block_size 对命中率和高效模式的影响，对不同部署配置有指导意义。

3. **TOP-10 连续 block 组合（E3）直接连接工程实践**：通过识别高频 block 序列，可以给出 system prompt 标准化、排列顺序优化和预热策略的具体建议，而不仅仅是统计描述。

4. **E5 prefix_hash_alignment** 将 block 分析结果转化为可读文本，使分析结论对非技术决策者也能直接理解。

### 设计问题与建议修正

| 问题 | 说明 | 建议 |
|---|---|---|
| **用户数量过滤阈值** | 请求数极少的用户（如只有 1~2 次请求）的命中率统计意义低，会污染曲线 | 先统计用户 total_blocks 分布，取 P5/P10 作为截断点，不宜硬编码 50 |
| **单轮分类准确率** | 通过关键词检测分类 Agent vs. text，存在误判 | 优先用 request_id 是否重复出现作为主要判断依据（更可靠），关键词检测仅作辅助 |
| **E1/E2 X 轴用 rank 而非 user_id** | user_id 明文不应出现在论文图中（隐私问题） | 使用 rank（降序序号）作为 x 轴，不暴露 user_id |

---

## 8. 实现缺口汇总

| 缺口 | 影响实验 | 解决方案 | 优先级 |
|---|---|---|---|
| `transformers` 未安装 | 全部新数据集实验 | `pip install transformers`（指定与业务 vLLM 相同的模型名） | P0 |
| `mmh3` 未安装 | 全部新数据集实验（chained hash） | `pip install mmh3` | P0 |
| 新数据集 loader 缺失 | 全部 | 开发 `io/business_loader.py`：读入 `user_id, request_id, timestamp, raw_prompt` 的 JSONL，调用 `build_block_records_from_raw_requests()`；将单轮 / Agent 请求分流 | P0 |
| 单轮请求分类器缺失 | F4, F13, E1, E3, E5 | 开发 `analysis/request_classifier.py`：request_id 重复检测（主）+ 关键词检测（辅） | P1 |
| per-user hit rate 分析模块缺失 | E1, E1-B | 开发 `analysis/user_hit_rate.py`：per-user 聚合 hit_rate + `compute_skewness_stats()` 返回 per-user `(hit_blocks, request_count)` | P1 |
| 偏斜分析绘图脚本缺失 | E1-B | 新建 `scripts/generate_skewness.py`：双 Y 轴（柱形 + 折线）绘制命中贡献 / 请求量累积分布图 | P1 |
| 多 block_size sweep 脚本缺失 | E1, E3 | 开发统一 sweep 入口脚本，循环 block_size ∈ {16, 32, 64, 128} 并汇总输出 | P2 |
| block text decoder 缺失 | E5 | 开发 `analysis/block_text_decoder.py`；在 Path B 中增加 `block_registry` 写入逻辑 | P2 |
| 跨用户复用热度图的 provenance 追踪缺失 | E2（暂缓） | 扩展 replay 或 index 层，维护 `{block_id → user_id}` 来源映射；当前 replay 不记录命中来源 | P3 |

---

## 9. 实验执行顺序

```
Phase 1 — 环境与依赖
  [1] pip install transformers mmh3
  [2] 确认 tokenizer 名称（与业务 vLLM 部署相同的模型）
  [3] 确认业务 vLLM 实例的 block_size（通常 16）

Phase 2 — 数据 loader 开发
  [4] 开发 io/business_loader.py（raw prompt JSONL → RequestRecord list，单轮 / Agent 分流）
  [5] 开发 analysis/request_classifier.py（single-turn text 过滤，Agent 记录但不处理）
  [6] 单元测试 loader 和 classifier

Phase 3 — 论文复现（新数据集版本，单轮请求）
  [7] F4 block 频率分布
  [8] F13 单轮 reuse_time CDF（global replay 语义）
  [9] reuse_rank（已有 reuse_rank.py）
  [10] top_ngrams（已有 top_ngrams.py，block_size=16 对应 F15）

Phase 4 — 新实验
  [11] 开发 analysis/user_hit_rate.py（per-user 聚合 hit_rate + skewness stats）
  [12] E1 单轮 per-user hit rate sweep（block_size ∈ {16,32,64,128}）
  [13] E1-B 复用偏斜分析：命中贡献累积分布图 + 请求量累积分布图（scripts/generate_skewness.py）
  [14] E3 TOP10 contiguous blocks sweep（各 block_size）
  [15] E5 block text decoder：TOP-10 序列还原原始文本（需 block_registry）

Phase 5 — 进阶实验（可选）
  [15] E2 跨用户复用热度图（需 provenance 追踪扩展，P3）
```

---

## 10. 新 loader 接口草案

业务数据集 JSONL 每行格式（字段名可在 loader 中映射）：

```json
{
  "user_id": "u123",
  "request_id": "req-456",
  "timestamp": 1718000000.0,
  "prompt": "You are a helpful assistant...\nUser: ..."
}
```

Loader 调用示例：

```python
from block_prefix_analyzer.io.business_loader import load_business_jsonl
from block_prefix_analyzer.v2.adapters.siphash_builder import ChainedBlockBuilder

single_turn_records = load_business_jsonl(
    "data/business/requests.jsonl",
    block_size=16,                          # 必须与业务 vLLM 一致
    block_builder=ChainedBlockBuilder(block_size=16),   # vLLM-aligned hash
    request_type="single_turn",             # 只返回单轮请求
)
```

`load_business_jsonl` 内部：
1. 按 `request_id` 统计出现次数，重复 = Agent session → 过滤（当前阶段）
2. 对单轮请求调用 `build_block_records_from_raw_requests()`
3. 将 `user_id` 存入 `record.metadata["user_id"]`

---

## 11. 未来工作 — Agent 场景（暂缓原因）

当前阶段不分析 Agent 多轮请求，原因如下：

**技术限制**：
- 业务数据集用 `request_id` 重复出现隐式标识 Agent session，但无法区分一个 session 内的**调用角色**（规划调用 vs. 工具调用 vs. 最终回答），也无法判断 session 边界是否清晰（同一 `request_id` 是否跨多个独立 Agent 任务）。
- Agent session 内部，随着轮次增加，prompt 累积历史上下文，prefix 命中率天然极高（接近 100%）；这一高命中率来自**上下文积累**（同 session 内），而非跨 session / 跨用户的真正 KV cache 共享。两种复用机制混合在一起，统计结果难以解读。

**后续条件**：待数据集补充 `session_id`（或 `parent_request_id`）字段，能够明确区分"同 session 内调用"和"跨 session 新起始"后，再开启以下实验：

| 暂缓实验 | 恢复前提 |
|---|---|
| **F14** Agent reuse_time CDF | 明确 session 边界（`session_id` 或 `parent_request_id`） |
| **E2** Agent per-call hit ratio CDF / 跨用户热度图 | 同上，且需区分 session 内复用 vs. 跨 session 复用 |
| **E4** Agent TOP-10 block 序列 + system prompt 设计含义 | 同上，且 E5 block text decoder 已完成 |

---

## 12. 未来工作 — Phase 3（多模型 + 容量 + 时序）

以下实验与本阶段单轮 text 理想命中率研究无关，列于此处供 Phase 3 参照。

### 12.1 关键修正（Phase 3 前必须落地）

**`model_id` 字段已在 Phase 2 中作为必须字段引入**（用于过滤 embedding 模型 + 分流生成式模型 replay）。Phase 3 在此基础上扩展到跨模型对比。

**Phase 3 适用的生成式模型列表（按调用量排序）**：

| 模型 | 调用量（参考） | 平均输入 | block_size=16 均块数 | Phase 3 priority |
|---|---|---|---|---|
| QWen-v3-32B-8K | 660,916 | ~2–3K tokens | ~150 | P0（Phase 2 主分析对象） |
| QWen-V3-32B-32K | 140,396 | ~10K tokens | ~625 | P1 |
| Deepseek-V3-671B-8K | 87,553 | 待测 | 待测 | P1 |
| QWen-V3.5-27B-128K | 待测 | **~87K tokens** | **~5,437** | ⏸ 需 Phase 2.5（RadixTrieIndex）后开放 |
| GLM-4.7-HCMAAS | 待测 | **~49K tokens** | **~3,062** | ⏸ 需 Phase 2.5（RadixTrieIndex）后开放 |
| QWen-V3.5-27B-64K | 待测 | **~46K tokens** | **~2,875** | ⏸ 需 Phase 2.5（RadixTrieIndex）后开放 |

**必须排除的 embedding 模型**（在 loader 白名单中过滤）：
- `bge-m3`（439,168 次）
- `QWen-V3-Embedding-0.6B`（175,362 次）

### 12.2 E6：按生成式模型分组的 prefix 命中率对比

**研究问题**：三个主力生成式模型的 prefix cache 收益差距有多大？哪个模型最值得投入优化？

**方法**：按 `model_id` 分组，每组独立 replay，计算全局 `content_prefix_reuse_rate` 和 per-user 分布。

**图表**：
- 条形图：各模型全局 prefix 命中率对比（同一图中并排）
- 叠加折线图：各模型 per-user 命中率排名曲线（X = rank，Y = hit_rate）

---

### 12.3 E7：8K vs 32K 上下文规格的 prefix 命中模式对比

**研究问题**：同底座模型（QWen-v3-32B）的 8K 和 32K 配置，prompt 长度分布和命中模式是否存在系统性差异？是否会影响"推荐用户迁移到长上下文"的决策？

**图表**：
- 双直方图：8K vs 32K 用户的 `total_blocks` 分布（prompt 长度代理）
- 双 CDF：8K vs 32K 的 `content_prefix_reuse_rate` per-user 分布

---

### 12.4 E8：Prefix 命中率的时间周期性分析

**研究问题**：流量随时间波动剧烈，prefix cache 命中率是否随之波动？高峰期缓存是更热还是更冷？

**数据来源**：F4 时间窗口框架 + 运营看板模型调用量时序。

| 子图 | X 轴 | Y 轴 | 目的 |
|---|---|---|---|
| 每小时命中率热力图 | 一天中的小时（0–23） | 一周中的天 | 发现"缓存友好"时间窗口 |
| 调用量 vs 命中率散点图 | 时间窗口调用量 | 时间窗口 prefix 命中率 | 判断高峰是否与高命中率正相关 |
| 分模型时序曲线 | 时间 | 各模型 prefix 命中率 | 不同模型命中率时序是否同步 |

**预期结论**：
- 若"高峰 → 更热"：同类请求短时集中爆发（定时批任务），高峰期优先分配 cache 有价值
- 若"高峰 → 更冷"：高峰由多样化新用户驱动，cache 压力最大时收益最低，需扩容

---

### 12.5 E9：租户调用量时序模式分类 + 与命中率的关联

**研究问题**：租户调用量随时间变化剧烈，是否存在可识别的模式类型（脉冲型 / 持续型 / 交互型 / 周期型）？不同模式的租户 prefix cache 命中率是否有系统性差异？

**分类依据**（简单两维分类，无需机器学习）：
- 峰值 / 均值比（衡量脉冲程度）
- 活跃小时占比（衡量持续性）

| 模式类型 | 特征 | 对 prefix cache 的含义 |
|---|---|---|
| Batch 脉冲型 | 短时大量 → 长时沉默 | 批次内请求相似，hit 率极高 |
| 稳定持续型 | 全天均匀，量中等 | 命中率取决于内容相似度 |
| 交互型 | 随机分散，间隔长 | 缓存已被淘汰，hit 率低 |
| 周期型 | 工作日固定时段活跃 | 日内缓存有效，跨日可能失效 |

**图表**：散点图（X = 峰均比，Y = prefix 命中率，颜色 = 模式类型）+ 各类型命中率箱线图

---

### 12.6 Phase 3 恢复前提

| 实验 | 恢复前提 |
|---|---|
| E6 | `model_id` 字段可用（Phase 2 已要求），且 transformers/mmh3 对齐完成 |
| E7 | 同 E6；QWen-v3-32B-8K 和 32K 的 tokenizer 确认为同一模型 |
| E8 | 运营看板可导出时序数据（与 replay 结果按时间窗口对齐） |
| E9 | `user_id` 可用（Phase 2 已要求）；租户时序数据可导出 |
| 长输入模型（128K/64K/49K avg） | Phase 2.5 完成（`RadixTrieIndex` 实现并通过等价性测试），全量内存降至 2–8 GB 范围内 |
