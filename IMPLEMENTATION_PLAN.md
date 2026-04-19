# IMPLEMENTATION_PLAN.md — V1 分步实施计划

本文档是 `block-prefix-analyzer` 到达 V1 可用状态的最小可评审路径。
每一步结束时仓库都必须保持 **可安装 + pytest 全绿**。
每一步对应 **一到两个 commit**，不混合架构改动与新功能。

---

## 当前进度
- [x] **Step 0 — 骨架与打包**
  - 目录结构、`pyproject.toml`、`.gitignore`、README / CLAUDE / 本文件
  - `RequestRecord` 数据模型与排序 key；`PrefixIndex` 抽象协议
  - 各模块桩 + 对应 test 文件（`pytest.skip` 占位）；`pytest` 全绿

- [x] **Step 1 — 数据模型与排序助手**
  - `sort_records()` 帮助函数；`time_unit` 元数据约定写入 docstring
  - 冻结语义：空 `block_ids` 保留记录但**不计入分母**；首条请求**计入分母、命中按 0**
  - `tests/test_types.py` 全部通过（稳定排序、空 block_ids、首记录语义）

- [x] **Step 2 — PrefixIndex 接口与简单 Trie**
  - `TrieIndex.insert` + `TrieIndex.longest_prefix_match` 实现（等值匹配，O(L)，无路径压缩）
  - `tests/test_trie.py` 全部通过（9 个具名测试用例，覆盖空 trie / 单插入 / 分叉 / 幂等 / 共享前缀 / 重复 block / 增量）

---

## 后续步骤

### Step 3 — JSONL Loader（test: loader 转绿）
**目标**：从 JSONL 读取最小字段集，严格分配 `arrival_index`，应用稳定排序。

**改动**：
- `src/block_prefix_analyzer/io/jsonl_loader.py`：
  - 强制字段：`request_id`, `timestamp`, `block_ids`。
  - 可选：`token_count`, `block_size`, `metadata`。
  - `arrival_index` 由 loader 按读取顺序分配（用户字段被忽略并记录 warning 级提示）。
  - 返回按 `(timestamp, arrival_index)` 排序的 `list[RequestRecord]`。

---

### Step 3 — JSONL Loader（test: loader 转绿）
**目标**：从 JSONL 读取最小字段集，严格分配 `arrival_index`，应用稳定排序。

**改动**：
- `src/block_prefix_analyzer/io/jsonl_loader.py`：
  - 强制字段：`request_id`, `timestamp`, `block_ids`。
  - 可选：`token_count`, `block_size`, `metadata`。
  - `arrival_index` 由 loader 按读取顺序分配（用户字段被忽略并记录 warning 级提示）。
  - 返回按 `(timestamp, arrival_index)` 排序的 `list[RequestRecord]`。
- `tests/test_jsonl_loader.py`：
  - 同时间戳样本的顺序保留。
  - 缺字段 / 类型错误时报明确异常。
  - 空 `block_ids` 的处理策略（保留并留给 metrics 决定是否剔除）。

---

### Step 4 — 回放引擎（test: replay 转绿）
**目标**：实现 V1 核心环 —— 对每条请求先算前缀指标，再插入索引；保证 no self-hit。

**改动**：
- `src/block_prefix_analyzer/replay.py`：
  - 函数式 API：`replay(records, index_factory=TrieIndex) -> Iterator[PerRequestResult]`。
  - `PerRequestResult` 至少包含：`request_id`, `total_blocks`, `prefix_hit_blocks`, `timestamp`, `arrival_index`。
  - 禁止调用方在 yield 之前看到索引已写入；实现上明确 "measure → insert" 顺序。
- `tests/test_replay.py`：
  - **首请求** 永远得到 `prefix_hit_blocks == 0`。
  - 两条完全相同的请求：第二条必须命中全部 block（用于排除"多算自己"）。
  - 相同 timestamp、不同 arrival_index 的请求：后到者能命中前者。
  - 长链式共享前缀场景下的增量命中正确。

---

### Step 5 — Metrics（test: metrics 转绿）
**目标**：把 per-request 结果聚合成报告指标，**micro + macro 双口径同时输出**。

**改动**：
- `src/block_prefix_analyzer/metrics.py`：
  - `prefix_aware_ideal_hit_ratio`（micro + macro）
  - `block_level_reusable_ratio`（micro + macro）
  - `token_level_prefix_hit_ratio`（在 `block_size` 与 `token_count` 齐全时启用）
  - `reuse_time_stats`（默认 `last_seen`；输出 mean/p50/p95）
  - 所有函数纯函数式，仅依赖 `PerRequestResult` 列表 + 原始 `RequestRecord`。
- `tests/test_metrics.py`：
  - crafted 3～5 条请求的手工期望值。
  - 空 `block_ids` 被正确从分母剔除。
  - 首请求不应把分母污染成 0（兜底行为测试）。

---

### Step 6 — Summary 与输出（test: summary 转绿）
**目标**：把 metrics 打印成人可读汇总，并提供 CSV / JSONL 写出。

**改动**：
- `src/block_prefix_analyzer/reports/summary.py`：
  - `format_summary(metrics) -> str`：表格化中文摘要。
  - `write_jsonl(results, path)` / `write_csv(results, path)`。
- `tests/test_summary.py`：
  - 输出 schema 固定（字段顺序、字段名、数值精度）。
  - 空输入的降级输出。

---

### Step 7 — 端到端 Golden 测试
**目标**：锁定 V1 对外可观察行为。

**改动**：
- 新增 `tests/fixtures/` 目录（本轮未建，Step 7 再建）。
- 2～3 个小的 JSONL 样本 + 期望输出 JSON。
- 任意修改导致 golden 差异都必须走 "解释 + 刷新 golden" 的两步流程。

---

## Out of scope（直到 V1 完成都不做）
- tokenizer / chat template / block hash 复现（V2）
- radix 压缩、磁盘索引、增量流式（V3）
- CLI、Web UI、可视化、实时服务
- 多进程 / 分布式
- 与任何在线推理框架的运行时耦合

---

## 未决问题（等待人类确认）
1. `block_ids` 空的请求是否计入分母？（当前倾向：剔除，可由标志位切换）
2. `reuse_time` 的时间戳单位是否由 metadata 里的 `time_unit` 字段声明？
3. 首个请求是否计入分母？（当前倾向：计入，hit 计 0，贴合冷启动）
4. 本地分支是 `master`，Spec 推荐 `main`；是否需要改名？
5. CSV / JSONL 输出的字段命名惯例（snake_case 已定，但具体字段名需要 Step 6 再确认）。
