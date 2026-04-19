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
  - `tests/test_trie.py` 全部通过（14 个具名测试用例）

- [x] **Step 3 — JSONL Loader**
  - `LoadError(ValueError)`：fail-fast，含 1-based 行号
  - 强制字段 `request_id` / `timestamp` / `block_ids`；`arrival_index` 始终由 loader 按非空行顺序分配
  - `metadata` 原样保留（`time_unit` 等 key 不修改）
  - `tests/test_jsonl_loader.py` 全部通过（24 个测试）

- [x] **Step 4 — 回放引擎**
  - `replay(records, index_factory=TrieIndex) -> Iterator[PerRequestResult]`
  - 冻结 "query → yield → insert" 顺序；首请求冷启动命中 0；无 self-hit 由 SpyIndex 验证
  - `tests/test_replay.py` 全部通过（22 个测试）

- [x] **Step 5 — Metrics 聚合**
  - `PerRequestResult` 新增 `reusable_block_count`（按位置计数，不去重）
  - replay 内维护 `seen_blocks` 集合，严格在 yield 之后更新，保持无 self-hit
  - `MetricsSummary`（frozen dataclass）：6 个计数/汇总字段 + 2 个预算比率
  - `compute_metrics(results) -> MetricsSummary`：纯聚合，不重扫 block 序列
  - 分母仅含 non-empty 请求；全 empty 时比率安全返回 0.0
  - `tests/test_replay.py` 扩充 7 个 reusable 语义测试（共 29 个）
  - `tests/test_metrics.py` 全部通过（18 个测试，含端到端集成测试）

- [x] **Step 6 — Summary 与输出层**
  - `format_summary(MetricsSummary) -> str`：固定两列格式，整数 12 位右对齐，比率转换为 `%.4f%`
  - `summary_to_dict()` / `write_json()`：`dataclasses.asdict()` 直接序列化，key = 字段名
  - `csv_header()` / `summary_to_csv_row()` / `write_csv()`：`dataclasses.fields()` 保证 header 与 row 永远对齐
  - `write_text()`：写出 `format_summary` 文本 + 尾换行
  - 标准库实现，无第三方依赖
  - `tests/test_summary.py` 全部通过（29 个测试）
  - **V1 主链路闭环**：132 passed, 0 skipped

---

## 后续步骤


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

## 冻结的设计决定（已确认）
1. **空 `block_ids`**：记录保留在流中，但**排除出所有命中率分母**。
2. **首条请求**：计入分母，命中记 0（冷启动语义）。
3. **分支名**：已改为 `main`。
4. **`metadata["time_unit"]`**：Loader 原样保留；analyzer 核心层不读不转换。
5. **`arrival_index`**：始终由 loader 按文件行序分配，JSON 字段被静默覆盖。

## 待 Step 6 确认的细节
- CSV / JSONL 输出的具体字段名（snake_case 已确认，字段集 Step 6 再定）。
- `block_level_reusable_ratio` 所需的 per-block 可见性数据结构（Step 5 设计时需补充到 `PerRequestResult` 或另行传入）。
