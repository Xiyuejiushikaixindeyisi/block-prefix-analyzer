# V3 实验计划

本文档记录 Phase 3 的三项主要实验方向及对应代码实施计划。
实施顺序按由易到难排列：数据适配 → Agent 多轮 → 全天时序扩展。

---

## 背景与现状

- **已完成分析**：Qwen3-32B-8K、Qwen3-32B-32K、DeepSeek-V3-32K、DeepSeek-V3-8K（09:00–11:00 窗口）
- **关键观察**：
  - 短上下文模型（8K）流量随时间波动剧烈；长上下文模型（32K）仅呈现昼夜周期特征
  - Qwen3-32B-8K 和 Qwen3-32B-32K 调用量最多；Qwen3-32B-32K 端到端时延更长，TTFT 优化需求最强，是后续重点研究对象
  - Qwen-V3.5-27B-64K / 128K 数据量不足，需重新采集
  - Agent 数据结构与单轮请求不同，需独立处理方案

---

## 方向一：Qwen-V3.5-27B-64K / 128K 数据重采集与配置适配

### 问题
- 现有数据量不足，无法支撑有效分析
- 单条 prompt 最长达 128K 字符，超出默认 CSV 字段限制（已修复至 10 MB）

### 目标
- 补充足量样本，与其他模型保持同等口径
- 验证超长 prompt 场景下 prefix cache 命中率特征
- 评估 RadixTrieIndex 在超长上下文下的内存与速度表现

### 技术要点
- `block_size` 保持 128，需与部署 vLLM 实例配置一致
- 单条 prompt 平均 block 数 ≥ 256 时，`RadixTrieIndex` 自动启用（现有逻辑已支持）
- 暂不引入多进程（V1 约束），先评估单进程耗时；若全天数据处理时间过长，可按时间窗口分片后合并
- 新增 `configs/maas/qwen_v3_5_27b_64k/` 和 `configs/maas/qwen_v3_5_27b_128k/` 配置目录

### 代码实施步骤

#### Step 1 — 初始化两个模型的分析配置（最简单，复用现有工具）
```bash
python scripts/init_maas_configs.py qwen_v3_5_27b_64k  "Qwen-V3.5-27B-64K"
python scripts/init_maas_configs.py qwen_v3_5_27b_128k "Qwen-V3.5-27B-128K"
```

#### Step 2 — 数据重采集后执行 CSV → JSONL 转换
```bash
python scripts/convert_csv_to_jsonl.py \
    --input  data/internal/qwen_v3_5_27b_64k/raw/<file>.csv \
    --output data/internal/qwen_v3_5_27b_64k/requests.jsonl \
    --col-request-id 0 --col-user-id 1 --col-raw-prompt 2 --col-timestamp 3 \
    --has-header --encoding utf-8-sig

python scripts/convert_csv_to_jsonl.py \
    --input  data/internal/qwen_v3_5_27b_128k/raw/<file>.csv \
    --output data/internal/qwen_v3_5_27b_128k/requests.jsonl \
    --col-request-id 0 --col-user-id 1 --col-raw-prompt 2 --col-timestamp 3 \
    --has-header --encoding utf-8-sig
```

#### Step 3 — 先跑 benchmark_index，评估内存压力
```bash
python scripts/benchmark_index.py configs/maas/qwen_v3_5_27b_64k/benchmark_index.yaml
python scripts/benchmark_index.py configs/maas/qwen_v3_5_27b_128k/benchmark_index.yaml
```

#### Step 4 — 完整分析
```bash
bash scripts/run_maas_analysis.sh qwen_v3_5_27b_64k
bash scripts/run_maas_analysis.sh qwen_v3_5_27b_128k
```

---

## 方向二：Agent 多轮请求处理与新增分析图

### 问题
- Agent 请求中，同一 `request_id` 下可能发生数百次调用
- 现有系统无会话机制，仅支持单轮 API 调用
- 单轮分析框架无法捕捉 Agent 会话内的上下文复用规律

### 目标
- 将 Agent 原始数据重建为多轮会话结构
- 实现 F9、F10、F14 三张新分析图

### 数据处理方案

**会话重建逻辑**
- 原始 `request_id` → 重命名为 `chat_id`（代表一个 Agent 会话）
- 同一 `chat_id` 内按 `timestamp` 升序排列，得到第 1 轮、第 2 轮…第 N 轮
- 生成真实 `request_id`：格式为 `{chat_id}_{turn_index}`（如 `agent-001_0`、`agent-001_1`）
- 输出 JSONL 保留 `chat_id` 字段，供后续多轮分析使用
- 单轮业务数据（无 `chat_id`）保持现有处理路径不变

### 新增分析图定义

**F14：多轮 Agent 请求的 KV Cache 复用时间分布**
- 定义：同一 `chat_id` 内，第 N 轮请求与其最近一次命中的历史请求之间的时间间隔
- X 轴：reuse time（分钟），与 F13 口径一致
- Y 轴：CDF 累计占比
- 分层展示：按轮次（turn 1、turn 2–5、turn 6+）或按会话长度分组

**F9：Agent 会话轮次分布（复现 TraceA）**
- X 轴：一个 session 中的 request 数量（轮次数）
- Y 轴：累计占比（CDF）
- 用途：描述 MaaS Agent 请求的会话深度分布，与 TraceA 结论对比

**F10：用户平均轮次与标准差分布（复现 TraceA）**
- 图 A：每用户平均轮次（Mean Turns）分布
- 图 B：每用户轮次标准差（Standard Deviation）分布
- 双纵轴设计：
  - 左轴：绝对数值（蓝色 Per-user 柱/散点）
  - 右轴：累计占比 0–1（红色 Cumulation 曲线，即洛伦兹曲线）
- 用途：刻画用户行为习惯的异质性，识别高频 Agent 用户集中度

### 代码实施步骤

#### Step 1 — 新增 Agent CSV → JSONL 会话重建脚本
新建 `scripts/convert_agent_csv_to_jsonl.py`

- 读取含 `request_id`（即 `chat_id`）、`user_id`、`prompt`、`timestamp` 的 Agent CSV
- 按 `(chat_id, timestamp)` 排序，生成 `turn_index`
- 输出字段：`user_id`、`chat_id`、`request_id`（= `{chat_id}_{turn_index}`）、`turn_index`、`timestamp`、`raw_prompt`
- 继承 `convert_csv_to_jsonl.py` 的 `csv.field_size_limit`、`--encoding`、`--has-header` 等已有能力

#### Step 2 — 新增 Agent 数据加载器
新建 `src/block_prefix_analyzer/io/agent_loader.py`

- 与 `business_loader.py` 结构一致
- 额外暴露 `chat_id` 和 `turn_index` 字段，供 F9/F10/F14 分析使用
- 加载时按 `(timestamp, arrival_index)` 稳定排序（与现有回放约定一致）

#### Step 3 — 实现 F9：会话轮次 CDF
新建 `src/block_prefix_analyzer/analysis/f9.py` 和 `src/block_prefix_analyzer/plotting/f9.py`

- 统计每个 `chat_id` 的 turn 总数，构建频次分布
- 输出 CDF 曲线：X = session 轮次数，Y = 累计占比

#### Step 4 — 实现 F10：用户轮次均值 / 标准差双图
新建 `src/block_prefix_analyzer/analysis/f10.py` 和 `src/block_prefix_analyzer/plotting/f10.py`

- 按 `user_id` 聚合，计算每用户的平均轮次和轮次标准差
- 图 A / 图 B 共用同一绘图框架，双纵轴（绝对值 + 洛伦兹累计曲线）

#### Step 5 — 实现 F14：多轮 KV Cache 复用时间分布
新建 `src/block_prefix_analyzer/analysis/f14.py` 和 `src/block_prefix_analyzer/plotting/f14.py`

- 在 F13 复用时间框架基础上，新增会话内轮次分层逻辑
- 复用 `replay` 回放结果，按 `chat_id + turn_index` 归组后计算 reuse time

#### Step 6 — 新增 Agent 分析驱动脚本与配置模板
- 新建 `scripts/generate_f9_agent.py`、`scripts/generate_f10_agent.py`、`scripts/generate_f14_agent.py`
- 在 `scripts/init_maas_configs.py` 的模板中新增 `f9_agent.yaml`、`f10_agent.yaml`、`f14_agent.yaml`

#### 实施优先级

| 优先级 | 内容 | 前置依赖 |
|--------|------|----------|
| P0 | Step 1：会话重建脚本 | Agent 数据采集完成 |
| P1 | Step 2：agent_loader | Step 1 |
| P1 | Step 3：F9 | Step 2 |
| P1 | Step 4：F10 | Step 2 |
| P2 | Step 5：F14 | Step 3/4 验证数据有效后 |
| P2 | Step 6：驱动脚本与配置模板 | Step 3–5 |

---

## 方向三：全天流量时序分析扩展

### 问题
- 当前 F4 分析仅覆盖 09:00–11:00 两小时窗口，时间代表性存疑
- 短上下文模型（8K）流量随时间波动剧烈，两小时窗口可能恰好处于峰值或低谷
- 长上下文模型（32K）全天曲线是否仅有昼夜周期特征，需全天数据验证

### 目标
- 量化 prefix cache 命中率在全天不同时段的稳定性
- 对比峰值 vs. 低谷时段命中率差异
- 为 Qwen3-32B-32K 的 TTFT 优化研究提供全天维度的依据

### 涉及模型
Qwen3-32B-8K、Qwen3-32B-32K、DeepSeek-V3-32K、DeepSeek-V3-8K（已有数据，需确认是否覆盖全天）

### 代码实施步骤

#### Step 1 — 确认全天数据可用性
- 检查各模型现有 JSONL 的 timestamp 范围
- 若仅有 09:00–11:00 数据，需重新采集全天数据（00:00–24:00）

#### Step 2 — 为每个模型新增全天 F4 配置
在 `configs/maas/<slug>/` 下新增 `f4_prefix_fullday.yaml` 和 `f4_reusable_fullday.yaml`

- `bin_size_seconds: 300`（5 分钟粒度，适配全天 288 个 bin）
- `output_dir` 使用 `outputs/maas/<slug>/f4_prefix_fullday` 等独立子目录，不覆盖现有 09–11 结果

#### Step 3 — 扩展 F4 图表的 X 轴显示
当前 `plotting/f4.py` 的 X 轴标签为相对时间（秒/分钟）。全天视图需支持：
- X 轴显示为绝对时刻（HH:MM 格式），需从第一条记录的 timestamp 推算
- 可通过在 `series` 中新增 `start_timestamp` 字段实现，不改动现有单接口

#### Step 4 — 新增分时段命中率汇总统计
新建 `scripts/generate_f4_daily_summary.py`

- 将全天 288 个 bin 按时段分组（深夜 00–06、早晨 06–09、白天 09–18、晚间 18–24）
- 输出每时段的平均命中率、最大/最小值、标准差
- 输出为 CSV + 可打印 Markdown 表格（便于直接写入报告）

#### Step 5 — Qwen3-32B-32K 专项深化分析
- 在全天 F4 基础上，结合用户请求量（request count per bin）绘制双轴图：
  - 左轴：prefix hit rate
  - 右轴：request count（反映流量压力）
- 用于直观展示"高流量时段命中率是否更高（热点复用效应）"

---

## 整体实施顺序总览

```
方向一（数据适配）
  └─ Step 1: init_maas_configs（当天可完成）
  └─ Step 2: 数据重采集后 CSV→JSONL 转换
  └─ Step 3: benchmark_index 内存评估
  └─ Step 4: 完整分析

方向二（Agent 多轮）
  └─ Step 1: convert_agent_csv_to_jsonl.py（新脚本）
  └─ Step 2: agent_loader.py（新模块）
  └─ Step 3: F9 analysis + plotting
  └─ Step 4: F10 analysis + plotting
  └─ Step 5: F14 analysis + plotting
  └─ Step 6: 驱动脚本 + 配置模板更新

方向三（全天时序）
  └─ Step 1: 确认全天数据可用性
  └─ Step 2: 新增全天 F4 YAML 配置
  └─ Step 3: plotting/f4.py 扩展 X 轴绝对时刻显示
  └─ Step 4: generate_f4_daily_summary.py（新脚本）
  └─ Step 5: Qwen3-32B-32K 专项双轴图
```

---

*文档创建于 2026-04-24，覆盖 Phase 3 三个方向共 14 个代码步骤。*
