# Dashboard Phase 2 计划 — APP 级分析报告

**版本**：v1.0 草案
**状态**：待用户审定后开始实施
**前置**：Dashboard Phase 1（模型级）已完工并合入 `origin/main`（commit `9e00c4d`）

> **命名约定**：本文档统一使用 **"Dashboard Phase 2"** 或 **"APP-level dashboard"**，
> 不写裸的 "Phase 2"，避免与 `docs/two_queue_ttl_experiment_plan.md` 中的"实验 Phase X"混淆。
> 文档、CLI、目录、变量命名一律使用 `app`，不使用 `user`（生产日志里的 `user_id`
> 字段语义实际是 APP ID，不是真实用户）。

---

## 1. 目标与边界

### 1.1 一句话目标
输入 `(model_id, app_id)`，复用 Phase 1 的 11 个分析产物，产出
`outputs/maas/<model>/apps/<app_id>/report.html`，叙事重心是
**这个 APP / 产品在这个模型上的 KV cache 复用画像**。

### 1.2 范围内
- 从月度会议申请 csv 派生 APP 注册表（前置工件）。
- APP 级 `report.json` 的 schema 扩展（向后兼容 model 级）。
- APP 级静态 HTML 报告（4 个 section + 相对位置卡片）。
- 按需 CLI 入口：`scripts/build_app_report.py --model X --app com.huawei.xxx`。

### 1.3 范围外（明确不做）
- **不做** Streamlit 侧的 APP 联动（按 feedback：静态 HTML 是默认交付，Streamlit 仅辅助）。
- **不做** department 维度的报告（继续保留 `scope.department_*` 为 null，留给 Dashboard Phase 3）。
- **不做** APP × Model 矩阵视图（每份报告聚焦单 APP 单模型）。
- **不修改** 11 个 analysis 模块（按 `docs/可视化.md §9` 冻结项）。
- **不重跑** `e5_block_text`（成本不可接受，沿用 `common_prefix`）。
- **不引入** 数据库 / 缓存层 / 实时刷新机制。
- **不实现** 历史 csv 的多版本对比（每月新 csv 直接覆盖，旧 registry 由 git history 提供）。

---

## 2. 输入数据

### 2.1 会议申请 csv

**字段顺序**（来自示例前几行，全中文表头）：

```
上会日期 | *议题名 | *主讲人 | *APP ID | 产品名称 | *申请资源类型 |
产品经理 | 系统支持代表 | 参与人员 | 会议材料 | 评审结论 | 任务类型 |
模型 | 资源类型 | 资源使用方式 | 保障配额（卡） | 预计使用时长 |
保障并发数（个） | 业务用途
```

**关键字段（注册表关心）**：
- `*APP ID`：形如 `com.huawei.driver.adn.net`，作为主键
- `产品名称`：人类可读名（如"ICT技术专委会-软件分委会项目"）
- `模型`：申报模型名（如 `Qwen-V3-32B`、`GLM4.7`）；与生产部署 `model_id` 可能不一致
- `业务用途`：辅助标签（如"生产"、"训练"），用于过滤
- `评审结论`：过滤条件
- `资源使用方式`：过滤条件
- `任务类型`：过滤条件

**过滤规则（APP 注册表入选条件）**：

```
评审结论    == "同意"
资源使用方式 == "共享模型（API调用）"
任务类型    == "推理"
```

> 不满足以上三条之一的行直接丢弃；不进入 APP 注册表。
> 训练任务、独占资源、未通过审批的 APP 均不在 KV cache 复用分析范围内。

**更新频率**：每月一次，由用户手动放置新版 csv 后重跑 `build_app_registry.py`。
旧版本不保留（覆盖式更新），历史版本通过 `git log configs/app_registry.csv` 追溯。

### 2.2 生产请求日志中的 `user_id` 字段

**关键事实**：现有 `business_loader.py` 把 jsonl 里的 `user_id` 字段挂到
`RequestRecord.metadata["user_id"]`，**该字段的实际语义是 APP ID**（应用维度），
不是真实自然人或部门。

**字段命名策略**（Q1 决议）：
- **保留** `business_loader.py` 中的字段名 `user_id`（避免重命名牵动 6 个模型已生成的所有 metadata）。
- 在 docstring 中文档化语义："`user_id` field carries the APP ID
  (e.g. `com.huawei.xxx`), not a natural-person identifier."
- APP 报告链路中所有新增代码使用 `app_id` 命名；
  在与 business_loader 对接时显式做一次 alias：`app_id = record.metadata["user_id"]`。

---

## 3. APP 注册表（前置工件）

### 3.1 生成流程

```
scripts/build_app_registry.py
  --csv <path-to-monthly-csv>
  [--output configs/app_registry.csv]
  [--encoding utf-8]

步骤：
  1. 用 stdlib `csv.DictReader` 读取 csv（中文表头），所有单元格保留字符串类型
     （避免 pandas 默认把 `"NA"` 转 `NaN`，破坏 §3.2 "NA 原样保留" 不变量；
     与 `report_builder.py` 的项目约定一致）
  2. 按 §2.1 三条过滤规则筛选行
  3. **保留全部历史行**：同一 app_id 可能多次申请（不同月份 / 不同模型），
     一行 = 一次申请记录；不去重、不取最新
  4. 按 (app_id, source_meeting_date) 稳定排序，输出 app_registry.csv（§3.2）
  5. 打印统计：总行数 / 通过过滤行数 / unique app_id 数 / 模型分布
```

**幂等性**：相同输入 csv 多次运行产出一致；时间字段写入但用于排序后去除时间噪声（不写入秒级时间戳）。

**为何保留全部历史**：一个 app_id 可能在不同月份申请不同模型（如先申请 Qwen-V3-32B，
后改申请 GLM4.7），最新值不一定是当前生产实际部署。本计划仅展示历史申请清单，
具体哪条对应当前生产由用户事后与业务对齐确认。

### 3.2 注册表 schema

文件路径：`configs/app_registry.csv`（commit 进仓库；预计 < 50KB）

**主键不再是 app_id**，而是 `(app_id, source_meeting_date, declared_model)`
组合键（同一 APP 多次申请记录共存）。

字段（11 列）：

```
app_id,product_name,declared_model,business_purpose,source_meeting_date,
product_manager,resource_type_requested,resource_type_actual,
guaranteed_quota_cards,guaranteed_concurrency,expected_duration
```

| 列名 | csv 来源列 | 用途 |
|---|---|---|
| `app_id` | `*APP ID` | 与生产日志 `metadata["user_id"]` 关联 |
| `product_name` | `产品名称` | 报告标题 / 租户展示 |
| `declared_model` | `模型` | 申报模型，不规范化；与 `model_id` 对照 |
| `business_purpose` | `业务用途` | 用例标签（生产 / 测试） |
| `source_meeting_date` | `上会日期` | 历史追溯，组合键之一 |
| `product_manager` | `产品经理` | 租户联系人 |
| `resource_type_requested` | `*申请资源类型` | 申报算力（如 `D910B3 共40卡`） |
| `resource_type_actual` | `资源类型` | 分配算力（如 `D910B4+D310P`） |
| `guaranteed_quota_cards` | `保障配额（卡）` | 卡数；可能为 `NA` |
| `guaranteed_concurrency` | `保障并发数（个）` | 并发上限；可能为 `NA` |
| `expected_duration` | `预计使用时长` | 计划周期（如 `一年`）|

**示例**（同一 APP 多次申请）：

```
com.huawei.driver.adn.net,ICT...,Qwen-V3-32B,生产,2026-01-06,潘宏波,D910B3 共40卡,D910B4+D310P,NA,100,一年
com.huawei.driver.adn.net,ICT...,GLM4.7,生产,2026-03-10,潘宏波,D910B4 共20卡,D910B4,NA,80,六个月
com.huawei.bpit.generalai.genfabric.aidevenv,通用AI,GLM4.7,生产,2026-01-06,刘惠鹏,D910B1,D910B3,16,NA,NA
```

**字段处理规则**：
- `NA` 字符串原样保留（不转 null）；下游展示时识别 `NA` 并显示为"未约定"
- 中文/英文/数字混排字符串原样保留，不做规范化
- 不在表中的列（如 `*主讲人`、`系统支持代表`、`参与人员`、`会议材料`、
  `*议题名`、`资源使用方式`、`评审结论`、`任务类型`）一律**不存入** registry
  - `资源使用方式`/`评审结论`/`任务类型` 已在过滤阶段消费完，无需保留
  - 其他列与租户展示无关，省略以保持精简

> **declared_model 与 model_id 故意冗余**：
> csv 里的"模型"是 APP 申报值（人类描述，如 `Qwen-V3-32B`），
> 而 dashboard 的 `model_id` 是部署 slug（如 `qwen_v3_5_27b_64k`）。
> 两者可能不一致——本身就是一个值得 flag 的发现，APP 报告会显式展示这两个值。

### 3.3 unregistered APP 的处理（Q2 决议）

生产日志里出现、但**注册表查询返回空列表**的 `app_id`（未审批 / 历史遗留 / 测试流量）：

- **Fallback 出报告**：仍然生成 HTML，不静默跳过。
- `scope.product_name` = `"<unregistered>"`
- `scope.declared_model` = `null`
- `scope.app_history` = `[]`（空列表）
- 报告头部显示一条 **warning 横幅**：
  > ⚠️ 此 APP ID 未在会议申请记录中找到。可能为未审批 / 历史遗留 APP / 测试流量。
- 静默跳过会漏掉异常流量；warning fallback 让运维侧能主动发现。

### 3.4 数据完整性验证（注册表生成后立即检查）

```
build_app_registry.py 末尾打印：
  - 注册表行数 N_rows
  - 注册表 unique app_id 数 N_apps
  - 当前 outputs/maas/ 下各模型日志中出现的 unique app_id 数 N_log
  - 交集 N_intersect / 仅在日志 N_log_only / 仅在注册表 N_reg_only
  - 若 N_log_only / N_log > 30%：打印 WARNING（提示注册表过期）
  - 多次申请的 APP 列表（同一 app_id 出现 ≥ 2 行）：打印 top-10 供人工核对
```

---

## 4. 报告 schema 扩展（v1.2）

### 4.1 `scope` 字段扩展

`report.json` 的 `scope` 字段在 v1.1 基础上**新增 4 个字段**，所有字段可空，
不破坏 model 级报告：

```jsonc
"scope": {
  "kind": "app",                           // v1.1: "model"; v1.2: 新增 "app"
  "model_id": "qwen_v3_5_27b_64k",         // 部署模型 slug（真实日志归属）
  "app_id": "com.huawei.driver.adn.net",   // 新增（v1.2）
  "product_name": "ICT技术专委会-软件分委会项目",  // 新增（v1.2）；unregistered 时为 "<unregistered>"
  "declared_model": "GLM4.7",              // 新增（v1.2）；最近一次申请的模型，仅作摘要
  "app_history": [                         // 新增（v1.2）：完整申请历史（按 source_meeting_date 升序）
    {
      "source_meeting_date": "2026-01-06",
      "declared_model": "Qwen-V3-32B",
      "business_purpose": "生产",
      "product_manager": "潘宏波",
      "resource_type_requested": "D910B3 共40卡",
      "resource_type_actual": "D910B4+D310P",
      "guaranteed_quota_cards": "NA",
      "guaranteed_concurrency": "100",
      "expected_duration": "一年"
    },
    { "source_meeting_date": "2026-03-10", "declared_model": "GLM4.7", ... }
  ],
  "user_id": null,                         // 弃用字段，保留向后兼容
  "department_id": null,                   // Phase 3 预留
  "department_name": null                  // Phase 3 预留
}
```

**字段语义**：
- `declared_model`：取 `app_history` 中最近一行（最大 `source_meeting_date`）的 `declared_model`，**仅用于报告摘要展示**；不代表当前生产部署。多模型申请的真实情况通过 `app_history` 完整透出。
- `app_history`：完整申请记录数组，按时间升序；unregistered APP 为 `[]`。

### 4.2 schema_version

- v1.1 → v1.2：仅扩展 `scope`，不动其他 5 个 section 的 schema
- 既有的 model 级报告（`kind="model"`）re-build 后字段顺序不变，
  只是 `scope` 多 3 个 null 字段，向后兼容
- dashboard 渲染时按 `kind` 分支选择模板

---

## 5. 章节设计（D3 推荐方案：4 个 section）

模型级 5 个 section 中**剔除"写入压力 / working set / 用户偏斜"等节点维度指标**，
聚焦 APP 自身可解释的 4 个维度。

### 5.1 Section A：命中率画像

| 指标 | 来源 | 备注 |
|---|---|---|
| 该 APP 的 ideal prefix hit rate | F4 重算（在 user_id 过滤后的子集上） | per-app 主指标，**block_size = 128** |
| 该 APP 的请求数 / block 总数 | 同上 | 规模上下文 |
| 模型整体 ideal prefix hit rate | model 级 report.json（block_size = 128） | 对照基线 |
| 模型内同 APP 的 ideal hit 中位 / p90 | **`e1_user_hit_rate/user_hit_bs128.csv`** 派生 | **同模型横向对照（锚定 bs=128）** |
| 该 APP 的 block_size sweep 折线 | F4 sweep 重算（4 档 16/32/64/128） | 与模型级 sweep 对齐口径 |

> **横向对照口径锚定**：相对位置卡片与本节中位/p90 计算**统一使用 block_size = 128**
> 的 `user_hit_bs128.csv`。dashboard 主图也是 128，避免 sweep 多档时的歧义。
> 64K+ 模型若 sweep 仅有单档（128），则口径自动一致；8K–32K 模型存在 4 档时，
> 其他 3 档（16/32/64）仅在该 APP 自己的 sweep 折线中展示，不参与横向对照。

### 5.2 Section B：流量节奏

| 指标 | 来源 | 备注 |
|---|---|---|
| 该 APP 的请求时序（按小时聚合） | 在 user_id 过滤后的子集上重算 | 路由策略输入信号 |
| 该 APP 的并发节奏分位 | 同上 | |
| 该 APP 活跃时段是否与全局流量高峰重合 | 与 model 级 traffic_pattern 对照 | 路由价值代理 |

### 5.3 Section C：时间局部性

| 指标 | 来源 | 备注 |
|---|---|---|
| 该 APP 的 F13 reuse_time CDF | F13 重算（user_id 过滤 + turn_index==0 pre-filter） | per-app 主指标 |
| 该 APP 的 reuse_time p50/p80/p95 | 同上 | TTL 锚点候选 |
| 模型整体 F13 分位（对照） | model 级 report.json | |

### 5.4 Section D：System Prompt 共识块

| 指标 | 来源 | 备注 |
|---|---|---|
| 该 APP 的 common_prefix consensus blocks | common_prefix 重算（限定 user_id 过滤后的子集） | **核心叙事**：揭示该 APP 的业务模式 |
| 共识 prefix 长度（blocks / chars） | 同上 | |
| `content_type_guess`（8 类） | 复用 Phase 1 推断逻辑 | json_schema / system_prompt / agent_tool_prompt 等 |
| 与模型整体 common_prefix 是否重叠 | 简单交集统计 | 揭示该 APP 是否使用通用 system prompt |

### 5.5 相对位置卡片（顶部摘要）

报告头部一张卡片，给出该 APP 在同模型内的相对定位（口径统一 block_size = 128）：

- 请求量百分位（top X%，基于 `e1b_skewness` per-user 数据派生）
- ideal hit rate vs 同模型 APP 中位（差距 ±Y pp）
- 共识 prefix 长度 vs 同模型平均
- 流量是否集中在高峰时段（路由价值标签：高 / 中 / 低）
- **申报模型一致性**：
  - 一致 ✓：`app_history` 中存在任一 `declared_model` 与当前 `model_id` 名称对齐（启发式匹配，如 `Qwen-V3-32B` ↔ `qwen_v3_32b_*`）
  - 不一致 ⚠：所有历史申报模型均与当前部署模型不匹配
  - 多模型申请：展示完整 `app_history` 表格（`source_meeting_date | declared_model`）

> 启发式匹配的规则（小写 + 去分隔符 + 子串匹配）在实施时落到
> `app_registry.match_declared_to_deployment(declared, model_id) -> bool`，
> 留有可单测的边界。

---

## 6. 实现路径

### 6.1 触发模式（D1）：按需 CLI

```
scripts/build_app_report.py
  --model qwen_v3_5_27b_64k
  --app com.huawei.driver.adn.net
  [--registry configs/app_registry.csv]
  [--data-root data/internal]
  [--outputs-root outputs/maas]
```

**不做** 批量预生成 top-N 报告。一次一份，运行时间约 1–3 分钟（仅重算 F13/F4/common_prefix 三项）。

### 6.2 计算路径（D2）：后置过滤为主

| 指标类别 | 计算路径 |
|---|---|
| `e1_user_hit_rate` 已是 per-user CSV | **直接读取**，按 `user_id == app_id` 过滤行 |
| `reuse_rank` 已是 per-request CSV，含 user_id | **直接读取并过滤** |
| `e1b_skewness` 已是 per-user CSV | **直接读取**派生百分位 |
| F4 / F13 / common_prefix 无 per-user 切片 | 在 raw jsonl 上做 user_id 过滤后**重算这三项** |
| traffic_pattern / reuse_distance 节点维度 | **不重算**，从 model 级报告读对照值 |

> **不走前置过滤**（即不生成 `users/<U>/requests.jsonl` 子集再跑全 11 个分析）的原因：
> - 前置过滤每个 APP 需重跑 11 个分析，累计 5–10 分钟
> - 后置过滤只需重算 3 项，且大部分指标可直接读 CSV，单 APP < 3 分钟
> - 11 个分析模块**不需要任何修改**（满足 Phase 1 §9 冻结约束）

### 6.3 渲染：复用 `render_static_report.py`

- 模板共用，按 `scope.kind` 分支：`"app"` 时使用 APP 报告标题、warning 横幅、相对位置卡片
- HTML 文件路径：`outputs/maas/<model>/apps/<app_id_safe>/report.html`
  - `app_id_safe = app_id.replace("/", "_")`（防御性，APP ID 含点号无问题）

---

## 7. 文件清单

### 7.1 新建文件

| 路径 | 用途 |
|---|---|
| `scripts/build_app_registry.py` | 月度 csv → `configs/app_registry.csv` |
| `scripts/build_app_report.py` | CLI 入口，编排 §6.2 计算路径 |
| `src/block_prefix_analyzer/reports/app_report.py` | APP 级 report.json 组装（v1.2 schema） |
| `src/block_prefix_analyzer/reports/app_filter.py` | 在 raw jsonl 流上做 user_id 过滤的工具函数 |
| `src/block_prefix_analyzer/reports/app_registry.py` | 注册表加载 / 查询 / unregistered fallback |
| `tests/test_app_registry.py` | csv 解析、过滤、unregistered 处理 |
| `tests/test_app_report.py` | 端到端：synthetic_demo 上的 APP 报告 |
| `tests/test_app_filter.py` | jsonl user_id 过滤的边界用例 |
| `configs/app_registry.csv` | 首版注册表产物（用户提供首份 csv 后生成并 commit） |

### 7.2 修改文件

| 路径 | 改动 |
|---|---|
| `src/block_prefix_analyzer/report_builder.py` | 抽出 4 个 section 计算函数，让 model 报告 / app 报告共用；不破坏既有 API |
| `scripts/render_static_report.py` | 新增 `kind="app"` 分支：标题、warning 横幅、相对位置卡片 |
| `src/block_prefix_analyzer/io/business_loader.py` | docstring 补充：`user_id` 实际为 APP ID |
| `README.md` | "Dashboard Phase 2" 小节：3 步流程（csv → registry → app report） |

### 7.3 schema_version bump

| 文件 | 改动 |
|---|---|
| `src/block_prefix_analyzer/report_builder.py` | `SCHEMA_VERSION = "1.2"` |
| 所有现存 `report.json` | 不主动重跑；下次 `build_model_report` 自然升级 |

---

## 8. 实现步骤（commit 切分）

每个 step 一个独立 commit；每步完成 `pytest` 必须保绿。

| Step | 内容 | Commit message |
|---|---|---|
| 0 | `app_registry.py` 模块 + csv 解析 + 过滤规则（保留全部历史）+ 单测 | `feat(app): csv-driven app registry loader` |
| 1 | `scripts/build_app_registry.py` CLI；用首份 csv 生成 `configs/app_registry.csv` | `feat(app): registry builder script and first registry artifact` |
| 2 | `report_builder.py` 抽 section 计算函数（refactor，model 报告输出不变） | `refactor(report): extract per-section computation for reuse` |
| 3 | `app_filter.py` + 单测（user_id 过滤的最小逻辑） | `feat(app): jsonl user_id filter utility` |
| 4a | `app_report.py` v1.2 scope 骨架（含 `app_history` 列表 + unregistered fallback）+ 单测 | `feat(app): app report scope v1.2 skeleton with history` |
| 4b | Section A：命中率画像（含同模型横向对照，锚定 bs=128）| `feat(app): hit rate section with cross-app comparison` |
| 4c | Section B：流量节奏（per-app 时序 + 高峰相位） | `feat(app): traffic cadence section` |
| 4d | Section C：时间局部性（F13 per-app 重算） | `feat(app): temporal locality section` |
| 4e | Section D：system prompt 共识块（common_prefix per-app 重算 + content_type_guess） | `feat(app): consensus prefix section` |
| 4f | 相对位置卡片（顶部摘要）+ `match_declared_to_deployment` 启发式匹配 | `feat(app): top relative-position summary card` |
| 5 | `render_static_report.py` 加 `kind="app"` 分支（标题 + warning 横幅 + 卡片） | `feat(app): render static html for app reports` |
| 6 | `scripts/build_app_report.py` CLI 编排 | `feat(app): cli for building per-app reports` |
| 7 | `synthetic_demo` 端到端跑通 + golden 对比 | `test(app): end-to-end app report on synthetic_demo` |
| 8 | README "Dashboard Phase 2" 小节 | `docs: document dashboard phase 2 (app-level)` |

> Step 4a–4f 共 6 个子步骤，每步独立 commit。完成 4a 后即可端到端调通骨架（4 个 section
> 暂为占位空 dict），4b–4e 逐个 section 填实，4f 收束相对位置卡片。这一拆分让中途回退不破坏整体可运行状态。

---

## 9. 验收标准

Dashboard Phase 2 完成的判定：

1. `python scripts/build_app_registry.py --csv <path>` 在用户提供的首份 csv 上无报错，
   产出 `configs/app_registry.csv`，过滤统计与人工核对一致
2. `python scripts/build_app_report.py --model qwen_v3_5_27b_64k --app <某 registered app>`
   在 < 5 分钟内产出 `outputs/maas/qwen_v3_5_27b_64k/apps/<app>/report.html`
3. 同上命令对 unregistered app_id 也能产出报告，warning 横幅可见
4. `report.json` 的 `scope.kind == "app"`，`product_name` 与注册表一致
5. 4 个 section 渲染正常（无 KeyError、无空白图）
6. 相对位置卡片显示 5 个指标，`declared_model` 与 `model_id` 一致性可视
7. `pytest` 全绿（含本计划新增 3 个测试文件）
8. README 末尾有 "Dashboard Phase 2" 使用说明小节
9. **既有 model 级 report.html 重新生成后内容无变化**（除 schema_version 升到 1.2，scope 多 3 个 null 字段）

---

## 10. 关键不变量（实施期间不得违反）

- **不修改** `src/block_prefix_analyzer/analysis/**` 下任何 11 个分析模块
- **不引入** pandas 进 `src/block_prefix_analyzer/` 核心（仅在 `scripts/`、`reports/` 层使用）
- **不破坏** 既有 model 级 report.json 的字段（仅扩展，不删改）
- **不读取** `data/internal/**` 之外的私有数据路径（注册表 csv 路径由 CLI 参数显式传入）
- **不静默跳过** unregistered app（必须 fallback 出报告 + warning）
- **不在** Streamlit dashboard 里加 APP 联动逻辑（静态 HTML 是默认交付）

---

## 11. 后续（不在本计划内，仅备忘）

- Dashboard Phase 3：department 维度（需业务侧提供 `app_id → department` 映射 CSV）
- 多月 csv 历史对比（注册表 diff，识别新增 / 注销 / 模型迁移的 APP）
- APP 报告与 `two_queue_ttl_experiment_plan.md` 的衔接（per-app gap_closed_ratio 是否值得加入）

以上三项均**不在 Dashboard Phase 2 范围内**，等本计划完工并稳定运行后再单独评估。
