# 分析维度与实验步骤

> KV cache 复用分析的**心智模型**与**实验操作手册**。先讲"看什么"（4 维 × 2 层级），
> 再讲"怎么跑"（命令行）。
> 实现细节互锁地放在 [docs/可视化.md](./可视化.md)（Phase 1 模型层）和
> [docs/dashboard_phase2_plan.md](./dashboard_phase2_plan.md)（Phase 2 APP 层）。

---

## 1. 设计核心：4 维分析 × 2 层级投影

每个生产数据集（一个 `.csv` 文件 = 一个模型部署的请求日志）都按**同一组 4 个维度**
被分析两遍：

```
生产 csv (data/internal/<MODEL>/raw/<MODEL>.csv)
        │   columns: chat_id, user_id, raw_prompt, timestamp
        │   注: user_id 实际承载 APP_id（生产平台约定）
        ▼
┌──────────────────── [模型层 / Phase 1] ────────────────────┐
│   ① 理想命中率   ② 流量业务模式   ③ 时间局部性   ④ 可复用内容 │
│   全模型聚合，回答 "这个模型作为整体长什么样"               │
└────────────────────────────────────────────────────────────┘
        │
        │   per-user_id 切片（每个 APP 一份）
        ▼
┌──────────────────── [APP 层 / Phase 2] ─────────────────────┐
│   ① 理想命中率   ② 流量节奏     ③ 时间局部性   ④ system prompt │
│   per-APP 重算 + 与模型层基线对比                             │
└─────────────────────────────────────────────────────────────┘
```

两个层级共用 4 个维度的目的：**任何 APP 报告的读者都不需要切换心智模型**。
模型层告诉你"全局形状"；APP 层告诉你"某个 APP 在这个形状里站在哪里"。

---

## 2. 4 维 × 2 层级对照表

| 维度 | 模型层（Phase 1）| APP 层（Phase 2）|
|---|---|---|
| **① 理想命中率** | F4 全模型 ideal_hit_ratio + e1 4 档 sweep + 用户级分布（top-10% Lorenz）+ reuse_rank 分布 | per-APP F4 重算（bs=128 单档）+ 模型基线对照 + 同模型 APP 中位/p80/p90（横向位置）|
| **② 流量业务模式 / 节奏** | 间隔分位（p50/p75/p80/p95）+ 全局请求时序 + 每秒新写入 unique block + working set + 会话结构（F9/F10）| per-APP 间隔分位 + 该 APP 时序（inline `volume_series`）+ 与模型高峰对齐度（`peak_alignment`） |
| **③ KV cache 时间局部性** | F13 单轮 reuse-time CDF + F14 多轮 reuse-time CDF + reuse_distance（cache 压力指标） | per-APP F13 重算 + 模型基线对照（仅分位数 p50/p75/p80/p95）|
| **④ 可复用内容（system prompt）** | common_prefix 全模型共识块 top-N + content_type_guess + 解码后的 prefix 文本 | per-APP common_prefix（`min_count=2`）+ 与模型 common_prefix 重叠度 + content_type_guess |

> **每个维度都"per-APP 重算"+ "与模型对照"**：APP 层不直接复用模型层数字，
> 而是在 user_id 过滤后的子集上跑同样的 analysis 模块，得到该 APP 自己的
> 数字 + 模型基线作为对照。这保证 APP 报告里的每个数字都"属于这个 APP"，
> 而不是模型平均。

---

## 3. 两块派生视图（不引入新输入，可选展示）

派生视图 = 完全从前 4 维数据计算出的二级摘要，不消耗新数据来源。

### 3.1 模型层派生：自动建议（`section_5_recommendations`）

9 条规则从前 4 个 section 数据自动派生，按 P0/P1/P2 + Warning 分组：

| Rule ID | 触发条件示例 | 优先级 |
|---|---|---|
| R-PIN-CHAIN | 长 system prompt + 高用户偏斜 | P0 / P1 |
| R-CACHE-PRESSURE | `available_cache_blocks << reuse_distance_p80` | P0 |
| R-CAPACITY-FIRST | 高 reuse_rate 但低实际命中率 | P0 |
| R-BATCH-TTL | `reuse_time_p80` 在批处理时间尺度 | P1 |
| R-LONG-TTL | `reuse_time_p95` 远超默认 TTL | P1 |
| R-MULTI-TENANT | 用户偏斜中等 + 多 system prompt | P1 |
| R-LOW-CEILING | `ideal_hit_ratio` 本身就低 | P2 |
| W-SAME-SECOND | 同秒并发主导（命中率虚高） | warning |
| W-REUSE-ZERO | 数据完全无 reuse | warning |

**作用**：一眼看出"这个模型现在最值得做什么优化"，不用每次都从 4 个 section 里挖证据。

**APP 层不做这块**：9 条规则的 evidence 多数是节点维度的（cache 容量 / 用户偏斜 /
Lorenz 等），per-APP 没意义。

### 3.2 APP 层派生：顶部 Relative Position 卡片

5 个 mini-metric 横排显示该 APP 在同模型内的相对定位：

| Mini-metric | 来源 | 用途 |
|---|---|---|
| 请求量分位（top X%）| `e1_user_hit_rate/user_hit_bs128.csv` 派生 | 这是不是该模型的主力 APP？ |
| 命中率 vs 模型中位（Δ ±Y pp）| `section_1.app_f4` vs `user_hit_distribution.p50` | 该 APP 命中率在分布里偏哪一侧？ |
| 共识 prefix 长度 vs 模型 | `section_4.app_consensus` vs `common_prefix/metadata.json` | 该 APP 的 system prompt 比模型平均短 / 长？ |
| 流量是否集中在高峰时段（高/中/低）| `section_2.peak_alignment.ratio` + 阈值 (0.30 / 0.10) | 路由优化是否值得为这个 APP 做？ |
| 申报模型一致性（✓ / ⚠）| `scope.app_history × scope.model_id`（启发式匹配）| 申报模型与实际部署是否吻合？|

**作用**：批量浏览几十上百份 APP 报告时，不用打开每份就能看出"这个 APP 是不是
routing 优化的重点候选"。

> **unregistered APP 没有最后一项**（registry 缺失时该子字段 = `null`）；
> 其他 4 项不依赖 registry，照常计算。

---

## 4. 业务元信息的角色 —— 加分项，不是必需项

会议纪要（每月一次的资源申请审批记录）通过 `configs/app_registry.csv` 给 APP
报告**贴标签**：

| 标签字段 | 来源 | registry 缺失时回退 |
|---|---|---|
| `product_name` | 月度会议 csv | `"<unregistered>"` |
| `declared_model` | 月度会议 csv | `null` |
| `product_manager` | 月度会议 csv | （申请历史表整段不渲染）|
| `business_purpose` | 月度会议 csv | 同上 |
| 申请历史（含 8 个字段）| 月度会议 csv | `scope.app_history = []` |

**关键不变量**：4 维分析的**所有数值**来自生产 csv 自身，**不需要会议纪要参与**。
registry 缺失时报告照常出，仅在顶部多一条 ⚠ 横幅，分析价值不降。

> 这一选择反映"分析骨干 = 数值"，业务标签 = 装饰性元信息。
> 详见 `docs/dashboard_phase2_plan.md` §3.3 unregistered fallback。

---

## 5. 实验步骤（完整命令行）

针对一个示例模型 `qwen_v3_8b_8k`，从零开始端到端跑通。**默认路径不依赖月度会议
xlsx**（§5.4 是可选追加）。

### 5.0 一次性环境准备（新机器）

```bash
git clone git@github.com:Xiyuejiushikaixindeyisi/block-prefix-analyzer.git
cd block-prefix-analyzer
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ui,test]"
pytest -q                     # 自检：1047 passed + 10 xfailed
```

### 5.1 放置生产数据集（每个目标模型一份）

```bash
mkdir -p data/internal/qwen_v3_8b_8k/raw/
cp <来源>/qwen_v3_8b_8k.csv data/internal/qwen_v3_8b_8k/raw/qwen_v3_8b_8k.csv
# 4 列必需：chat_id, user_id, raw_prompt, timestamp
# user_id 实际承载 APP_id；编码 utf-8 / utf-8-sig 都可
```

`data/internal/` 已在 `.gitignore`，私密数据不会进 git。

### 5.2 模型层分析（一行命令产出 4 维 × 完整模型报告）

```bash
scripts/run_dashboard_pipeline.sh qwen_v3_8b_8k
```

内部自动 6 步（每步幂等可断点续跑）：YAML 配置初始化 → CSV→JSONL → 单轮子集
→ **并行**跑 11 个分析模块 → 聚合 `report.json` → 渲染 `report.html`。

⏱ **预计耗时**：8K 模型 / 10 万级请求 → 5–15 分钟。
🔧 **可调环境变量**：`FORCE=1`（强制重跑）/ `PARALLEL=1`（串行 / 内存紧）/
`DATA_ROOT=/data/internal`（数据在仓库外）。

#### 模型层产出

```
outputs/maas/qwen_v3_8b_8k/
    f4_prefix/  e1_user_hit_rate/  reuse_rank/             # ① 理想命中率
    traffic_pattern/  f9_agent/  f10_agent/  e1b_skewness/ # ② 流量
    f13_prefix/  f14_prefix/  reuse_distance/              # ③ 时间局部性
    common_prefix/                                          # ④ 可复用内容
    report.json                  # 聚合 v1.2 schema
    report.html                  ★ 单文件交付（4 个 section + 9 条建议）
```

打开模型层报告：

```bash
xdg-open outputs/maas/qwen_v3_8b_8k/report.html
# 或下载到本地双击
```

### 5.3 APP 层分析（无 registry 路径，推荐先这样跑）

`build_app_report.py` 默认 `--registry configs/app_registry.csv`；该文件不存在时
**自动**走 unregistered fallback，无需任何额外参数 / 安装。

```bash
MODEL=qwen_v3_8b_8k

# 5.3a 单个 APP（指定 user_id）
python scripts/build_app_report.py --model $MODEL --app com.huawei.driver.adn.net

# 5.3b 该模型下所有请求量 ≥ 10 的 APP（批量 loop）
python -c "
from block_prefix_analyzer.reports.app_filter import count_app_ids
for app, n in count_app_ids('data/internal/$MODEL/requests.jsonl').most_common():
    if n >= 10: print(app)
" | while read app; do
    echo \"═════ $app ═════\"
    python scripts/build_app_report.py --model $MODEL --app \"$app\" || true
done
```

调整门槛：把 `n >= 10` 改成 `n >= 1` 出全部（含长尾，多数会是空报告）；或
`n >= 100` 只出主力 APP。

⏱ **预计耗时**：每 APP 1–3 分钟。100 APP × 2 min ≈ 3.5 小时（串行）。

#### APP 层产出（每个 APP 一份独立目录）

```
outputs/maas/qwen_v3_8b_8k/apps/<app_id>/
    filtered_requests.jsonl      # 该 APP 的请求子集（per-APP 重算输入）
    report.json                  # v1.2 schema, kind="app"
    report.html                  ★ 单文件（顶部 RP 卡片 + 4 个 section + ⚠ 横幅）
```

报告内容（每份独立打开即可读）：

- 顶部 **Relative Position 卡片**（§3.2 列出 5 项）
- **Section A — 命中率画像**：该 APP F4 hit rate / 模型基线 / 同模型 APP 分布
- **Section B — 流量节奏**：该 APP 时序 / 间隔分位 / 高峰对齐度
- **Section C — 时间局部性**：该 APP F13 reuse-time 分位 / 模型基线
- **Section D — System Prompt 共识**：top-20 共识块（含 content_type_guess）+ 与模型重叠
- **申请历史表**：仅 registered APP；unregistered 时此表不渲染、顶部显 ⚠ 横幅

### 5.4 （可选）补充业务标签：从月度会议 xlsx 生成 registry

仅在你需要 APP 报告里看到产品名 / 申报模型 / 申请历史时跑这一步。

```bash
pip install openpyxl                                         # 一次性

# xlsx → csv（dtype=str 保留 "NA" 字符串，禁止 pandas 转 NaN）
mkdir -p data/internal/meetings/
cp <来源>/2026-05.xlsx data/internal/meetings/2026-05.xlsx
python -c "
import pandas as pd
pd.read_excel('data/internal/meetings/2026-05.xlsx', dtype=str).to_csv(
    'data/internal/meetings/2026-05.csv', index=False, encoding='utf-8'
)"
# 多 sheet 时加 sheet_name='<名字>' 显式指定（默认读第一个）

# csv → registry
python scripts/build_app_registry.py \
    --csv data/internal/meetings/2026-05.csv \
    --output configs/app_registry.csv
```

之后**重跑** §5.3 的 APP 报告 loop：registry 内容会自动读取生效，
`scope.product_name` / `scope.declared_model` / `scope.app_history` /
顶部 Relative Position 卡片的"申报模型一致性"子字段全部填实。

> **隐私**：`configs/app_registry.csv` 含产品经理姓名 / 业务用途，如不希望入 origin：
> `echo "configs/app_registry.csv" >> .gitignore`。

### 5.5 浏览所有产出

```bash
# 模型层（一份）
xdg-open outputs/maas/qwen_v3_8b_8k/report.html

# APP 层（多份；按文件名 = APP_ID 定位）
ls outputs/maas/qwen_v3_8b_8k/apps/*/report.html | wc -l    # 总数
xdg-open outputs/maas/qwen_v3_8b_8k/apps/<某APP>/report.html
```

远端机器上看，把整个目录打包：

```bash
tar czf qwen_v3_8b_8k_reports.tar.gz outputs/maas/qwen_v3_8b_8k/
```

---

## 6. 一键 copy-paste 完整流程（最少阻力版）

不带 xlsx，直接从生产 csv 一路到 APP 报告：

```bash
# === 一次性 ===
git clone git@github.com:Xiyuejiushikaixindeyisi/block-prefix-analyzer.git
cd block-prefix-analyzer
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ui,test]"
pytest -q

# === 每个模型 ===
MODEL=qwen_v3_8b_8k

# 1. 放数据
mkdir -p data/internal/$MODEL/raw/
# (手动) cp <来源>/$MODEL.csv data/internal/$MODEL/raw/$MODEL.csv

# 2. 模型层
scripts/run_dashboard_pipeline.sh $MODEL

# 3. APP 层（≥ 10 请求）
python -c "
from block_prefix_analyzer.reports.app_filter import count_app_ids
for app, n in count_app_ids('data/internal/$MODEL/requests.jsonl').most_common():
    if n >= 10: print(app)
" | while read app; do
    python scripts/build_app_report.py --model $MODEL --app "$app" || true
done

# 4. 浏览
xdg-open outputs/maas/$MODEL/report.html
ls outputs/maas/$MODEL/apps/*/report.html | wc -l
```

后续若要补 registry 业务标签：跑 §5.4，然后再走一遍上面的 loop 即可。

---

## 7. 故障排查速查

| 症状 | 检查 |
|---|---|
| `RAW_CSV not found` | `ls data/internal/<MODEL>/raw/<MODEL>.csv` |
| 11 分析有失败 | `tail /tmp/_pipeline_<MODEL>/<analysis>.log` |
| 注册表脚本警告 "registry may be stale" | 大量日志 user_id 不在注册表 → 注册表过期 / 测试流量；看末尾 "in logs only" 列表 |
| APP 报告 sections 全空 | 该 APP 请求数 < 2 → consensus / F13 都需要 ≥ 2，正常长尾 |
| HTML 缺图 / 空白 | 检查 `outputs/maas/<MODEL>/<analysis>/*.png`；不存在说明对应模块运行失败 |
| 中文乱码（xlsx 转 csv 时）| 转换命令缺 `encoding='utf-8'` |
| `ModuleNotFoundError: openpyxl` | `pip install openpyxl`（仅 §5.4 需要）|

### 端到端最小验证

无私密数据时用仓库自带 `synthetic_demo`（合成数据，已 commit）走通整条链路：

```bash
scripts/run_dashboard_pipeline.sh synthetic_demo
python scripts/build_app_report.py --model synthetic_demo --app u02
xdg-open outputs/maas/synthetic_demo/apps/u02/report.html
```

绿了说明环境 OK；问题在你的私密数据 / 路径配置。

---

## 8. 参考文档

| 主题 | 文档 |
|---|---|
| 模型层（Phase 1）完整设计与决策 | [docs/可视化.md](./可视化.md) |
| APP 层（Phase 2）完整设计与决策 | [docs/dashboard_phase2_plan.md](./dashboard_phase2_plan.md) |
| 项目协作守则 + V1/V2 范围约束 | [CLAUDE.md](../CLAUDE.md) |
| 整体快速上手 | [README.md](../README.md) |
