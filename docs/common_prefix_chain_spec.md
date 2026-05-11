# `find_common_prefix_chain` — Spec v1.0

> **状态（2026-05-11）**：spec 已锁定，待实施。本文档是 5-commit 实施工作（见 §11）
> 的契约源——任何实施细节如与此文档冲突，**应先更新本文档**再修改代码。
>
> **背景**：现有 `find_common_prefix` 是 position-wise majority，可能拼出"幽灵链"
> （任何请求中都不真实存在的 block_id 序列）。详见
> [docs/可视化.md](./可视化.md) 决策表后 caveat 块。本规范定义其替代品。
>
> **算法定位**：path-closed trie greedy main path = "soft LCP"（Longest Common Prefix
> with frequency floor）。**不是 LCS** —— LCS 允许中间跳过元素，与 vLLM 连续 hash 链
> 物理不符；详见 [docs/analysis_dimensions.md §3.1](./analysis_dimensions.md) 旁支讨论。

---

## 1. 目标 & 非目标

### 目标
1. 输出链**一定真实存在**于至少 `min_count` 个请求中（path-closed 不变量）
2. 三阈值组合判定何时停止扩展，每条 stop 都可解释（`stop_reason` + 分叉点替代项）
3. 与 vLLM 物理 prefix cache 链式语义对齐（连续从 position 0 起 + block_id 链一致）
4. 输入接口 / `decoded_text` 行为 / `block_registry` 用法**与旧函数一致**——便于换算法不换上层调用

### 非目标
- 多链 / top-K main path（scope creep，留给将来）
- block_id 模糊匹配 / token-level diff（与 vLLM 物理不符）
- 取代 `top_ngrams.py`（那是位置无关的 n-gram 频次分析，不同语义）

---

## 2. 公开 API

```python
# src/block_prefix_analyzer/analysis/common_prefix.py
def find_common_prefix_chain(
    records: list[RequestRecord],
    block_registry: BlockRegistry,
    *,
    block_size: int,
    min_count: int = 10,
    branch_threshold: float = 0.05,
    coverage_threshold: float = 0.0,
    max_blocks: int = 100_000,
) -> CommonPrefixChainResult:
    ...
```

### 参数

| 参数 | 默认 | 必填 | 说明 |
|---|---|---|---|
| `records` | — | ✅ | RequestRecord 列表（同 `find_common_prefix`） |
| `block_registry` | — | ✅ | block_id → text 反查表（同旧函数） |
| `block_size` | — | ✅ | 仅供 metadata 报告，不参与分析 |
| `min_count` | 10 (model) / 2 (app) | ❌ | 绝对阈值：`heaviest.freq >= min_count`。默认参数下的主过滤器 |
| `branch_threshold` | **0.05**（软地板）| ❌ | 相对父：`heaviest.freq / parent.freq >= branch_threshold`。5% 极小，典型数据几乎不触发，只兜底"主流坍塌为 < 5%"的极端碎片化分叉。**可手动调高**到 0.5 / 0.7 做更严格的"清晰主流"分析 |
| `coverage_threshold` | **0.0**（no-op）| ❌ | 相对全局：`heaviest.freq / total_records >= coverage_threshold`。默认禁用；显式调高（如 0.05）剪长尾 |
| `max_blocks` | 100,000 | ❌ | 安全帽（同旧函数） |

### 默认值的设计意图

- **`min_count`** 是主过滤器：绝对人数门槛，简单可解释。
- **`branch_threshold = 0.05`** 是**软地板**——5% 极小，典型 Agent / TEXT 数据上每步 branch_ratio 接近 100%，软地板不咬；只在病态碎片化场景（主流子节点占父 < 5%、即超过 95% 的请求散到其他分支）才触发，避免 trie 在毫无共识的位置硬走最大子节点产生噪声链。**用户可以手动调高**到 0.5 / 0.7 做更严格的"清晰主流"分析。
- **`coverage_threshold = 0.0`** 完全禁用：当前阶段最关心"是否存在真实可复用 chain"，不主动剪长尾。研究"高覆盖共识链"时再 opt-in（如 0.05）。

不要把 `branch_threshold` / `coverage_threshold` 视作"魔法答案"——它们的默认值优先保证"不破坏既有信号"，研究时按需手动调整。

---

## 3. 输出 dataclass

```python
from typing import Literal

@dataclass(frozen=True)
class ChainBlock:
    position: int                  # 0-based depth in the chain
    block_id: BlockId
    freq: int                      # # records that walked through this node
    parent_freq: int               # # records that walked through previous node (root.freq = total_records)
    global_coverage_pct: float     # freq / total_records * 100
    branch_ratio_pct: float        # freq / parent_freq * 100


StopReason = Literal[
    "no_records",            # input was empty or all block_ids were empty
    "min_count",             # heaviest child's freq < min_count
    "branch_threshold",      # heaviest.freq / parent.freq < branch_threshold
    "coverage_threshold",    # heaviest.freq / total_records < coverage_threshold
    "max_blocks",            # chain length hit the safety cap
    "no_children",           # walked to a trie leaf (rare; all records ended at exactly this depth)
]


@dataclass(frozen=True)
class BranchAlternative:
    block_id: BlockId
    freq: int
    fraction_of_parent: float      # competing branches at the stop node, sorted desc
    decoded_text_preview: str      # first ~50 chars of this alternative's text


@dataclass(frozen=True)
class CommonPrefixChainResult:
    consensus_blocks: list[ChainBlock]
    prefix_length_blocks: int             # = len(consensus_blocks)
    prefix_length_chars: int              # = len(decoded_text)
    decoded_text: str

    block_size: int
    total_records: int                    # # records with non-empty block_ids (denominator)

    # Threshold provenance (reproducibility)
    min_count_threshold: int
    branch_threshold: float
    coverage_threshold: float

    # Stop diagnostics
    stop_reason: StopReason
    stop_position: int                    # usually = len(consensus_blocks); 0 if no_records
    branch_alternatives: list[BranchAlternative]   # top-5 competitors at the stop node
```

---

## 4. 算法

```
# Phase 1: build trie
total_records = #(r in records if r.block_ids)   # non-empty only
root = TrieNode(freq=total_records)
for r in records:
    if not r.block_ids:
        continue                              # skip empty-block records (D3)
    node = root
    for bid in r.block_ids[:max_blocks]:
        node = node.children.setdefault(bid, TrieNode(freq=0))
        node.freq += 1

# Phase 2: greedy walk main path
chain, current = [], root
stop_reason = "no_children"                   # default if while loop exits cleanly
while current.children:
    bid, heaviest = max(current.children.items(), key=lambda kv: kv[1].freq)

    if heaviest.freq < min_count:
        stop_reason = "min_count"; break
    if heaviest.freq / current.freq < branch_threshold:
        stop_reason = "branch_threshold"; break
    if total_records and heaviest.freq / total_records < coverage_threshold:
        stop_reason = "coverage_threshold"; break

    chain.append(ChainBlock(
        position=len(chain),
        block_id=bid,
        freq=heaviest.freq,
        parent_freq=current.freq,
        global_coverage_pct=heaviest.freq / total_records * 100,
        branch_ratio_pct=heaviest.freq / current.freq * 100,
    ))
    current = heaviest

# Phase 3: capture diagnostics + decode
if total_records == 0:
    stop_reason = "no_records"

branch_alternatives = top_N(
    [(bid, child) for bid, child in current.children.items()
     if not chain or bid != chain[-1].block_id],   # exclude main path winner
    by freq desc, N=5,
)
decoded_text = "".join(block_registry.get(cb.block_id, f"<MISSING:{cb.block_id}>")
                       for cb in chain)
```

### 复杂度
- Build：`O(Σ |r.block_ids|)` ≈ O(N × L)
- Walk：O(L)（沿主链一次到底）
- 内存：O(unique trie nodes) —— 高共识 Agent 数据近 O(L)；最坏（所有请求 block_ids 全不重叠）O(N × L)，触帽时被 max_blocks 限住

---

## 5. 阈值组合的语义

3 个阈值**全部必须满足**才继续，**任一失败即停**，`stop_reason` 记录哪个失败的。

| 阈值 | 失败时含义 | 何时打开 |
|---|---|---|
| `min_count` | 主流子节点的绝对人数已少到无意义 | 默认即开（10 / 2）|
| `branch_threshold` | 走到这里的请求**继续走主流的比例**不够高（真分叉了）| 默认 0.05 软地板，典型数据不触发；研究模型业务线分叉时手动调高（0.5/0.7）|
| `coverage_threshold` | 主流子节点占**全局**比例已低到不能代表整体 | 默认 0.0 关闭；想剪长尾、只留高覆盖共识时手动开启（如 0.05）|

### 默认参数下的行为
`min_count=10, branch=0.05, coverage=0.0` → 走 trie 最重子节点链，直到 `freq < min_count`、或主流坍塌到 < 5%、或叶子。这是"宽松多数派 LCP" + 软地板防碎片化，与旧 `find_common_prefix(min_count=10)` 的设计意图最接近，但**输出链一定真实存在**（修复"幽灵链"）。

### 显式调高的典型用法

| 场景 | 推荐参数 |
|---|---|
| 找"分叉清晰的主流业务线" | `branch_threshold=0.6` 或 `0.7` |
| 剪掉占比 < 5% 的长尾共识 | `coverage_threshold=0.05` |
| Agent 模型的强共识严格分析 | `branch_threshold=0.8, coverage_threshold=0.5` |
| per-APP（小子集）兜底 | `min_count=2`, 其余默认（同当前 APP 行为） |

---

## 6. `stop_reason` 取值（6 种）

| stop_reason | 触发条件 |
|---|---|
| `no_records` | 输入 records 为空或全是 `block_ids == []` |
| `min_count` | 当前节点主流子的 freq < min_count |
| `branch_threshold` | branch_ratio < branch_threshold（真分叉了）|
| `coverage_threshold` | global_coverage < coverage_threshold（主流但占比太小）|
| `max_blocks` | chain 长度触 max_blocks 帽 |
| `no_children` | 走到叶子（所有请求恰好在此结束 → strict LCP 等于全部请求长度）|

---

## 7. `branch_alternatives` —— 诊断金矿

stop 时记录主流之外的 top-5 候选，每个含 `decoded_text_preview`（前 ~50 char）。

### 用例
- `stop_reason="branch_threshold"` + `alternatives=[(B, 50%, "你是问答助手..."), (C, 50%, "你是代码审查助手...")]` → **报告告诉读者：这个模型有两条业务线，A 共识到这里就分流了，建议分别跑 APP 报告或降低 branch_threshold**
- `stop_reason="min_count"` + `alternatives=[]` → 真到长尾了，没什么可看
- `stop_reason="coverage_threshold"` + `alternatives=[(X, 8%, "...")]` → 主流虽然占父 60%，但全局只占 8%，读者可以选择降 coverage_threshold 看更长链

### Top-N
N = 5（fixed in v1.0；未来可参数化）。**不含**当前 main chain 的延续节点（避免冗余）。

---

## 8. 不变量（实现 + 测试都要 pin）

新函数 MUST guarantee：

1. **Path-closed**：`consensus_blocks` 的 block_id 序列一定是**至少 `min_count` 个 records 的真实 prefix**。这是修复"幽灵链" bug 的根本契约。
2. **总分母（D3）**：`total_records = #records that have non-empty block_ids`。空 block_ids 不进 trie 也不进分母（与 CLAUDE.md V1 hit-rate 分母约定一致）。
3. **freq 单调非增**：chain 上 `consensus_blocks[i].freq >= consensus_blocks[i+1].freq`（深度越大经过的人越少或相等）。
4. **branch_ratio_pct ∈ (0, 100]**：永远是相对 chain 上的前一节点；root 的 freq = total_records，所以 position 0 的 branch_ratio_pct = global_coverage_pct。
5. **decoded_text 只用 chain 上的 block_id 解码**，不掺入 alternatives 的内容。
6. **`stop_reason` 不可空**：任何执行路径都必须给一个值。
7. **`branch_alternatives` 不含 winner**：是对 stop 节点子节点中**除去 main chain 续走方向**的 top-N 排序。
8. **threshold provenance**：3 个阈值都原样写进 result，便于可重现。
9. **空输入 short-circuit**：`total_records == 0` 必须在算法入口判定，避免后续除零。

---

## 9. 测试 fixture 目录（11 个最小集）

每个 fixture = `(records, params, expected_result)` 元组，inline 在
`tests/test_common_prefix_chain.py` 里（不引入额外 fixtures dir）。

| # | 名字 | 输入 | params | 预期 stop_reason | 关键断言 |
|---|---|---|---|---|---|
| 1 | `strict_lcp_full_chain` | 5 个完全相同 4-block 请求 | min_count=2 | `no_children` | chain 长 4，每节点 freq=5，branch_ratio=100% |
| 2 | `min_count_blocks_at_branch` | 5 个 `[A,B]` + 5 个 `[A,C]` | min_count=2 | `min_count` | chain=`[A]`（B/C 各 5 但都 < min_count=6 不会触发；改 min_count=6 测）—— 简化为 `min_count=6` 跑此 fixture |
| 3 | `branch_winner_below_threshold` | 6 个 `[A,B]` + 4 个 `[A,C]` | min_count=2, branch=0.7 | `branch_threshold` | chain=`[A]`（B 占 60% < 70% 不达标），alternatives=[B(60%), C(40%)] |
| 4 | `branch_winner_above_threshold` | 7 个 `[A,B]` + 3 个 `[A,C]` | min_count=2, branch=0.7 | `min_count` | chain=`[A,B]`（70% ≥ 70%），下一步无新 children → 实际 fixture 用 `[A,B,end]` 让 B 之后无子 |
| 5 | `coverage_threshold_kicks_in` | 100 个共享 `[A,B]` 后分成 60 走 C / 40 走 D | min_count=2, branch=0.5, coverage=0.7 | `coverage_threshold` | chain=`[A,B]`（C 占父 60% 但全局 60% < 70% 阈值）|
| 6 | `periodic_content_no_aliasing` | "abc"×50 prompts × 5 个相同请求（旧算法的歧义场景）| min_count=2 | `no_children` | chain 真实存在于 5 个请求；不会拼幽灵链 |
| 7 | `single_record_below_min_count` | 1 个请求 | min_count=2 | `min_count` | chain=`[]`，stop@0 |
| 8 | `empty_input` | `records=[]` | min_count=2 | `no_records` | chain=`[]`、prefix_length_chars=0、total_records=0 |
| 9 | `legacy_position_wise_diverges` | r1..r4=`[A,B/Y,...]` r5=`[B,X,...]` (§1 反例) | min_count=2 | `min_count` 或 `no_children` —— 关键是 chain ≠ position-wise 拼出的 `[A,X]` | 验证 trie 结果与旧算法不同：trie chain 是 `[A]` 或 `[A,B]`，绝不是 `[A,X]` |
| 10 | `varying_lengths` | 不同长度请求 + 短的请求提前结束 | min_count=2 | `min_count` | freq 在请求结束的位置不会被错误计入；不变量 #3 freq 单调非增 |
| 11 | `block_registry_decode` | chain 已知 + registry 已知 | min_count=2 | （任意）| 验证 decoded_text 严格按 chain.block_id 顺序拼接 + missing block_id 用 `<MISSING:...>` 占位 |

每个 fixture 都附额外子 assert 检查：`branch_alternatives` 至少 N≤5、threshold provenance 字段被原样回传、`prefix_length_chars == len(decoded_text)`。

---

## 10. JSON schema 影响（v1.2 → v1.3）

### `common_prefix/metadata.json` 字段（新算法产出）

```jsonc
{
  // 既有保留
  "trace_name": ..., "input_file": ..., "block_size": ...,
  "total_records": ..., "min_count_threshold": ...,
  "prefix_length_blocks": ..., "prefix_length_chars": ...,
  "mean_coverage_pct": ...,                  // ← 含义微调（见下）

  // 新增
  "algorithm": "trie_greedy_v1",             // 与旧 "position_wise_majority" 区分
  "branch_threshold": 0.05,
  "coverage_threshold": 0.0,
  "stop_reason": "min_count",
  "stop_position": 47,
  "branch_alternatives": [
    {"block_id": "...", "freq": 12, "fraction_of_parent": 0.18,
     "decoded_text_preview": "..."},
    ...
  ]
}
```

`mean_coverage_pct` 含义：旧算法是各位置 coverage_pct 的均值；新算法改为**链上 `global_coverage_pct` 的均值**（更可解释 + 单调下降可绘曲线）。

### `coverage_profile.csv` 列变更

| 旧 | 新 |
|---|---|
| `position, block_id, count, coverage_pct` | `position, block_id, freq, parent_freq, global_coverage_pct, branch_ratio_pct` |

### `report.json["section_4_content"].consensus_blocks` 元素

| 旧字段 | 新字段 | 备注 |
|---|---|---|
| `count` | `freq` | 命名对齐 trie 词汇；**保留旧名一个版本**做 alias 读取（D4）|
| `coverage_pct` | `global_coverage_pct` + `branch_ratio_pct` | **2 个数字**替代 1 个；旧名一个版本兼容读 |
| — | `parent_freq` | 新增 |
| `text_preview` / `truncated` / `content_type_guess` / `rank` | （**保留**）| 与旧一致 |

### `schema_version`：v1.2 → **v1.3**（D5）

`reports/report_builder.SCHEMA_VERSION = "1.3"`。Model 报告 + APP 报告同步升。

### 兼容期 alias 读取（D4）

在迁移期（commit 3 引入、commit 5 撤除），`reports/sections.py` 与 `reports/app_compute.py` 读 consensus_blocks 时：

```python
freq = b.get("freq") or b.get("count")
coverage_pct = b.get("global_coverage_pct") or b.get("coverage_pct")
```

让旧 report.json（schema_version=1.2）仍能被 v1.3 渲染。Commit 5 撤除 alias 后，旧报告需要重生成。

---

## 11. 实施步骤（5 个独立 commit）

每步独立提交，每步 `pytest` 全绿。**implementation 不引入新设计**——严格按本 spec 走。

| commit | 内容 | 关注点 |
|---|---|---|
| 1 | `feat(analysis): add find_common_prefix_chain (trie-greedy)` | 新函数 + 完整 §9 测试套（11 fixtures + invariants）+ docstring。**旧函数不动**，新老共存。 |
| 2 | `feat(common_prefix): script switches to trie-greedy by default` | `scripts/generate_common_prefix.py` 改用新函数；`metadata.json` 输出 §10 新字段；`coverage_profile.csv` 改新列。**重跑 synthetic_demo 验证 byte-level 输出符合 spec**。 |
| 3 | `refactor(reports): adapt sections / app_compute to v1.3 schema` | `build_section_4_content` + `compute_app_common_prefix` 走新字段名；alias 读取写好；`SCHEMA_VERSION = "1.3"`；renderer 加 `stop_reason` + `branch_alternatives` 展示。 |
| 4 | `feat(render): show stop_reason + branch_alternatives in section 4` + 撤 caveat | `_section_4` 加新展示；撤掉 [docs/可视化.md](./可视化.md) / [docs/dashboard_phase2_plan.md](./dashboard_phase2_plan.md) / [docs/analysis_dimensions.md](./analysis_dimensions.md) 的 caveat 块（修复落地）。 |
| 5 | `refactor(common_prefix): remove deprecated find_common_prefix` | 老函数 + 兼容 alias 读取一并删除。**前提**：repo 中所有 .json artifact 已重生成（或被丢弃）。 |

迁移期总跨度：commit 1–4 为一个 PR / 一组连续 commit；commit 5 至少**间隔 1 个版本**才能合入（让外部消费者有时间适配）。

---

## 12. 决策锁定（5/5）

| # | 决策 | 锁定值 | 备注 |
|---|---|---|---|
| **D1** | `branch_threshold` 默认 | **0.05**（软地板，可手动调整）| 5% 极小，典型数据不咬，只兜底极端碎片化；研究时手动调高（0.5/0.7）做"清晰主流"分析 |
| **D2** | `coverage_threshold` 默认 | **0.0**（no-op，opt-in 调研旋钮）| 当前阶段最关心"是否存在真实可复用 chain"，不主动剪长尾 |
| **D3** | `total_records` 分母 | **只算非空 block_ids 的 records** | 与 CLAUDE.md V1 hit-rate 分母一致 |
| **D4** | 字段命名（`count` → `freq`）| **硬切换** + 一个版本兼容读取 alias → 1 版后删旧名 | 见 §10 alias 实现 |
| **D5** | `schema_version` 升级 | v1.2 → **v1.3** | 字段不向后兼容地变名 |

---

## 13. 实施时的小提示（供下一个 session 参考）

- 新函数实现优先用 dict-of-dict TrieNode（递归树），不用 path-keyed flat dict（内存爆炸）
- 不变量 #3（freq 单调非增）+ #4（branch_ratio ∈ (0, 100]）可以做 property-based 测试，覆盖随机生成的 records
- `BranchAlternative.decoded_text_preview` 长度参数化（默认 50），通过 `_PREVIEW_LEN` 常量
- `stop_reason="no_records"` 必须在 build trie 之前判定，避免后续 `0/0` 异常
- v1.3 schema 落地时，记得 model 报告 / APP 报告**同时**升级——`report_builder.SCHEMA_VERSION` 是单一源
- `algorithm` 字段命名空间化（`trie_greedy_v1`）便于将来引入 v2（如带 sibling 链路、token 级别合并）

---

**spec 截止**：2026-05-11。
**下一步**：另起 session "开始 commit 1：实施 `find_common_prefix_chain`"。
