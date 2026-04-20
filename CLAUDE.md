# CLAUDE.md — Working rules for Claude Code inside this repo

这份文档是未来每一次 Claude Code 在本仓库工作时必须遵循的协作守则。
它浓缩了 `PROJECT_SPEC_FOR_CLAUDE.md` 的关键约束，避免后续轮次重新讨论。

---

## 1. 协作语言
- **思考用英文**，用户可见的回复 / 计划 / 评审 / 总结一律使用 **简体中文**。
- 代码注释、commit message、docstring 使用英文。

## 2. 工程原则
- 稳定、可扩展、模块隔离、可独立测试 优先于"看起来聪明"的捷径。
- 不做过早优化；V1 只追求 **正确性 + 确定性 + 可读性**。
- 不引入未被当前任务需要的抽象；不添加未验证的性能技巧。
- 依赖方向清晰：`index/` 不依赖 `reports/`；`reports/` 不承载核心业务逻辑；
  `adapters/`（V2 才会出现）必须保持可选，不得污染 V1。

## 3. 每次改动必须遵守
1. **先说边界**：当前这一步要做什么、不做什么，用中文讲清楚。
2. **列清单**：要新建 / 修改哪些文件，为什么需要它们。
3. **小步提交**：每个 commit 只对应一个逻辑关注点；架构重构与新功能不混在一起。
4. **测试保绿**：改动完成后 `pytest` 必须可运行通过。
5. **写完即停**：本轮边界达成后停下，等用户确认再继续。

## 4. V1 范围硬约束（当前阶段）
- 输入必须已包含 `block_hash_ids`（或等价的 block 序列）；**不要** 实现 tokenizer / chat template / block builder。
- 索引实现用 **简单 Trie**；**不要** 实现 radix 压缩。
- **不要** 引入 Redis / SQLite / 分布式 / 多进程 / 外部服务。
- **不要** 预先加 CLI、配置系统、Web 服务等未被需要的外壳。
- 回放必须严格遵循 **"先算指标 → 再插入索引"** 的顺序，禁止 self-hit。
- 请求排序必须显式按 `(timestamp, arrival_index)` 稳定升序。

## 5. 指标定义约定（V1 冻结）
| 指标 | 定义 |
|---|---|
| block-level reusable ratio | 如果某 block hash 在 **任何更早请求中** 出现过，则视为可复用。最宽口径。 |
| prefix-aware ideal hit ratio | 只有 **从请求起点开始、连续命中** 的 block 才计入；第一次 miss 之后的所有 block 都不再算命中。主要指标。 |
| token-level prefix hit ratio | 把可复用前缀的 block 数换算回 token 数；需要明确最后一个 partial block 的处理策略。 |
| reuse time | 被后续请求复用时 `current_time - last_seen_time`（V1 默认 `last_seen`）。 |
| block lifespan | 从首次出现到最终被复用 / 最终出现的时间跨度，V1 可延后。 |

聚合口径同时输出 **micro**（`Σ hit / Σ total`）与 **macro**（均值）两种，避免口径误解。

## 6. Git 规范
- `main`（或 `master`）保持可运行、测试绿。
- commit 信息使用 Conventional Commits 风格：`chore:` / `feat:` / `fix:` / `test:` / `docs:` / `refactor:`。
- 每个 commit 只做一件事；避免把重构和新功能塞进同一个 commit。
- 未经用户明确同意，**不主动 push、不创建 PR、不改 git config、不执行破坏性操作**（reset --hard、branch -D、push --force 等）。

## 7. 测试规范
- 每个核心模块有独立测试文件。
- 优先写 **小而确定** 的 crafted fixture，避免大块不透明数据。
- 每个 bug 修复应附带新增或增强的测试用例。

## 8. 文档规范
- 指标定义 / 数据模型 / 公共接口的 docstring 必须写清 **前置假设与不变量**。
- 只在 "WHY 不明显" 时写注释；不写对代码字面意思的重复解释。
- 不新建 Markdown 文档，除非用户明确要求（READM/CLAUDE/IMPLEMENTATION_PLAN 除外）。

## 9. 用户明确否决过的做法（禁止区）
- 从 tokenizer / chat template 开始做
- 在 V1 引入 radix 压缩、Redis、SQLite、分布式
- 通过 `get_best_prefix(request_id)` 这类"随机查询"模型组织核心逻辑（应该是时间序回放）

## 10. V2 两条分析路径的区分（必须遵守）

V2 存在两条完全不同的路径，**任何时候都不得混淆**：

### 路径 A — TraceA replay 路径
- 输入：已有 `hash_ids` 的 JSONL 记录（TraceA 公开数据集）
- 数据加载：`io/traceA_loader.py` 直接映射 `hash_ids → block_ids`
- **不经过** V2 chat template / tokenizer / block builder
- Layer 2–3 的 pending 状态**不影响**此路径
- F4 / F13 / F14 / F15 对 TraceA 的分析均走此路径

### 路径 B — raw request 完整对齐路径
- 输入：原始 `messages: list[Message]`，需经过 chat template → tokenizer → block builder
- 当前状态：Layer 1（chat template 渲染）已验证；Layer 2（tokenizer）/ Layer 3（block hash）仍 pending
- **不能宣称与 vLLM 完全对齐**，直到 `HFTokenizerAdapter` + `ChainedBlockBuilder` 的 xfail 测试转为 pass

### 写代码 / 文档时必须遵守
1. **不要把路径 B 的 pending 层写成"已完成框架对齐"。**
2. `alignment_status="pending_framework"` 的 fixture 必须保留 `expected_token_ids=None` 和 `expected_block_ids=None`，直到真实值被填入。
3. `xfail` 测试代表"诚实的 pending 承诺"，不得删除或改为 skip（除非真正填入了 golden 值）。
4. 在分析报告 / commit message 中，必须区分"对 TraceA hash_ids 的分析结论"和"对 raw request 完整链路的对齐结论"。
