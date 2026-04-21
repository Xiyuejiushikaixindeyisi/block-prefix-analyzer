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

### 5a. 三个概念严格区分（禁止混用）

| 概念 | 代码字段名 | 定义 | 能否等同于 vLLM prefix hit？ |
|---|---|---|---|
| **content_block_reuse_event** | `content_reused_blocks_anywhere` | 任意位置的 block 只要在严格过去的历史池出现过即为 reuse event，不要求位于连续前缀段内 | **否**——非前缀连续命中不等价于 vLLM prefix cache hit |
| **content_prefix_reuse_blocks** | `content_prefix_reuse_blocks` | 从请求开头开始连续命中的 block 数，第一次 miss 后停止计数 | **是（无限容量等价）**——见下方核心原因 |
| **finite_capacity_vllm_prefix_hit** | _(未实现)_ | 在特定 cache size + LRU/LFU 驱逐策略下的实际 prefix cache hit | 需额外模拟驱逐；`content_prefix_reuse_blocks` 是其上界 |

**核心原因（2025-04 更新）**

**vLLM V1 APC（Automatic Prefix Caching）机制：**

> "vLLM chooses a hash-based approach for implementing prefix caching."  — 官方设计文档

数据结构：`BlockPool + free queue + block_hash → KVCacheBlock 哈希表`（**不是** radix tree）。

链式 block hash 公式：
```
block_hash[i] = H(parent_block_hash   = block_hash[i-1],
                  curr_block_token_ids = tuple(tokens[i*B : (i+1)*B]),
                  extra_keys           = (lora_id?, mm_identifiers?, cache_salt?))
```
vLLM 以 `block_hash[i]` 作为物理块的唯一标识，在 BlockPool 的哈希表中查找已缓存的 KVCacheBlock。相同 hash 的块共享同一物理内存。命中时从前往后线性匹配这串 hash，遇到第一个 miss 停止。**因此 vLLM 只能共享"前缀链完全一致"的块**——任何中间 block 不同则后续链全部发散，无法共享。

**为何不用 radix tree：**
前缀依赖信息已被"压缩进"链式哈希——每个 block 的 hash 里含有 parent hash，后续 block 的 hash 天然编码了完整前缀历史。不需要显式树节点来保存 token 前缀路径，hash map 一次查找即可确认"这个 block 在这条前缀链上是否命中"。

**hash map 性能特性（来源：`single_type_kv_cache_manager.py` line 282）：**
- 查找复杂度 O(k)，k = 命中的 full block 数，**不是** token 数
- block 粒度大（典型 16 或 128 token/block），4096 token 前缀最多 32 次 map lookup
- `find_longest_cache_hit()` 顺序扫描，遇到第一个 miss 立即停止，不做全局搜索

**hash map 内存特性（来源：`block_pool.py`）：**
- 上界 = BlockPool 初始化时预分配的固定 KVCacheBlock 数量（line 141），不会随请求数/token 数无限增长
- 淘汰时在 `_maybe_evict_cached_block()`（line 307）从 map 中移除，有界
- **不做内容去重（no dedup）**：两个 block 内容完全相同，也不会合并为同一物理 block（为保持 block_id append-only，来源：官方 API 文档 + block_pool.py line 45）
- 重复 hash 时 value 从单个 `KVCacheBlock` 变为 `{block_id: KVCacheBlock}` 映射（line 72）

**vLLM V1 实现细节（来源：kv_cache_utils.py / cache.py）：**

| 项 | 细节 |
|---|---|
| 哈希函数 H | 由 `CacheConfig.prefix_caching_hash_algo` 配置；默认 `sha256`（pickle 序列化后 SHA-256），另有 `sha256_cbor`（CBOR 序列化，跨语言可复现） |
| 第一个 block 的 parent | `NONE_HASH`：若设置 `PYTHONHASHSEED` 则由其派生，否则 `os.urandom(32)`；**不是固定常量、不是空字节** |
| `extra_keys` 类型 | `Optional[tuple[Any, ...]]`；无 LoRA、无多模态、无 `cache_salt` 时为 `None` |
| `extra_keys` 包含什么 | LoRA int ID、多模态 `mm_feature.identifier`（仅覆盖到当前 block 的特征）、`cache_salt`（仅加入第一个 block，`start_token_idx == 0`） |
| `extra_keys` **不包含**什么 | **模型名、dtype、kv_cache dtype、模型权重 hash 均不在此**；模型隔离通过 vLLM 实例（进程）级 BlockPool 隔离实现，而非 hash 字段 |
| `block_size` | 可配置（1/8/16/32/64/128），默认 16；多数 attention backend 要求为 16 的倍数；Ascend 开启 prefix caching 时强制 128 |
| 尾 block 处理 | **不参与哈希，不进入 prefix cache**；只有 token 数恰好等于 `block_size` 的完整 block 才生成 block hash |

**TraceA hash_ids 与 vLLM APC 的关系：**
```
TraceA:  hash_id[i] = SipHash-2-4(salt, tokens[i*16:(i+1)*16])    // 仅依赖 block 内容，独立哈希
vLLM:    block_hash[i] = H(block_hash[i-1], tuple(tokens[i*B:(i+1)*B]), extra_keys)  // 链式哈希
```

前缀匹配的等价性证明（归纳）：
```
若 hash_ids_A[0..k-1] == hash_ids_B[0..k-1]
  ⟺ (SipHash 无碰撞) tokens_A[0..16k] == tokens_B[0..16k]
  ⟹ vllm_hash_A[0] = H(seed, tokens[0:16],  extra) = vllm_hash_B[0]  ✓
     vllm_hash_A[1] = H(vllm_hash_A[0], tokens[16:32], extra)
                    = H(vllm_hash_B[0], tokens[16:32], extra)
                    = vllm_hash_B[1]  ✓
     ...（归纳至 k-1）...
  ⟹ 无限容量 vLLM APC 在位置 0..k-1 全部命中
```

非前缀匹配为何不等价：
```
hash_ids_A[j] == hash_ids_B[j]  但  hash_ids_A[0..j-1] ≠ hash_ids_B[0..j-1]
  ⟹ vllm_hash_A[j-1] ≠ vllm_hash_B[j-1]
  ⟹ vllm_hash_A[j] = H(vllm_hash_A[j-1], tokens, extra)
                    ≠ H(vllm_hash_B[j-1], tokens, extra)
                    = vllm_hash_B[j]
  ⟹ vLLM 不共享此物理块（hash 不同）
```

**结论：**
- `content_reused_blocks_anywhere` ≠ vLLM APC hit（非前缀块因链式 hash 不同，vLLM 无法共享）
- `content_prefix_reuse_blocks` **= 无限容量 vLLM APC 命中数的等价度量**（same-model 假设下）
- 有限容量（含 LRU/LFU 驱逐）的实际命中率 ≤ `content_prefix_reuse_blocks` / `total_blocks`

**尾 block 截断（代码对齐 vLLM 实现）：**

vLLM 只对完整 block（token 数 == `block_size`）生成 block_hash 并进入 prefix cache；不足 `block_size` 的尾 block 不参与哈希、不被缓存。

- **Path A（TraceA loader）**：在 `io/traceA_loader.py` 中对齐。当 `input_length`（即 `token_count`）可用时，加载时截断 `block_ids = hash_ids[:input_length // TRACE_A_BLOCK_SIZE]`，剔除末尾的部分 block hash。若 `input_length` 缺失，`block_ids` 保留原始 `hash_ids` 全部条目。`TRACE_A_BLOCK_SIZE = 16`，与数据集文件名 `blksz_16` 一致。
- **Path B（V2 pipeline）**：`ChainedBlockBuilder.build()` 和 `SimpleBlockBuilder.build()` 均只对完整 block 生成 block_id（`while i + block_size <= n`），自动跳过不足 `block_size` 的尾部 token，无需额外截断。

**block_size 强制校验（新数据集必须显式指定）：**

`build_block_records_from_raw_requests()` 现要求显式传入 `block_size=<N>` 或 `block_builder=<builder>` 之一；两者均缺失时抛 `ValueError`。原因：新业务数据集的 block_size 必须与部署 vLLM 实例的 block_size 保持一致，任何默认值都可能静默产生错误结果。

**NONE_HASH 与 `ChainedBlockBuilder.initial_hash`：**

vLLM 第一个 block 的 parent hash（`NONE_HASH`）由 `PYTHONHASHSEED` 派生或 `os.urandom(32)` 生成，**不是固定常量 0**。`ChainedBlockBuilder` 默认 `initial_hash=0` 是分析占位值，含义如下：
- 对**同一数据集内部**的相对命中率分析：使用固定种子（任何值）均可获得正确结果，`initial_hash=0` 足够。
- 对**精确复现特定 vLLM 部署**的 block_id 值：需从该部署获取 NONE_HASH 并通过 `ChainedBlockBuilder(initial_hash=<NONE_HASH>)` 传入。否则计算出的 block_id 值与 vLLM 不同，但命中率统计仍然准确。

**Same-model 假设（必须满足，且必须在数据集层面保证）：**
KV cache 中存储的 K/V tensor 不仅由 token 决定，还由模型权重 W_K、W_V 决定：
```
K[i] = W_K × embedding(token[i])
V[i] = W_V × embedding(token[i])
```
不同模型处理相同 token 序列产生的 KV tensor 完全不同，**不可跨模型共享**。
vLLM 通过**进程级 BlockPool 隔离**（而非 hash 字段）来防止跨模型复用；`extra_keys` 中不含模型标识。

**因此，输入数据集必须满足：所有 request 均来自同一模型的同一部署实例。** 混入不同模型的请求会导致 prefix 命中率计算结果不具备物理意义（两条请求的 token 相同但 KV tensor 不同，不能复用）。
`content_prefix_reuse_blocks` 仅对 **单一模型单次部署** 的请求集合有效；
跨模型（即使 token 序列相同）无 KV 复用可言。
TraceA 数据集来自单一 Qwen 模型，该假设在 TraceA 分析中自动满足。

**Tokenizer 与 prefix cache 命中率分析的关系：**
- vLLM 的链式哈希直接作用于 **token IDs**，不依赖原始文本
- `content_prefix_reuse_blocks` 直接对比 `hash_ids`（token block 的哈希），**不需要 tokenizer 参与**
- 只要数据集已提供 token IDs（或等价的 block 哈希），tokenizer **完全不是**复现 prefix cache 命中率的前提
- Tokenizer 仅在输入是**原始文本（raw text）**、需要先转换为 token IDs 时才进入链路
- 因此：计算理想 prefix 命中率所需的最小信息 = token IDs + block_size + vLLM 链式哈希参数（H、NONE_HASH、extra_keys）
- Path B 的 "tokenizer pending" 状态**不影响** prefix cache 命中率分析；Path A（TraceA，已有 hash_ids）完全绕开 tokenizer

**命名约束：**
- `content_prefix_reuse_blocks` 可标注为 **"ideal prefix cache hit (infinite capacity, same model)"**
- 禁止标注为 **"vLLM prefix cache hit"**（未说明无限容量假设时），以防误导有限容量场景
- `content_reused_blocks_anywhere` 禁止以任何形式称为 prefix cache hit

### 5b. 其余指标
| 指标 | 定义 |
|---|---|
| `content_block_reuse_ratio` | `Σ content_reused_blocks_anywhere / Σ total_blocks`，最宽口径 |
| `content_prefix_reuse_rate` | `Σ content_prefix_reuse_blocks / Σ total_blocks`，从起点连续命中 |
| `content_prefix_reuse_token_ratio` | `content_prefix_reuse_tokens / total_tokens`，token 粒度 |
| reuse time | `current_time - last_seen_time`（V1 默认 `last_seen`） |

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
- 输入为**原始文本**时的完整链路：`raw_text → tokenizer → token_ids → block_splitter → chained_hash`
- 输入为**token IDs** 时可跳过 tokenizer，直接走：`token_ids → block_splitter → chained_hash`
- 当前状态：Layer 1（chat template 渲染）已验证；tokenizer 对齐 / `ChainedBlockBuilder` 仍 pending
- **不能宣称与 vLLM 完全对齐**，直到 `ChainedBlockBuilder`（使用 vLLM 链式哈希参数）的 xfail 测试转为 pass
- **适用场景**：处理无预计算 `hash_ids` 的新请求；Path A 已有 `hash_ids` 则无需此路径
- **注意**：即使 Path B 完成，KV cache 复用分析也必须限定在同一模型部署内
- **vLLM 链式哈希参数**（Path B 对齐时必须确认）：
  - 哈希函数：`prefix_caching_hash_algo`（默认 `sha256`）
  - 第一个 block 的 parent：`NONE_HASH`（`PYTHONHASHSEED` 派生或 `os.urandom(32)`）
  - `extra_keys`：LoRA ID、多模态特征 identifier、cache_salt；**不含模型名/dtype**

### 写代码 / 文档时必须遵守
1. **不要把路径 B 的 pending 层写成"已完成框架对齐"。**
2. `alignment_status="pending_framework"` 的 fixture 必须保留 `expected_token_ids=None` 和 `expected_block_ids=None`，直到真实值被填入。
3. `xfail` 测试代表"诚实的 pending 承诺"，不得删除或改为 skip（除非真正填入了 golden 值）。
4. 在分析报告 / commit message 中，必须区分"对 TraceA hash_ids 的分析结论"和"对 raw request 完整链路的对齐结论"。
