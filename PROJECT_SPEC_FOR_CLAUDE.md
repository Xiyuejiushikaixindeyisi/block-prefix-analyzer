# Block Prefix Analyzer — Project Spec for Claude Code

## 0. Collaboration rules for this project
This repository will be developed with **Claude Code**, but the collaboration style must follow these rules:

1. **Think in English, reply to the user in Chinese.**
   - Internal analysis, design reasoning, and code comments can remain in English if helpful.
   - All user-facing summaries, review notes, implementation plans, and explanations should be written in **clear Chinese**.

2. **Prioritize stability, extensibility, and isolation.**
   - The project should prefer a clean, layered design over clever shortcuts.
   - Each functional module should be as isolated as reasonably possible.
   - Each core module should have its **own independent test file(s)**.

3. **Use Git as a first-class engineering constraint.**
   - Work in small, reviewable steps.
   - Avoid mixing architecture changes, feature implementation, and refactors in one large change.
   - After each reviewable step, the repo should remain in a clean, testable state.

4. **Do not optimize too early.**
   - V1 should focus on correctness, deterministic behavior, and maintainability.
   - Performance optimizations should only happen after the correctness baseline is stable.

---

## 1. Project goal
Build an **offline analyzer for block-sequence prefix reuse**. The analyzer should process requests in chronological order and compute:

1. **max reusable prefix length** for each request
2. **prefix-aware ideal hit ratio** under an infinite-capacity prefix cache assumption
3. **block-level reusable ratio** under the looser assumption “a block is reusable if it appeared before”
4. optional derived metrics such as:
   - token-level prefix hit ratio
   - reuse time
   - block lifespan

The long-term goal is to make the analysis **semantically aligned with vLLM / vLLM-Ascend**, but the first version must prioritize:

- correctness of metric definitions
- deterministic offline replay
- testability
- modularity
- future extensibility

This tool is **independent from online serving** and should be usable on public datasets or internal sampled traces.

---

## 2. Development environment
- OS: **Windows + WSL**
- Project root folder: `block-prefix-analyzer`
- Coding assistant: **Claude Code**
- Initial state: **empty folder**
- Version control: **Git from day one**

Assume development happens entirely inside WSL.

---

## 3. Important design corrections to the original plan
The original plan is directionally good, but several changes are needed so the project remains reviewable and low-risk.

### 3.1 Do **not** start from tokenizer/template integration
Tokenizer alignment, chat template alignment, and block hash reproduction are important, but they should **not** be the first coding milestone.

Reason:
- they depend on external framework details
- they make debugging harder
- they are not required for validating the core prefix-analysis logic

**Correct first milestone:**
Start from traces that already contain `block_hash_ids` or an equivalent block sequence field.

That means the project should first solve:
- ordered request replay
- reusable-prefix matching
- ideal-hit accounting
- statistics output

Only after that is stable should we add adapters for:
- tokenizer
- chat template builder
- block builder / hash reproducer

### 3.2 First version should use a **plain trie-compatible design**, not a fully optimized radix tree
Although the project may eventually use a compressed radix tree, the first implementation should optimize for **clarity and correctness**.

Recommended approach:
- define a clean `PrefixIndex` interface
- implement a **simple Trie** first
- keep “radix compression” as a later optimization

Reason:
- fixed-size block ids already make plain trie logic straightforward
- correctness is easier to test
- path compression introduces complexity that is unnecessary in V1

### 3.3 The core processing unit should be **stream replay**, not random request lookup
The original interface ideas like `get_best_prefix(request_id)` are not the best fit.

This analyzer should primarily work as a **chronological replay engine**:

For each request in order:
1. compute its reusable prefix against previously inserted requests
2. record metrics
3. insert its block sequence into the index

This avoids ambiguity and matches the semantics of “previously seen” reuse.

### 3.4 Metric definitions must be frozen early
The project will fail if metric definitions remain fuzzy.

The first implementation must explicitly define:

#### A. block-level reusable ratio
A block counts as reusable if the same block hash appeared in any earlier request.
This is the loosest metric.

#### B. prefix-aware ideal hit ratio
Only the **contiguous prefix from the start of the request** counts as a hit.
Once the first miss appears, later blocks in the same request do **not** count as prefix hits.
This is the main metric.

#### C. token-level prefix hit ratio
Map the reusable prefix blocks back to token counts.
Need explicit handling for the final partial block.

#### D. reuse time
For a block reused by a later request:

`reuse_time = current_request_time - previous_reuse_reference_time`

V1 should support at least:
- `last_seen` reuse time
- deterministic tie-breaking using input order when timestamps are equal

#### E. lifespan
A block’s lifespan is the time from first appearance to its final reuse / final appearance under the chosen definition.
This can be a later-stage metric if needed.

### 3.5 Ordering rules must be deterministic
Many traces only have second-level timestamps.
Therefore request ordering must be:
1. timestamp ascending
2. stable original input order ascending

This rule must be implemented explicitly and documented.

### 3.6 Self-hit must be impossible
When processing a request, metrics must be computed **before** inserting that request into the prefix index.
Otherwise the request may incorrectly match itself.

### 3.7 Separate “semantic alignment” from “framework dependency”
The project should be organized in layers:

1. **core analyzer** — no vLLM dependency
2. **trace adapters** — parse datasets into canonical request records
3. **framework alignment adapters** — tokenizer/template/block builder reproduction
4. **evaluation / reports** — summary tables, CSV, plots if needed

This keeps the core usable even when framework details are unavailable.

---

## 4. Final scope for V1 / V2 / V3

### V1 — block-sequence analyzer only
Input already contains block sequence data, e.g. `block_hash_ids`.

Must support:
- chronological replay
- max reusable prefix length per request
- block-level reusable ratio
- prefix-aware ideal hit ratio
- summary stats
- CSV / JSONL outputs
- unit tests and golden tests

Must **not** include yet:
- tokenizer reproduction
- chat template reproduction
- Redis / SQLite
- distributed design
- aggressive optimization

### V2 — framework-aligned block builder
Add adapter pipeline:
- raw request text/messages
- tokenizer alignment
- chat template alignment
- block split and block hash reproduction

Then verify that V2-generated block sequences match framework behavior.

### V3 — optimization and scaling
Optional later work:
- radix compression
- memory optimization
- streaming incremental ingestion
- mmap / SQLite / on-disk index
- parallel preprocessing

---

## 5. Canonical data model
Define a canonical request record independent of data source.

```python
@dataclass
class RequestRecord:
    request_id: str
    timestamp: int | float
    arrival_index: int           # stable input order
    block_ids: list[int | str]
    token_count: int | None = None
    block_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Notes:
- `arrival_index` is mandatory for stable ordering
- `block_ids` is the minimum required field for V1
- `token_count` is optional in V1, useful for token-level metrics
- `metadata` can store workload type, turn number, department, chat id, etc.

---

## 6. Core architecture
Recommended package layout:

```text
block-prefix-analyzer/
  README.md
  CLAUDE.md
  PROJECT_SPEC_FOR_CLAUDE.md
  IMPLEMENTATION_PLAN.md
  pyproject.toml
  .gitignore
  src/
    block_prefix_analyzer/
      __init__.py
      types.py
      replay.py
      metrics.py
      io/
        __init__.py
        jsonl_loader.py
        csv_loader.py
      index/
        __init__.py
        base.py
        trie.py
      reports/
        __init__.py
        summary.py
        writers.py
      adapters/
        __init__.py
        # V2 starts here
        vllm_tokenizer.py
        chat_template.py
        block_builder.py
  tests/
    test_types.py
    test_trie.py
    test_replay.py
    test_metrics.py
    test_jsonl_loader.py
    test_summary.py
    fixtures/
```

Rules:
- Keep the package layered and dependency direction clear.
- `index/` should not import reporting code.
- `reports/` should not own core business logic.
- `adapters/` must remain optional and must not pollute V1.
- Tests should be split by module, not collapsed into one giant file.

---

## 7. Testing strategy
The project should be designed for **independent validation of each module**.

### 7.1 Test isolation requirements
At minimum:
- `test_types.py` validates canonical record assumptions
- `test_trie.py` validates trie insertion / prefix lookup semantics
- `test_replay.py` validates chronological replay and no-self-hit behavior
- `test_metrics.py` validates metric definitions on small crafted examples
- `test_jsonl_loader.py` validates trace parsing and ordering preservation
- `test_summary.py` validates aggregated outputs

### 7.2 Testing philosophy
- Use small deterministic fixtures.
- Prefer crafted examples over large opaque fixtures for unit tests.
- Add golden tests for a few end-to-end V1 examples.
- Every bug fix should ideally add or strengthen one test.

### 7.3 Non-goal for early testing
- Do not add large integration test harnesses too early.
- Do not benchmark performance before correctness is stable.

---

## 8. Git workflow requirements
Git is part of the project design, not an afterthought.

### 8.1 Required principles
- Initialize Git immediately.
- Make **small, reviewable commits**.
- One commit should preferably correspond to one logical step.
- Do not mix unrelated refactors with new features.
- The repo should remain runnable and testable after each step.

### 8.2 Suggested commit sequence for early development
1. `chore: initialize project skeleton`
2. `test: add fixtures and module-level test scaffolding`
3. `feat: add canonical request record and ordering helpers`
4. `feat: add prefix index interface and trie implementation`
5. `feat: add replay engine for reusable prefix accounting`
6. `feat: add summary metrics and writers`
7. `test: add golden tests for v1 replay semantics`

### 8.3 Branching guidance
If branch management is introduced, prefer:
- `main` as the stable reviewed branch
- short-lived feature branches for larger steps

But if the repo is still very small, direct work on `main` with disciplined commits is acceptable.

---

## 9. Implementation guidance for Claude Code
When you propose changes, you should:

1. explain the boundary of the current step in Chinese
2. list the files to be created or changed
3. explain why each file exists
4. prefer the smallest complete slice that keeps tests green
5. stop after the requested boundary is reached

You should actively avoid:
- speculative abstractions that are not yet needed
- hidden coupling between modules
- giant files that combine multiple responsibilities
- implementing V2/V3 logic during V1 scaffolding

---

## 10. Desired first deliverable
The first coding round should produce only:
- repository skeleton
- package structure
- minimal `pyproject.toml`
- minimal test setup
- `IMPLEMENTATION_PLAN.md`
- clear notes about ambiguities or risks

It should **not** attempt to finish the analyzer in one shot.

---

## 11. Current implementation status / divergence from original plan

> **Added**: reflects the state as of the V2 readiness gate (commit 56043b3).
> The original spec above remains the canonical design blueprint; this section
> records what has been built and where the current implementation diverges or
> extends beyond the original plan.

### What is complete

- **V1** — fully implemented per spec: chronological replay, two hit metrics,
  `MetricsSummary`, golden tests (145 tests).
- **TraceA paper repro** — F4 dual plot (reusable + prefix-aware) generated
  from the public TraceA dataset (43,058 records, block_size=16).
- **V2-min pipeline** — `normalize → render → tokenize → block build → V1 record`;
  internal consistency verified (87 tests); SHA-256 placeholder hash.
- **V2-full metrics** — `token_level_prefix_hit_ratio`, `mean_reuse_time`,
  `lifespan` (`compute_block_lifespans`), session/category helpers.
- **V2 readiness gate** — `QwenChatTemplate` (Layer 1 VERIFIED), xfail
  guards for Layer 2–3, readiness checklist C1–C6 tested (421 total tests).

### Divergence from original plan

| Original plan | Current state |
|---|---|
| `adapters/vllm_tokenizer.py` | `v2/adapters/hf_tokenizer.py` — interface only, PENDING |
| `adapters/block_builder.py` | `v2/adapters/block_builder.py` (SHA-256) + `siphash_builder.py` (mmh3 stub) |
| V2 "verify block sequences match framework" | Layer 1 verified; Layers 2–3 xfail (pending `transformers` + `mmh3`) |
| `adapters/` as flat directory under `block_prefix_analyzer/` | Implemented as `v2/adapters/` sub-package |

### Two-path distinction (not in original spec)

The original spec did not distinguish between:
- **Path A**: traces already carrying pre-computed `hash_ids` (TraceA)
- **Path B**: raw requests that need the full V2 template→tokenizer→hash chain

This distinction is critical. Path A is fully operational for F13–F15 analysis.
Path B requires `transformers` + `mmh3` for absolute framework alignment.
See `V2_READINESS.md` and `CLAUDE.md §10` for details.
