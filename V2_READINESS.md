# V2 Readiness Gate — Validated Scope

> Status: **GATE MET (internal-only config)** — F13–F15 analysis can proceed
> with the TraceA dataset using the internal-only configuration.
> Full Qwen2/vLLM framework alignment is PENDING.

---

## 0. Two Distinct Analysis Paths

This project serves two fundamentally different use cases.  **They must not be
conflated** in code, documentation, or analysis conclusions.

### Path A — TraceA replay (operational NOW)

```
TraceA JSONL  ──►  traceA_loader.py  ──►  RequestRecord (block_ids = hash_ids)
                                                    │
                                                    ▼
                                            V1 replay + enriched_replay
                                                    │
                                                    ▼
                                     F4 / F13 / F14 / F15 analysis
```

- Input already contains `hash_ids` computed by vLLM in production.
- `traceA_loader.py` maps `hash_ids → block_ids` directly.
- **Does NOT pass through** V2 chat template / tokenizer / block builder.
- Layer 2–3 pending status does **not** affect this path.
- All F4/F13/F14/F15 analysis of TraceA data uses this path.

### Path B — raw request full-alignment (partially PENDING)

```
RawRequest(messages)  ──►  ChatTemplateAdapter  ──►  TokenizerAdapter
                                                              │
                                                              ▼
                                                    BlockBuilder  ──►  block_ids
                                                                            │
                                                                            ▼
                                                                   V1 replay + metrics
```

- Layer 1 (chat template rendering): **VERIFIED** for both MinimalChatTemplate
  and QwenChatTemplate.
- Layer 2 (tokenizer): **PENDING** — requires `pip install transformers`.
- Layer 3 (block hash): **PENDING** — requires `pip install mmh3`.
- Cannot claim "exact vLLM alignment" until Layers 2–3 are verified.

---

## 1. Target Configuration

| Layer | Component | Status |
|---|---|---|
| Chat template | `QwenChatTemplate` — Qwen2 `<\|im_start\|>` format | Layer 1 **VERIFIED** |
| Tokenizer | `HFTokenizerAdapter("Qwen/Qwen2-7B-Instruct")` | **PENDING** (requires `transformers`) |
| Block builder | `ChainedBlockBuilder` with MurmurHash3 prefix-chaining | **PENDING** (requires `mmh3`) |
| Block size | 16 (vLLM default for Qwen2) | Configured |
| Fallback (internal) | `MinimalChatTemplate` + `CharTokenizer` + `SimpleBlockBuilder(SHA-256)` | **VERIFIED (internal-only)** |

---

## 2. Layer-by-Layer Verification Status

### Layer 1 — Chat Template Rendering

| Fixture | MinimalChatTemplate | QwenChatTemplate |
|---|---|---|
| single_user | VERIFIED | VERIFIED |
| system_user | VERIFIED | VERIFIED |
| multi_turn | VERIFIED | VERIFIED |
| empty_system_content | VERIFIED | VERIFIED |
| unicode_content | VERIFIED | N/A |
| exact_block_boundary_bs4 | VERIFIED | N/A |

**VERIFIED** means: the `expected_rendered_prompt` field in the fixture was derived
directly from the Jinja template source and is byte-for-byte reproducible by
running the adapter code.

`QwenChatTemplate` output matches the official Qwen2 Jinja template published
in `tokenizer_config.json` at `Qwen/Qwen2-7B-Instruct`. No transformers dependency
required for Layer 1.

### Layer 2 — Token IDs

- **MinimalChatTemplate + CharTokenizer**: VERIFIED (internal-only).
  `CharTokenizer` maps each character to `ord(c)`.  Deterministic but NOT a real BPE tokenizer.
- **QwenChatTemplate + HFTokenizerAdapter**: **PENDING**.
  Requires `pip install transformers` and downloading the Qwen2 vocabulary.
  Golden `expected_token_ids` fields in `qwen_fixtures.py` are `None` until filled.

### Layer 3 — Block IDs and Leftover

- **SimpleBlockBuilder (SHA-256, per-block independent)**: VERIFIED (internal-only).
  Hash function is SHA-256 truncated to 8 bytes (little-endian `uint64`).
  Does NOT match vLLM's prefix-cache key space.
- **ChainedBlockBuilder (MurmurHash3 with prefix chaining)**: **PENDING**.
  Interface is implemented; real `mmh3` hash function requires `pip install mmh3`.
  Golden `expected_block_ids` fields are `None` until filled.

---

## 3. Frozen Metric Definitions (V2 locked)

These definitions are **frozen** as of V2-min; must not be changed without a
corresponding update to all golden fixtures and the CLAUDE.md definition table.

| Metric | Definition |
|---|---|
| **block-level reusable ratio** | For each position `i` in `block_ids`, if `block_ids[i]` appeared in any strictly earlier request, position `i` is counted. Widest aperture. |
| **prefix-aware ideal hit ratio** | Only contiguous blocks from the request start that match the prefix trie count. First miss terminates the run. Main metric. |
| **token-level prefix hit ratio** | `content_prefix_reuse_tokens / total_tokens` where `content_prefix_reuse_tokens = content_prefix_reuse_blocks × block_size`. If all full blocks are hit, the partial last block (leftover) is also credited. |
| **reuse_time** | `current_request.timestamp − last_seen_ts[block_id]`. Per-block, mean reported per request. Intra-request duplicates contribute one sample only. |
| **lifespan** | `last_reuse_timestamp − first_seen_timestamp` per block across the full trace. Zero if the block is never reused by a later request. Offline metric. |

---

## 4. Applicability by Path

### Path A — TraceA replay (what can be done now)

TraceA carries pre-computed vLLM `hash_ids`; the V2 template/tokenizer/hash
chain is **bypassed entirely**.  All analysis below is currently feasible:

| Analysis | Status | Note |
|---|---|---|
| F4 reusable ratio over time | ✅ Done | 59.40% overall on TraceA |
| F4 prefix-aware hit ratio over time | ✅ Done | 57.79% overall on TraceA |
| F13 single-turn reuse_time distribution | ✅ Ready | session helpers in place |
| F14 multi-turn / follow-up reuse_time | ✅ Ready | category/session helpers in place |
| F15 reuse_time by request category | ✅ Ready | `get_category()` available |
| lifespan distribution | ✅ Ready | `compute_block_lifespans()` available |

### Path B — raw request full-alignment (what requires more work)

These conclusions require Path B to be fully verified (Layers 2–3 passing):

| Analysis | Status | Blocker |
|---|---|---|
| Comparison to paper absolute token-level numbers | ❌ PENDING | Qwen2 tokenizer Layer 2 |
| Exact vLLM cache-key alignment | ❌ PENDING | ChainedBlockBuilder + mmh3 Layer 3 |
| Certifying that synthetic traces match vLLM behaviour | ❌ PENDING | Both layers |

---

## 5. What Remains Pending Before Full Framework Alignment

1. **Install `transformers`** and verify Qwen2 tokenizer golden values (Layer 2).
2. **Install `mmh3`** and verify ChainedBlockBuilder block IDs (Layer 3).
3. Fill in `expected_token_ids`, `expected_block_ids`, `expected_leftover_token_count`
   in `tests/v2_alignment/qwen_fixtures.py`.
4. Change `alignment_status` from `"pending_framework"` to `"framework_verified"`.
5. Once Layer 2–3 are verified, the 10 xfail tests in `test_v2_readiness.py` will
   convert to PASS automatically.

---

## 6. Readiness Conclusion

> **Conclusion 1: V2 has met the minimum gate to begin F13–F15 analysis.**

Specifically:
- All three hit-metric definitions are locked, tested, and human-verifiable (C1 ✅).
- Layer-1 (chat template) is fully verified for both MinimalChatTemplate and QwenChatTemplate (C2a ✅).
- Layers 2–3 are pending framework alignment, clearly marked as xfail (C2b/C2c 🔶 pending).
- V1 core chain has a non-regression golden test (C3 ✅).
- A hand-crafted trace verifies all four metric types with human-readable expected values (C4 ✅).
- Path equivalence between V2 pipeline and hand-authored block_ids is confirmed (C5 ✅).
- Session/category helpers are implemented and tested (C6 ✅).
- This document exists (C7 ✅).

The pending xfail items (C2b, C2c) do **not** block F13–F15 work on TraceA,
because TraceA records already contain pre-computed block IDs from vLLM and
do not pass through the V2 chat template / tokenizer pipeline.
