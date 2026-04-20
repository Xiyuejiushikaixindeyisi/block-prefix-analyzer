# V2 Readiness Gate — Validated Scope

> Status: **GATE MET (internal-only config)** — F13–F15 analysis can proceed
> with the TraceA dataset using the internal-only configuration.
> Full Qwen2/vLLM framework alignment is PENDING.

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
| **token-level prefix hit ratio** | `prefix_hit_tokens / total_tokens` where `prefix_hit_tokens = prefix_hit_blocks × block_size`. If all full blocks are hit, the partial last block (leftover) is also credited. |
| **reuse_time** | `current_request.timestamp − last_seen_ts[block_id]`. Per-block, mean reported per request. Intra-request duplicates contribute one sample only. |
| **lifespan** | `last_reuse_timestamp − first_seen_timestamp` per block across the full trace. Zero if the block is never reused by a later request. Offline metric. |

---

## 4. What Can Be Analyzed Now

With the **internal-only configuration** (MinimalChatTemplate + CharTokenizer +
SimpleBlockBuilder) and the **TraceA dataset** (43,058 records, block_size=16):

| Analysis | Feasibility |
|---|---|
| F4 (reusable ratio over time) | ✅ Done — 59.40% overall |
| F4 (prefix-aware hit ratio over time) | ✅ Done — 57.79% overall |
| F13 (per-session hit distribution) | ✅ Ready — session helpers in place |
| F14 (per-category/type breakdown) | ✅ Ready — category helpers in place |
| F15 (reuse_time / lifespan distribution) | ✅ Ready — enriched_replay + compute_block_lifespans |
| Comparison to paper absolute numbers | ❌ Not yet — requires Qwen2 tokenizer + vLLM hash |
| Exact vLLM cache-key alignment | ❌ Not yet — requires ChainedBlockBuilder + mmh3 |

The TraceA dataset already carries `hash_ids` (pre-computed vLLM block IDs), so
for **F13–F15 on TraceA**, the V2 pipeline is NOT the bottleneck — the loader
(`traceA_loader.py`) uses the original `hash_ids` directly, bypassing the
chat template and tokenizer layers entirely.  The readiness gate above applies
to **synthetic / re-tokenized workloads**, not to TraceA replay.

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
