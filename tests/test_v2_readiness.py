"""V2 readiness gate: minimum verification checklist before F13-F15.

This file enforces the "V2 completeness gate" described in V2_READINESS.md.
Each section corresponds to one item in the validated-scope checklist.

Items and their current status
--------------------------------
C1  Three hit definitions locked          PASS  (internal-only config)
C2a Template consistency (layer 1)        PASS  (MinimalChatTemplate + QwenChatTemplate layer-1)
C2b Tokenizer consistency (layer 2)       XFAIL (pending: transformers not installed)
C2c Block-hash consistency (layer 3)      XFAIL (pending: mmh3 not installed)
C3  V1 no regression (golden)             PASS
C4  Hand-crafted trace (human-verifiable) PASS
C5  Path equivalence                      PASS
C6  session/category semantics            PASS  (see also test_v2_session.py)

Any test in this file that fails (not xfail) means the gate is NOT met.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.replay import replay as v1_replay
from block_prefix_analyzer.types import RequestRecord
from block_prefix_analyzer.v2.adapters.block_builder import SimpleBlockBuilder
from block_prefix_analyzer.v2.adapters.chat_template import MinimalChatTemplate
from block_prefix_analyzer.v2.adapters.qwen_chat_template import QwenChatTemplate
from block_prefix_analyzer.v2.adapters.tokenizer import CharTokenizer
from block_prefix_analyzer.v2.metrics import (
    EnrichedPerRequestResult,
    compute_block_lifespans,
    enriched_replay,
)
from block_prefix_analyzer.v2.pipeline import build_block_records_from_raw_requests
from block_prefix_analyzer.v2.schema import Message, RawRequest
from block_prefix_analyzer.v2.session import (
    get_category,
    group_by_session,
    is_followup_request,
    is_root_request,
)
from tests.v2_alignment.fixtures import (
    VERIFIED_FIXTURES,
    V2AlignmentFixture,
)
from tests.v2_alignment.qwen_fixtures import QWEN_LAYER1_VERIFIED


# ===========================================================================
# C1: Three hit definitions are frozen
# (block-level reusable ratio, prefix-aware ideal hit ratio, token-level ratio)
# ===========================================================================
#
# Hand-crafted trace:
#   t=0   r1: [1, 2, 3]        cold start
#   t=10  r2: [1, 2, 4]        prefix_hit=2 (1,2 match); reusable=2 (1,2 seen)
#   t=20  r3: [1, 3, 2]        prefix_hit=1 (1 matches, then 3≠2); reusable=3 (all seen)
#
# Human-verifiable expected values:
#   r1: prefix_hit=0, reusable=0
#   r2: prefix_hit=2, reusable=2, reuse_time=mean(10,10)=10.0
#   r3: prefix_hit=1, reusable=3, reuse_time=mean(10,20,10)=13.333...
#
# Lifespan:
#   block 1: first=0, last_reuse=20 → 20
#   block 2: first=0, last_reuse=20 → 20
#   block 3: first=0, last_reuse=20 → 20
#   block 4: first=10, last_reuse=None → 0

_HANDCRAFTED_RECORDS = [
    RequestRecord("r1", 0.0,  0, [1, 2, 3], token_count=3, block_size=1),
    RequestRecord("r2", 10.0, 1, [1, 2, 4], token_count=3, block_size=1),
    RequestRecord("r3", 20.0, 2, [1, 3, 2], token_count=3, block_size=1),
]


class TestC1_ThreeHitDefinitions:
    """C1: All three hit metrics coexist on the same trace; verify each."""

    def _results(self):
        return list(enriched_replay(_HANDCRAFTED_RECORDS))

    # ---- block-level prefix-aware ideal hit ratio ----

    def test_r1_cold_start_prefix_hit_zero(self):
        assert self._results()[0].prefix_hit_blocks == 0

    def test_r2_prefix_hit_two(self):
        assert self._results()[1].prefix_hit_blocks == 2

    def test_r3_prefix_hit_one(self):
        # [1, 3, 2]: 1 matches, but 3 ≠ 2 (second position in trie after r1, r2)
        assert self._results()[2].prefix_hit_blocks == 1

    # ---- block-level reusable ratio ----

    def test_r1_cold_start_reusable_zero(self):
        assert self._results()[0].reusable_block_count == 0

    def test_r2_reusable_two(self):
        assert self._results()[1].reusable_block_count == 2

    def test_r3_reusable_three(self):
        assert self._results()[2].reusable_block_count == 3

    # ---- token-level prefix hit ratio ----

    def test_r1_token_ratio_zero(self):
        assert self._results()[0].token_level_prefix_hit_ratio == 0.0

    def test_r2_token_ratio_two_thirds(self):
        assert abs(self._results()[1].token_level_prefix_hit_ratio - 2 / 3) < 1e-9

    def test_r3_token_ratio_one_third(self):
        assert abs(self._results()[2].token_level_prefix_hit_ratio - 1 / 3) < 1e-9


# ===========================================================================
# C2a: Chat template Layer-1 consistency
# Each fixture independently verifiable; failures localize to template layer.
# ===========================================================================

class TestC2a_TemplateConsistency:
    """C2a: MinimalChatTemplate and QwenChatTemplate Layer-1 golden values."""

    @pytest.mark.parametrize("fx", VERIFIED_FIXTURES, ids=lambda f: f.name)
    def test_minimal_template_renders_expected(self, fx: V2AlignmentFixture):
        tmpl = MinimalChatTemplate()
        actual = tmpl.render(fx.messages)
        assert actual == fx.expected_rendered_prompt, (
            f"MinimalChatTemplate Layer-1 mismatch for {fx.name!r}.\n"
            f"  expected: {fx.expected_rendered_prompt!r}\n"
            f"  actual:   {actual!r}"
        )

    @pytest.mark.parametrize("fx", QWEN_LAYER1_VERIFIED, ids=lambda f: f.name)
    def test_qwen_template_layer1_verified(self, fx: V2AlignmentFixture):
        """Layer-1 VERIFIED: QwenChatTemplate output matches Qwen2 Jinja template."""
        tmpl = QwenChatTemplate()
        actual = tmpl.render(fx.messages)
        assert actual == fx.expected_rendered_prompt, (
            f"QwenChatTemplate Layer-1 mismatch for {fx.name!r}.\n"
            f"  expected: {fx.expected_rendered_prompt!r}\n"
            f"  actual:   {actual!r}"
        )


# ===========================================================================
# C2b: Tokenizer Layer-2 consistency (PENDING — xfail until transformers available)
# ===========================================================================

class TestC2b_TokenizerConsistency:
    """C2b: Real tokenizer alignment — xfail until 'transformers' is installed."""

    @pytest.mark.xfail(
        reason="Layer-2 pending: pip install transformers to enable Qwen2 tokenizer",
        strict=False,
    )
    def test_hf_tokenizer_importable(self):
        import transformers  # noqa: F401

    @pytest.mark.xfail(
        reason="Layer-2 pending: Qwen2 tokenizer golden values not yet filled in",
        strict=False,
    )
    @pytest.mark.parametrize("fx", QWEN_LAYER1_VERIFIED, ids=lambda f: f.name)
    def test_qwen_token_ids_pending(self, fx: V2AlignmentFixture):
        """Slot for Layer-2 token golden values; expected to xfail until filled."""
        assert fx.expected_token_ids is not None, (
            f"Fixture {fx.name!r}: expected_token_ids is None (pending_framework). "
            "Fill in after running with HFTokenizerAdapter('Qwen/Qwen2-7B-Instruct')."
        )


# ===========================================================================
# C2c: Block hash Layer-3 consistency (PENDING — xfail until mmh3 available)
# ===========================================================================

class TestC2c_BlockHashConsistency:
    """C2c: Chained-hash block builder alignment — xfail until mmh3 installed."""

    @pytest.mark.xfail(
        reason="Layer-3 pending: pip install mmh3 to enable ChainedBlockBuilder",
        strict=False,
    )
    def test_mmh3_importable(self):
        import mmh3  # noqa: F401

    @pytest.mark.xfail(
        reason="Layer-3 pending: Qwen2 block IDs not yet filled in",
        strict=False,
    )
    @pytest.mark.parametrize("fx", QWEN_LAYER1_VERIFIED, ids=lambda f: f.name)
    def test_qwen_block_ids_pending(self, fx: V2AlignmentFixture):
        """Slot for Layer-3 block-hash golden values."""
        assert fx.expected_block_ids is not None, (
            f"Fixture {fx.name!r}: expected_block_ids is None (pending_framework)."
        )


# ===========================================================================
# C3: V1 no regression — golden test for V1 main chain
# ===========================================================================

class TestC3_V1NoRegression:
    """C3: V1 core chain is unaffected by V2 additions."""

    def _v1_records(self):
        """Hand-authored V1-compatible records (no V2 pipeline involved)."""
        return [
            RequestRecord("cold", 0.0,  0, [10, 20, 30]),
            RequestRecord("hit2", 1.0,  1, [10, 20, 40]),
            RequestRecord("hit1", 2.0,  2, [10, 50, 60]),
            RequestRecord("miss", 3.0,  3, [99, 98, 97]),
        ]

    def test_v1_replay_result_count(self):
        results = list(v1_replay(self._v1_records()))
        assert len(results) == 4

    def test_v1_cold_start_zero_hit(self):
        results = list(v1_replay(self._v1_records()))
        assert results[0].prefix_hit_blocks == 0
        assert results[0].reusable_block_count == 0

    def test_v1_second_request_prefix_two(self):
        results = list(v1_replay(self._v1_records()))
        assert results[1].prefix_hit_blocks == 2
        assert results[1].reusable_block_count == 2

    def test_v1_third_request_prefix_one(self):
        results = list(v1_replay(self._v1_records()))
        assert results[2].prefix_hit_blocks == 1
        assert results[2].reusable_block_count == 1

    def test_v1_all_miss_request(self):
        results = list(v1_replay(self._v1_records()))
        assert results[3].prefix_hit_blocks == 0
        assert results[3].reusable_block_count == 0

    def test_v1_replay_order_is_by_timestamp(self):
        shuffled = [
            RequestRecord("r3", 3.0, 2, [99]),
            RequestRecord("r1", 1.0, 0, [1]),
            RequestRecord("r2", 2.0, 1, [2]),
        ]
        results = list(v1_replay(shuffled))
        assert [r.timestamp for r in results] == [1.0, 2.0, 3.0]


# ===========================================================================
# C4: Hand-crafted trace — all four metrics human-verifiable
# ===========================================================================

class TestC4_HandCraftedTrace:
    """C4: Human-verifiable trace covering all four metric types.

    Trace layout (block_size=1, token_count=3):
        t=0   r1: [1, 2, 3]   cold start
        t=10  r2: [1, 2, 4]   prefix=2, reusable=2, mean_reuse_time=10.0
        t=20  r3: [1, 3, 2]   prefix=1, reusable=3, mean_reuse_time=13.33...

    Lifespan (blocks 1,2,3 all reused at t=20; block 4 never reused):
        block 1 → 20, block 2 → 20, block 3 → 20, block 4 → 0
    """

    def _results(self):
        return list(enriched_replay(_HANDCRAFTED_RECORDS))

    def test_prefix_hit_all_requests(self):
        rs = self._results()
        assert [r.prefix_hit_blocks for r in rs] == [0, 2, 1]

    def test_reusable_count_all_requests(self):
        rs = self._results()
        assert [r.reusable_block_count for r in rs] == [0, 2, 3]

    def test_reuse_time_r2_exactly_10(self):
        rs = self._results()
        assert rs[1].mean_reuse_time == 10.0

    def test_reuse_time_r3_mean(self):
        rs = self._results()
        # block 1 last_seen at t=10 → diff=10; block 3 last_seen at t=0 → diff=20;
        # block 2 last_seen at t=10 → diff=10; mean = 40/3
        expected = 40.0 / 3
        assert abs(rs[2].mean_reuse_time - expected) < 1e-9

    def test_lifespan_reused_blocks(self):
        spans = {s.block_id: s.lifespan for s in compute_block_lifespans(_HANDCRAFTED_RECORDS)}
        assert spans[1] == 20.0
        assert spans[2] == 20.0
        assert spans[3] == 20.0

    def test_lifespan_never_reused_block_zero(self):
        spans = {s.block_id: s.lifespan for s in compute_block_lifespans(_HANDCRAFTED_RECORDS)}
        assert spans[4] == 0.0


# ===========================================================================
# C5: Path equivalence
# V2 pipeline path and hand-authored block_ids path agree in V1 replay
# ===========================================================================

class TestC5_PathEquivalence:
    """C5: V2 pipeline output and hand-authored block_ids give identical V1 results.

    Uses FIXTURE_SINGLE_USER golden block_ids to cross-validate:
      Path A: raw request → build_block_records_from_raw_requests → replay
      Path B: hand-authored RequestRecord with known block_ids → replay
    """

    def _v2_records(self):
        from tests.v2_alignment.fixtures import FIXTURE_SINGLE_USER as FX
        raw = [RawRequest(
            request_id="r1",
            timestamp=0.0,
            messages=FX.messages,
        )]
        return build_block_records_from_raw_requests(
            raw,
            chat_template=MinimalChatTemplate(),
            tokenizer=CharTokenizer(),
            block_builder=SimpleBlockBuilder(block_size=FX.block_size),
        )

    def _hand_authored_records(self):
        from tests.v2_alignment.fixtures import FIXTURE_SINGLE_USER as FX
        return [RequestRecord(
            request_id="r1",
            timestamp=0.0,
            arrival_index=0,
            block_ids=list(FX.expected_block_ids),
        )]

    def test_v2_pipeline_block_ids_match_golden(self):
        from tests.v2_alignment.fixtures import FIXTURE_SINGLE_USER as FX
        records = self._v2_records()
        assert records[0].block_ids == FX.expected_block_ids

    def test_path_equivalence_prefix_hit(self):
        """Both paths produce the same prefix_hit_blocks for a repeated request."""
        v2_records = self._v2_records() * 2   # repeat: second should be full hit
        # Re-assign arrival indices and timestamps to avoid duplicates
        v2_records[0] = RequestRecord("r1", 0.0, 0, v2_records[0].block_ids)
        v2_records[1] = RequestRecord("r2", 1.0, 1, v2_records[1].block_ids)

        hand_records = self._hand_authored_records()
        from tests.v2_alignment.fixtures import FIXTURE_SINGLE_USER as FX
        hand_records_repeat = [
            RequestRecord("r1", 0.0, 0, list(FX.expected_block_ids)),
            RequestRecord("r2", 1.0, 1, list(FX.expected_block_ids)),
        ]

        v2_results = list(v1_replay(v2_records))
        hand_results = list(v1_replay(hand_records_repeat))

        assert v2_results[1].prefix_hit_blocks == hand_results[1].prefix_hit_blocks

    def test_v2_pipeline_block_size_stored(self):
        records = self._v2_records()
        assert records[0].block_size == 16  # FIXTURE_SINGLE_USER uses block_size=16


# ===========================================================================
# C6: session/category semantics (smoke tests; full coverage in test_v2_session.py)
# ===========================================================================

class TestC6_SessionCategorySemantics:
    """C6: Typed session/category helpers work on V2 pipeline output."""

    def _make_session_records(self):
        raw = [
            RawRequest("root", 0.0,
                       [Message(role="user", content="hi")],
                       session_id="s1", turn=1),
            RawRequest("followup", 1.0,
                       [Message(role="user", content="more")],
                       session_id="s1", parent_request_id="root", turn=2),
            RawRequest("other", 2.0,
                       [Message(role="user", content="other")],
                       session_id="s2", turn=1),
        ]
        return build_block_records_from_raw_requests(raw)

    def test_root_request_identified(self):
        records = self._make_session_records()
        roots = [r for r in records if is_root_request(r)]
        # "root" (turn=1, no parent) and "other" (turn=1, no parent) are roots
        assert len(roots) == 2

    def test_followup_request_identified(self):
        records = self._make_session_records()
        followups = [r for r in records if is_followup_request(r)]
        assert len(followups) == 1
        assert followups[0].request_id == "followup"

    def test_group_by_session_two_groups(self):
        records = self._make_session_records()
        groups = group_by_session(records)
        assert set(groups.keys()) == {"s1", "s2"}
        assert len(groups["s1"]) == 2
        assert len(groups["s2"]) == 1

    def test_get_category_from_pipeline_output(self):
        raw = [RawRequest("r1", 0.0,
                          [Message(role="user", content="x")],
                          category="text-1")]
        records = build_block_records_from_raw_requests(raw)
        assert get_category(records[0]) == "text-1"


# ===========================================================================
# Adapter interface contracts (not alignment, but pluggability)
# ===========================================================================

class TestAdapterContracts:
    """Verify that all adapters satisfy their Protocol contracts."""

    def test_qwen_template_name(self):
        assert QwenChatTemplate().name() == "qwen2_chat_template"

    def test_minimal_template_name(self):
        assert MinimalChatTemplate().name() == "minimal_chat_template"

    def test_qwen_template_empty_messages(self):
        rendered = QwenChatTemplate().render([])
        assert rendered == "<|im_start|>assistant\n"

    def test_qwen_template_no_generation_prompt(self):
        msgs = [Message(role="user", content="hi")]
        rendered = QwenChatTemplate(add_generation_prompt=False).render(msgs)
        assert rendered == "<|im_start|>user\nhi<|im_end|>\n"
        assert "<|im_start|>assistant" not in rendered

    def test_chained_block_builder_interface(self):
        """ChainedBlockBuilder is constructable and produces BlockBuildResult."""
        from block_prefix_analyzer.v2.adapters.siphash_builder import ChainedBlockBuilder
        from block_prefix_analyzer.v2.adapters.block_builder import BlockBuildResult

        def fake_hash(tokens: list[int], prev: int) -> int:
            return hash(tuple(tokens)) ^ prev

        builder = ChainedBlockBuilder(block_size=2, hash_fn=fake_hash)
        result = builder.build([1, 2, 3, 4, 5])
        assert isinstance(result, BlockBuildResult)
        assert len(result.block_ids) == 2    # 4 tokens / block_size=2 = 2 blocks
        assert result.leftover_token_count == 1

    def test_chained_builder_prefix_chaining_differs_from_simple(self):
        """Chained hashes differ from independent per-block hashes."""
        from block_prefix_analyzer.v2.adapters.siphash_builder import ChainedBlockBuilder

        calls = []
        def recording_hash(tokens: list[int], prev: int) -> int:
            calls.append((tuple(tokens), prev))
            return prev + sum(tokens)

        builder = ChainedBlockBuilder(block_size=2, hash_fn=recording_hash)
        builder.build([1, 2, 3, 4])
        assert len(calls) == 2
        # Second block's prev_hash = output of first block (not 0)
        assert calls[1][1] != 0
