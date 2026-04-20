"""V2 alignment consistency tests.

Validation target matrix
==========================

This test file verifies three layers of consistency for the V2-min default
configuration (MinimalChatTemplate + CharTokenizer + SimpleBlockBuilder).

Layer 1 — Chat Template Consistency
  Test that: ChatTemplateAdapter.render(messages) == expected_rendered_prompt
  Why:       A prompt change causes ALL downstream tokens and blocks to change.
  Locating:  Use diff_text() to see exactly which character diverges.

Layer 2 — Tokenizer Consistency
  Test that: TokenizerAdapter.encode(rendered_prompt) == expected_token_ids
  Why:       Token ID changes cause block hash changes even if the prompt is correct.
  Locating:  Use diff_token_ids() to see the first mismatched position and ID.

Layer 3a — Block Count Consistency
  Test that: len(build(token_ids).block_ids) == expected_n_blocks
  Why:       Wrong block count means the block_size splitting logic is broken.

Layer 3b — Block Hash Consistency
  Test that: build(token_ids).block_ids == expected_block_ids
  Why:       Hash mismatch means the token content or hash function changed.

Layer 3c — Leftover Token Consistency
  Test that: build(token_ids).leftover_token_count == expected_leftover
  Why:       Leftover counts affect the "incomplete last block" exclusion rule.

Layer 4 — End-to-End Pipeline Consistency
  Test that: build_block_records_from_raw_requests(request) matches all expected.

Layer 5 — V1 Replay Integration Consistency
  Test that: the V2 output can flow through V1 replay → compute_metrics
  and that two identical requests produce a non-zero prefix hit rate.

BOUNDARY: "verified configuration"
====================================
The golden values in tests/v2_alignment/fixtures.py are valid ONLY for:
  - MinimalChatTemplate   (NOT Qwen / LLaMA / any real model template)
  - CharTokenizer         (NOT HuggingFace / vLLM tokenizer)
  - SimpleBlockBuilder with _sha256_block_default (NOT SipHash used by vLLM)
  - Python 3.10+ (ord() and hashlib.sha256 are stable)

To add framework-aligned golden values:
  1. Add a real tokenizer adapter (e.g. HFTokenizerAdapter).
  2. Add a real template adapter (e.g. QwenChatTemplate).
  3. Compute golden outputs with real adapters.
  4. Add a new V2AlignmentFixture with config_tag="qwen_vllm_aligned".
  5. Change alignment_status to "framework_verified".
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.metrics import compute_metrics
from block_prefix_analyzer.replay import replay
from block_prefix_analyzer.v2.adapters.block_builder import SimpleBlockBuilder
from block_prefix_analyzer.v2.adapters.chat_template import MinimalChatTemplate
from block_prefix_analyzer.v2.adapters.tokenizer import CharTokenizer
from block_prefix_analyzer.v2.diagnostics import (
    alignment_report,
    diff_block_ids,
    diff_leftover,
    diff_text,
    diff_token_ids,
)
from block_prefix_analyzer.v2.pipeline import build_block_records_from_raw_requests
from block_prefix_analyzer.v2.schema import Message, RawRequest
from tests.v2_alignment.fixtures import VERIFIED_FIXTURES, V2AlignmentFixture

# Shared adapters for all tests in this module
_TEMPLATE = MinimalChatTemplate()
_TOKENIZER = CharTokenizer()


def _builder(block_size: int = 16) -> SimpleBlockBuilder:
    return SimpleBlockBuilder(block_size=block_size)


# ---------------------------------------------------------------------------
# Layer 1: Chat template consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", VERIFIED_FIXTURES, ids=lambda f: f.name)
def test_template_rendering_matches_golden(fx: V2AlignmentFixture) -> None:
    """Layer 1: render(messages) must match the stored golden prompt."""
    actual = _TEMPLATE.render(fx.messages)
    if actual != fx.expected_rendered_prompt:
        report = diff_text(fx.expected_rendered_prompt, actual, label="rendered_prompt")
        pytest.fail(
            f"[{fx.name}] Layer 1 (template) MISMATCH\n"
            f"Config: {fx.config_tag}\n"
            f"{report}"
        )


def test_template_layer1_golden_is_human_readable() -> None:
    """Each fixture's expected_rendered_prompt must be a non-empty string."""
    for fx in VERIFIED_FIXTURES:
        assert isinstance(fx.expected_rendered_prompt, str), fx.name
        assert len(fx.expected_rendered_prompt) > 0, fx.name


def test_template_single_user_contains_role_markers() -> None:
    from tests.v2_alignment.fixtures import FIXTURE_SINGLE_USER as fx
    rendered = _TEMPLATE.render(fx.messages)
    assert "<|user|>" in rendered
    assert "<|assistant|>" in rendered
    assert rendered.endswith("<|assistant|>")


def test_template_system_user_role_order() -> None:
    from tests.v2_alignment.fixtures import FIXTURE_SYSTEM_USER as fx
    rendered = _TEMPLATE.render(fx.messages)
    assert rendered.index("<|system|>") < rendered.index("<|user|>")
    assert rendered.index("<|user|>") < rendered.index("<|assistant|>")


def test_template_empty_system_content_omits_blank_line() -> None:
    from tests.v2_alignment.fixtures import FIXTURE_EMPTY_SYSTEM_CONTENT as fx
    rendered = _TEMPLATE.render(fx.messages)
    # The rendered prompt should NOT have a double newline after <|system|>
    assert "<|system|>\n\n" not in rendered
    assert rendered == fx.expected_rendered_prompt


# ---------------------------------------------------------------------------
# Layer 2: Tokenizer consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", VERIFIED_FIXTURES, ids=lambda f: f.name)
def test_tokenizer_matches_golden(fx: V2AlignmentFixture) -> None:
    """Layer 2: encode(rendered_prompt) must match the stored golden token IDs."""
    if fx.expected_token_ids is None:
        pytest.skip(f"[{fx.name}] expected_token_ids is pending")
    rendered = _TEMPLATE.render(fx.messages)
    actual_ids = _TOKENIZER.encode(rendered)
    if actual_ids != fx.expected_token_ids:
        report = diff_token_ids(fx.expected_token_ids, actual_ids)
        pytest.fail(
            f"[{fx.name}] Layer 2 (tokenizer) MISMATCH\n"
            f"Config: {fx.config_tag}\n"
            f"{report}\n"
            f"Note: render output was: {rendered!r}"
        )


def test_tokenizer_token_count_matches_rendered_length() -> None:
    """CharTokenizer produces exactly one token per character."""
    from tests.v2_alignment.fixtures import FIXTURE_SINGLE_USER as fx
    rendered = _TEMPLATE.render(fx.messages)
    tokens = _TOKENIZER.encode(rendered)
    assert len(tokens) == len(rendered), (
        f"CharTokenizer must produce one token per character: "
        f"len(tokens)={len(tokens)}, len(rendered)={len(rendered)}"
    )


def test_tokenizer_unicode_code_points() -> None:
    """CJK characters produce their Unicode code points."""
    from tests.v2_alignment.fixtures import FIXTURE_UNICODE_CONTENT as fx
    if fx.expected_token_ids is None:
        pytest.skip("pending")
    tokens = _TOKENIZER.encode(fx.expected_rendered_prompt)
    assert 20320 in tokens, "ord('你') == 20320 must appear in token IDs"
    assert 22909 in tokens, "ord('好') == 22909 must appear in token IDs"


def test_tokenizer_layer2_golden_length_matches_prompt_length() -> None:
    """Verify the golden token_ids list has the right length for each fixture."""
    for fx in VERIFIED_FIXTURES:
        if fx.expected_token_ids is None:
            continue
        expected_len = len(fx.expected_rendered_prompt)
        actual_len = len(fx.expected_token_ids)
        assert actual_len == expected_len, (
            f"[{fx.name}] golden token_ids length {actual_len} != "
            f"rendered_prompt length {expected_len}"
        )


# ---------------------------------------------------------------------------
# Layer 3a: Block count consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", VERIFIED_FIXTURES, ids=lambda f: f.name)
def test_block_count_matches_golden(fx: V2AlignmentFixture) -> None:
    """Layer 3a: number of complete blocks must match golden."""
    if fx.expected_block_ids is None:
        pytest.skip(f"[{fx.name}] expected_block_ids is pending")
    if fx.expected_token_ids is None:
        pytest.skip(f"[{fx.name}] expected_token_ids is pending")
    result = _builder(fx.block_size).build(fx.expected_token_ids)
    expected_count = len(fx.expected_block_ids)
    actual_count = len(result.block_ids)
    if actual_count != expected_count:
        pytest.fail(
            f"[{fx.name}] Layer 3a (block count) MISMATCH\n"
            f"  block_size={fx.block_size}, n_tokens={len(fx.expected_token_ids)}\n"
            f"  expected {expected_count} blocks, got {actual_count}"
        )


def test_block_count_formula_n_tokens_div_block_size() -> None:
    """Block count = floor(n_tokens / block_size)."""
    for fx in VERIFIED_FIXTURES:
        if fx.expected_token_ids is None or fx.expected_block_ids is None:
            continue
        n = len(fx.expected_token_ids)
        expected_n_blocks = n // fx.block_size
        actual_n_blocks = len(fx.expected_block_ids)
        assert actual_n_blocks == expected_n_blocks, (
            f"[{fx.name}] expected floor({n}/{fx.block_size})={expected_n_blocks} "
            f"blocks, golden has {actual_n_blocks}"
        )


# ---------------------------------------------------------------------------
# Layer 3b: Block hash consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", VERIFIED_FIXTURES, ids=lambda f: f.name)
def test_block_hashes_match_golden(fx: V2AlignmentFixture) -> None:
    """Layer 3b: each block's hash value must match the golden."""
    if fx.expected_block_ids is None or fx.expected_token_ids is None:
        pytest.skip(f"[{fx.name}] pending")
    result = _builder(fx.block_size).build(fx.expected_token_ids)
    if result.block_ids != fx.expected_block_ids:
        report = diff_block_ids(fx.expected_block_ids, result.block_ids)
        pytest.fail(
            f"[{fx.name}] Layer 3b (block hashes) MISMATCH\n"
            f"Config: {fx.config_tag}, block_size={fx.block_size}\n"
            f"{report}"
        )


# ---------------------------------------------------------------------------
# Layer 3c: Leftover token count consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", VERIFIED_FIXTURES, ids=lambda f: f.name)
def test_leftover_token_count_matches_golden(fx: V2AlignmentFixture) -> None:
    """Layer 3c: leftover token count must match golden."""
    if fx.expected_leftover_token_count is None or fx.expected_token_ids is None:
        pytest.skip(f"[{fx.name}] pending")
    result = _builder(fx.block_size).build(fx.expected_token_ids)
    if result.leftover_token_count != fx.expected_leftover_token_count:
        report = diff_leftover(fx.expected_leftover_token_count, result.leftover_token_count)
        pytest.fail(f"[{fx.name}] Layer 3c (leftover) MISMATCH\n{report}")


def test_leftover_formula_n_tokens_mod_block_size() -> None:
    """Leftover = n_tokens % block_size."""
    for fx in VERIFIED_FIXTURES:
        if fx.expected_token_ids is None or fx.expected_leftover_token_count is None:
            continue
        expected = len(fx.expected_token_ids) % fx.block_size
        assert fx.expected_leftover_token_count == expected, (
            f"[{fx.name}] golden leftover {fx.expected_leftover_token_count} "
            f"!= {len(fx.expected_token_ids)} % {fx.block_size} = {expected}"
        )


# ---------------------------------------------------------------------------
# Layer 4: End-to-end pipeline consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fx", VERIFIED_FIXTURES, ids=lambda f: f.name)
def test_pipeline_end_to_end_matches_all_golden_layers(fx: V2AlignmentFixture) -> None:
    """Layer 4: full pipeline from RawRequest to RequestRecord.

    Checks all three layers in one shot and emits a combined report on failure.
    """
    raw = RawRequest(
        request_id=fx.name,
        timestamp=0.0,
        messages=fx.messages,
    )
    records = build_block_records_from_raw_requests(
        [raw],
        chat_template=_TEMPLATE,
        tokenizer=_TOKENIZER,
        block_builder=_builder(fx.block_size),
    )
    assert len(records) == 1
    rec = records[0]
    meta = rec.metadata

    # Layer 1 check
    actual_prompt = meta.get("v2_rendered_prompt", "")
    l1 = diff_text(fx.expected_rendered_prompt, actual_prompt)

    # Layer 2 check
    actual_tokens = _TOKENIZER.encode(actual_prompt)
    l2 = diff_token_ids(fx.expected_token_ids or [], actual_tokens)

    # Layer 3 checks
    l3a = diff_block_ids(fx.expected_block_ids or [], list(rec.block_ids))
    l3b = diff_leftover(
        fx.expected_leftover_token_count or 0,
        meta.get("v2_leftover_tokens", -1),
    )

    failed_layers = []
    if actual_prompt != fx.expected_rendered_prompt:
        failed_layers.append("Layer 1 (template)")
    if fx.expected_token_ids and actual_tokens != fx.expected_token_ids:
        failed_layers.append("Layer 2 (tokenizer)")
    if fx.expected_block_ids and list(rec.block_ids) != fx.expected_block_ids:
        failed_layers.append("Layer 3b (block hashes)")
    if fx.expected_leftover_token_count is not None:
        if meta.get("v2_leftover_tokens") != fx.expected_leftover_token_count:
            failed_layers.append("Layer 3c (leftover)")

    if failed_layers:
        report = alignment_report(
            fx.name,
            rendered_prompt_result=l1,
            token_ids_result=l2,
            block_ids_result=l3a,
            leftover_result=l3b,
        )
        pytest.fail(
            f"[{fx.name}] End-to-end mismatch in: {', '.join(failed_layers)}\n{report}"
        )


# ---------------------------------------------------------------------------
# Layer 5: V1 replay integration consistency
# ---------------------------------------------------------------------------

def test_v2_v1_replay_integration_same_content_yields_prefix_hit() -> None:
    """Identical V2 requests produce full prefix hit on the second request."""
    msgs = [
        Message(role="system", content="Be helpful."),
        Message(role="user", content="What is the capital of France?"),
    ]
    raw = [
        RawRequest(request_id="r1", timestamp=0.0, messages=msgs),
        RawRequest(request_id="r2", timestamp=60.0, messages=msgs),
    ]
    records = build_block_records_from_raw_requests(
        raw,
        chat_template=_TEMPLATE,
        tokenizer=_TOKENIZER,
        block_builder=_builder(16),
    )
    results = list(replay(records))
    # r1: cold start
    assert results[0].prefix_hit_blocks == 0, "First request must be cold start"
    # r2: full prefix hit (identical content → identical blocks)
    assert results[1].prefix_hit_blocks == results[1].total_blocks, (
        f"Identical content should produce full prefix hit: "
        f"prefix_hit={results[1].prefix_hit_blocks}, total={results[1].total_blocks}"
    )


def test_v2_v1_replay_integration_different_content_partial_hit() -> None:
    """Requests sharing a common prefix produce a partial hit."""
    system = Message(role="system", content="You are helpful.")
    raw = [
        RawRequest(request_id="r1", timestamp=0.0, messages=[
            system, Message(role="user", content="Hello A")
        ]),
        RawRequest(request_id="r2", timestamp=60.0, messages=[
            system, Message(role="user", content="Hello B")
        ]),
    ]
    records = build_block_records_from_raw_requests(
        raw, chat_template=_TEMPLATE, tokenizer=_TOKENIZER, block_builder=_builder(4)
    )
    results = list(replay(records))
    r2 = results[1]
    # The system prompt blocks are shared → at least some prefix hits
    assert r2.prefix_hit_blocks > 0, "Shared prefix should produce at least one hit"
    # But the user content differs → not a full hit
    assert r2.prefix_hit_blocks < r2.total_blocks, "Different user content should not give full hit"


def test_v2_v1_compute_metrics_hit_rate_bounded() -> None:
    """MetricsSummary hit rates are always in [0, 1]."""
    from tests.v2_alignment.fixtures import FIXTURE_MULTI_TURN as fx
    raw = [
        RawRequest(request_id="r1", timestamp=0.0, messages=fx.messages),
        RawRequest(request_id="r2", timestamp=60.0, messages=fx.messages),
    ]
    records = build_block_records_from_raw_requests(
        raw, block_builder=_builder(fx.block_size)
    )
    summary = compute_metrics(list(replay(records)))
    assert 0.0 <= summary.overall_prefix_hit_rate <= 1.0
    assert 0.0 <= summary.overall_block_level_reusable_ratio <= 1.0


def test_v2_v1_metadata_preserved_in_record() -> None:
    """Debug metadata from V2 pipeline is accessible in RequestRecord.metadata."""
    from tests.v2_alignment.fixtures import FIXTURE_SINGLE_USER as fx
    raw = [RawRequest(request_id="r1", timestamp=0.0, messages=fx.messages)]
    records = build_block_records_from_raw_requests(raw, block_builder=_builder(fx.block_size))
    meta = records[0].metadata
    assert meta["v2_rendered_prompt"] == fx.expected_rendered_prompt
    assert meta["v2_token_count"] == len(fx.expected_token_ids or [])
    assert meta["v2_leftover_tokens"] == fx.expected_leftover_token_count


# ---------------------------------------------------------------------------
# Alignment boundary documentation test
# ---------------------------------------------------------------------------

def test_pending_fixture_is_marked_correctly() -> None:
    """Pending fixtures must have alignment_status='pending_framework' and None golden fields."""
    from tests.v2_alignment.fixtures import FIXTURE_PENDING_FRAMEWORK as fx
    assert fx.alignment_status == "pending_framework"
    assert fx.expected_token_ids is None
    assert fx.expected_block_ids is None
    assert fx.expected_leftover_token_count is None


def test_verified_fixtures_have_complete_golden_values() -> None:
    """All VERIFIED_FIXTURES must have no None golden fields."""
    for fx in VERIFIED_FIXTURES:
        assert fx.expected_rendered_prompt not in (None, "", "PENDING"), fx.name
        assert fx.expected_token_ids is not None, fx.name
        assert fx.expected_block_ids is not None, fx.name
        assert fx.expected_leftover_token_count is not None, fx.name


def test_fixture_config_tags_are_explicit() -> None:
    """Each fixture declares which configuration produced its golden values."""
    for fx in VERIFIED_FIXTURES:
        assert fx.config_tag != "", fx.name
        assert fx.alignment_status in ("internal_only", "pending_framework", "framework_verified"), fx.name
