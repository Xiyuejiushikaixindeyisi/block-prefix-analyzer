"""Tests for ``find_common_prefix_chain`` (Spec §9, 11 fixtures + Spec §8 invariants).

Each test maps directly to a row in
``docs/common_prefix_chain_spec.md`` §9. The Spec §8 invariants are
checked both implicitly (per-fixture assertions) and via dedicated
invariant-only tests at the bottom.

Block IDs are small integers (1, 2, 3, ...) for readability; they map
to recognizable strings via REGISTRY so decoded_text assertions can
spell out the expected output.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.analysis.common_prefix import (
    BranchAlternative,
    ChainBlock,
    CommonPrefixChainResult,
    find_common_prefix_chain,
)
from block_prefix_analyzer.types import RequestRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(request_id: str, block_ids: list, *, idx: int = 0,
         ts: float = 0.0) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        timestamp=ts,
        block_ids=list(block_ids),
        arrival_index=idx,
    )


# Compact aliases for fixture readability.
A, B, C, D, E = 1, 2, 3, 4, 5
P, Q, R, S, T = 6, 7, 8, 9, 10
X, Y, Z = 11, 12, 13

REGISTRY = {
    A: "alpha", B: "beta", C: "gamma", D: "delta", E: "epsilon",
    P: "pp", Q: "qq", R: "rr", S: "ss", T: "tt",
    X: "xray", Y: "yankee", Z: "zulu",
}


# ===========================================================================
# Spec §9 — 11 named fixtures
# ===========================================================================

# Fixture 1 ------------------------------------------------------------------

def test_strict_lcp_full_chain():
    """5 identical 4-block requests → walk full chain, stop_reason=no_children."""
    records = [_rec(f"r{i}", [A, B, C, D], idx=i) for i in range(5)]
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=2,
    )
    assert result.stop_reason == "no_children"
    assert [cb.block_id for cb in result.consensus_blocks] == [A, B, C, D]
    for cb in result.consensus_blocks:
        assert cb.freq == 5
        assert cb.parent_freq == 5
        assert cb.global_coverage_pct == 100.0
        assert cb.branch_ratio_pct == 100.0
    assert result.decoded_text == "alphabetagammadelta"
    assert result.prefix_length_blocks == 4
    assert result.prefix_length_chars == len(result.decoded_text)
    assert result.total_records == 5
    assert result.stop_position == 4
    # No children at stop node → no alternatives.
    assert result.branch_alternatives == []


# Fixture 2 ------------------------------------------------------------------

def test_min_count_blocks_at_branch():
    """5 [A,B] + 5 [A,C] with min_count=6 → A passes (10 ≥ 6), then both
    B (5) and C (5) fail min_count → stop at chain=[A]."""
    records = (
        [_rec(f"b{i}", [A, B], idx=i) for i in range(5)] +
        [_rec(f"c{i}", [A, C], idx=i + 5) for i in range(5)]
    )
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=6,
    )
    assert result.stop_reason == "min_count"
    assert [cb.block_id for cb in result.consensus_blocks] == [A]
    assert result.consensus_blocks[0].freq == 10
    assert result.stop_position == 1
    # B and C are both alternatives at stop node A.
    alt_ids = sorted(alt.block_id for alt in result.branch_alternatives)
    assert alt_ids == [B, C]


# Fixture 3 ------------------------------------------------------------------

def test_branch_winner_below_threshold():
    """6 [A,B] + 4 [A,C], branch_threshold=0.7 → B occupies 6/10 = 60% of A's
    children, below 70% → stop at chain=[A]."""
    records = (
        [_rec(f"b{i}", [A, B], idx=i) for i in range(6)] +
        [_rec(f"c{i}", [A, C], idx=i + 6) for i in range(4)]
    )
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128,
        min_count=2, branch_threshold=0.7,
    )
    assert result.stop_reason == "branch_threshold"
    assert [cb.block_id for cb in result.consensus_blocks] == [A]
    alt_ids = sorted(alt.block_id for alt in result.branch_alternatives)
    assert alt_ids == [B, C]
    # Heaviest alt (B) reports 60% of parent.
    b_alt = next(a for a in result.branch_alternatives if a.block_id == B)
    assert b_alt.fraction_of_parent == pytest.approx(0.6)
    assert b_alt.decoded_text_preview == "beta"


# Fixture 4 ------------------------------------------------------------------

def test_branch_winner_above_threshold():
    """7 [A,B] + 3 [A,C], branch_threshold=0.7 → B is exactly 70% (≥ 0.7),
    advances. After [A,B] no children remain → stop_reason=no_children."""
    records = (
        [_rec(f"b{i}", [A, B], idx=i) for i in range(7)] +
        [_rec(f"c{i}", [A, C], idx=i + 7) for i in range(3)]
    )
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128,
        min_count=2, branch_threshold=0.7,
    )
    assert result.stop_reason == "no_children"
    assert [cb.block_id for cb in result.consensus_blocks] == [A, B]
    assert result.consensus_blocks[1].branch_ratio_pct == pytest.approx(70.0)
    assert result.consensus_blocks[1].global_coverage_pct == pytest.approx(70.0)


# Fixture 5 ------------------------------------------------------------------

def test_coverage_threshold_kicks_in():
    """100 share [A,B], then 60 walk C / 40 walk D. With branch=0.5 (C
    qualifies at 60%) and coverage=0.7, C's global coverage 60/100 = 60%
    falls below 70% → stop at [A,B]."""
    records = (
        [_rec(f"abc{i}", [A, B, C], idx=i) for i in range(60)] +
        [_rec(f"abd{i}", [A, B, D], idx=i + 60) for i in range(40)]
    )
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128,
        min_count=2, branch_threshold=0.5, coverage_threshold=0.7,
    )
    assert result.stop_reason == "coverage_threshold"
    assert [cb.block_id for cb in result.consensus_blocks] == [A, B]
    assert result.consensus_blocks[0].global_coverage_pct == 100.0
    assert result.consensus_blocks[1].global_coverage_pct == 100.0


# Fixture 6 ------------------------------------------------------------------

def test_periodic_content_no_aliasing():
    """5 identical requests with periodic block_ids [P, P, P]. The trie has
    distinct nodes per depth even when block_id repeats — so the chain
    output is the genuine 3-step path, freq=5 at every depth."""
    records = [_rec(f"r{i}", [P, P, P], idx=i) for i in range(5)]
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=16, min_count=2,
    )
    assert result.stop_reason == "no_children"
    assert len(result.consensus_blocks) == 3
    for cb in result.consensus_blocks:
        assert cb.block_id == P
        assert cb.freq == 5
    assert result.decoded_text == "pppppp"


# Fixture 7 ------------------------------------------------------------------

def test_single_record_below_min_count():
    """One record can't satisfy min_count=2 → empty chain at root."""
    records = [_rec("solo", [A, B, C], idx=0)]
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=2,
    )
    assert result.stop_reason == "min_count"
    assert result.consensus_blocks == []
    assert result.stop_position == 0
    assert result.total_records == 1
    # Root has one child (A) which becomes the (failed) alternative.
    assert len(result.branch_alternatives) == 1
    assert result.branch_alternatives[0].block_id == A


# Fixture 8 ------------------------------------------------------------------

def test_empty_input():
    result = find_common_prefix_chain(
        [], REGISTRY, block_size=128, min_count=2,
    )
    assert result.stop_reason == "no_records"
    assert result.consensus_blocks == []
    assert result.prefix_length_chars == 0
    assert result.total_records == 0
    assert result.branch_alternatives == []


def test_all_empty_block_ids_yields_no_records():
    """Records exist but every block_ids is empty → same as empty input."""
    records = [_rec("r0", [], idx=0), _rec("r1", [], idx=1)]
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=1,
    )
    assert result.stop_reason == "no_records"
    assert result.total_records == 0
    assert result.consensus_blocks == []


# Fixture 9 ------------------------------------------------------------------

def test_legacy_position_wise_diverges():
    """Spec §1 counter-example. Position-wise majority would emit a chain
    [A, X] that no real request has. Trie-greedy must NOT do that.

    Records:
        r1: [A, B, P]
        r2: [A, Y, Q]
        r3: [A, B, R]
        r4: [A, Y, X]
        r5: [B, X, P]
    """
    records = [
        _rec("r1", [A, B, P], idx=0),
        _rec("r2", [A, Y, Q], idx=1),
        _rec("r3", [A, B, R], idx=2),
        _rec("r4", [A, Y, X], idx=3),
        _rec("r5", [B, X, P], idx=4),
    ]
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=2,
    )
    chain_ids = [cb.block_id for cb in result.consensus_blocks]

    # Critical: the chain MUST NOT contain the [A, X] adjacency that
    # position-wise majority emits. Verify by checking each (i, i+1)
    # pair exists consecutively in some real record.
    for i in range(len(chain_ids) - 1):
        a, b = chain_ids[i], chain_ids[i + 1]
        appears_consecutively = any(
            r.block_ids[j] == a and r.block_ids[j + 1] == b
            for r in records
            for j in range(len(r.block_ids) - 1)
        )
        assert appears_consecutively, (
            f"Ghost edge ({a}, {b}) at position {i} of chain {chain_ids} — "
            f"this pair never occurs consecutively in any record."
        )

    # Concrete expectation for this fixture: A→B is the heaviest at root,
    # then under A's node B (2) and Y (2) tie; max picks one. Either way
    # the picked one must be a real continuation (B or Y), not X.
    assert chain_ids[0] == A
    assert chain_ids[1] in (B, Y)


# Fixture 10 -----------------------------------------------------------------

def test_varying_lengths_freq_monotone_non_increasing():
    """Records of different lengths. Trie freq must monotonically decrease
    along the chain (Invariant #3). With 4 requests of lengths 3/2/1/4
    starting [A, B, ...], freqs are A=4, B=3, C=2, D=1 (1 < min_count=2)."""
    records = [
        _rec("r1", [A, B, C], idx=0),
        _rec("r2", [A, B], idx=1),
        _rec("r3", [A], idx=2),
        _rec("r4", [A, B, C, D], idx=3),
    ]
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=2,
    )
    freqs = [cb.freq for cb in result.consensus_blocks]
    assert freqs == [4, 3, 2]
    for i in range(len(freqs) - 1):
        assert freqs[i] >= freqs[i + 1], f"freq not monotone: {freqs}"
    assert result.stop_reason == "min_count"


# Fixture 11 -----------------------------------------------------------------

def test_block_registry_decode_uses_chain_only():
    records = [_rec(f"r{i}", [A, B], idx=i) for i in range(2)]
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=2,
    )
    assert result.decoded_text == "alphabeta"
    assert result.prefix_length_chars == len("alphabeta")


def test_block_registry_decode_handles_missing_ids():
    records = [_rec(f"r{i}", [A, B], idx=i) for i in range(2)]
    partial = {A: "alpha"}                 # B missing on purpose
    result = find_common_prefix_chain(
        records, partial, block_size=128, min_count=2,
    )
    assert "alpha" in result.decoded_text
    assert f"<MISSING:{B}>" in result.decoded_text
    # Length still matches the actual (placeholder-substituted) text.
    assert result.prefix_length_chars == len(result.decoded_text)


# ===========================================================================
# Spec §8 — invariant-only tests
# ===========================================================================

def test_invariant_path_closed():
    """For every chain block, the FULL prefix [chain[0]..chain[i]] must
    appear in at least min_count records (Invariant #1)."""
    records = (
        [_rec(f"abc{i}", [A, B, C], idx=i) for i in range(10)] +
        [_rec(f"abd{i}", [A, B, D], idx=i + 10) for i in range(5)]
    )
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=5,
    )
    for cb in result.consensus_blocks:
        full_prefix = [c.block_id for c in result.consensus_blocks[:cb.position + 1]]
        matching = sum(
            1 for r in records
            if r.block_ids[:len(full_prefix)] == full_prefix
        )
        assert matching >= 5, (
            f"Path-closed violated at position {cb.position}: prefix "
            f"{full_prefix} appears in only {matching} records (< 5)."
        )


def test_invariant_threshold_provenance():
    """All three thresholds are echoed back in the result (Invariant #8)."""
    records = [_rec(f"r{i}", [A, B], idx=i) for i in range(3)]
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128,
        min_count=3, branch_threshold=0.5, coverage_threshold=0.1,
    )
    assert result.min_count_threshold == 3
    assert result.branch_threshold == 0.5
    assert result.coverage_threshold == 0.1


def test_invariant_branch_alternatives_top_n_capped():
    """When the stop node has more than 5 children, branch_alternatives is
    limited to the top 5 by freq (Spec §7 N=5 fixed)."""
    # 7 distinct first blocks, each with 1 record → all freq 1 < min_count=2.
    records = [_rec(f"r{i}", [i + 1], idx=i) for i in range(7)]
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=2,
    )
    assert result.stop_reason == "min_count"
    assert len(result.branch_alternatives) == 5         # capped, not 7


def test_invariant_branch_ratio_pct_in_range():
    """branch_ratio_pct ∈ (0, 100] for every chain step (Invariant #4)."""
    records = (
        [_rec(f"abc{i}", [A, B, C], idx=i) for i in range(7)] +
        [_rec(f"abd{i}", [A, B, D], idx=i + 7) for i in range(3)]
    )
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=2,
    )
    for cb in result.consensus_blocks:
        assert 0.0 < cb.branch_ratio_pct <= 100.0, (
            f"branch_ratio_pct out of range at pos {cb.position}: "
            f"{cb.branch_ratio_pct}"
        )


def test_invariant_decoded_text_uses_chain_only():
    """decoded_text doesn't accidentally include text from branch
    alternatives (Invariant #5)."""
    records = (
        [_rec(f"abc{i}", [A, B], idx=i) for i in range(6)] +
        [_rec(f"ad{i}", [A, D], idx=i + 6) for i in range(4)]
    )
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=2,
    )
    # Chain advances [A, B] (B wins 6 vs D's 4).
    assert [cb.block_id for cb in result.consensus_blocks] == [A, B]
    # Decoded text contains alpha + beta but NOT delta (D's text).
    assert result.decoded_text == "alphabeta"
    assert "delta" not in result.decoded_text


def test_invariant_first_position_branch_ratio_equals_global_coverage():
    """At position 0, parent_freq = total_records, so branch_ratio_pct ==
    global_coverage_pct (Invariant #4 corollary)."""
    records = (
        [_rec(f"a{i}", [A], idx=i) for i in range(8)] +
        [_rec(f"b{i}", [B], idx=i + 8) for i in range(2)]
    )
    result = find_common_prefix_chain(
        records, REGISTRY, block_size=128, min_count=2,
    )
    assert result.consensus_blocks[0].branch_ratio_pct == \
           result.consensus_blocks[0].global_coverage_pct


def test_invariant_stop_reason_always_set():
    """stop_reason is always one of the 6 enum values (Invariant #6)."""
    cases = [
        ([], "no_records"),
        ([_rec("solo", [A], idx=0)], "min_count"),
        ([_rec(f"r{i}", [A], idx=i) for i in range(5)], "no_children"),
    ]
    for records, expected in cases:
        result = find_common_prefix_chain(
            records, REGISTRY, block_size=128, min_count=2,
        )
        assert result.stop_reason == expected
