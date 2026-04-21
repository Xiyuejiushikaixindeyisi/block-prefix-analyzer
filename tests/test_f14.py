"""Tests for F14 — multi-turn follow-up version of F13 (Trace A only).

Definitions under test:
  source_scope  : follow-up turns only (parent_chat_id >= 0)
  pool_scope    : ALL earlier requests (single-turn + roots + follow-ups)
  inset         : CUSTOM forward-looking — follow-up reusable by any future request

Test areas:
  1. Source scope: only follow-up turns (parent_chat_id >= 0) generate CDF events
  2. CDF computed over full events before axis crop (x_axis_max does NOT filter events)
  3. Inset is forward-looking (any future request), not backward any-hit
  4. ALL requests warm the pool (single-turn and roots included)
  5. Within-request dedup: repeated block_id counts once per request
  6. reuse_time uses last_seen semantics (current_ts - last_seen_ts)
  7. Forward inset denominator = ALL follow-up requests (not just reusable ones)
  8. CDF per-type is monotonically non-decreasing, range [0, 1]
  9. identify_multi_turn_request_ids returns follow-up turns (parent_chat_id >= 0)
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.analysis.f14 import (
    F14Output,
    compute_f14,
    identify_multi_turn_request_ids,
    save_f14_breakdown_csv,
    save_f14_cdf_csv,
    save_f14_metadata_json,
)
from block_prefix_analyzer.types import RequestRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(
    request_id: int | str,
    timestamp: float,
    block_ids: list[int],
    *,
    parent_chat_id: int = -1,
    req_type: str = "text",
    arrival_index: int | None = None,
) -> RequestRecord:
    ai = arrival_index if arrival_index is not None else (
        int(request_id) if str(request_id).isdigit() else 0
    )
    return RequestRecord(
        request_id=str(request_id),
        timestamp=float(timestamp),
        arrival_index=ai,
        block_ids=block_ids,
        metadata={"parent_chat_id": parent_chat_id, "type": req_type},
    )


# ---------------------------------------------------------------------------
# Test 1: source scope — only follow-up turns generate CDF events
# ---------------------------------------------------------------------------

class TestArea1_SourceScope:
    """Root turns (parent_chat_id == -1) and single-turn requests must NOT
    generate events; only follow-ups (parent_chat_id >= 0) do."""

    def test_root_turn_does_not_generate_events(self):
        # r1 is a multi-turn root (parent=-1); r2 is its follow-up.
        # r1 warms the pool but must not itself generate events.
        r1 = _rec(1, 0.0, [10, 20], parent_chat_id=-1)   # root — no events
        r2 = _rec(2, 1.0, [10, 30], parent_chat_id=1)    # follow-up — generates events
        out = compute_f14([r1, r2])
        # Only r2 is a follow-up → only r2 can generate events.
        assert out.multi_turn_request_count == 1           # r2 only
        # r2 hits blocks 10 (from r1) → 1 event.
        assert out.series.content_block_reuse_event_count_total == 1

    def test_single_turn_does_not_generate_events(self):
        # r1 is single-turn; r2/r3 are a multi-turn session.
        r1 = _rec(1, 0.0, [99])                           # single-turn
        r2 = _rec(2, 1.0, [99], parent_chat_id=-1)        # multi-turn root
        r3 = _rec(3, 2.0, [99], parent_chat_id=2)         # follow-up
        out = compute_f14([r1, r2, r3])
        # r3 is the only follow-up → multi_turn_request_count == 1.
        assert out.multi_turn_request_count == 1
        # r3 hits block 99; last_seen from r2 (t=1) since all-request pool.
        assert out.series.content_block_reuse_event_count_total == 1

    def test_multi_turn_ids_returns_follow_up_turns_only(self):
        r1 = _rec(1, 0.0, [1])                            # single-turn
        r2 = _rec(2, 1.0, [2], parent_chat_id=-1)         # multi-turn root
        r3 = _rec(3, 2.0, [3], parent_chat_id=2)          # follow-up
        mt_ids = identify_multi_turn_request_ids([r1, r2, r3])
        assert mt_ids == {"3"}                             # root r2 excluded
        assert "1" not in mt_ids
        assert "2" not in mt_ids


# ---------------------------------------------------------------------------
# Test 2: CDF full events before axis crop
# ---------------------------------------------------------------------------

class TestArea2_CdfBeforeAxisCrop:
    """x_axis_max_minutes must NOT filter events — CDF uses ALL events."""

    def test_events_beyond_axis_max_are_counted_in_cdf(self):
        # r1/r2 form one session; r3/r4 another with 90-min gap.
        r1 = _rec(1, 0.0,    [10], parent_chat_id=-1)
        r2 = _rec(2, 60.0,   [10], parent_chat_id=1)
        r3 = _rec(3, 0.0,    [20], parent_chat_id=-1)
        r4 = _rec(4, 5400.0, [20], parent_chat_id=3)  # 90 min gap
        out = compute_f14([r1, r2, r3, r4], x_axis_max_minutes=24.0)
        assert out.series.content_block_reuse_event_count_total == 2
        assert out.series.content_block_reuse_event_count_over_56min == 1

    def test_cdf_rows_count_equals_total_events(self):
        r1 = _rec(1, 0.0,   [5, 6], parent_chat_id=-1)
        r2 = _rec(2, 60.0,  [5, 6], parent_chat_id=1)
        r3 = _rec(3, 120.0, [5],    parent_chat_id=2)
        out = compute_f14([r1, r2, r3])
        assert len(out.series.cdf_rows) == out.series.content_block_reuse_event_count_total

    def test_tight_axis_does_not_reduce_cdf_events(self):
        r1 = _rec(1, 0.0,    [7], parent_chat_id=-1)
        r2 = _rec(2, 3600.0, [7], parent_chat_id=1)  # 60 min gap
        out_wide = compute_f14([r1, r2], x_axis_max_minutes=120.0)
        out_tight = compute_f14([r1, r2], x_axis_max_minutes=1.0)
        assert (out_wide.series.content_block_reuse_event_count_total ==
                out_tight.series.content_block_reuse_event_count_total)


# ---------------------------------------------------------------------------
# Test 3: inset is forward-looking (any future request), not backward any-hit
# ---------------------------------------------------------------------------

class TestArea3_InsetForwardLooking:
    """Inset = custom forward-looking: follow-up reusable by any future request."""

    def test_backward_hit_does_not_make_request_reusable(self):
        # r2 hits block backward (from r1 in pool) but has no future reuser.
        r1 = _rec(1, 0.0, [10], parent_chat_id=-1)
        r2 = _rec(2, 1.0, [10], parent_chat_id=1)
        out = compute_f14([r1, r2])
        assert out.backward_reusable_count == 1   # backward hit exists
        assert out.forward_reusable_count == 0    # no future request → not reusable

    def test_future_request_of_any_type_makes_follow_up_reusable(self):
        # r2 (follow-up) starts with block 10; r3 also starts with block 10 → prefix sharing.
        r1 = _rec(1, 0.0, [10, 20], parent_chat_id=-1)
        r2 = _rec(2, 1.0, [10, 30], parent_chat_id=1)
        r3 = _rec(3, 2.0, [10, 99]) # starts with block 10 == r2.block_ids[0]
        out = compute_f14([r1, r2, r3])
        assert out.forward_reusable_count >= 1

    def test_non_prefix_future_request_does_not_make_follow_up_reusable(self):
        # r3 has block 30 (same as r2's block at pos 1) but starts with 99 — no prefix sharing.
        r1 = _rec(1, 0.0, [10, 20], parent_chat_id=-1)
        r2 = _rec(2, 1.0, [10, 30], parent_chat_id=1)  # follow-up, starts with 10
        r3 = _rec(3, 2.0, [99, 30])  # starts with 99 ≠ r2.block_ids[0]=10
        out = compute_f14([r1, r2, r3])
        assert out.forward_reusable_count == 0

    def test_only_future_requests_count_not_past(self):
        # r_past is before the follow-up; sharing a block with it does not count.
        r_past = _rec(1, 0.0, [99])                    # past single-turn
        r_root = _rec(2, 1.0, [99], parent_chat_id=-1)
        r_fu   = _rec(3, 2.0, [99], parent_chat_id=2)
        out = compute_f14([r_past, r_root, r_fu])
        # No future requests after r_fu → 0 reusable.
        assert out.forward_reusable_count == 0

    def test_inset_denominator_is_all_follow_up_requests(self):
        # r2 is the only follow-up (r1 is root); nothing reusable.
        r1 = _rec(1, 0.0, [1], parent_chat_id=-1)
        r2 = _rec(2, 1.0, [2], parent_chat_id=1)
        out = compute_f14([r1, r2])
        assert out.multi_turn_request_count == 1   # r2 only
        assert out.forward_reusable_count == 0
        total_frac = sum(r.fraction for r in out.series.breakdown_rows)
        assert total_frac == 0.0


# ---------------------------------------------------------------------------
# Test 4: ALL requests warm the pool (pool_scope = all_earlier_requests)
# ---------------------------------------------------------------------------

class TestArea4_AllRequestsWarmPool:
    """Single-turn requests and root turns MUST update the pool timestamp."""

    def test_single_turn_updates_pool_timestamp(self):
        # r_root at t=0 puts block 5 in pool (ts=0).
        # r_st (single-turn) at t=1 updates block 5 in pool (ts=1).
        # r_fu (follow-up) at t=2 hits block 5; last_seen = 1 (from r_st).
        r_root = _rec(2, 0.0, [5], parent_chat_id=-1)
        r_st   = _rec(1, 1.0, [5])                    # single-turn
        r_fu   = _rec(3, 2.0, [5], parent_chat_id=2)
        out = compute_f14([r_root, r_st, r_fu])
        events = out.series.events
        assert len(events) == 1
        # reuse_time = 2.0 - 1.0 = 1.0 (last_seen updated by r_st at t=1)
        assert abs(events[0].reuse_time_seconds - 1.0) < 1e-9

    def test_root_turn_updates_pool_for_subsequent_follow_up(self):
        # r1 (root) puts block 7 in pool; r2 (follow-up) hits it.
        r1 = _rec(1, 0.0,  [7], parent_chat_id=-1)
        r2 = _rec(2, 30.0, [7], parent_chat_id=1)
        out = compute_f14([r1, r2])
        events = out.series.events
        assert len(events) == 1
        assert abs(events[0].reuse_time_seconds - 30.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 5: within-request dedup
# ---------------------------------------------------------------------------

class TestArea5_WithinRequestDedup:
    """Repeated block_id within a single request counts as one event."""

    def test_duplicate_block_ids_count_once(self):
        r1 = _rec(1, 0.0, [10],         parent_chat_id=-1)
        r2 = _rec(2, 1.0, [10, 10, 10], parent_chat_id=1)
        out = compute_f14([r1, r2])
        assert out.series.content_block_reuse_event_count_total == 1


# ---------------------------------------------------------------------------
# Test 6: reuse_time uses last_seen semantics
# ---------------------------------------------------------------------------

class TestArea6_ReuseTimeLastSeen:
    def test_reuse_time_equals_current_minus_last_seen(self):
        r1 = _rec(1, 100.0, [42], parent_chat_id=-1)
        r2 = _rec(2, 200.0, [42], parent_chat_id=1)   # last_seen=100 → rt=100
        r3 = _rec(3, 350.0, [42], parent_chat_id=2)   # last_seen=200 → rt=150
        out = compute_f14([r1, r2, r3])
        times = sorted(e.reuse_time_seconds for e in out.series.events)
        assert abs(times[0] - 100.0) < 1e-9
        assert abs(times[1] - 150.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 7: inset denominator = ALL follow-up requests
# ---------------------------------------------------------------------------

class TestArea7_InsetDenominator:
    def test_breakdown_fractions_use_all_follow_up_as_denominator(self):
        # 2 follow-up turns; only one is reusable by a future request.
        r1  = _rec(1, 0.0, [1],  parent_chat_id=-1)   # root
        r2  = _rec(2, 1.0, [10], parent_chat_id=1)    # follow-up 1
        r3  = _rec(3, 2.0, [20], parent_chat_id=1)    # follow-up 2
        r4  = _rec(4, 3.0, [10])                       # future single-turn (any type)
        out = compute_f14([r1, r2, r3, r4])
        # r2 shares block 10 with r4 (future) → reusable.
        # r3 has block 20, no future reuser → not reusable.
        assert out.multi_turn_request_count == 2
        assert out.forward_reusable_count == 1
        total_frac = sum(row.fraction for row in out.series.breakdown_rows)
        assert abs(total_frac - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Test 8: CDF monotonicity
# ---------------------------------------------------------------------------

class TestArea8_CdfMonotonicity:
    def test_cdf_monotonically_non_decreasing(self):
        r1 = _rec(1, 0.0,   [1, 2, 3], parent_chat_id=-1)
        r2 = _rec(2, 60.0,  [1, 2],    parent_chat_id=1)
        r3 = _rec(3, 180.0, [1, 3],    parent_chat_id=2)
        r4 = _rec(4, 600.0, [2, 3],    parent_chat_id=3)
        out = compute_f14([r1, r2, r3, r4])
        by_type: dict[str, list[float]] = {}
        for row in out.series.cdf_rows:
            by_type.setdefault(row.request_type, []).append(row.cdf)
        for t, cdfs in by_type.items():
            for i in range(1, len(cdfs)):
                assert cdfs[i] >= cdfs[i - 1], f"CDF not monotone for type {t}"
            assert cdfs[-1] == pytest.approx(1.0)

    def test_cdf_values_in_unit_interval(self):
        r1 = _rec(1, 0.0,  [5], parent_chat_id=-1)
        r2 = _rec(2, 30.0, [5], parent_chat_id=1)
        out = compute_f14([r1, r2])
        for row in out.series.cdf_rows:
            assert 0.0 < row.cdf <= 1.0


# ---------------------------------------------------------------------------
# Test 9: identify_multi_turn_request_ids
# ---------------------------------------------------------------------------

class TestArea9_IdentifyMultiTurnIds:
    def test_returns_only_follow_up_turns(self):
        r1 = _rec(1, 0.0, [1])                          # single-turn
        r2 = _rec(2, 1.0, [2])                          # single-turn
        r3 = _rec(3, 2.0, [3], parent_chat_id=-1)       # multi-turn root → excluded
        r4 = _rec(4, 3.0, [4], parent_chat_id=3)        # follow-up → included
        mt = identify_multi_turn_request_ids([r1, r2, r3, r4])
        assert mt == {"4"}

    def test_all_single_turn_gives_empty_result(self):
        r1 = _rec(1, 0.0, [1])
        r2 = _rec(2, 1.0, [2])
        mt = identify_multi_turn_request_ids([r1, r2])
        assert len(mt) == 0

    def test_roots_are_excluded_follow_ups_included(self):
        r1 = _rec(1, 0.0, [1], parent_chat_id=-1)   # root → excluded
        r2 = _rec(2, 1.0, [2], parent_chat_id=1)    # follow-up → included
        r3 = _rec(3, 2.0, [3], parent_chat_id=1)    # follow-up → included
        mt = identify_multi_turn_request_ids([r1, r2, r3])
        assert mt == {"2", "3"}
        assert "1" not in mt
