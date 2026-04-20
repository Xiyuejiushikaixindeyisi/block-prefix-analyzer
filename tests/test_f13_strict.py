"""Tests for F13 strict paper-definition analysis.

Test areas (10 required):
  1. single-turn identification helpers (root predicate + identify_root_requests)
  2. Multi-turn roots ARE excluded; only sessions with length==1 are single-turn
  3. Main CDF uses block-level reusable events, not prefix events
  4. reuse_time uses last_seen semantics
  5. Within-request dedup: repeated block_id counts once
  6. Inset uses single-turn-only pool (backward-looking)
  7. Multi-turn follow-ups do NOT update the pool
  8. CDF is monotonically non-decreasing per type, range [0, 1]
  9. Output schema: F13Series fields and CSV column names are stable
 10. image -> Multimedia label mapping is correct
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from block_prefix_analyzer.analysis.f13_strict import (
    BREAKDOWN_DEFINITION,
    DEDUPE_RULE,
    EVENT_DEFINITION,
    POOL_DEFINITION_BREAKDOWN,
    POOL_DEFINITION_CDF,
    REUSE_TIME_DEFINITION,
    SINGLE_TURN_DEFINITION,
    _is_root_request,
    compute_f13_strict_series,
    identify_root_requests,
    save_strict_breakdown_csv,
    save_strict_cdf_csv,
    save_strict_metadata_json,
)
from block_prefix_analyzer.types import RequestRecord


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _rec(
    request_id: int | str,
    timestamp: float,
    block_ids: list[int],
    *,
    parent_chat_id: int = -1,
    req_type: str = "text",
) -> RequestRecord:
    """Create a RequestRecord with TraceA-style metadata."""
    return RequestRecord(
        request_id=str(request_id),
        timestamp=float(timestamp),
        arrival_index=int(request_id) if str(request_id).isdigit() else 0,
        block_ids=block_ids,
        metadata={
            "parent_chat_id": parent_chat_id,
            "type": req_type,
        },
    )


def _rec_no_parent_field(request_id, timestamp, block_ids, req_type="text") -> RequestRecord:
    """Create a RequestRecord with parent_chat_id field entirely absent."""
    return RequestRecord(
        request_id=str(request_id),
        timestamp=float(timestamp),
        arrival_index=int(request_id),
        block_ids=block_ids,
        metadata={"type": req_type},
    )


# ---------------------------------------------------------------------------
# Shared fixture
#
# Timeline:
#   t=0   r1: root (pid=-1), text, blocks=[1, 2, 3]  — root of 2-turn session
#   t=10  r2: follow-up of r1 (pid=1), text, blocks=[1, 2, 4]
#   t=20  r3: root (pid=-1), text, blocks=[1, 5, 6]  — single-turn session
#   t=30  r4: root (pid=-1), image, blocks=[7, 8, 9] — single-turn session
#   t=40  r5: root (pid=-1), text, blocks=[1, 2, 4, 10] — single-turn session
#
# Single-turn (session length == 1): {r3, r4, r5}
#   r1 EXCLUDED — it is the root of a 2-request session (has follow-up r2)
#   r2 EXCLUDED — follow-up (pid=1)
#
# Replay (single-turn pool only, r1 and r2 excluded):
#   r3 t=20: pool empty → 0 events; pool becomes {1:20, 5:20, 6:20}
#   r4 t=30: pool has {1,5,6}; r4 blocks={7,8,9} → 0 events; pool += {7:30,8:30,9:30}
#   r5 t=40: pool has {1,5,6,7,8,9}; r5 blocks={1,2,4,10} → eligible={1}
#            block 2 NOT in pool (r1 excluded); block 4 NOT in pool (r2 excluded)
#            event: block 1, rt=40-20=20s
#
# Summary:
#   reuse_events_total = 1  (r5: 1 event)
#   backward_reusable = {r5}
#   single_turn_request_count = 3  (r3, r4, r5)
#
# Forward-inset (which single-turn requests are reused by a future single-turn):
#   r3 (blocks={1,5,6}): future={r4,r5}; r5 has block 1 → REUSABLE
#   r4 (blocks={7,8,9}): future={r5}; no overlap → NOT reusable
#   r5 (blocks={1,2,4,10}): no future → NOT reusable
#   forward_reusable_request_count = 1  (r3 only)
# ---------------------------------------------------------------------------

@pytest.fixture
def base_records():
    return [
        _rec(1, 0,  [1, 2, 3], parent_chat_id=-1, req_type="text"),   # r1 root
        _rec(2, 10, [1, 2, 4], parent_chat_id=1,  req_type="text"),   # r2 follow-up
        _rec(3, 20, [1, 5, 6], parent_chat_id=-1, req_type="text"),   # r3 root
        _rec(4, 30, [7, 8, 9], parent_chat_id=-1, req_type="image"),  # r4 root
        _rec(5, 40, [1, 2, 4, 10], parent_chat_id=-1, req_type="text"),  # r5 root
    ]


# ===========================================================================
# Area 1: helper predicates (_is_root_request, identify_root_requests)
#         These remain available but are no longer used as the main filter.
#         The main filter (_identify_single_turn_request_ids) uses session
#         reconstruction and len(session)==1.
# ===========================================================================

class TestArea1_SingleTurnDefinition:
    def test_neg1_is_root(self):
        r = _rec(1, 0, [], parent_chat_id=-1)
        assert _is_root_request(r) is True

    def test_positive_pid_is_not_root(self):
        r = _rec(2, 0, [], parent_chat_id=1)
        assert _is_root_request(r) is False

    def test_large_positive_pid_is_not_root(self):
        r = _rec(3, 0, [], parent_chat_id=99999)
        assert _is_root_request(r) is False

    def test_absent_parent_field_is_root(self):
        r = _rec_no_parent_field(4, 0, [])
        assert _is_root_request(r) is True

    def test_identify_root_requests_returns_correct_set(self, base_records):
        roots = identify_root_requests(base_records)
        assert roots == {"1", "3", "4", "5"}  # r1, r3, r4, r5
        assert "2" not in roots               # r2 is follow-up


# ===========================================================================
# Area 2: multi-turn session roots ARE excluded from single-turn set.
#         Only requests whose session has exactly 1 request are single-turn.
# ===========================================================================

class TestArea2_MultiTurnRootsExcluded:
    def test_multi_turn_root_excluded_from_single_turn(self):
        # r_root starts a two-turn session (r_child follows up)
        # → r_root is NOT single-turn even though parent_chat_id == -1
        from block_prefix_analyzer.analysis.f13 import _identify_single_turn_request_ids
        r_root = _rec(10, 0, [1, 2], parent_chat_id=-1, req_type="text")
        r_child = _rec(11, 5, [1, 3], parent_chat_id=10, req_type="text")
        st_ids = _identify_single_turn_request_ids([r_root, r_child])
        assert "10" not in st_ids  # root of multi-turn session → excluded
        assert "11" not in st_ids  # follow-up → excluded
        assert len(st_ids) == 0

    def test_multi_turn_root_not_in_pool(self):
        # a1: single-turn; b1 root of 2-turn session (b2 follow-up)
        # b1 is excluded → its blocks do NOT warm the pool
        # Therefore c1 sees an empty pool and generates no events.
        a1 = _rec(1, 0,  [100, 200], parent_chat_id=-1, req_type="text")  # single-turn
        b1 = _rec(2, 10, [100, 300], parent_chat_id=-1, req_type="text")  # multi-turn root
        b2 = _rec(3, 20, [100, 400], parent_chat_id=2,  req_type="text")  # follow-up
        c1 = _rec(4, 30, [100, 200], parent_chat_id=-1, req_type="text")  # single-turn

        series = compute_f13_strict_series([a1, b1, b2, c1])
        # single-turn = {a1, c1} (b1 excluded — multi-turn root)
        assert series.single_turn_request_count == 2
        # c1 sees pool from a1 only: {100, 200} → eligible={100, 200} → 2 events
        assert series.content_block_reuse_event_count_total == 2


# ===========================================================================
# Area 3: main CDF uses block-level reusable events (not prefix-aware)
# ===========================================================================

class TestArea3_BlockLevelReusableEvents:
    def test_non_prefix_block_generates_event(self):
        # r1: blocks=[1, 2, 3]
        # r2: blocks=[99, 1] — block 99 is first (no prefix hit), block 1 is at pos 1
        # Under prefix-aware: prefix_hit=0 → no events (first block 99 not seen)
        # Under reusable: block 1 IS in pool → 1 event
        r1 = _rec(1, 0, [1, 2, 3], parent_chat_id=-1)
        r2 = _rec(2, 10, [99, 1], parent_chat_id=-1)
        series = compute_f13_strict_series([r1, r2])
        assert series.content_block_reuse_event_count_total == 1  # block 1 is reusable despite non-prefix position

    def test_event_definition_tag(self, base_records):
        series = compute_f13_strict_series(base_records)
        assert "content_block_reuse" in series.event_definition
        assert "prefix" not in series.event_definition


# ===========================================================================
# Area 4: reuse_time uses last_seen semantics
# ===========================================================================

class TestArea4_ReuseTimeLastSeen:
    def test_reuse_time_is_current_minus_last_seen(self):
        r1 = _rec(1, 0,  [10], parent_chat_id=-1)   # block 10 first seen at t=0
        r2 = _rec(2, 30, [10], parent_chat_id=-1)   # block 10 seen again at t=30 → rt=30
        r3 = _rec(3, 80, [10], parent_chat_id=-1)   # block 10 last seen at t=30 → rt=80-30=50
        series = compute_f13_strict_series([r1, r2, r3])
        times = sorted(e.reuse_time_seconds for e in series.events)
        assert times == [30.0, 50.0]

    def test_reuse_time_minutes_conversion(self):
        r1 = _rec(1, 0,    [55], parent_chat_id=-1)
        r2 = _rec(2, 3360, [55], parent_chat_id=-1)  # 56 minutes = 3360 seconds
        series = compute_f13_strict_series([r1, r2])
        assert len(series.events) == 1
        assert series.events[0].reuse_time_seconds == pytest.approx(3360.0)
        assert series.events[0].reuse_time_minutes == pytest.approx(56.0)

    def test_reuse_time_definition_tag(self):
        assert "last_seen" in REUSE_TIME_DEFINITION


# ===========================================================================
# Area 5: within-request dedup — repeated block_id counts once
# ===========================================================================

class TestArea5_WithinRequestDedup:
    def test_duplicate_block_in_request_produces_one_event(self):
        r1 = _rec(1, 0,  [7], parent_chat_id=-1)
        # r2 contains block 7 twice — should generate only 1 event
        r2 = _rec(2, 10, [7, 7, 8], parent_chat_id=-1)
        series = compute_f13_strict_series([r1, r2])
        events_for_r2 = [e for e in series.events if e.request_id == "2"]
        assert len(events_for_r2) == 1  # block 7 counted once

    def test_dedup_rule_tag(self):
        assert "set_dedup" in DEDUPE_RULE
        assert "one_event_per_unique_block" in DEDUPE_RULE


# ===========================================================================
# Area 6: inset uses FORWARD-LOOKING definition (reusable by future single-turn)
#
# Base fixture forward-reusable analysis (single-turn = {r3, r4, r5}):
#   r3 (t=20, blocks={1,5,6}): future={r4,r5}; r5 has block 1 → REUSABLE
#   r4 (t=30, blocks={7,8,9}): future={r5}; no overlap         → NOT reusable
#   r5 (t=40, blocks={1,2,4,10}): no future                    → NOT reusable
# Forward-reusable set = {r3}, count = 1, not-reusable = {r4, r5}, count = 2
# ===========================================================================

class TestArea6_InsetForwardLooking:
    def test_inset_uses_forward_reusable_count(self, base_records):
        series = compute_f13_strict_series(base_records)
        # Single-turn = {r3, r4, r5} (r1 excluded as multi-turn root)
        # Forward-reusable: r3 only (block 1 appears in future r5)
        assert series.forward_reusable_request_count == 1
        assert series.single_turn_request_count == 3
        assert (series.single_turn_request_count - series.forward_reusable_request_count) == 2

    def test_inset_denominator_is_all_single_turn_requests(self, base_records):
        series = compute_f13_strict_series(base_records)
        total = series.forward_reusable_request_count + (
            series.single_turn_request_count - series.forward_reusable_request_count
        )
        assert total == series.single_turn_request_count == 3

    def test_inset_fraction_sums_to_reusable_fraction(self, base_records):
        series = compute_f13_strict_series(base_records)
        total_frac = sum(row.fraction for row in series.breakdown_rows)
        expected = series.forward_reusable_request_count / series.single_turn_request_count
        assert total_frac == pytest.approx(expected)

    def test_pool_definition_tags(self):
        assert "single_turn" in POOL_DEFINITION_CDF
        assert "forward_looking" in POOL_DEFINITION_BREAKDOWN


# ===========================================================================
# Area 7: multi-turn follow-ups do NOT update the pool
# ===========================================================================

class TestArea7_FollowUpsExcludedFromPool:
    def test_block_from_followup_not_counted_as_reusable(self):
        # r1: root, blocks=[1, 2]
        # r2: follow-up, blocks=[3, 4]  ← block 3 only from follow-up
        # r3: root, blocks=[3, 5]       ← block 3 should NOT generate CDF event (r2 not in pool)
        r1 = _rec(1, 0,  [1, 2], parent_chat_id=-1)
        r2 = _rec(2, 10, [3, 4], parent_chat_id=1)   # follow-up
        r3 = _rec(3, 20, [3, 5], parent_chat_id=-1)  # root
        series = compute_f13_strict_series([r1, r2, r3])
        # CDF: block 3 only appeared in r2 (follow-up), which doesn't update pool → 0 events
        assert series.content_block_reuse_event_count_total == 0
        # Forward inset: r1 blocks={1,2} vs r3 blocks={3,5} → no overlap → 0 forward-reusable
        assert series.forward_reusable_request_count == 0

    def test_followup_block_isolated_from_pool(self, base_records):
        # r5 blocks=[1, 2, 4, 10]
        # block 2 only in r1 (multi-turn root → excluded from pool)
        # block 4 only in r2 (follow-up → excluded from pool)
        # Only block 1 (from r3, single-turn) is in pool when r5 arrives
        series = compute_f13_strict_series(base_records)
        r5_events = [e for e in series.events if e.request_id == "5"]
        assert len(r5_events) == 1  # only block 1 hits (not block 2 or 4)

    def test_non_single_turn_not_in_count(self, base_records):
        series = compute_f13_strict_series(base_records)
        # r3, r4, r5 are single-turn (session len==1)
        # r1 excluded (has follow-up r2); r2 excluded (follow-up)
        assert series.single_turn_request_count == 3


# ===========================================================================
# Area 8: CDF monotonically non-decreasing, range [0, 1]
# ===========================================================================

class TestArea8_CDFMonotonicity:
    def test_cdf_monotone_per_type(self, base_records):
        series = compute_f13_strict_series(base_records)
        by_type: dict[str, list[float]] = {}
        for row in series.cdf_rows:
            by_type.setdefault(row.display_label, []).append(row.cdf)
        for label, cdfs in by_type.items():
            for i in range(1, len(cdfs)):
                assert cdfs[i] >= cdfs[i - 1], f"CDF not monotone for {label}"

    def test_cdf_range_0_to_1(self, base_records):
        series = compute_f13_strict_series(base_records)
        for row in series.cdf_rows:
            assert 0.0 < row.cdf <= 1.0

    def test_last_cdf_per_type_is_1(self, base_records):
        series = compute_f13_strict_series(base_records)
        by_type: dict[str, float] = {}
        for row in series.cdf_rows:
            by_type[row.display_label] = row.cdf
        for label, last_cdf in by_type.items():
            assert last_cdf == pytest.approx(1.0), f"Last CDF != 1 for {label}"

    def test_cold_start_no_events_no_cdf_rows(self):
        r1 = _rec(1, 0, [1, 2], parent_chat_id=-1)
        series = compute_f13_strict_series([r1])
        assert series.content_block_reuse_event_count_total == 0
        assert series.cdf_rows == []


# ===========================================================================
# Area 9: output schema — F13Series fields and CSV column names stable
# ===========================================================================

class TestArea9_OutputSchema:
    def test_f13series_required_fields(self, base_records):
        series = compute_f13_strict_series(base_records)
        assert hasattr(series, "event_definition")
        assert hasattr(series, "events")
        assert hasattr(series, "cdf_rows")
        assert hasattr(series, "breakdown_rows")
        assert hasattr(series, "single_turn_request_count")
        assert hasattr(series, "forward_reusable_request_count")
        assert hasattr(series, "backward_event_hit_request_count")
        assert hasattr(series, "content_block_reuse_event_count_total")
        assert hasattr(series, "content_block_reuse_event_count_over_56min")
        assert hasattr(series, "x_axis_max_minutes")

    def test_cdf_csv_columns(self, base_records, tmp_path):
        series = compute_f13_strict_series(base_records)
        p = tmp_path / "cdf.csv"
        save_strict_cdf_csv(series, p)
        with p.open() as f:
            header = f.readline().strip().split(",")
        assert header == ["request_type", "display_label",
                          "reuse_time_seconds", "reuse_time_minutes", "cdf"]

    def test_breakdown_csv_columns(self, base_records, tmp_path):
        series = compute_f13_strict_series(base_records)
        p = tmp_path / "bd.csv"
        save_strict_breakdown_csv(series, p)
        with p.open() as f:
            header = f.readline().strip().split(",")
        assert header == [
            "request_type", "display_label",
            "content_reused_request_count",
            "content_reused_request_fraction_over_all_single_turn_requests",
        ]

    def test_metadata_json_required_keys(self, base_records, tmp_path):
        series = compute_f13_strict_series(base_records)
        p = tmp_path / "meta.json"
        save_strict_metadata_json(
            series, p,
            trace_name="test", input_file="test.jsonl",
            note_public_adaptation="test note",
        )
        meta = json.loads(p.read_text())
        required_keys = [
            "trace_name", "input_file", "single_turn_definition",
            "event_definition", "reuse_time_definition",
            "dedupe_within_request_rule", "breakdown_definition",
            "pool_definition_for_cdf", "pool_definition_for_breakdown",
            "type_label_mapping", "x_axis_max_minutes",
            "single_turn_request_count", "forward_reusable_request_count",
            "forward_non_reusable_request_count", "content_block_reuse_event_count_total",
            "content_block_reuse_event_count_over_56min", "note_public_adaptation",
        ]
        for key in required_keys:
            assert key in meta, f"Missing metadata key: {key}"

    def test_over_56min_counted_not_excluded(self):
        # Event at t=3361s = 56.017 min, just over 56min limit
        r1 = _rec(1, 0,    [99], parent_chat_id=-1)
        r2 = _rec(2, 3361, [99], parent_chat_id=-1)
        series = compute_f13_strict_series([r1, r2], x_axis_max_minutes=56.0)
        assert series.content_block_reuse_event_count_total == 1      # counted in total
        assert series.content_block_reuse_event_count_over_56min == 1  # also flagged
        assert len(series.cdf_rows) == 1                # stays in CDF


# ===========================================================================
# Area 10: image -> Multimedia mapping
# ===========================================================================

class TestArea10_TypeLabelMapping:
    def test_image_maps_to_multimedia(self):
        r1 = _rec(1, 0,  [1, 2], parent_chat_id=-1, req_type="image")
        r2 = _rec(2, 10, [1, 3], parent_chat_id=-1, req_type="image")
        series = compute_f13_strict_series([r1, r2])
        assert series.content_block_reuse_event_count_total == 1
        evt = series.events[0]
        assert evt.request_type == "image"
        assert evt.display_label == "Multimedia"

    def test_image_breakdown_label_is_multimedia(self):
        r1 = _rec(1, 0,  [1], parent_chat_id=-1, req_type="image")
        r2 = _rec(2, 10, [1], parent_chat_id=-1, req_type="image")
        series = compute_f13_strict_series([r1, r2])
        image_rows = [row for row in series.breakdown_rows if row.request_type == "image"]
        assert len(image_rows) == 1
        assert image_rows[0].display_label == "Multimedia"

    def test_all_four_type_mappings(self):
        from block_prefix_analyzer.analysis.f13 import DEFAULT_TYPE_LABEL_MAPPING
        assert DEFAULT_TYPE_LABEL_MAPPING["text"] == "Text"
        assert DEFAULT_TYPE_LABEL_MAPPING["file"] == "File"
        assert DEFAULT_TYPE_LABEL_MAPPING["image"] == "Multimedia"
        assert DEFAULT_TYPE_LABEL_MAPPING["search"] == "Search"

    def test_mixed_types_all_mapped(self):
        r1 = _rec(1, 0,  [1], parent_chat_id=-1, req_type="text")
        r2 = _rec(2, 5,  [2], parent_chat_id=-1, req_type="file")
        r3 = _rec(3, 10, [1], parent_chat_id=-1, req_type="text")  # reuse from r1
        r4 = _rec(4, 15, [2], parent_chat_id=-1, req_type="file")  # reuse from r2
        series = compute_f13_strict_series([r1, r2, r3, r4])
        labels = {e.display_label for e in series.events}
        assert "Text" in labels
        assert "File" in labels
