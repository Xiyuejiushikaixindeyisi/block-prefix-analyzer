"""Tests for F13 analysis: reuse-time CDF for single-turn sessions.

Covers all 11 required test items:
  1.  Reusable figure uses block-level reusable events (even non-prefix blocks)
  2.  Prefix figure only uses events from the contiguous prefix-hit segment
  3.  Both figures share the same single-turn filtering logic
  4.  Both figures use last_seen reuse_time semantics
  5.  Both figures dedup blocks within a request
  6.  Both figures compute CDF over all events, then clip x-axis
  7.  Reusable and prefix produce different CDFs on the same sample
  8.  Reusable and prefix produce different request-level inset statistics
  9.  image -> Multimedia type label mapping
 10.  Output schema is stable (CdfRow, BreakdownRow field names)
 11.  No dependency on raw-request full-alignment path (V2 pipeline)

All scenarios use hand-crafted RequestRecord objects with integer block_ids.
No TraceA file is loaded; no V2 pipeline is invoked.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.analysis.f13 import (
    BreakdownRow,
    CdfRow,
    DEFAULT_TYPE_LABEL_MAPPING,
    F13Series,
    ReuseEventRow,
    compute_f13_series,
    save_breakdown_csv,
    save_cdf_csv,
    save_metadata_json,
)
from block_prefix_analyzer.types import RequestRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(
    rid: str,
    ts: float,
    block_ids: list[int],
    *,
    req_type: str = "text",
    parent_chat_id: int = -1,
    arrival_index: int = 0,
) -> RequestRecord:
    return RequestRecord(
        request_id=rid,
        timestamp=ts,
        arrival_index=arrival_index,
        block_ids=block_ids,
        metadata={"type": req_type, "parent_chat_id": parent_chat_id},
    )


def _compute(records, event_def, x_max=56.0):
    return compute_f13_series(records, event_definition=event_def, x_axis_max_minutes=x_max)


# ---------------------------------------------------------------------------
# Standard scenario A
#
# Session structure:
#   root (id="root"):  t=0,  blocks=[1,2,3], parent=-1 → has child "child"
#   child(id="child"): t=5,  blocks=[4,5],   parent="root" (int parent=root's chat_id)
#   st   (id="30"):    t=10, blocks=[1,4,2], parent=-1  → single-turn (root, no child)
#
# Note: parent_chat_id is stored as integer matching chat_id.
# "root" has chat_id=10; "child" has parent_chat_id=10; "st" has parent_chat_id=-1.
#
# After r_root inserted:  trie=[1,2,3];  last_seen={1:0, 2:0, 3:0}
# After r_child inserted: trie+=[4,5];   last_seen+={4:5, 5:5}
#
# For r_st=[1,4,2] at t=10:
#   trie prefix match: 1 matches, next=4 but trie children of 1 is {2} → prefix_hit=1
#   reusable eligible: {1,4,2} all in last_seen → 3 events
#   prefix eligible:   segment=[1] → {1} in last_seen → 1 event
# ---------------------------------------------------------------------------

def _make_scenario_a():
    """Return (r_root, r_child, r_st) with proper parent_chat_id linkage."""
    r_root  = _rec("10", 0.0,  [1, 2, 3], req_type="text", parent_chat_id=-1, arrival_index=0)
    r_child = _rec("20", 5.0,  [4, 5],    req_type="text", parent_chat_id=10,  arrival_index=1)
    r_st    = _rec("30", 10.0, [1, 4, 2], req_type="text", parent_chat_id=-1, arrival_index=2)
    return [r_root, r_child, r_st]


# ---------------------------------------------------------------------------
# Test 1: Reusable figure uses block-level events (including non-prefix blocks)
# ---------------------------------------------------------------------------

def test_reusable_events_include_non_prefix_blocks():
    records = _make_scenario_a()
    series = _compute(records, "reusable")
    # Expect 3 reuse events: blocks 1, 4, 2 (all seen before r_st, including non-prefix block 4 and 2)
    assert series.reuse_event_count_total == 3
    event_times = sorted(e.reuse_time_seconds for e in series.events)
    # block 1 → 10-0=10, block 4 → 10-5=5, block 2 → 10-0=10
    assert sorted(event_times) == sorted([10.0, 5.0, 10.0])


def test_reusable_events_contain_non_prefix_block_4():
    records = _make_scenario_a()
    series = _compute(records, "reusable")
    # Block 4 is in r_st at position 1 (after prefix miss) but should appear in reusable events
    reuse_times = sorted(e.reuse_time_seconds for e in series.events)
    assert 5.0 in reuse_times  # block 4: reuse_time = 10-5 = 5


# ---------------------------------------------------------------------------
# Test 2: Prefix figure only uses contiguous prefix-hit segment
# ---------------------------------------------------------------------------

def test_prefix_events_only_from_prefix_segment():
    records = _make_scenario_a()
    series = _compute(records, "prefix")
    # prefix_hit=1 for r_st=[1,4,2] → only block 1 is in segment → 1 event
    assert series.reuse_event_count_total == 1
    assert series.events[0].reuse_time_seconds == 10.0  # block 1: 10-0=10


def test_prefix_events_exclude_non_prefix_blocks():
    records = _make_scenario_a()
    series = _compute(records, "prefix")
    # Block 4 (position 1) and block 2 (position 2) are NOT in prefix segment
    for e in series.events:
        assert e.reuse_time_seconds != 5.0, "Block 4 (reuse_time=5) must NOT appear in prefix events"


# ---------------------------------------------------------------------------
# Test 3: Both figures share the same single-turn filtering
# ---------------------------------------------------------------------------

def test_single_turn_filter_same_for_both_definitions():
    records = _make_scenario_a()
    series_r = _compute(records, "reusable")
    series_p = _compute(records, "prefix")
    # Both should see the same single-turn request count
    assert series_r.single_turn_request_count == series_p.single_turn_request_count
    # In scenario A, only r_st (id="30") is single-turn
    assert series_r.single_turn_request_count == 1


def test_multi_turn_requests_excluded_from_events():
    records = _make_scenario_a()
    series = _compute(records, "reusable")
    # r_root ("10") has child "20" → not single-turn → must not generate events as current request
    # r_child ("20") has parent → not single-turn
    # r_st ("30") is the only single-turn request
    for e in series.events:
        assert e.request_id == "30"


# ---------------------------------------------------------------------------
# Test 4: reuse_time uses last_seen semantics
# ---------------------------------------------------------------------------

def test_reuse_time_uses_last_seen_not_first_seen():
    """Block A seen at t=0 (root), again at t=5 (child). Single-turn at t=10.
    reuse_time should be 10-5=5, not 10-0=10."""
    r_root  = _rec("1", 0.0, [99], parent_chat_id=-1, arrival_index=0)
    r_child = _rec("2", 5.0, [99], parent_chat_id=1,   arrival_index=1)
    r_st    = _rec("3", 10.0, [99], parent_chat_id=-1, arrival_index=2)

    series_r = _compute([r_root, r_child, r_st], "reusable")
    assert len(series_r.events) == 1
    assert series_r.events[0].reuse_time_seconds == 5.0, (
        f"Expected 5.0 (last_seen=5), got {series_r.events[0].reuse_time_seconds}"
    )


def test_prefix_reuse_time_also_uses_last_seen():
    r_root  = _rec("1", 0.0, [7],  parent_chat_id=-1, arrival_index=0)
    r_child = _rec("2", 3.0, [7],  parent_chat_id=1,   arrival_index=1)
    r_st    = _rec("3", 9.0, [7],  parent_chat_id=-1, arrival_index=2)

    series_p = _compute([r_root, r_child, r_st], "prefix")
    assert len(series_p.events) == 1
    assert series_p.events[0].reuse_time_seconds == 6.0  # 9-3=6


# ---------------------------------------------------------------------------
# Test 5: Both figures dedup blocks within a request
# ---------------------------------------------------------------------------

def test_reusable_deduplication_within_request():
    """Block 1 appears twice in r_st; must produce only one reuse event."""
    r_seed  = _rec("1", 0.0,  [1], parent_chat_id=-1, arrival_index=0)
    r_seed2 = _rec("2", 0.0,  [],  parent_chat_id=1,   arrival_index=1)   # child → r_seed not single-turn
    r_st    = _rec("3", 10.0, [1, 1, 2], parent_chat_id=-1, arrival_index=2)

    series = _compute([r_seed, r_seed2, r_st], "reusable")
    # block 1 seen twice in r_st → deduped to 1 event; block 2 is new → 0 events
    assert series.reuse_event_count_total == 1
    assert series.events[0].reuse_time_seconds == 10.0


def test_prefix_deduplication_within_prefix_segment():
    """Even within the prefix segment, duplicate blocks count as one event."""
    # [1, 1] with prefix_hit=2 (both 1s match trie path [1,1,...])
    # After r_seed inserts [1, 1], the trie has path 1→1→...
    r_seed  = _rec("1", 0.0, [1, 1], parent_chat_id=-1, arrival_index=0)
    r_child = _rec("2", 0.0, [],     parent_chat_id=1,   arrival_index=1)
    r_st    = _rec("3", 5.0, [1, 1], parent_chat_id=-1, arrival_index=2)

    series = _compute([r_seed, r_child, r_st], "prefix")
    # Prefix segment = [1, 1] → unique = {1} → 1 event
    assert series.reuse_event_count_total == 1


# ---------------------------------------------------------------------------
# Test 6: CDF computed over ALL events; clip is only for counting / plotting
# ---------------------------------------------------------------------------

def test_cdf_includes_events_over_x_axis_limit():
    """Events beyond 56 min must be included in CDF but counted in over_limit counter."""
    # reuse_time = 4000s = 66.7 min > 56 min
    r_seed  = _rec("1", 0.0,    [42], parent_chat_id=-1, arrival_index=0)
    r_child = _rec("2", 0.0,    [],   parent_chat_id=1,   arrival_index=1)
    r_st    = _rec("3", 4000.0, [42], parent_chat_id=-1, arrival_index=2)

    series = _compute([r_seed, r_child, r_st], "reusable", x_max=56.0)
    assert series.reuse_event_count_total == 1
    assert series.reuse_event_count_over_56min == 1
    # CDF of the single event should reach 1.0 (computed over all events)
    assert len(series.cdf_rows) == 1
    assert series.cdf_rows[0].cdf == 1.0
    assert series.cdf_rows[0].reuse_time_seconds == 4000.0


def test_cdf_not_renormalized_after_clipping():
    """Mix of events: some within 56 min, some beyond. CDF goes to 1.0 at the last event."""
    r_seed  = _rec("1", 0.0, [10, 20], parent_chat_id=-1, arrival_index=0)
    r_child = _rec("2", 0.0, [],       parent_chat_id=1,   arrival_index=1)
    r_st    = _rec("3", 10.0, [10, 20], parent_chat_id=-1, arrival_index=2)   # within 56 min
    r_st2   = _rec("4", 5000.0, [10],   parent_chat_id=-1, arrival_index=3)  # beyond 56 min

    series = _compute([r_seed, r_child, r_st, r_st2], "reusable", x_max=56.0)
    # r_st: blocks 10 and 20 both reused (2 events); r_st2: block 10 reused (1 event)
    assert series.reuse_event_count_total == 3
    assert series.reuse_event_count_over_56min == 1
    max_cdf = max(r.cdf for r in series.cdf_rows)
    assert max_cdf == 1.0, "CDF must reach 1.0 (not re-normalized)"


# ---------------------------------------------------------------------------
# Test 7: Reusable and prefix produce different CDFs on the same sample
# ---------------------------------------------------------------------------

def test_reusable_and_prefix_cdfs_differ():
    """Same records → different event counts → different CDFs."""
    records = _make_scenario_a()
    series_r = _compute(records, "reusable")
    series_p = _compute(records, "prefix")
    # reusable: 3 events; prefix: 1 event → different CDFs
    assert series_r.reuse_event_count_total != series_p.reuse_event_count_total
    r_cdf_pts = [(row.reuse_time_seconds, row.cdf) for row in series_r.cdf_rows]
    p_cdf_pts = [(row.reuse_time_seconds, row.cdf) for row in series_p.cdf_rows]
    assert r_cdf_pts != p_cdf_pts


# ---------------------------------------------------------------------------
# Test 8: Inset statistics differ between reusable and prefix
# ---------------------------------------------------------------------------

def _make_scenario_b():
    """Single-turn request where reusable event exists but prefix_hit=0.

    r_root: blocks=[11], parent=-1 → has child → not single-turn
    r_child: blocks=[22], parent=r_root → not single-turn
    r_st: blocks=[33, 11], parent=-1 → single-turn

    For r_st=[33, 11]:
      After r_root: trie=[11]; after r_child: trie+=[22]
      prefix_hit of [33, 11]: 33 not in trie at root → prefix_hit=0
      reusable: {33, 11} ∩ last_seen = {11} → 1 reuse event
      prefix:   segment=[] (prefix_hit=0) → 0 reuse events
    """
    r_root  = _rec("101", 0.0,  [11], parent_chat_id=-1,  arrival_index=0)
    r_child = _rec("102", 1.0,  [22], parent_chat_id=101,  arrival_index=1)
    r_st    = _rec("103", 10.0, [33, 11], parent_chat_id=-1, arrival_index=2)
    return [r_root, r_child, r_st]


def test_reusable_inset_shows_request_with_reuse():
    records = _make_scenario_b()
    series = _compute(records, "reusable")
    assert series.request_count_with_reuse == 1
    assert series.request_count_without_reuse == 0


def test_prefix_inset_shows_no_request_with_reuse():
    records = _make_scenario_b()
    series = _compute(records, "prefix")
    # prefix_hit=0 → no prefix events → no request counted in inset
    assert series.request_count_with_reuse == 0
    assert series.request_count_without_reuse == 1


def test_reusable_and_prefix_breakdown_differ():
    records = _make_scenario_b()
    series_r = _compute(records, "reusable")
    series_p = _compute(records, "prefix")
    r_counts = [row.count for row in series_r.breakdown_rows]
    p_counts = [row.count for row in series_p.breakdown_rows]
    assert r_counts != p_counts or series_r.request_count_with_reuse != series_p.request_count_with_reuse


# ---------------------------------------------------------------------------
# Test 9: image → Multimedia type label mapping
# ---------------------------------------------------------------------------

def test_image_type_mapped_to_multimedia():
    r_seed  = _rec("1", 0.0,  [5], req_type="text",  parent_chat_id=-1, arrival_index=0)
    r_child = _rec("2", 0.0,  [],  req_type="text",  parent_chat_id=1,   arrival_index=1)
    r_img   = _rec("3", 5.0,  [5], req_type="image", parent_chat_id=-1, arrival_index=2)

    series = _compute([r_seed, r_child, r_img], "reusable")
    # r_img is single-turn and has 1 reuse event
    assert series.reuse_event_count_total == 1
    event = series.events[0]
    assert event.request_type == "image"
    assert event.display_label == "Multimedia"


def test_image_type_in_breakdown_has_multimedia_label():
    r_seed  = _rec("1", 0.0, [5], req_type="text",  parent_chat_id=-1, arrival_index=0)
    r_child = _rec("2", 0.0, [],  req_type="text",  parent_chat_id=1,   arrival_index=1)
    r_img   = _rec("3", 5.0, [5], req_type="image", parent_chat_id=-1, arrival_index=2)

    series = _compute([r_seed, r_child, r_img], "reusable")
    labels = [row.display_label for row in series.breakdown_rows]
    assert "Multimedia" in labels


def test_default_type_label_mapping_has_multimedia():
    assert DEFAULT_TYPE_LABEL_MAPPING["image"] == "Multimedia"


# ---------------------------------------------------------------------------
# Test 10: Output schema is stable
# ---------------------------------------------------------------------------

def test_cdf_row_has_required_fields():
    records = _make_scenario_a()
    series = _compute(records, "reusable")
    assert len(series.cdf_rows) > 0
    row = series.cdf_rows[0]
    assert hasattr(row, "request_type")
    assert hasattr(row, "display_label")
    assert hasattr(row, "reuse_time_seconds")
    assert hasattr(row, "reuse_time_minutes")
    assert hasattr(row, "cdf")


def test_breakdown_row_has_required_fields():
    records = _make_scenario_a()
    series = _compute(records, "reusable")
    assert len(series.breakdown_rows) > 0
    row = series.breakdown_rows[0]
    assert hasattr(row, "request_type")
    assert hasattr(row, "display_label")
    assert hasattr(row, "count")
    assert hasattr(row, "fraction")


def test_f13_series_has_required_fields():
    records = _make_scenario_a()
    series = _compute(records, "reusable")
    assert hasattr(series, "event_definition")
    assert hasattr(series, "single_turn_request_count")
    assert hasattr(series, "request_count_with_reuse")
    assert hasattr(series, "request_count_without_reuse")
    assert hasattr(series, "reuse_event_count_total")
    assert hasattr(series, "reuse_event_count_over_56min")
    assert hasattr(series, "x_axis_max_minutes")


def test_reuse_time_minutes_matches_seconds(tmp_path):
    records = _make_scenario_a()
    series = _compute(records, "reusable")
    for e in series.events:
        assert abs(e.reuse_time_minutes - e.reuse_time_seconds / 60.0) < 1e-9
    for r in series.cdf_rows:
        assert abs(r.reuse_time_minutes - r.reuse_time_seconds / 60.0) < 1e-9


def test_cdf_row_cdf_values_in_0_1():
    records = _make_scenario_a()
    series = _compute(records, "reusable")
    for row in series.cdf_rows:
        assert 0.0 < row.cdf <= 1.0


def test_breakdown_fraction_sum_le_one():
    records = _make_scenario_a()
    series = _compute(records, "reusable")
    total = sum(row.fraction for row in series.breakdown_rows)
    assert total <= 1.0 + 1e-9


def test_event_definition_recorded_in_series():
    records = _make_scenario_a()
    assert _compute(records, "reusable").event_definition == "reusable"
    assert _compute(records, "prefix").event_definition == "prefix"


def test_save_cdf_csv_creates_file(tmp_path):
    series = _compute(_make_scenario_a(), "reusable")
    p = tmp_path / "cdf.csv"
    save_cdf_csv(series, p)
    lines = p.read_text().splitlines()
    assert lines[0] == "request_type,display_label,reuse_time_seconds,reuse_time_minutes,cdf"
    assert len(lines) > 1


def test_save_breakdown_csv_reusable_columns(tmp_path):
    series = _compute(_make_scenario_a(), "reusable")
    p = tmp_path / "breakdown_r.csv"
    save_breakdown_csv(series, p)
    header = p.read_text().splitlines()[0]
    assert "reusable_request_count" in header
    assert "reusable_request_fraction_over_all_single_turn_requests" in header


def test_save_breakdown_csv_prefix_columns(tmp_path):
    series = _compute(_make_scenario_a(), "prefix")
    p = tmp_path / "breakdown_p.csv"
    save_breakdown_csv(series, p)
    header = p.read_text().splitlines()[0]
    assert "prefix_reusable_request_count" in header
    assert "prefix_reusable_request_fraction_over_all_single_turn_requests" in header


def test_save_metadata_json_required_keys(tmp_path):
    series = _compute(_make_scenario_a(), "reusable")
    p = tmp_path / "metadata.json"
    save_metadata_json(series, p, trace_name="test", input_file="test.jsonl")
    import json
    meta = json.loads(p.read_text())
    required_keys = [
        "trace_name", "input_file", "event_definition", "single_turn_definition",
        "reuse_time_definition", "dedupe_within_request_rule", "type_label_mapping",
        "x_axis_max_minutes", "single_turn_request_count", "request_count_with_reuse",
        "request_count_without_reuse", "reuse_event_count_total",
        "reuse_event_count_over_56min", "note_public_adaptation",
    ]
    for key in required_keys:
        assert key in meta, f"metadata.json missing required key: {key!r}"


# ---------------------------------------------------------------------------
# Test 11: No dependency on raw-request full-alignment path
# ---------------------------------------------------------------------------

def test_no_v2_pipeline_import_in_f13():
    """f13.py must not import V2 pipeline, adapters, or tokenizer modules."""
    import importlib
    import importlib.util
    import pathlib

    f13_path = pathlib.Path("src/block_prefix_analyzer/analysis/f13.py")
    source = f13_path.read_text()
    forbidden_imports = [
        "from block_prefix_analyzer.v2.pipeline",
        "from block_prefix_analyzer.v2.adapters",
        "from block_prefix_analyzer.v2.normalizer",
    ]
    for imp in forbidden_imports:
        assert imp not in source, (
            f"f13.py must not depend on V2 pipeline path. Found: {imp!r}"
        )


def test_compute_f13_accepts_plain_request_records():
    """All inputs are RequestRecord objects with block_ids — no V2 pipeline needed."""
    records = [
        RequestRecord("1", 0.0, 0, [1, 2], metadata={"type": "text", "parent_chat_id": -1}),
        RequestRecord("2", 1.0, 1, [3, 4], metadata={"type": "text", "parent_chat_id": -1}),
    ]
    series = compute_f13_series(records, event_definition="reusable")
    assert isinstance(series, F13Series)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_records_returns_zero_counts():
    series = _compute([], "reusable")
    assert series.single_turn_request_count == 0
    assert series.reuse_event_count_total == 0
    assert series.cdf_rows == []
    assert series.breakdown_rows == []


def test_all_multi_turn_no_single_turn_events():
    """All requests belong to multi-turn sessions → no reuse events extracted."""
    r_root  = _rec("1", 0.0,  [1, 2], parent_chat_id=-1, arrival_index=0)
    r_child = _rec("2", 5.0,  [1, 3], parent_chat_id=1,   arrival_index=1)
    series = _compute([r_root, r_child], "reusable")
    assert series.single_turn_request_count == 0
    assert series.reuse_event_count_total == 0


def test_cold_start_single_turn_no_events():
    """Single-turn request is the very first request → nothing seen before → 0 events."""
    r_st = _rec("1", 0.0, [5, 6, 7], parent_chat_id=-1, arrival_index=0)
    series = _compute([r_st], "reusable")
    assert series.single_turn_request_count == 1
    assert series.reuse_event_count_total == 0
    assert series.request_count_without_reuse == 1
