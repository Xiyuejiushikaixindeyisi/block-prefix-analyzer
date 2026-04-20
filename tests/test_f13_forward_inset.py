"""Tests for F13 forward-looking inset computation.

Required test areas:
  1. Direction test: backward-hit ≠ forward-reusable — proves semantic distinction.
  2. Future reuse test: request IS marked reusable when a later root reuses its block.
  3. Denominator test: inset denominator = all root requests (not just reusable ones).
  4. No future reuser test: last request in trace is never forward-reusable.
  5. Multi-turn follow-ups excluded: follow-ups are not roots, not counted.
  6. Block-set overlap semantics: any shared block counts (not prefix-only).
  7. num_future_reusers count: correct count of distinct future reusers.
  8. first_future_reuse_delay: correct delay to earliest future reuser.
  9. CSV schema: correct column names and types.
 10. Type mapping: image -> Multimedia in ForwardReuseRecord.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from block_prefix_analyzer.analysis.f13_forward_inset import (
    ForwardReuseRecord,
    compute_forward_inset,
    forward_inset_to_breakdown_rows,
    save_forward_inset_csv,
)
from block_prefix_analyzer.analysis.f13_strict import compute_f13_strict
from block_prefix_analyzer.types import RequestRecord


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rec(
    request_id: int,
    timestamp: float,
    block_ids: list[int],
    *,
    parent_chat_id: int = -1,
    req_type: str = "text",
) -> RequestRecord:
    return RequestRecord(
        request_id=str(request_id),
        timestamp=float(timestamp),
        arrival_index=request_id,
        block_ids=block_ids,
        metadata={"parent_chat_id": parent_chat_id, "type": req_type},
    )


# ---------------------------------------------------------------------------
# Test 1: Direction test — backward hit ≠ forward reusable
# ---------------------------------------------------------------------------

class TestDirectionDistinction:
    """Proves backward-looking and forward-looking give DIFFERENT sets of reusable requests."""

    def test_backward_and_forward_differ(self):
        """
        Trace:
          r1 t=0:  root, blocks=[1, 2]       ← first; no prior blocks
          r2 t=10: root, blocks=[3, 4]       ← sees nothing backward (pool has 1,2; no overlap)
          r3 t=20: root, blocks=[1, 5]       ← sees block 1 backward (from r1)

        Backward-looking reusable = {r3}      (r3 hit the pool)
        Forward-looking reusable  = {r1, r2}  (r1 reused by r3; r2 has no future reuser →
                                               actually just r1 has block 1 reused by r3)

        Wait, let me trace carefully:
        r1 blocks={1,2}: future roots are r2 (blocks={3,4}→no overlap) and r3 (blocks={1,5}→block 1 overlap). r1 IS forward-reusable.
        r2 blocks={3,4}: future roots are r3 (blocks={1,5}→no overlap). r2 is NOT forward-reusable.
        r3 blocks={1,5}: no future roots. r3 is NOT forward-reusable.

        Backward: {r3}   (only r3 has a backward hit)
        Forward:  {r1}   (only r1 has a future reuser)

        So the SETS are completely disjoint: r3 is backward-reusable but NOT forward-reusable.
        r1 is forward-reusable but NOT backward-reusable.
        """
        r1 = _rec(1, 0,  [1, 2])
        r2 = _rec(2, 10, [3, 4])
        r3 = _rec(3, 20, [1, 5])
        records = [r1, r2, r3]

        # Forward-looking
        fwd = compute_forward_inset(records)
        fwd_by_id = {r.request_id: r for r in fwd}
        assert fwd_by_id["1"].is_reusable_by_future_root is True   # r1 reused by r3
        assert fwd_by_id["2"].is_reusable_by_future_root is False  # r2 not reused
        assert fwd_by_id["3"].is_reusable_by_future_root is False  # r3 is last

        # Backward-looking (via compute_f13_strict)
        out = compute_f13_strict(records)
        # request_count_with_reuse is now forward-looking = 1 (only r1)
        assert out.series.request_count_with_reuse == 1
        # backward count: only r3 = 1
        assert out.backward_reusable_count == 1

        # The two reusable SETS are different: {r1} vs {r3}
        # (They happen to have the same SIZE but completely different members.)

    def test_request_backward_reusable_but_not_forward(self):
        """r3 is backward-reusable but NOT forward-reusable (no future root reuses it)."""
        r1 = _rec(1, 0,  [1, 2])
        r3 = _rec(3, 20, [1, 5])
        records = [r1, r3]

        fwd = compute_forward_inset(records)
        fwd_by_id = {r.request_id: r for r in fwd}
        # r3 arrived after r1 (backward-reusable for CDF) but has NO future root
        assert fwd_by_id["3"].is_reusable_by_future_root is False

    def test_request_forward_reusable_but_not_backward(self):
        """r1 is forward-reusable but NOT backward-reusable (it's the cold-start request)."""
        r1 = _rec(1, 0,  [1, 2])
        r2 = _rec(2, 10, [1, 5])
        records = [r1, r2]

        fwd = compute_forward_inset(records)
        fwd_by_id = {r.request_id: r for r in fwd}
        # r1 arrived first (no backward hit) but r2 reuses block 1 from r1
        assert fwd_by_id["1"].is_reusable_by_future_root is True


# ---------------------------------------------------------------------------
# Test 2: Future reuse detection
# ---------------------------------------------------------------------------

class TestFutureReuseDetection:
    def test_future_root_reusing_one_block_marks_source_reusable(self):
        r1 = _rec(1, 0,  [10, 20, 30])
        r2 = _rec(2, 10, [10, 40, 50])  # reuses block 10 from r1
        fwd = compute_forward_inset([r1, r2])
        by_id = {r.request_id: r for r in fwd}
        assert by_id["1"].is_reusable_by_future_root is True
        assert by_id["1"].reused_block_count == 1  # only block 10 is shared

    def test_future_followup_does_not_count(self):
        """A follow-up request reusing r1's block does NOT make r1 forward-reusable."""
        r1 = _rec(1, 0,  [10], parent_chat_id=-1)
        r2 = _rec(2, 10, [10], parent_chat_id=1)   # follow-up: NOT a root
        fwd = compute_forward_inset([r1, r2])
        by_id = {r.request_id: r for r in fwd}
        # r2 is excluded (not a root), so r1 has no future root reuser
        assert by_id["1"].is_reusable_by_future_root is False

    def test_first_reused_by_is_earliest_future_root(self):
        r1 = _rec(1, 0,  [7])
        r2 = _rec(2, 10, [7])   # first future reuser of r1
        r3 = _rec(3, 20, [7])   # second future reuser of r1
        fwd = compute_forward_inset([r1, r2, r3])
        by_id = {r.request_id: r for r in fwd}
        assert by_id["1"].first_reused_by_request_id == "2"
        assert by_id["1"].first_future_reuse_delay_seconds == pytest.approx(10.0)
        assert by_id["1"].num_future_reusers == 2


# ---------------------------------------------------------------------------
# Test 3: Denominator = all root requests
# ---------------------------------------------------------------------------

class TestDenominator:
    def test_denominator_is_all_root_requests(self):
        r1 = _rec(1, 0,  [1])   # cold-start: not reusable (no future root)
        r2 = _rec(2, 10, [2])   # no future root
        r3 = _rec(3, 20, [3])   # no future root
        fwd = compute_forward_inset([r1, r2, r3])
        breakdown = forward_inset_to_breakdown_rows(fwd, total_root_count=3)
        # All 3 requests are not reusable; all fractions should be 0
        assert all(row.fraction == 0.0 for row in breakdown)

    def test_denominator_includes_not_reusable_requests(self):
        r1 = _rec(1, 0,  [1])
        r2 = _rec(2, 10, [1])  # r1 is reusable by r2
        r3 = _rec(3, 20, [99]) # r2 is NOT reusable (r3 doesn't share blocks)
        fwd = compute_forward_inset([r1, r2, r3])
        breakdown = forward_inset_to_breakdown_rows(fwd, total_root_count=3)
        total_frac = sum(row.fraction for row in breakdown)
        # Only r1 is forward-reusable (1/3 ≈ 0.333)
        assert total_frac == pytest.approx(1 / 3)

    def test_denominator_from_compute_f13_strict(self):
        r1 = _rec(1, 0,  [1])
        r2 = _rec(2, 10, [1])
        out = compute_f13_strict([r1, r2])
        total = out.series.request_count_with_reuse + out.series.request_count_without_reuse
        assert total == out.series.single_turn_request_count == 2


# ---------------------------------------------------------------------------
# Test 4: Last request never forward-reusable
# ---------------------------------------------------------------------------

class TestLastRequestNotReusable:
    def test_last_root_in_trace_is_never_forward_reusable(self):
        r1 = _rec(1, 0,  [1, 2])
        r2 = _rec(2, 10, [3, 4])
        r3 = _rec(3, 20, [5, 6])  # last root, no future root
        fwd = compute_forward_inset([r1, r2, r3])
        by_id = {r.request_id: r for r in fwd}
        assert by_id["3"].is_reusable_by_future_root is False
        assert by_id["3"].num_future_reusers == 0

    def test_single_root_request_not_forward_reusable(self):
        r1 = _rec(1, 0, [1, 2, 3])
        fwd = compute_forward_inset([r1])
        assert len(fwd) == 1
        assert fwd[0].is_reusable_by_future_root is False


# ---------------------------------------------------------------------------
# Test 5: Multi-turn follow-ups excluded
# ---------------------------------------------------------------------------

class TestFollowUpsExcluded:
    def test_followups_not_in_forward_records(self):
        r1 = _rec(1, 0,  [1, 2], parent_chat_id=-1)
        r2 = _rec(2, 10, [1, 3], parent_chat_id=1)   # follow-up
        r3 = _rec(3, 20, [1, 4], parent_chat_id=-1)  # root
        fwd = compute_forward_inset([r1, r2, r3])
        ids = {r.request_id for r in fwd}
        assert "2" not in ids   # follow-up excluded
        assert ids == {"1", "3"}

    def test_followup_sharing_block_does_not_make_source_reusable(self):
        """A follow-up with block 1 (from r1) does NOT make r1 forward-reusable."""
        r1 = _rec(1, 0,  [1], parent_chat_id=-1)
        fu = _rec(2, 10, [1], parent_chat_id=1)   # follow-up, has block 1
        fwd = compute_forward_inset([r1, fu])
        by_id = {r.request_id: r for r in fwd}
        assert by_id["1"].is_reusable_by_future_root is False


# ---------------------------------------------------------------------------
# Test 6: Block-set overlap semantics (any shared block counts)
# ---------------------------------------------------------------------------

class TestBlockSetOverlap:
    def test_non_prefix_shared_block_counts(self):
        """block at position 2 of r2 matches block at position 0 of r1 — still counts."""
        r1 = _rec(1, 0,  [99])         # block 99 at pos 0
        r2 = _rec(2, 10, [11, 22, 99]) # block 99 at pos 2 (non-prefix)
        fwd = compute_forward_inset([r1, r2])
        by_id = {r.request_id: r for r in fwd}
        assert by_id["1"].is_reusable_by_future_root is True

    def test_no_shared_block_means_not_reusable(self):
        r1 = _rec(1, 0,  [1, 2, 3])
        r2 = _rec(2, 10, [4, 5, 6])
        fwd = compute_forward_inset([r1, r2])
        by_id = {r.request_id: r for r in fwd}
        assert by_id["1"].is_reusable_by_future_root is False


# ---------------------------------------------------------------------------
# Test 7: num_future_reusers
# ---------------------------------------------------------------------------

class TestNumFutureReusers:
    def test_multiple_future_reusers_counted(self):
        r1 = _rec(1, 0,  [42])
        r2 = _rec(2, 10, [42])
        r3 = _rec(3, 20, [42])
        r4 = _rec(4, 30, [42])
        fwd = compute_forward_inset([r1, r2, r3, r4])
        by_id = {r.request_id: r for r in fwd}
        assert by_id["1"].num_future_reusers == 3  # r2, r3, r4
        assert by_id["2"].num_future_reusers == 2  # r3, r4
        assert by_id["3"].num_future_reusers == 1  # r4
        assert by_id["4"].num_future_reusers == 0

    def test_distinct_reusers_counted_once_per_request(self):
        """r1 shares 2 blocks with r2; r2 should be counted once (not twice)."""
        r1 = _rec(1, 0,  [10, 20])
        r2 = _rec(2, 10, [10, 20, 30])  # shares both blocks 10 and 20 with r1
        fwd = compute_forward_inset([r1, r2])
        by_id = {r.request_id: r for r in fwd}
        assert by_id["1"].num_future_reusers == 1   # r2 counted once
        assert by_id["1"].reused_block_count == 2   # both blocks are reused


# ---------------------------------------------------------------------------
# Test 8: first_future_reuse_delay
# ---------------------------------------------------------------------------

class TestFirstFutureReuseDelay:
    def test_delay_is_timestamp_difference(self):
        r1 = _rec(1, 100.0, [5])
        r2 = _rec(2, 160.0, [5])   # delay = 60s
        fwd = compute_forward_inset([r1, r2])
        by_id = {r.request_id: r for r in fwd}
        assert by_id["1"].first_future_reuse_delay_seconds == pytest.approx(60.0)
        assert by_id["1"].first_reused_by_request_id == "2"

    def test_delay_is_minimum_across_blocks(self):
        """r1 has blocks [A, B]; r2 (t=10) reuses B; r3 (t=5) reuses A.
        first_future_reuse_delay should be 5 (r3 is earlier)."""
        r1 = _rec(1, 0,  [100, 200])
        r3 = _rec(3, 5,  [200, 999])  # reuses block 200 from r1, earlier than r2
        r2 = _rec(2, 10, [100, 888])  # reuses block 100 from r1, later than r3
        fwd = compute_forward_inset([r1, r2, r3])
        by_id = {r.request_id: r for r in fwd}
        assert by_id["1"].first_future_reuse_delay_seconds == pytest.approx(5.0)
        assert by_id["1"].first_reused_by_request_id == "3"

    def test_none_when_not_reusable(self):
        r1 = _rec(1, 0, [1])
        fwd = compute_forward_inset([r1])
        assert fwd[0].first_future_reuse_delay_seconds is None
        assert fwd[0].first_reused_by_request_id is None


# ---------------------------------------------------------------------------
# Test 9: CSV schema
# ---------------------------------------------------------------------------

class TestCsvSchema:
    EXPECTED_COLUMNS = [
        "request_id", "timestamp", "request_type", "display_label",
        "is_root_request", "is_reusable_by_future_root",
        "first_reused_by_request_id", "first_future_reuse_delay_seconds",
        "num_future_reusers", "reused_block_count", "reused_block_approx_tokens",
    ]

    def test_csv_header_columns(self, tmp_path):
        r1 = _rec(1, 0, [1])
        fwd = compute_forward_inset([r1])
        path = tmp_path / "fwd.csv"
        save_forward_inset_csv(fwd, path)
        with path.open() as f:
            header = f.readline().strip().split(",")
        assert header == self.EXPECTED_COLUMNS

    def test_csv_row_count_equals_root_count(self, tmp_path):
        records = [
            _rec(1, 0,  [1], parent_chat_id=-1),
            _rec(2, 10, [2], parent_chat_id=1),   # follow-up
            _rec(3, 20, [3], parent_chat_id=-1),
        ]
        fwd = compute_forward_inset(records)
        path = tmp_path / "fwd.csv"
        save_forward_inset_csv(fwd, path)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # only 2 root requests


# ---------------------------------------------------------------------------
# Test 10: image -> Multimedia
# ---------------------------------------------------------------------------

class TestTypeLabelMapping:
    def test_image_request_gets_multimedia_label(self):
        r1 = _rec(1, 0,  [1], req_type="image")
        r2 = _rec(2, 10, [1], req_type="image")
        fwd = compute_forward_inset([r1, r2])
        by_id = {r.request_id: r for r in fwd}
        assert by_id["1"].request_type == "image"
        assert by_id["1"].display_label == "Multimedia"

    def test_image_breakdown_row_has_multimedia_label(self):
        r1 = _rec(1, 0,  [1], req_type="image")
        r2 = _rec(2, 10, [1], req_type="image")
        fwd = compute_forward_inset([r1, r2])
        rows = forward_inset_to_breakdown_rows(fwd, total_root_count=2)
        image_rows = [row for row in rows if row.request_type == "image"]
        assert len(image_rows) == 1
        assert image_rows[0].display_label == "Multimedia"

    def test_forward_reusable_image_counted_in_breakdown(self):
        r1 = _rec(1, 0,  [77], req_type="image")
        r2 = _rec(2, 10, [77], req_type="image")  # r2 reuses r1's block
        fwd = compute_forward_inset([r1, r2])
        rows = forward_inset_to_breakdown_rows(fwd, total_root_count=2)
        image_rows = [row for row in rows if row.request_type == "image"]
        assert image_rows[0].count == 1        # only r1 is forward-reusable
        assert image_rows[0].fraction == pytest.approx(0.5)  # 1/2
