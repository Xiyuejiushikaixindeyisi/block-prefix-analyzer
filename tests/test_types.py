"""Tests for the canonical :class:`RequestRecord` and ordering helpers.

Covers:
* Correct field construction and independent mutable defaults.
* :func:`ordering_key` produces the canonical ``(timestamp, arrival_index)`` tuple.
* :func:`sort_records` sorts deterministically; ties broken by ``arrival_index``.
* Empty ``block_ids`` is valid (metrics denominator exclusion is enforced by
  the metrics module, not here — see test_metrics.py Step 5).
* After sorting, the first record is the one with the lowest
  ``(timestamp, arrival_index)``; it carries the "first request" semantics
  (prefix_hit == 0, included in denominator) that the replay/metrics modules
  depend on.
"""
from __future__ import annotations

from block_prefix_analyzer.types import RequestRecord, ordering_key, sort_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(
    request_id: str,
    timestamp: float,
    arrival_index: int,
    block_ids: list | None = None,
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        timestamp=timestamp,
        arrival_index=arrival_index,
        block_ids=block_ids if block_ids is not None else [1, 2, 3],
    )


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------

def test_defaults_are_none_or_empty() -> None:
    r = _make("r", 0.0, 0)
    assert r.token_count is None
    assert r.block_size is None
    assert r.metadata == {}


def test_metadata_default_is_independent_per_instance() -> None:
    a = _make("a", 0.0, 0)
    b = _make("b", 0.0, 1)
    a.metadata["x"] = 1
    assert b.metadata == {}


# ---------------------------------------------------------------------------
# ordering_key
# ---------------------------------------------------------------------------

def test_ordering_key_returns_tuple() -> None:
    r = _make("r", 1.5, 7)
    assert ordering_key(r) == (1.5, 7)


def test_ordering_key_different_timestamps() -> None:
    earlier = _make("a", 1.0, 0)
    later = _make("b", 2.0, 0)
    assert ordering_key(earlier) < ordering_key(later)


def test_ordering_key_same_timestamp_different_arrival() -> None:
    first = _make("a", 1.0, 0)
    second = _make("b", 1.0, 1)
    assert ordering_key(first) < ordering_key(second)


# ---------------------------------------------------------------------------
# sort_records
# ---------------------------------------------------------------------------

def test_sort_records_does_not_mutate_input() -> None:
    records = [_make("a", 2.0, 0), _make("b", 1.0, 0)]
    original_first_id = records[0].request_id
    sort_records(records)
    assert records[0].request_id == original_first_id


def test_sort_records_by_timestamp() -> None:
    records = [_make("a", 3.0, 0), _make("b", 1.0, 0), _make("c", 2.0, 0)]
    result = sort_records(records)
    assert [r.request_id for r in result] == ["b", "c", "a"]


def test_sort_records_tie_broken_by_arrival_index() -> None:
    # All share timestamp 1.0; ordering must follow arrival_index strictly.
    r0 = _make("first", 1.0, 0)
    r2 = _make("third", 1.0, 2)
    r1 = _make("second", 1.0, 1)

    result = sort_records([r2, r0, r1])
    assert [r.request_id for r in result] == ["first", "second", "third"]


def test_sort_records_mixed_timestamps_and_ties() -> None:
    # Realistic scenario: some requests share a second-level timestamp.
    same_ts_last = _make("b", 1.0, 99)
    same_ts_first = _make("c", 1.0, 5)
    same_ts_middle = _make("d", 1.0, 6)
    later = _make("a", 2.0, 0)

    result = sort_records([later, same_ts_last, same_ts_middle, same_ts_first])
    assert [r.request_id for r in result] == ["c", "d", "b", "a"]


def test_sort_records_single_element() -> None:
    r = _make("only", 0.0, 0)
    assert sort_records([r]) == [r]


def test_sort_records_empty_list() -> None:
    assert sort_records([]) == []


# ---------------------------------------------------------------------------
# Empty block_ids
# ---------------------------------------------------------------------------

def test_empty_block_ids_is_permitted() -> None:
    # V1 allows empty block_ids; the record is kept in the stream.
    # The metrics module decides denominator exclusion (see test_metrics.py).
    r = _make("empty", 0.0, 0, block_ids=[])
    assert r.block_ids == []


def test_empty_block_ids_record_participates_in_sort() -> None:
    # Records with empty block_ids sort normally; they are not dropped at load time.
    normal = _make("normal", 1.0, 0, block_ids=[1, 2])
    empty_blocks = _make("empty", 0.0, 0, block_ids=[])
    result = sort_records([normal, empty_blocks])
    assert result[0].request_id == "empty"
    assert result[1].request_id == "normal"


# ---------------------------------------------------------------------------
# First-record semantics (data-level contract)
# ---------------------------------------------------------------------------

def test_first_record_after_sort_is_lowest_key() -> None:
    # After sorting, index-0 is the record with the smallest
    # (timestamp, arrival_index).  The replay engine will produce
    # content_prefix_reuse_blocks == 0 for it (first request, no prior state).
    # This test pins the sort identity so replay tests can rely on it.
    a = _make("a", 5.0, 0)
    b = _make("b", 1.0, 10)   # lower timestamp → first
    c = _make("c", 1.0, 2)    # same timestamp, lower arrival_index → first

    result = sort_records([a, b, c])
    first = result[0]
    assert first.request_id == "c"
    # Downstream invariant (documented here, enforced in test_replay.py):
    # the replay engine must assign content_prefix_reuse_blocks == 0 to this record.
    assert ordering_key(first) == min(ordering_key(r) for r in [a, b, c])


def test_metadata_time_unit_advisory_field() -> None:
    # Loaders may store time_unit in metadata so derived metrics can
    # interpret duration values correctly.  The analyzer never reads this
    # field itself; this test just confirms the convention is storable.
    r = RequestRecord(
        request_id="r",
        timestamp=1_713_500_000,
        arrival_index=0,
        block_ids=[1],
        metadata={"time_unit": "s"},
    )
    assert r.metadata["time_unit"] == "s"
