"""Tests for V2 session and category helpers.

Coverage:
  is_root_request  — no parent + turn ≤ 1 / ≥ 2
  is_followup_request — inverse of is_root
  get_category     — direct, constructed, None
  group_by_session — grouping, sorting, unsessioned key
"""
from __future__ import annotations

from block_prefix_analyzer.types import RequestRecord
from block_prefix_analyzer.v2.session import (
    get_category,
    group_by_session,
    is_followup_request,
    is_root_request,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(
    rid: str,
    ts: float = 0.0,
    arrival_index: int = 0,
    **meta_kwargs,
) -> RequestRecord:
    return RequestRecord(
        request_id=rid,
        timestamp=ts,
        arrival_index=arrival_index,
        block_ids=[],
        metadata={k: v for k, v in meta_kwargs.items() if v is not None},
    )


# ---------------------------------------------------------------------------
# is_root_request
# ---------------------------------------------------------------------------

def test_root_no_parent_no_turn() -> None:
    assert is_root_request(_rec("r1")) is True


def test_root_no_parent_turn_1() -> None:
    assert is_root_request(_rec("r1", turn=1)) is True


def test_root_no_parent_turn_0() -> None:
    assert is_root_request(_rec("r1", turn=0)) is True


def test_not_root_when_parent_set() -> None:
    assert is_root_request(_rec("r1", parent_request_id="r0")) is False


def test_not_root_turn_2() -> None:
    assert is_root_request(_rec("r1", turn=2)) is False


def test_not_root_turn_2_even_without_parent() -> None:
    r = _rec("r2", turn=2)
    assert is_root_request(r) is False


def test_parent_overrides_turn_1() -> None:
    """If parent is set, it's always a followup regardless of turn."""
    r = _rec("r1", parent_request_id="r0", turn=1)
    assert is_root_request(r) is False


# ---------------------------------------------------------------------------
# is_followup_request
# ---------------------------------------------------------------------------

def test_followup_inverse_of_root() -> None:
    root = _rec("r1")
    followup = _rec("r2", parent_request_id="r1", turn=2)
    assert is_followup_request(root) is False
    assert is_followup_request(followup) is True


# ---------------------------------------------------------------------------
# get_category
# ---------------------------------------------------------------------------

def test_get_category_from_metadata() -> None:
    r = _rec("r1", category="text-1")
    assert get_category(r) == "text-1"


def test_get_category_constructed_from_type_turn() -> None:
    r = RecordWithMeta("r1", type_val="search", turn=3)
    assert get_category(r) == "search-3"


def test_get_category_none_when_no_fields() -> None:
    r = _rec("r1")
    assert get_category(r) is None


def test_get_category_category_takes_priority() -> None:
    """metadata['category'] wins over type+turn construction."""
    r = RecordWithMeta("r1", category_val="explicit", type_val="x", turn=1)
    assert get_category(r) == "explicit"


def test_get_category_none_when_only_type() -> None:
    r = RecordWithMeta("r1", type_val="search")
    assert get_category(r) is None


def test_get_category_none_when_only_turn() -> None:
    r = RecordWithMeta("r1", turn=2)
    assert get_category(r) is None


# ---------------------------------------------------------------------------
# group_by_session
# ---------------------------------------------------------------------------

def test_group_by_session_basic() -> None:
    r1 = _rec("r1", ts=0.0, arrival_index=0, session_id="s1")
    r2 = _rec("r2", ts=1.0, arrival_index=1, session_id="s1")
    r3 = _rec("r3", ts=2.0, arrival_index=2, session_id="s2")
    groups = group_by_session([r1, r2, r3])
    assert set(groups.keys()) == {"s1", "s2"}
    assert len(groups["s1"]) == 2
    assert len(groups["s2"]) == 1


def test_group_by_session_unsessioned_key_empty_string() -> None:
    r = _rec("r1")  # no session_id in metadata
    groups = group_by_session([r])
    assert "" in groups
    assert groups[""][0].request_id == "r1"


def test_group_by_session_sorted_by_timestamp() -> None:
    r1 = _rec("r1", ts=5.0, arrival_index=0, session_id="s1")
    r2 = _rec("r2", ts=1.0, arrival_index=1, session_id="s1")
    groups = group_by_session([r1, r2])
    assert groups["s1"][0].timestamp == 1.0
    assert groups["s1"][1].timestamp == 5.0


def test_group_by_session_empty_input() -> None:
    groups = group_by_session([])
    assert groups == {}


def test_group_by_session_all_same_session() -> None:
    records = [_rec(f"r{i}", ts=float(i), arrival_index=i, session_id="only") for i in range(5)]
    groups = group_by_session(records)
    assert list(groups.keys()) == ["only"]
    assert len(groups["only"]) == 5


# ---------------------------------------------------------------------------
# Helper: record with mixed metadata fields
# ---------------------------------------------------------------------------

class RecordWithMeta(RequestRecord):
    def __init__(
        self,
        rid: str,
        *,
        type_val: str | None = None,
        turn: int | None = None,
        category_val: str | None = None,
    ) -> None:
        meta: dict = {}
        if type_val is not None:
            meta["type"] = type_val
        if turn is not None:
            meta["turn"] = turn
        if category_val is not None:
            meta["category"] = category_val
        super().__init__(
            request_id=rid,
            timestamp=0.0,
            arrival_index=0,
            block_ids=[],
            metadata=meta,
        )
