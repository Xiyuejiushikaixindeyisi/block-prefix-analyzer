"""Session and category helpers for V2 request metadata.

Provides typed predicates and grouping utilities for the session/category
fields stored in ``RequestRecord.metadata`` by the V2 pipeline.  These are
the building blocks for per-session and per-category analytics (F13–F15).

Field conventions (set by V2 pipeline from RawRequest)
-------------------------------------------------------
``parent_request_id`` : str | None
    ID of the preceding turn's request.  ``None`` for root requests.
``session_id`` : str | None
    Opaque session identifier grouping related requests.
``category`` : str | None
    Category label forwarded directly from ``RawRequest.category``.
``type`` : str | None
    Request type (from ``RawRequest.metadata["type"]``).
``turn`` : int | None
    Conversation turn number, 1-indexed.

None of these fields are consumed by the V1 core (replay, metrics, reports).
"""
from __future__ import annotations

from collections import defaultdict

from block_prefix_analyzer.types import RequestRecord, ordering_key


def is_root_request(record: RequestRecord) -> bool:
    """Return ``True`` if this is the first turn of a session.

    A request is considered root when ``parent_request_id`` is absent
    **and** ``turn`` is absent or ≤ 1.  If ``parent_request_id`` is set,
    the request is always a followup regardless of turn number.
    """
    meta = record.metadata
    if meta.get("parent_request_id") is not None:
        return False
    turn = meta.get("turn")
    if turn is not None:
        return int(turn) <= 1
    return True


def is_followup_request(record: RequestRecord) -> bool:
    """Return ``True`` if this request continues an existing session."""
    return not is_root_request(record)


def get_category(record: RequestRecord) -> str | None:
    """Return the category label for a request.

    Lookup order:
    1. ``metadata["category"]`` — forwarded directly from ``RawRequest.category``.
    2. ``"{type}-{turn}"`` — constructed when both ``metadata["type"]`` and
       ``metadata["turn"]`` are present.
    3. ``None`` — if neither source is available.
    """
    meta = record.metadata
    if "category" in meta and meta["category"] is not None:
        return str(meta["category"])
    type_val = meta.get("type")
    turn_val = meta.get("turn")
    if type_val is not None and turn_val is not None:
        return f"{type_val}-{turn_val}"
    return None


def group_by_session(
    records: list[RequestRecord],
) -> dict[str, list[RequestRecord]]:
    """Group records by ``session_id``; unsessioned records go under key ``''``.

    Each group is sorted by the canonical ``(timestamp, arrival_index)`` key.

    Parameters
    ----------
    records:
        Unsorted list of :class:`~block_prefix_analyzer.types.RequestRecord`.

    Returns
    -------
    dict[str, list[RequestRecord]]
        Mapping of ``session_id → sorted list of records``.  The empty-string
        key ``''`` collects all records whose ``metadata["session_id"]`` is
        absent or ``None``.
    """
    groups: dict[str, list[RequestRecord]] = defaultdict(list)
    for record in records:
        sid = record.metadata.get("session_id") or ""
        groups[sid].append(record)
    return {k: sorted(v, key=ordering_key) for k, v in groups.items()}
