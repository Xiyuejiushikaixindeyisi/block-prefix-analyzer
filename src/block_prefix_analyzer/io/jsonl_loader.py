"""JSONL trace loader.

Reads a JSONL file (one JSON object per line), validates each line, and
returns a list of :class:`~block_prefix_analyzer.types.RequestRecord` sorted
by the canonical ``(timestamp, arrival_index)`` order.

Frozen V1 schema (per line)
----------------------------
Required fields:

* ``request_id`` — any JSON scalar; normalised to ``str``.
* ``timestamp``  — JSON number (``int`` or ``float``); ``bool`` is rejected.
* ``block_ids``  — JSON array whose elements are all ``int`` or ``str``;
  ``bool`` elements are rejected.

Optional fields:

* ``metadata``    — JSON object; missing value defaults to ``{}``.
  ``metadata["time_unit"]`` is preserved unchanged if present.
* ``token_count`` — JSON integer or ``null``/missing; ``bool`` is rejected.
* ``block_size``  — JSON integer or ``null``/missing; ``bool`` is rejected.

``arrival_index`` assignment
-----------------------------
``arrival_index`` is **always** assigned by this loader in non-blank-line read
order (0-indexed), regardless of any ``arrival_index`` field present in the
input.  Any input value is silently overridden.  This guarantees global
uniqueness and compatibility with :func:`~block_prefix_analyzer.types.sort_records`.

Error handling
--------------
Any invalid line raises :class:`LoadError` (a ``ValueError`` subclass) with a
message that includes the **1-based line number** and the reason for failure.
Lines with invalid JSON, missing required fields, or wrong types all raise
immediately — there is no silent-skip mode.

Blank lines are skipped without consuming an ``arrival_index`` slot.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..types import BlockId, RequestRecord, sort_records


class LoadError(ValueError):
    """Raised when a JSONL line fails validation.

    Always includes the 1-based line number and a human-readable reason so
    callers can surface actionable diagnostics without re-reading the file.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[RequestRecord]:
    """Load and validate a JSONL trace file.

    Returns a list of :class:`RequestRecord` sorted by
    ``(timestamp, arrival_index)`` (canonical chronological order).

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file.  Each non-blank line must be a JSON
        object conforming to the V1 schema described in this module's
        docstring.

    Raises
    ------
    LoadError
        On the first line that fails JSON parsing, has a missing required
        field, or has a field with an invalid type.
    OSError
        If the file cannot be opened.
    """
    records: list[RequestRecord] = []
    arrival_idx = 0

    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue  # blank lines skipped; arrival_idx not incremented

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise LoadError(
                    f"line {lineno}: invalid JSON — {exc.msg} "
                    f"(col {exc.colno})"
                ) from exc

            record = _parse_row(row, lineno=lineno, arrival_index=arrival_idx)
            records.append(record)
            arrival_idx += 1

    return sort_records(records)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_row(row: object, *, lineno: int, arrival_index: int) -> RequestRecord:
    """Validate ``row`` and build a :class:`RequestRecord`.

    ``lineno`` is 1-based and used only for error messages.
    ``arrival_index`` is assigned by the caller (``load_jsonl``).
    """
    if not isinstance(row, dict):
        raise LoadError(
            f"line {lineno}: expected a JSON object, "
            f"got {type(row).__name__!r}"
        )

    # --- request_id (required; any scalar → str) --------------------------
    _require(row, "request_id", lineno)
    request_id = str(row["request_id"])

    # --- timestamp (required; number, not bool) ---------------------------
    _require(row, "timestamp", lineno)
    ts = row["timestamp"]
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        raise LoadError(
            f"line {lineno}: 'timestamp' must be a number, "
            f"got {type(ts).__name__!r}"
        )

    # --- block_ids (required; list of int|str, no bool, no float) ---------
    _require(row, "block_ids", lineno)
    raw_ids = row["block_ids"]
    if not isinstance(raw_ids, list):
        raise LoadError(
            f"line {lineno}: 'block_ids' must be a JSON array, "
            f"got {type(raw_ids).__name__!r}"
        )
    block_ids: list[BlockId] = []
    for i, bid in enumerate(raw_ids):
        if isinstance(bid, bool) or not isinstance(bid, (int, str)):
            raise LoadError(
                f"line {lineno}: 'block_ids[{i}]' must be int or str, "
                f"got {type(bid).__name__!r}"
            )
        block_ids.append(bid)

    # --- token_count (optional; int or null) ------------------------------
    token_count: int | None = None
    if "token_count" in row:
        tc = row["token_count"]
        if tc is not None:
            if isinstance(tc, bool) or not isinstance(tc, int):
                raise LoadError(
                    f"line {lineno}: 'token_count' must be an integer or null, "
                    f"got {type(tc).__name__!r}"
                )
            token_count = tc

    # --- block_size (optional; int or null) -------------------------------
    block_size: int | None = None
    if "block_size" in row:
        bs = row["block_size"]
        if bs is not None:
            if isinstance(bs, bool) or not isinstance(bs, int):
                raise LoadError(
                    f"line {lineno}: 'block_size' must be an integer or null, "
                    f"got {type(bs).__name__!r}"
                )
            block_size = bs

    # --- metadata (optional; JSON object) ---------------------------------
    metadata: dict = {}
    if "metadata" in row:
        md = row["metadata"]
        if not isinstance(md, dict):
            raise LoadError(
                f"line {lineno}: 'metadata' must be a JSON object, "
                f"got {type(md).__name__!r}"
            )
        metadata = md  # preserve as-is; time_unit and other keys kept intact

    return RequestRecord(
        request_id=request_id,
        timestamp=ts,
        arrival_index=arrival_index,
        block_ids=block_ids,
        token_count=token_count,
        block_size=block_size,
        metadata=metadata,
    )


def _require(row: dict, field: str, lineno: int) -> None:
    if field not in row:
        raise LoadError(
            f"line {lineno}: missing required field {field!r}"
        )
