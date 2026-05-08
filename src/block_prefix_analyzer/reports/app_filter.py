"""Filter a raw production ``requests.jsonl`` down to a single APP subset.

Used by the APP-level report builder (Dashboard Phase 2 Step 4) to feed a
per-APP subset into the existing F4 / F13 / common_prefix analysis
pipelines without modifying them. The ``user_id`` field in business-loader
JSONL is semantically the APP ID (per ``docs/dashboard_phase2_plan.md``
§2.2), so the filter is a simple equality match.

Public API
----------
``iter_filtered_records(path, app_id)``
    Yields parsed dict records whose ``user_id`` matches ``app_id``.
    Malformed lines, empty lines, and records lacking ``user_id`` are
    silently skipped — no exceptions for a bad source file.

``write_filtered_jsonl(src, dst, app_id)``
    Streams the matching subset of ``src`` into ``dst``, preserving each
    line's original bytes so downstream analysis tools see the exact
    formatting they expect. Returns :class:`FilterStats`.

``count_app_ids(path)``
    Single-pass diagnostic: returns ``Counter[str_user_id -> count]``
    for the whole file. Useful for "which APPs exist in this log" before
    invoking the per-APP report.

Skipping rules (frozen)
-----------------------
* Empty (whitespace-only) lines: skipped, not counted toward totals.
* Malformed JSON: skipped, counted in ``malformed_count``.
* Missing ``user_id`` (key absent or value ``None``/``""``): skipped,
  counted in ``missing_user_id_count``.

``write_filtered_jsonl`` and ``iter_filtered_records`` use the same
skipping rules so per-APP totals are consistent across the two entry
points.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilterStats:
    """Counts produced by :func:`write_filtered_jsonl`.

    Attributes
    ----------
    total_lines:
        Number of non-empty lines read from the source file.
    kept_count:
        Number of lines whose ``user_id`` matched and were written to ``dst``.
    malformed_count:
        Number of non-empty lines that failed ``json.loads``.
    missing_user_id_count:
        Number of parseable records lacking a usable ``user_id``
        (key absent, value ``None``, or empty string).
    """

    total_lines: int
    kept_count: int
    malformed_count: int
    missing_user_id_count: int


def _is_usable_uid(uid: object) -> bool:
    return uid is not None and uid != ""


def iter_filtered_records(
    path: Path | str, app_id: str, *, encoding: str = "utf-8"
) -> Iterator[dict]:
    """Yield parsed records from ``path`` whose ``user_id`` equals ``app_id``.

    Comparison is done via ``str()`` coercion on both sides, so an integer
    ``user_id`` in the source file matches a string ``app_id`` (and vice
    versa). Order is preserved.
    """
    path = Path(path)
    target = str(app_id)
    with path.open("r", encoding=encoding) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            uid = obj.get("user_id") if isinstance(obj, dict) else None
            if not _is_usable_uid(uid):
                continue
            if str(uid) == target:
                yield obj


def write_filtered_jsonl(
    src: Path | str,
    dst: Path | str,
    app_id: str,
    *,
    encoding: str = "utf-8",
) -> FilterStats:
    """Stream the matching subset of ``src`` into ``dst``.

    Original line bytes are preserved verbatim so downstream tools that
    read the file see the exact formatting (key order, whitespace, escape
    sequences) of the source. A trailing newline is appended only if the
    last copied line lacks one.

    Creates ``dst.parent`` if missing.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    target = str(app_id)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    total = kept = malformed = missing = 0
    with (
        src_path.open("r", encoding=encoding) as fin,
        dst_path.open("w", encoding=encoding) as fout,
    ):
        for raw in fin:
            stripped = raw.strip()
            if not stripped:
                continue
            total += 1
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                malformed += 1
                continue
            uid = obj.get("user_id") if isinstance(obj, dict) else None
            if not _is_usable_uid(uid):
                missing += 1
                continue
            if str(uid) == target:
                if not raw.endswith("\n"):
                    raw = raw + "\n"
                fout.write(raw)
                kept += 1
    return FilterStats(
        total_lines=total,
        kept_count=kept,
        malformed_count=malformed,
        missing_user_id_count=missing,
    )


def count_app_ids(
    path: Path | str, *, encoding: str = "utf-8"
) -> Counter[str]:
    """Return ``Counter`` mapping each present ``str(user_id)`` to its row count.

    Empty lines, malformed JSON, and records lacking a usable ``user_id``
    are skipped silently (matching :func:`iter_filtered_records`).
    """
    counts: Counter[str] = Counter()
    path = Path(path)
    with path.open("r", encoding=encoding) as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            uid = obj.get("user_id") if isinstance(obj, dict) else None
            if _is_usable_uid(uid):
                counts[str(uid)] += 1
    return counts
