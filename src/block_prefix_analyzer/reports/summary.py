"""Summary formatting and stable export helpers for replay metrics.

This module is a **pure output layer**: it only receives an already-computed
:class:`~block_prefix_analyzer.metrics.MetricsSummary` and renders it in
various formats.  It never runs replay, re-aggregates metrics, reads JSONL
files, or touches any prefix index.

Exported capabilities
---------------------
``format_summary``
    Render a :class:`MetricsSummary` as stable multi-line plain text.

``summary_to_dict``
    Convert a :class:`MetricsSummary` to a plain ``dict`` ready for
    ``json.dumps``.

``csv_header``
    Return the stable list of CSV column names (matches
    :class:`MetricsSummary` field order).

``summary_to_csv_row``
    Convert a :class:`MetricsSummary` to a flat list of values aligned with
    ``csv_header()``.

``write_text``, ``write_json``, ``write_csv``
    Convenience helpers that write one of the above representations to a
    file using only the standard library.
"""
from __future__ import annotations

import csv as _csv
import dataclasses
import json
from pathlib import Path

from ..metrics import MetricsSummary

# ---------------------------------------------------------------------------
# Internal format constants
# ---------------------------------------------------------------------------

_LABEL_WIDTH = 36
_INT_VALUE_WIDTH = 12
_FLOAT_VALUE_WIDTH = 11   # numeric part only; "%" appended separately
_SEP_WIDTH = _LABEL_WIDTH + 2 + _FLOAT_VALUE_WIDTH + 1  # = 50

_HEADER_LINE = "Block Prefix Analyzer \u2014 Replay Summary"
_SEP_LINE = "=" * _SEP_WIDTH


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def format_summary(summary: MetricsSummary) -> str:
    """Render ``summary`` as stable multi-line plain text.

    Output format
    -------------
    * Line 1: fixed title
    * Line 2: ``=`` separator (50 characters)
    * Lines 3-8: integer fields — label left-aligned in 36 chars, value
      right-aligned in 12 chars.
    * Lines 9-10: ratio fields — value multiplied by 100 and formatted as
      ``%.4f%`` in an 11-char numeric field.

    The field order is fixed and matches :class:`MetricsSummary` declaration
    order:

    1. ``request_count``
    2. ``non_empty_request_count``
    3. ``cold_start_request_count``
    4. ``total_blocks``
    5. ``total_prefix_hit_blocks``
    6. ``total_reusable_blocks``
    7. ``overall_prefix_hit_rate``
    8. ``overall_block_level_reusable_ratio``

    The returned string does **not** end with a trailing newline.
    """
    w = _LABEL_WIDTH

    def _int_line(label: str, value: int) -> str:
        return f"{label:<{w}}: {value:>{_INT_VALUE_WIDTH}}"

    def _pct_line(label: str, value: float) -> str:
        return f"{label:<{w}}: {value * 100:>{_FLOAT_VALUE_WIDTH}.4f}%"

    lines = [
        _HEADER_LINE,
        _SEP_LINE,
        _int_line("request_count",                    summary.request_count),
        _int_line("non_empty_request_count",           summary.non_empty_request_count),
        _int_line("cold_start_request_count",          summary.cold_start_request_count),
        _int_line("total_blocks",                      summary.total_blocks),
        _int_line("total_prefix_hit_blocks",           summary.total_prefix_hit_blocks),
        _int_line("total_reusable_blocks",             summary.total_reusable_blocks),
        _pct_line("overall_prefix_hit_rate",           summary.overall_prefix_hit_rate),
        _pct_line("overall_block_level_reusable_ratio", summary.overall_block_level_reusable_ratio),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def summary_to_dict(summary: MetricsSummary) -> dict[str, int | float]:
    """Convert ``summary`` to a plain ``dict`` suitable for JSON serialisation.

    Keys are the :class:`MetricsSummary` field names (snake_case).  Values
    are the raw Python ``int`` / ``float`` values — ratios are *not*
    converted to percentages.  The input object is never mutated.
    """
    return dataclasses.asdict(summary)


def write_json(summary: MetricsSummary, path: str | Path) -> None:
    """Write ``summary`` as a JSON object to ``path`` (overwrites if exists).

    The output is indented with 2 spaces and ends with a newline.
    """
    data = summary_to_dict(summary)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def csv_header() -> list[str]:
    """Return the stable list of CSV column names.

    The order matches the :class:`MetricsSummary` field declaration order
    exactly and will not change within V1.  Use this together with
    :func:`summary_to_csv_row` to build multi-row CSV files.
    """
    return [f.name for f in dataclasses.fields(MetricsSummary)]


def summary_to_csv_row(summary: MetricsSummary) -> list[int | float]:
    """Return ``summary`` values as a flat list aligned with :func:`csv_header`.

    Ratios are raw floats (not converted to percentages).  The list length
    always equals ``len(csv_header())``.
    """
    return [getattr(summary, f.name) for f in dataclasses.fields(MetricsSummary)]


def write_csv(summary: MetricsSummary, path: str | Path) -> None:
    """Write ``summary`` as a CSV file with a header row (overwrites if exists).

    The output contains exactly two rows: the header row and the data row.
    Line endings follow the platform default for CSV (``newline=""`` is used
    as recommended by :mod:`csv`).
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        writer.writerow(csv_header())
        writer.writerow(summary_to_csv_row(summary))


# ---------------------------------------------------------------------------
# Text file helper
# ---------------------------------------------------------------------------

def write_text(summary: MetricsSummary, path: str | Path) -> None:
    """Write the formatted text summary to ``path`` (overwrites if exists).

    Appends a trailing newline after the last line.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(format_summary(summary))
        fh.write("\n")
