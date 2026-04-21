"""Request classifier for business dataset diagnostics.

Classifies each :class:`~block_prefix_analyzer.types.RequestRecord` as
``"single_turn"`` or ``"agent_likely"`` using two signals (checked in order):

1. **request_id repetition** (primary, reliable):
   A ``request_id`` that appears more than once in the input set is classified
   ``"agent_likely"`` — repeated IDs indicate an Agent session where multiple
   LLM calls share the same top-level request identifier.

2. **Keyword detection in rendered prompt** (secondary, supplementary):
   If ``metadata["v2_rendered_prompt"]`` contains a known Agent-pattern
   keyword (case-insensitive), the record is classified ``"agent_likely"``
   even when its ``request_id`` is unique. Requires the loader to have been
   called with ``include_debug_metadata=True``.

Usage note
----------
This module is a **diagnostic / validation tool**; it is NOT called internally
by :func:`~block_prefix_analyzer.io.business_loader.load_business_jsonl`.
Use it after loading to verify that the dataset matches expectations::

    records = load_business_jsonl(path, block_size=128,
                                  include_debug_metadata=True)
    summary = classification_summary(records)
    # Expected for Phase-2 dataset: {"single_turn": N, "agent_likely": 0}

Single-turn / Agent splitting (Phase 3)
----------------------------------------
:func:`filter_single_turn` is the planned hook for Phase-3 filtering.
The current Phase-2 business dataset contains only single-turn text requests;
``filter_single_turn`` should return all records unchanged for this dataset.
"""
from __future__ import annotations

from collections import Counter
from typing import Literal, Sequence

from block_prefix_analyzer.types import RequestRecord

Classification = Literal["single_turn", "agent_likely"]

_DEFAULT_AGENT_KEYWORDS: tuple[str, ...] = (
    "tool_call",
    "tool_result",
    "opencode",
    "claudecode",
    "<tool_use>",
    "<function_call>",
    "<tool_response>",
)


def classify_requests(
    records: Sequence[RequestRecord],
    *,
    agent_keywords: Sequence[str] = _DEFAULT_AGENT_KEYWORDS,
) -> dict[str, Classification]:
    """Return a ``{request_id: classification}`` mapping for every record.

    Parameters
    ----------
    records:
        Output of :func:`~block_prefix_analyzer.io.business_loader.load_business_jsonl`
        or any list of :class:`~block_prefix_analyzer.types.RequestRecord`.
    agent_keywords:
        Strings whose presence in ``metadata["v2_rendered_prompt"]``
        (case-insensitive) triggers ``"agent_likely"``.  Pass ``()`` to
        disable keyword detection (e.g. when ``include_debug_metadata=False``).

    Returns
    -------
    dict[str, Classification]
        All ``request_id`` values present in *records* are included.
        When a ``request_id`` appears multiple times all occurrences
        receive the same label.
    """
    # Primary signal: any request_id appearing > 1 time → agent session
    id_counts: Counter[str] = Counter(r.request_id for r in records)
    repeated_ids: frozenset[str] = frozenset(
        rid for rid, cnt in id_counts.items() if cnt > 1
    )

    lower_keywords = tuple(kw.lower() for kw in agent_keywords)

    result: dict[str, Classification] = {}
    for record in records:
        rid = record.request_id

        # Primary signal wins immediately
        if rid in repeated_ids:
            result[rid] = "agent_likely"
            continue

        # Secondary signal: keyword scan (only when rendered prompt available)
        if rid not in result:
            rendered: str = record.metadata.get("v2_rendered_prompt", "")
            if lower_keywords and rendered and any(
                kw in rendered.lower() for kw in lower_keywords
            ):
                result[rid] = "agent_likely"
            else:
                result[rid] = "single_turn"

    return result


def filter_single_turn(
    records: Sequence[RequestRecord],
    *,
    agent_keywords: Sequence[str] = _DEFAULT_AGENT_KEYWORDS,
) -> list[RequestRecord]:
    """Return only records classified as ``"single_turn"``.

    Convenience wrapper around :func:`classify_requests`.

    Parameters
    ----------
    records:
        Input records (typically all records from the loader).
    agent_keywords:
        Forwarded to :func:`classify_requests`. Pass ``()`` to disable
        keyword detection when ``include_debug_metadata=False`` was used.
    """
    labels = classify_requests(records, agent_keywords=agent_keywords)
    return [r for r in records if labels.get(r.request_id) == "single_turn"]


def classification_summary(
    records: Sequence[RequestRecord],
    *,
    agent_keywords: Sequence[str] = _DEFAULT_AGENT_KEYWORDS,
) -> dict[str, int]:
    """Return per-label record counts for a quick sanity check.

    Example output::

        {"single_turn": 45230, "agent_likely": 0}

    Each *record* (not unique request_id) is counted once, so a repeated
    ``request_id`` with 50 occurrences contributes 50 to ``"agent_likely"``.
    """
    labels = classify_requests(records, agent_keywords=agent_keywords)
    counts: dict[str, int] = {"single_turn": 0, "agent_likely": 0}
    for record in records:
        label = labels.get(record.request_id, "single_turn")
        counts[label] += 1
    return counts
