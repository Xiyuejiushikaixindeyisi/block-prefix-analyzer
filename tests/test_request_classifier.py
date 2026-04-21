"""Tests for analysis/request_classifier.py.

All fixtures are hand-constructed RequestRecord objects. No I/O, no network.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.analysis.request_classifier import (
    Classification,
    classify_requests,
    classification_summary,
    filter_single_turn,
)
from block_prefix_analyzer.types import RequestRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(
    request_id: str,
    *,
    timestamp: float = 0.0,
    block_ids: list[int] | None = None,
    rendered_prompt: str | None = None,
) -> RequestRecord:
    meta: dict = {}
    if rendered_prompt is not None:
        meta["v2_rendered_prompt"] = rendered_prompt
    return RequestRecord(
        request_id=request_id,
        timestamp=timestamp,
        arrival_index=0,
        block_ids=block_ids or [1, 2, 3],
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# classify_requests — primary signal (request_id repetition)
# ---------------------------------------------------------------------------

def test_unique_request_ids_are_single_turn() -> None:
    records = [_record("r1"), _record("r2"), _record("r3")]
    labels = classify_requests(records)
    assert labels == {"r1": "single_turn", "r2": "single_turn", "r3": "single_turn"}


def test_repeated_request_id_classified_agent_likely() -> None:
    records = [_record("r1"), _record("r1"), _record("r2")]
    labels = classify_requests(records)
    assert labels["r1"] == "agent_likely"
    assert labels["r2"] == "single_turn"


def test_all_repeated_all_agent_likely() -> None:
    records = [_record("r1"), _record("r1"), _record("r2"), _record("r2")]
    labels = classify_requests(records)
    assert all(v == "agent_likely" for v in labels.values())


def test_single_record_is_single_turn() -> None:
    labels = classify_requests([_record("solo")])
    assert labels["solo"] == "single_turn"


def test_empty_input_returns_empty_dict() -> None:
    assert classify_requests([]) == {}


# ---------------------------------------------------------------------------
# classify_requests — secondary signal (keyword detection)
# ---------------------------------------------------------------------------

def test_keyword_in_prompt_classified_agent_likely() -> None:
    records = [_record("r1", rendered_prompt="use tool_call to run bash")]
    labels = classify_requests(records)
    assert labels["r1"] == "agent_likely"


def test_keyword_detection_is_case_insensitive() -> None:
    records = [_record("r1", rendered_prompt="Invoke TOOL_CALL here")]
    labels = classify_requests(records)
    assert labels["r1"] == "agent_likely"


def test_no_keyword_no_promotion() -> None:
    records = [_record("r1", rendered_prompt="just a normal question")]
    labels = classify_requests(records)
    assert labels["r1"] == "single_turn"


def test_keyword_detection_skipped_when_no_rendered_prompt() -> None:
    records = [_record("r1")]  # no v2_rendered_prompt in metadata
    labels = classify_requests(records)
    assert labels["r1"] == "single_turn"


def test_empty_keywords_disables_keyword_detection() -> None:
    records = [_record("r1", rendered_prompt="uses tool_call heavily")]
    labels = classify_requests(records, agent_keywords=())
    assert labels["r1"] == "single_turn"


def test_primary_signal_overrides_keyword_check() -> None:
    # request_id repeated → agent_likely regardless of prompt content
    records = [
        _record("r1", rendered_prompt="no keywords here"),
        _record("r1", rendered_prompt="no keywords here"),
    ]
    labels = classify_requests(records)
    assert labels["r1"] == "agent_likely"


def test_custom_keywords_respected() -> None:
    records = [_record("r1", rendered_prompt="call mybot now")]
    labels = classify_requests(records, agent_keywords=("mybot",))
    assert labels["r1"] == "agent_likely"


# ---------------------------------------------------------------------------
# filter_single_turn
# ---------------------------------------------------------------------------

def test_filter_returns_only_single_turn_records() -> None:
    records = [
        _record("r1"),          # single_turn
        _record("r2"),          # single_turn
        _record("r3"),          # repeated → agent
        _record("r3"),          # repeated → agent
    ]
    filtered = filter_single_turn(records)
    ids = [r.request_id for r in filtered]
    assert "r1" in ids
    assert "r2" in ids
    assert "r3" not in ids


def test_filter_all_single_turn_unchanged() -> None:
    records = [_record("r1"), _record("r2"), _record("r3")]
    assert len(filter_single_turn(records)) == 3


def test_filter_all_agent_returns_empty() -> None:
    records = [_record("r1"), _record("r1")]
    assert filter_single_turn(records) == []


def test_filter_empty_input_returns_empty() -> None:
    assert filter_single_turn([]) == []


def test_filter_keyword_based_exclusion() -> None:
    records = [
        _record("r1", rendered_prompt="normal prompt"),
        _record("r2", rendered_prompt="invokes tool_call"),
    ]
    filtered = filter_single_turn(records)
    ids = [r.request_id for r in filtered]
    assert "r1" in ids
    assert "r2" not in ids


def test_filter_keyword_disabled_keeps_all() -> None:
    records = [
        _record("r1", rendered_prompt="normal prompt"),
        _record("r2", rendered_prompt="invokes tool_call"),
    ]
    filtered = filter_single_turn(records, agent_keywords=())
    assert len(filtered) == 2


# ---------------------------------------------------------------------------
# classification_summary
# ---------------------------------------------------------------------------

def test_summary_counts_all_records() -> None:
    records = [
        _record("r1"),
        _record("r2"),
        _record("r3"), _record("r3"),  # 2 agent_likely records
    ]
    summary = classification_summary(records)
    assert summary["single_turn"] == 2
    assert summary["agent_likely"] == 2


def test_summary_all_single_turn() -> None:
    records = [_record("r1"), _record("r2"), _record("r3")]
    summary = classification_summary(records)
    assert summary == {"single_turn": 3, "agent_likely": 0}


def test_summary_empty_input() -> None:
    summary = classification_summary([])
    assert summary == {"single_turn": 0, "agent_likely": 0}


# ---------------------------------------------------------------------------
# Phase-2 dataset assumption: all records should be single_turn
# ---------------------------------------------------------------------------

def test_phase2_dataset_no_agent_records() -> None:
    """Verifies Phase-2 assumption: pure single-turn dataset → zero agent_likely."""
    records = [
        _record("req-001", rendered_prompt="What is Python?"),
        _record("req-002", rendered_prompt="Explain transformers in NLP."),
        _record("req-003", rendered_prompt="How do I sort a list?"),
    ]
    summary = classification_summary(records)
    assert summary["agent_likely"] == 0
    assert summary["single_turn"] == len(records)
