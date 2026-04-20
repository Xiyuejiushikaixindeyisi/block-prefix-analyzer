"""End-to-end golden tests for the V1 replay pipeline.

These tests exercise the full main chain:

    load_jsonl → replay → compute_metrics → MetricsSummary

Each test loads a small crafted JSONL fixture, runs the complete pipeline,
and compares the output against a stored golden file.

Golden file format
------------------
Each ``.golden.json`` contains two sections:

* ``replay_rows``: the list of per-request replay results (``PerRequestResult``
  fields as a dict), in canonical sort order.
* ``metrics``: the ``MetricsSummary`` dict produced by ``compute_metrics``.

Updating goldens
----------------
When a behaviour change is intentional, regenerate the goldens with::

    UPDATE_GOLDENS=1 pytest tests/test_golden.py -v

This overwrites all golden files and then passes.  Review the diff before
committing — every golden change is a semantic change.

Fixtures
--------
``minimal.jsonl``
    Three requests; the second fully matches the first; the third forks at
    the last block.  Tests linear prefix accumulation.

``forking.jsonl``
    Four requests sharing a common system-prompt prefix ([10, 20]), then
    diverging.  Tests multi-branch trie state and incremental prefix growth.

``with_empty.jsonl``
    Three requests: one normal, one with empty ``block_ids`` (excluded from
    denominator), one that has zero prefix hits but two reusable positions
    (tests the semantic gap between prefix_hit and content_reused_blocks_anywhere).
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from block_prefix_analyzer.io.jsonl_loader import load_jsonl
from block_prefix_analyzer.metrics import compute_metrics
from block_prefix_analyzer.replay import replay

_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Golden test infrastructure
# ---------------------------------------------------------------------------

def _pipeline(fixture_name: str) -> dict:
    """Run the full V1 pipeline on a fixture and return a comparable dict."""
    records = load_jsonl(_FIXTURES / fixture_name)
    rows = list(replay(records))
    metrics = compute_metrics(rows)
    return {
        "replay_rows": [dataclasses.asdict(r) for r in rows],
        "metrics": dataclasses.asdict(metrics),
    }


def _check_or_update(actual: dict, golden_path: Path) -> None:
    """Compare ``actual`` against the golden file, or update the file.

    Set the environment variable ``UPDATE_GOLDENS=1`` to regenerate the
    golden file instead of comparing.  The test passes either way so that
    a single ``UPDATE_GOLDENS=1 pytest`` run can refresh all goldens at once.
    """
    if os.environ.get("UPDATE_GOLDENS") == "1":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(
            json.dumps(actual, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return  # golden updated; test passes

    if not golden_path.exists():
        pytest.fail(
            f"Golden file not found: {golden_path}\n"
            f"Generate it with: UPDATE_GOLDENS=1 pytest {__file__}"
        )

    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    assert actual == expected, (
        f"Golden mismatch for '{golden_path.name}'.\n"
        f"If this change is intentional, run:\n"
        f"    UPDATE_GOLDENS=1 pytest {__file__}\n"
        f"then review and commit the updated golden file."
    )


# ---------------------------------------------------------------------------
# Fixture 1: minimal
#
# Three requests; second fully hits first; third forks at last block.
#
# Expected replay:
#   req-1: total=3, prefix_hit=0, reusable=0  (cold start)
#   req-2: total=3, prefix_hit=3, reusable=3  (full match)
#   req-3: total=3, prefix_hit=2, reusable=2  (fork at position 2)
#
# Expected MetricsSummary:
#   total_blocks=9, prefix_hit=5, reusable=5, rate=5/9, reusable_ratio=5/9
# ---------------------------------------------------------------------------

def test_golden_minimal() -> None:
    actual = _pipeline("minimal.jsonl")
    _check_or_update(actual, _FIXTURES / "minimal.golden.json")


# ---------------------------------------------------------------------------
# Fixture 2: forking
#
# Four requests sharing [10, 20] then diverging.
#
# Expected replay:
#   sys:    total=3, prefix_hit=0, reusable=0  (cold start)
#   user-a: total=3, prefix_hit=2, reusable=2  ([10,20] match; 41 unseen)
#   user-b: total=3, prefix_hit=2, reusable=2  ([10,20] match; 42 unseen)
#   combo:  total=4, prefix_hit=3, reusable=3  ([10,20,30] match; 50 unseen)
#
# Expected MetricsSummary:
#   total_blocks=13, prefix_hit=7, reusable=7, rate=7/13, reusable_ratio=7/13
# ---------------------------------------------------------------------------

def test_golden_forking() -> None:
    actual = _pipeline("forking.jsonl")
    _check_or_update(actual, _FIXTURES / "forking.golden.json")


# ---------------------------------------------------------------------------
# Fixture 3: with_empty
#
# sys=[1,2,3]; heartbeat=[]; user=[9,2,3] (prefix miss but reusable positions)
#
# Expected replay:
#   sys:       total=3, prefix_hit=0, reusable=0  (cold start)
#   heartbeat: total=0, prefix_hit=0, reusable=0  (empty, excluded from denom)
#   user:      total=3, prefix_hit=0, reusable=2  (9 unseen; 2,3 seen)
#
# Expected MetricsSummary:
#   total_blocks=6 (heartbeat excluded), prefix_hit=0, reusable=2
#   rate=0.0, reusable_ratio=2/6=1/3
# ---------------------------------------------------------------------------

def test_golden_with_empty() -> None:
    actual = _pipeline("with_empty.jsonl")
    _check_or_update(actual, _FIXTURES / "with_empty.golden.json")


# ---------------------------------------------------------------------------
# Additional semantic assertions on golden data
#
# These tests read the stored goldens and verify specific invariants so that
# a golden diff is immediately interpretable without mentally running the pipeline.
# They fail only after the golden files exist.
# ---------------------------------------------------------------------------

def _load_golden(name: str) -> dict:
    p = _FIXTURES / name
    if not p.exists():
        pytest.skip(f"Golden not yet generated: {name}")
    return json.loads(p.read_text(encoding="utf-8"))


def test_golden_minimal_first_row_is_cold_start() -> None:
    g = _load_golden("minimal.golden.json")
    first = g["replay_rows"][0]
    assert first["content_prefix_reuse_blocks"] == 0, "first request must be a cold start"
    assert first["content_reused_blocks_anywhere"] == 0, "first request has no reusable blocks"


def test_golden_minimal_second_row_full_match() -> None:
    g = _load_golden("minimal.golden.json")
    second = g["replay_rows"][1]
    assert second["content_prefix_reuse_blocks"] == second["total_blocks"]


def test_golden_minimal_third_row_partial_match() -> None:
    g = _load_golden("minimal.golden.json")
    third = g["replay_rows"][2]
    assert 0 < third["content_prefix_reuse_blocks"] < third["total_blocks"]


def test_golden_minimal_metrics_consistency() -> None:
    g = _load_golden("minimal.golden.json")
    m = g["metrics"]
    total_hit = sum(r["content_prefix_reuse_blocks"] for r in g["replay_rows"])
    assert m["total_content_prefix_reuse_blocks"] == total_hit
    denom = sum(r["total_blocks"] for r in g["replay_rows"] if r["total_blocks"] > 0)
    assert m["total_blocks"] == denom
    assert abs(m["content_prefix_reuse_rate"] - total_hit / denom) < 1e-12


def test_golden_forking_shared_prefix_both_users_hit_two() -> None:
    g = _load_golden("forking.golden.json")
    rows = {r["request_id"]: r for r in g["replay_rows"]}
    assert rows["user-a"]["content_prefix_reuse_blocks"] == 2
    assert rows["user-b"]["content_prefix_reuse_blocks"] == 2


def test_golden_forking_combo_hits_three() -> None:
    g = _load_golden("forking.golden.json")
    rows = {r["request_id"]: r for r in g["replay_rows"]}
    assert rows["combo"]["content_prefix_reuse_blocks"] == 3


def test_golden_with_empty_denominator_excludes_heartbeat() -> None:
    g = _load_golden("with_empty.golden.json")
    m = g["metrics"]
    # heartbeat has total_blocks=0 → excluded; denom = sys(3) + user(3) = 6
    assert m["total_blocks"] == 6
    assert m["non_empty_request_count"] == 2
    assert m["request_count"] == 3


def test_golden_with_empty_reusable_exceeds_prefix_hit() -> None:
    g = _load_golden("with_empty.golden.json")
    rows = {r["request_id"]: r for r in g["replay_rows"]}
    user = rows["user"]
    # user starts with unseen block 9 → no prefix hit; but 2,3 are reusable
    assert user["content_prefix_reuse_blocks"] == 0
    assert user["content_reused_blocks_anywhere"] == 2
    assert user["content_reused_blocks_anywhere"] > user["content_prefix_reuse_blocks"]


def test_golden_with_empty_content_prefix_reuse_rate_is_zero() -> None:
    g = _load_golden("with_empty.golden.json")
    assert g["metrics"]["content_prefix_reuse_rate"] == 0.0


def test_golden_with_empty_reusable_ratio_is_one_third() -> None:
    g = _load_golden("with_empty.golden.json")
    ratio = g["metrics"]["content_block_reuse_ratio"]
    assert abs(ratio - 1 / 3) < 1e-12
