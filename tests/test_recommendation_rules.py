"""Tests for :mod:`block_prefix_analyzer.recommendation`.

Each rule has a paired (fire-positive, fire-negative) golden test plus a few
engine-level tests covering the registry, sort order, and end-to-end.

The fixture builders below mint minimal report dicts that satisfy *only*
the fields each rule reads. This makes failures localised — a broken rule
shows up in its own test rather than in a shared mega-fixture.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.recommendation import (
    Recommendation,
    get_registry,
    run_all_rules,
    sort_recommendations,
)
from block_prefix_analyzer.recommendation.rules import (
    rule_batch_ttl,
    rule_cache_pressure,
    rule_capacity_first,
    rule_long_ttl,
    rule_low_ceiling,
    rule_multi_tenant,
    rule_pin_chain,
    rule_warn_reuse_zero,
    rule_warn_same_second,
)


# ---------------------------------------------------------------------------
# Minimal fixture helpers
# ---------------------------------------------------------------------------

def _empty_report() -> dict:
    return {
        "section_1_ideal_hit": None,
        "section_2_traffic": None,
        "section_3_locality": None,
        "section_4_content": None,
    }


def _with_ideal_hit(report: dict, ratio: float) -> dict:
    report["section_1_ideal_hit"] = {
        "f4_overall": {"ideal_hit_ratio": ratio}
    }
    return report


def _with_working_set(report: dict, windows: list[int],
                       blocks: list[int]) -> dict:
    report["section_2_traffic"] = report.get("section_2_traffic") or {}
    report["section_2_traffic"]["working_set"] = {
        "windows_min": windows,
        "unique_blocks": blocks,
    }
    return report


def _with_f13_stats(report: dict, **stats) -> dict:
    report["section_3_locality"] = report.get("section_3_locality") or {}
    report["section_3_locality"]["f13_single_turn"] = {
        "stats_seconds": stats
    }
    return report


def _with_reuse_distance(report: dict, p80: int,
                          available_cache: int | None = None) -> dict:
    report["section_3_locality"] = report.get("section_3_locality") or {}
    report["section_3_locality"]["reuse_distance"] = {
        "stats_blocks": {"p80": p80},
        "available_cache_blocks": available_cache,
    }
    return report


def _with_session_structure(report: dict, mean_turns: float,
                             top10_share: float) -> dict:
    report["section_2_traffic"] = report.get("section_2_traffic") or {}
    report["section_2_traffic"]["session_structure"] = {
        "f9_turn_count_cdf": {"mean_turns": mean_turns},
        "f10_user_turn_stats": {"lorenz_top10_pct_share_of_turns": top10_share},
    }
    return report


def _with_request_interval(report: dict, p50: float) -> dict:
    report["section_2_traffic"] = report.get("section_2_traffic") or {}
    report["section_2_traffic"]["request_interval_seconds"] = {"p50": p50}
    return report


def _with_common_prefix(report: dict, prefix_blocks: int,
                         head_coverage: float) -> dict:
    # v1.3 schema: count → freq, coverage_pct → global_coverage_pct.
    report["section_4_content"] = {
        "prefix_length_blocks": prefix_blocks,
        "consensus_blocks": [
            {"position": 0, "global_coverage_pct": head_coverage,
             "branch_ratio_pct": head_coverage,
             "freq": 100, "parent_freq": 100,
             "rank": 1, "text_preview": "x"}
        ],
    }
    return report


# ---------------------------------------------------------------------------
# R-PIN-CHAIN
# ---------------------------------------------------------------------------

def test_pin_chain_fires():
    r = _empty_report()
    _with_ideal_hit(r, 0.55)
    _with_common_prefix(r, prefix_blocks=56, head_coverage=87.5)
    rec = rule_pin_chain(r)
    assert rec is not None
    assert rec.rule_id == "R-PIN-CHAIN"
    assert rec.priority == "P0"
    assert rec.confidence == "high"
    assert rec.type == "recommendation"
    assert "56" in rec.conclusion


def test_pin_chain_skips_when_ideal_hit_too_high():
    r = _empty_report()
    _with_ideal_hit(r, 0.85)            # outside [0.4, 0.7]
    _with_common_prefix(r, prefix_blocks=56, head_coverage=87.5)
    assert rule_pin_chain(r) is None


def test_pin_chain_skips_when_prefix_too_short():
    r = _empty_report()
    _with_ideal_hit(r, 0.55)
    _with_common_prefix(r, prefix_blocks=10, head_coverage=87.5)
    assert rule_pin_chain(r) is None


# ---------------------------------------------------------------------------
# R-CAPACITY-FIRST
# ---------------------------------------------------------------------------

def test_capacity_first_fires():
    r = _empty_report()
    _with_ideal_hit(r, 0.82)
    _with_working_set(r, [60, 120], [150_000, 280_000])
    rec = rule_capacity_first(r)
    assert rec is not None
    assert rec.rule_id == "R-CAPACITY-FIRST"
    assert rec.priority == "P1"
    assert "150,000" in rec.evidence[1] or "150000" in rec.evidence[1]


def test_capacity_first_skips_when_working_set_small():
    r = _empty_report()
    _with_ideal_hit(r, 0.82)
    _with_working_set(r, [60, 120], [50_000, 80_000])
    assert rule_capacity_first(r) is None


# ---------------------------------------------------------------------------
# R-LOW-CEILING
# ---------------------------------------------------------------------------

def test_low_ceiling_fires():
    r = _with_ideal_hit(_empty_report(), 0.18)
    rec = rule_low_ceiling(r)
    assert rec is not None
    assert rec.rule_id == "R-LOW-CEILING"
    assert rec.priority == "P2"
    assert rec.confidence == "high"


def test_low_ceiling_skips_when_above_threshold():
    r = _with_ideal_hit(_empty_report(), 0.30)
    assert rule_low_ceiling(r) is None


# ---------------------------------------------------------------------------
# R-BATCH-TTL
# ---------------------------------------------------------------------------

def test_batch_ttl_fires():
    r = _empty_report()
    _with_ideal_hit(r, 0.55)
    _with_f13_stats(r, p50=0.5, p75=2.0, p80=3.0, p95=8.0)
    rec = rule_batch_ttl(r)
    assert rec is not None
    assert rec.rule_id == "R-BATCH-TTL"
    assert rec.priority == "P1"
    assert rec.confidence == "medium"


def test_batch_ttl_skips_when_p80_too_high():
    r = _empty_report()
    _with_ideal_hit(r, 0.55)
    _with_f13_stats(r, p50=0.5, p75=2.0, p80=10.0, p95=30.0)   # p80 > 5
    assert rule_batch_ttl(r) is None


# ---------------------------------------------------------------------------
# R-LONG-TTL
# ---------------------------------------------------------------------------

def test_long_ttl_fires():
    r = _empty_report()
    _with_ideal_hit(r, 0.62)
    _with_f13_stats(r, p50=10.0, p75=30.0, p80=46.0, p95=255.0)
    rec = rule_long_ttl(r)
    assert rec is not None
    assert rec.rule_id == "R-LONG-TTL"
    assert "46" in rec.conclusion


def test_long_ttl_skips_when_ideal_hit_too_low():
    r = _empty_report()
    _with_ideal_hit(r, 0.45)
    _with_f13_stats(r, p50=10.0, p75=30.0, p80=46.0, p95=255.0)
    assert rule_long_ttl(r) is None


# ---------------------------------------------------------------------------
# R-MULTI-TENANT
# ---------------------------------------------------------------------------

def test_multi_tenant_fires():
    r = _empty_report()
    _with_session_structure(r, mean_turns=4.2, top10_share=0.72)
    rec = rule_multi_tenant(r)
    assert rec is not None
    assert rec.rule_id == "R-MULTI-TENANT"
    assert rec.priority == "P2"


def test_multi_tenant_skips_when_share_too_low():
    r = _empty_report()
    _with_session_structure(r, mean_turns=4.2, top10_share=0.45)
    assert rule_multi_tenant(r) is None


def test_multi_tenant_skips_when_lorenz_field_missing():
    r = _empty_report()
    r["section_2_traffic"] = {
        "session_structure": {
            "f9_turn_count_cdf": {"mean_turns": 4.0},
            "f10_user_turn_stats": {"total_users": 25},   # no lorenz key
        }
    }
    assert rule_multi_tenant(r) is None


# ---------------------------------------------------------------------------
# R-CACHE-PRESSURE
# ---------------------------------------------------------------------------

def test_cache_pressure_fires_with_explicit_capacity():
    r = _with_reuse_distance(_empty_report(), p80=8000, available_cache=4000)
    rec = rule_cache_pressure(r)
    assert rec is not None
    assert rec.rule_id == "R-CACHE-PRESSURE"
    assert rec.priority == "P1"
    assert "2.00" in rec.evidence[2] or "2.0" in rec.evidence[2]


def test_cache_pressure_fires_via_fallback_threshold():
    r = _with_reuse_distance(_empty_report(), p80=60_000, available_cache=None)
    rec = rule_cache_pressure(r)
    assert rec is not None
    assert "兜底" in rec.evidence[1]


def test_cache_pressure_skips_when_p80_below_capacity():
    r = _with_reuse_distance(_empty_report(), p80=2000, available_cache=4000)
    assert rule_cache_pressure(r) is None


def test_cache_pressure_skips_when_no_capacity_and_below_fallback():
    r = _with_reuse_distance(_empty_report(), p80=30_000, available_cache=None)
    assert rule_cache_pressure(r) is None


# ---------------------------------------------------------------------------
# W-SAME-SECOND
# ---------------------------------------------------------------------------

def test_warn_same_second_fires():
    r = _with_request_interval(_empty_report(), p50=0)
    rec = rule_warn_same_second(r)
    assert rec is not None
    assert rec.type == "warning"
    assert rec.priority is None


def test_warn_same_second_skips_when_nonzero():
    r = _with_request_interval(_empty_report(), p50=0.4)
    assert rule_warn_same_second(r) is None


# ---------------------------------------------------------------------------
# W-REUSE-ZERO
# ---------------------------------------------------------------------------

def test_warn_reuse_zero_fires_via_f13():
    r = _empty_report()
    _with_f13_stats(r, p50=0, p75=1, p80=3, p95=10)
    rec = rule_warn_reuse_zero(r)
    assert rec is not None
    assert rec.type == "warning"
    assert rec.priority is None


def test_warn_reuse_zero_fires_via_reuse_distance_fallback():
    r = _empty_report()
    r["section_3_locality"] = {
        "reuse_distance": {
            "reuse_time_stats": {"p50": 0},
        }
    }
    rec = rule_warn_reuse_zero(r)
    assert rec is not None


def test_warn_reuse_zero_skips_when_nonzero():
    r = _empty_report()
    _with_f13_stats(r, p50=2, p75=5, p80=10, p95=30)
    assert rule_warn_reuse_zero(r) is None


# ---------------------------------------------------------------------------
# Engine — registry, sort order, run_all_rules
# ---------------------------------------------------------------------------

def test_registry_lists_all_nine_rules():
    rules = get_registry()
    assert len(rules) == 9
    assert all(callable(r) for r in rules)


def test_run_all_rules_on_empty_report_returns_empty_list():
    assert run_all_rules(_empty_report()) == []


def test_sort_order_p0_then_p1_then_p2_then_warning():
    items = [
        Recommendation("X-WARN", "warning", None, "low", "w", (), "a"),
        Recommendation("R-P2", "recommendation", "P2", "low", "x", (), "a"),
        Recommendation("R-P0", "recommendation", "P0", "low", "y", (), "a"),
        Recommendation("R-P1", "recommendation", "P1", "low", "z", (), "a"),
    ]
    out = sort_recommendations(items)
    assert [r.rule_id for r in out] == ["R-P0", "R-P1", "R-P2", "X-WARN"]


def test_sort_within_same_priority_is_alphabetical():
    items = [
        Recommendation("R-Z", "recommendation", "P1", "high", "x", (), "a"),
        Recommendation("R-A", "recommendation", "P1", "high", "x", (), "a"),
    ]
    assert [r.rule_id for r in sort_recommendations(items)] == ["R-A", "R-Z"]


def test_run_all_rules_end_to_end():
    """Build a synthetic report that should fire LOW_CEILING + W-SAME-SECOND."""
    r = _empty_report()
    _with_ideal_hit(r, 0.18)
    _with_request_interval(r, p50=0)

    out = run_all_rules(r)
    rule_ids = [rec.rule_id for rec in out]
    assert "R-LOW-CEILING" in rule_ids
    assert "W-SAME-SECOND" in rule_ids
    # Recommendation comes before warning.
    assert rule_ids.index("R-LOW-CEILING") < rule_ids.index("W-SAME-SECOND")


def test_recommendation_to_dict_round_trip():
    rec = Recommendation(
        rule_id="R-X",
        type="recommendation",
        priority="P0",
        confidence="high",
        conclusion="hello",
        evidence=("a", "b"),
        action="do something",
    )
    d = rec.to_dict()
    assert d["priority"] == "P0"
    assert d["evidence"] == ["a", "b"]
    assert d["rule_id"] == "R-X"
