"""Tests for ``scripts/dashboard.py`` non-streamlit helpers.

Streamlit-rendering code is exercised manually (``streamlit run``) and not
covered here — Streamlit's AppTest harness needs a streamlit runtime and
adds dependencies orthogonal to phase-1 logic. The helpers we DO test are
the discovery / load functions because dashboard correctness on real
report sets depends on them.

The script is loaded via :mod:`importlib` because ``scripts/`` is not a
package.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "dashboard.py"


@pytest.fixture(scope="module")
def dashboard():
    spec = importlib.util.spec_from_file_location("dashboard", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# discover_reports
# ---------------------------------------------------------------------------

def test_discover_reports_lists_only_dirs_with_report_json(tmp_path: Path, dashboard):
    outputs_root = tmp_path / "out"
    outputs_root.mkdir()

    # Two valid: model_a / model_b each with report.json
    for slug in ("model_a", "model_b"):
        (outputs_root / slug).mkdir()
        (outputs_root / slug / "report.json").write_text("{}")

    # Empty dir → skipped
    (outputs_root / "empty").mkdir()

    # Dir with sub-analysis but no report.json → skipped
    (outputs_root / "no_report" / "f4_prefix").mkdir(parents=True)
    (outputs_root / "no_report" / "f4_prefix" / "metadata.json").write_text("{}")

    # Top-level file → ignored
    (outputs_root / "stray.txt").write_text("x")

    assert dashboard.discover_reports(outputs_root) == ["model_a", "model_b"]


def test_discover_reports_returns_empty_when_root_missing(tmp_path: Path, dashboard):
    assert dashboard.discover_reports(tmp_path / "missing") == []


def test_discover_reports_sorted(tmp_path: Path, dashboard):
    outputs_root = tmp_path / "out"
    outputs_root.mkdir()
    for slug in ("zeta", "alpha", "mu"):
        (outputs_root / slug).mkdir()
        (outputs_root / slug / "report.json").write_text("{}")
    assert dashboard.discover_reports(outputs_root) == ["alpha", "mu", "zeta"]


# ---------------------------------------------------------------------------
# load_report
# ---------------------------------------------------------------------------

def test_load_report_returns_dict(tmp_path: Path, dashboard):
    outputs_root = tmp_path / "out"
    (outputs_root / "demo").mkdir(parents=True)
    payload = {"schema_version": "1.1", "scope": {"model_id": "demo"}}
    (outputs_root / "demo" / "report.json").write_text(json.dumps(payload))

    loaded = dashboard.load_report(outputs_root, "demo")
    assert loaded == payload


def test_load_report_returns_none_when_missing(tmp_path: Path, dashboard):
    outputs_root = tmp_path / "out"
    outputs_root.mkdir()
    assert dashboard.load_report(outputs_root, "missing") is None


def test_load_report_returns_none_on_corrupt_json(tmp_path: Path, dashboard):
    outputs_root = tmp_path / "out"
    (outputs_root / "demo").mkdir(parents=True)
    (outputs_root / "demo" / "report.json").write_text("{not valid")
    assert dashboard.load_report(outputs_root, "demo") is None


# ---------------------------------------------------------------------------
# histogram_frame (Step 10 helper)
# ---------------------------------------------------------------------------

def test_histogram_frame_basic(dashboard):
    import pandas as pd
    s = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    df = dashboard.histogram_frame(s, bins=10)
    assert df["count"].sum() == len(s)
    assert len(df) == 10                     # 10 bins requested


def test_histogram_frame_empty_input(dashboard):
    import pandas as pd
    df = dashboard.histogram_frame(pd.Series([], dtype=float))
    assert df.empty


def test_histogram_frame_constant_input_returns_single_bucket(dashboard):
    import pandas as pd
    df = dashboard.histogram_frame(pd.Series([0.42] * 50))
    assert len(df) == 1
    assert df["count"].iloc[0] == 50


def test_histogram_frame_drops_non_numeric(dashboard):
    import pandas as pd
    s = pd.Series([0.1, "bad", 0.5, None, 0.9])
    df = dashboard.histogram_frame(s, bins=4)
    assert df["count"].sum() == 3            # only the 3 numeric values


# ---------------------------------------------------------------------------
# _read_csv_safely (Step 11 helper)
# ---------------------------------------------------------------------------

def test_read_csv_safely_returns_dataframe(tmp_path: Path, dashboard):
    csv_path = tmp_path / "sub" / "data.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("a,b\n1,2\n3,4\n")
    df = dashboard._read_csv_safely(tmp_path, "sub/data.csv")
    assert df is not None
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_read_csv_safely_returns_none_for_missing(tmp_path: Path, dashboard):
    assert dashboard._read_csv_safely(tmp_path, "missing.csv") is None


def test_read_csv_safely_returns_none_for_empty_path(tmp_path: Path, dashboard):
    assert dashboard._read_csv_safely(tmp_path, None) is None
    assert dashboard._read_csv_safely(tmp_path, "") is None


def test_read_csv_safely_returns_none_on_corrupt_csv(tmp_path: Path, dashboard):
    bad = tmp_path / "bad.csv"
    # File present but unreadable as csv (invalid bytes).
    bad.write_bytes(b"\xff\xfe\xfd not csv \x00\x01")
    # Pandas may or may not raise on this; the safe wrapper either way must
    # not propagate the exception. Returning None or an empty df is fine.
    out = dashboard._read_csv_safely(tmp_path, "bad.csv")
    assert out is None or out.empty


# ---------------------------------------------------------------------------
# events_to_cdf (Step 12 helper)
# ---------------------------------------------------------------------------

def test_events_to_cdf_basic(dashboard):
    import pandas as pd
    s = pd.Series([10, 20, 30, 40, 50])
    df = dashboard.events_to_cdf(s)
    # Index = sorted distinct values; cdf monotone, ends at 1.0.
    assert list(df.index) == [10, 20, 30, 40, 50]
    assert df["cdf"].iloc[0] == pytest.approx(0.2)
    assert df["cdf"].iloc[-1] == pytest.approx(1.0)
    assert df["cdf"].is_monotonic_increasing


def test_events_to_cdf_collapses_duplicates(dashboard):
    import pandas as pd
    s = pd.Series([10, 10, 20, 20, 30])
    df = dashboard.events_to_cdf(s)
    # 5 events; CDF at 10 = 2/5, at 20 = 4/5, at 30 = 1.0
    assert list(df.index) == [10, 20, 30]
    assert df["cdf"].tolist() == pytest.approx([0.4, 0.8, 1.0])


def test_events_to_cdf_empty_input(dashboard):
    import pandas as pd
    df = dashboard.events_to_cdf(pd.Series([], dtype=float))
    assert df.empty


def test_events_to_cdf_drops_non_numeric(dashboard):
    import pandas as pd
    s = pd.Series([10, "bad", 20, None, 30])
    df = dashboard.events_to_cdf(s)
    assert list(df.index) == [10, 20, 30]
    assert df["cdf"].iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# consensus_blocks_to_frame (Step 13 helper)
# ---------------------------------------------------------------------------

def test_consensus_blocks_to_frame_empty(dashboard):
    df = dashboard.consensus_blocks_to_frame([])
    assert df.empty
    assert list(df.columns) == [
        "rank", "position", "count", "coverage_pct", "type", "text_preview",
    ]


def test_consensus_blocks_to_frame_basic(dashboard):
    blocks = [
        {"rank": 1, "position": 0, "count": 100, "coverage_pct": 95.5,
         "content_type_guess": "system_prompt", "text_preview": "你是 ..."},
        {"rank": 2, "position": 1, "count": 90, "coverage_pct": 88.2,
         "content_type_guess": "code", "text_preview": "def foo(): ..."},
    ]
    df = dashboard.consensus_blocks_to_frame(blocks)
    assert len(df) == 2
    assert df.iloc[0]["rank"] == 1
    assert df.iloc[0]["coverage_pct"] == 95.5
    # type column carries an emoji prefix from CONTENT_TYPE_EMOJI.
    assert "system_prompt" in df.iloc[0]["type"]
    assert df.iloc[0]["type"].startswith("🧭")
    assert df.iloc[1]["type"].startswith("💻")


def test_consensus_blocks_to_frame_handles_missing_fields(dashboard):
    blocks = [
        {"rank": 1},                                # missing everything else
        {"position": 0, "content_type_guess": "unknown_label"},  # unknown type
    ]
    df = dashboard.consensus_blocks_to_frame(blocks)
    assert len(df) == 2
    # Missing text_preview becomes empty string, not None.
    assert df.iloc[0]["text_preview"] == ""
    # Unknown type still renders with the fallback bullet.
    assert df.iloc[1]["type"].startswith("·")
    assert "unknown_label" in df.iloc[1]["type"]


def test_consensus_blocks_to_frame_falls_back_to_other_when_type_missing(dashboard):
    df = dashboard.consensus_blocks_to_frame([
        {"rank": 1, "text_preview": "x"}
    ])
    assert df.iloc[0]["type"].endswith("other")


# ---------------------------------------------------------------------------
# group_recommendations (Step 14 helper)
# ---------------------------------------------------------------------------

def test_group_recommendations_empty(dashboard):
    out = dashboard.group_recommendations([])
    # All four keys present even when there's nothing.
    assert set(out.keys()) == {"P0", "P1", "P2", "warning"}
    assert all(v == [] for v in out.values())


def test_group_recommendations_routes_by_priority_and_type(dashboard):
    recs = [
        {"rule_id": "R-A", "type": "recommendation", "priority": "P1"},
        {"rule_id": "R-B", "type": "recommendation", "priority": "P0"},
        {"rule_id": "W-A", "type": "warning",        "priority": None},
        {"rule_id": "R-C", "type": "recommendation", "priority": "P2"},
        {"rule_id": "R-D", "type": "recommendation", "priority": "P1"},
    ]
    out = dashboard.group_recommendations(recs)
    assert [r["rule_id"] for r in out["P0"]] == ["R-B"]
    assert [r["rule_id"] for r in out["P1"]] == ["R-A", "R-D"]
    assert [r["rule_id"] for r in out["P2"]] == ["R-C"]
    assert [r["rule_id"] for r in out["warning"]] == ["W-A"]


def test_group_recommendations_unknown_priority_falls_through_to_p2(dashboard):
    recs = [{"rule_id": "R-X", "type": "recommendation", "priority": "P9"}]
    out = dashboard.group_recommendations(recs)
    assert [r["rule_id"] for r in out["P2"]] == ["R-X"]
    assert out["P0"] == out["P1"] == out["warning"] == []


def test_group_recommendations_warning_takes_precedence_over_priority(dashboard):
    """A row tagged as warning routes to warning even if priority is set."""
    recs = [{"rule_id": "W-X", "type": "warning", "priority": "P0"}]
    out = dashboard.group_recommendations(recs)
    assert out["P0"] == []
    assert [r["rule_id"] for r in out["warning"]] == ["W-X"]


def test_priority_group_titles_match_expected_keys(dashboard):
    assert set(dashboard.PRIORITY_GROUP_TITLES.keys()) == {
        "P0", "P1", "P2", "warning",
    }


def test_priority_badge_includes_warning_for_none(dashboard):
    assert dashboard.PRIORITY_BADGE[None].startswith("⚠️")
    assert dashboard.PRIORITY_BADGE["P0"].startswith("🔴")


# ---------------------------------------------------------------------------
# _resolve_outputs_root (env var override)
# ---------------------------------------------------------------------------

def test_resolve_outputs_root_honors_env_var(tmp_path: Path, dashboard,
                                              monkeypatch):
    monkeypatch.setenv("BPA_OUTPUTS_ROOT", str(tmp_path))
    assert dashboard._resolve_outputs_root() == tmp_path


def test_resolve_outputs_root_expands_user(dashboard, monkeypatch):
    monkeypatch.setenv("BPA_OUTPUTS_ROOT", "~/some/outputs")
    out = dashboard._resolve_outputs_root()
    assert "~" not in str(out)


def test_resolve_outputs_root_default_falls_back(dashboard, monkeypatch):
    monkeypatch.delenv("BPA_OUTPUTS_ROOT", raising=False)
    out = dashboard._resolve_outputs_root()
    assert out.name == "maas"
    assert out.parent.name == "outputs"
