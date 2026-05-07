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
