"""Tests for ``scripts/build_model_report.py``.

Covers
------
* discover_models scans only directories with at least one analysis-style
  subdir (containing metadata.json) and skips empty / file-only entries.
* build_one writes outputs/<model>/report.json with the v1.1 schema and
  the recommendation engine output spliced into section_5_recommendations.
* build_one finds the JSONL via data_root and computes data_version when
  the file is present.
* CLI --model end-to-end run via subprocess produces a valid report.

The script is loaded via :mod:`importlib` because ``scripts/`` is not a
package.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_model_report.py"


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location(
        "build_model_report", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_outputs(model_dir: Path) -> None:
    """Drop a small but rule-firing fixture under ``model_dir/``."""
    f4 = model_dir / "f4_prefix" / "metadata.json"
    f4.parent.mkdir(parents=True, exist_ok=True)
    f4.write_text(json.dumps({
        "trace_name": "demo",
        "input_file": "data/internal/demo/requests.jsonl",
        "block_size": 128,
        "total_blocks_sum": 1000,
        "hit_blocks_sum": 720,
        "ideal_overall_hit_ratio": 0.72,
        "hit_definition": "content_prefix_reuse_blocks",
    }))

    tp = model_dir / "traffic_pattern" / "metadata.json"
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(json.dumps({
        "trace_name": "demo",
        "input_file": "data/internal/demo/requests.jsonl",
        "block_size": 128,
        "bin_size_s": 60,
        "working_set_windows_min": [60, 120],
        "totals": {
            "total_requests": 185,
            "total_unique_blocks": 5000,
            "duration_s": 7200.0,
            "first_timestamp_s": 0.0,
        },
        # p50 == 0 → fires W-SAME-SECOND
        "interval_percentiles": {"p50": 0, "p75": 1.0, "p80": 2.0, "p95": 5.0},
        # 60min working_set > 100K + ideal_hit 0.72 → fires R-CAPACITY-FIRST
        "working_set": {"60": 150000, "120": 280000},
    }))


# ---------------------------------------------------------------------------
# discover_models
# ---------------------------------------------------------------------------

def test_discover_models_returns_only_analyzed_dirs(tmp_path: Path, builder):
    outputs_root = tmp_path / "outputs"
    outputs_root.mkdir()

    # Model with one analysis sub-dir → discovered
    _write_minimal_outputs(outputs_root / "model_a")

    # Empty model dir → skipped
    (outputs_root / "model_b").mkdir()

    # Model with sub-dir but no metadata.json → skipped
    (outputs_root / "model_c" / "f4_prefix").mkdir(parents=True)

    # Top-level file → skipped
    (outputs_root / "stray.txt").write_text("ignore me")

    models = builder.discover_models(outputs_root)
    assert models == ["model_a"]


def test_discover_models_returns_empty_when_root_missing(tmp_path: Path, builder):
    assert builder.discover_models(tmp_path / "does-not-exist") == []


# ---------------------------------------------------------------------------
# build_one
# ---------------------------------------------------------------------------

def test_build_one_writes_v11_report(tmp_path: Path, builder):
    outputs_root = tmp_path / "outputs"
    data_root = tmp_path / "data"
    _write_minimal_outputs(outputs_root / "demo")

    out_path = builder.build_one("demo", outputs_root, data_root)
    assert out_path == outputs_root / "demo" / "report.json"
    assert out_path.exists()

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.1"
    assert report["scope"]["model_id"] == "demo"
    # Without a data file under data_root, data_version is None.
    assert report["meta"]["data_version"] is None
    # Rules fired: R-CAPACITY-FIRST + W-SAME-SECOND.
    rule_ids = sorted(r["rule_id"] for r in report["section_5_recommendations"])
    assert "R-CAPACITY-FIRST" in rule_ids
    assert "W-SAME-SECOND" in rule_ids


def test_build_one_uses_data_root_for_data_version(tmp_path: Path, builder):
    outputs_root = tmp_path / "outputs"
    data_root = tmp_path / "data"
    _write_minimal_outputs(outputs_root / "demo")
    (data_root / "demo").mkdir(parents=True)
    (data_root / "demo" / "requests.jsonl").write_text('{"hi": 1}\n')

    out_path = builder.build_one("demo", outputs_root, data_root)
    report = json.loads(out_path.read_text(encoding="utf-8"))
    dv = report["meta"]["data_version"]
    assert dv is not None
    assert dv.startswith("sha256:")
    assert len(dv.split(":", 1)[1]) == 16


def test_build_one_raises_if_outputs_dir_missing(tmp_path: Path, builder):
    outputs_root = tmp_path / "outputs"
    outputs_root.mkdir()
    with pytest.raises(FileNotFoundError, match="outputs directory not found"):
        builder.build_one("missing-model", outputs_root, tmp_path / "data")


def test_build_one_recommendations_sorted_p_first_warnings_last(tmp_path: Path, builder):
    outputs_root = tmp_path / "outputs"
    _write_minimal_outputs(outputs_root / "demo")

    out_path = builder.build_one("demo", outputs_root, tmp_path / "data")
    report = json.loads(out_path.read_text(encoding="utf-8"))
    recs = report["section_5_recommendations"]
    types = [r["type"] for r in recs]
    # All recommendations come before any warning.
    if "warning" in types:
        first_warn_idx = types.index("warning")
        assert all(t == "warning" for t in types[first_warn_idx:])


# ---------------------------------------------------------------------------
# CLI smoke (subprocess) — exercises argparse + main()
# ---------------------------------------------------------------------------

def test_cli_model_end_to_end(tmp_path: Path):
    outputs_root = tmp_path / "outputs"
    data_root = tmp_path / "data"
    _write_minimal_outputs(outputs_root / "demo")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--model", "demo",
         "--outputs-root", str(outputs_root),
         "--data-root", str(data_root)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    out_path = outputs_root / "demo" / "report.json"
    assert out_path.exists()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.1"
    assert report["scope"]["model_id"] == "demo"


def test_cli_all_discovers_and_builds(tmp_path: Path):
    outputs_root = tmp_path / "outputs"
    data_root = tmp_path / "data"
    _write_minimal_outputs(outputs_root / "model_a")
    _write_minimal_outputs(outputs_root / "model_b")
    (outputs_root / "empty_model").mkdir()        # skipped by discover_models

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--all",
         "--outputs-root", str(outputs_root),
         "--data-root", str(data_root)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    assert (outputs_root / "model_a" / "report.json").exists()
    assert (outputs_root / "model_b" / "report.json").exists()
    assert not (outputs_root / "empty_model" / "report.json").exists()
    assert "Building reports for 2 models" in proc.stdout


def test_cli_all_exits_with_error_when_no_models(tmp_path: Path):
    outputs_root = tmp_path / "outputs"
    outputs_root.mkdir()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--all",
         "--outputs-root", str(outputs_root),
         "--data-root", str(tmp_path / "data")],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 1
    assert "No analyzed models" in proc.stderr


def test_cli_requires_model_or_all(tmp_path: Path):
    """Argparse mutually-exclusive group: must pass exactly one of --model/--all."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode != 0
    assert "required" in proc.stderr.lower() or "one of the arguments" in proc.stderr.lower()
