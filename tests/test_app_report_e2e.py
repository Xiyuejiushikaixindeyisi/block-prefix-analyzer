"""End-to-end test for the per-APP report pipeline (Dashboard Phase 2 Step 7).

Covers:

* A self-contained synthetic dataset whose computed values are entirely
  predictable, so we can assert *golden* numerics across every section
  + the relative-position card. Runs in CI with no external data.
* An opportunistic test that runs the same pipeline against the real
  ``data/internal/synthetic_demo/`` trace when present (skipped in CI).
* Plan §9 acceptance criteria (kind=="app" report, sections 1–4 + RP
  populated, HTML self-contained with at least one inlined chart).
"""
from __future__ import annotations

import csv as _csv
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_app_report.py"


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_app_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic golden fixture
# ---------------------------------------------------------------------------

# block_size=16 + raw_prompt = nine distinct zero-padded numeric blocks:
# - exactly 9 full blocks per request (no tail leftover)
# - block_ids are pairwise distinct so F13 reuse events count per position
#   (a periodic prompt like "abc"*N would dedupe by block_id and confuse the
#   golden assertions).
GOLDEN_BLOCK_SIZE = 16
GOLDEN_BLOCKS_PER_REQUEST = 9
GOLDEN_PROMPT = "".join(f"{i:016d}" for i in range(GOLDEN_BLOCKS_PER_REQUEST))
GOLDEN_APP_ID = "com.app.golden"
GOLDEN_PRODUCT = "Golden Test Product"
GOLDEN_DECLARED = "Qwen-V3-32B"
GOLDEN_MODEL = "qwen_v3_32b_8k"      # matches declared via heuristic

_REGISTRY_HEADER = (
    "app_id,product_name,declared_model,business_purpose,source_meeting_date,"
    "product_manager,resource_type_requested,resource_type_actual,"
    "guaranteed_quota_cards,guaranteed_concurrency,expected_duration"
)


def _write_registry(tmp_path: Path) -> Path:
    p = tmp_path / "configs" / "app_registry.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    row = ",".join([
        GOLDEN_APP_ID, GOLDEN_PRODUCT, GOLDEN_DECLARED, "生产", "2026-01-06",
        "PM Golden", "D910B3", "D910B4", "NA", "100", "一年",
    ])
    p.write_text(_REGISTRY_HEADER + "\n" + row + "\n", encoding="utf-8")
    return p


def _write_raw_jsonl(tmp_path: Path) -> Path:
    """Three matching APP requests + two distractor requests for cross-app context.

    APP requests share an identical raw_prompt → identical block_ids per
    SimpleBlockBuilder → identical 9-block prefix. Spread 60 s apart so
    reuse_time is exactly 60 s for every reuse event.
    """
    p = tmp_path / "data" / GOLDEN_MODEL / "requests.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, ts in enumerate([0.0, 60.0, 120.0]):
        rows.append({
            "user_id": GOLDEN_APP_ID,
            "request_id": f"golden-r{i}",
            "timestamp": ts,
            "raw_prompt": GOLDEN_PROMPT,
        })
    # Distractor app for cross-app baseline statistics.
    for i, ts in enumerate([10.0, 70.0]):
        rows.append({
            "user_id": "com.app.other",
            "request_id": f"other-r{i}",
            "timestamp": ts,
            "raw_prompt": "xyz" * 50,    # different content → no shared prefix
        })
    p.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )
    return p


def _seed_model_outputs(tmp_path: Path) -> Path:
    """Lay down minimal Phase-1 model outputs that the per-APP pipeline reads.

    Numbers chosen so cross-baseline assertions are easy:
    * model F4 ratio = 0.50 (clearly different from this APP's 0.6667)
    * e1 cross-app distribution has 2 users so median is well-defined
    * common_prefix model length = 50 blocks (vs APP's 9)
    * traffic_pattern volume.csv: bin@120 is the lone peak (>= p90)
    """
    out = tmp_path / "outputs" / GOLDEN_MODEL
    for sub in ("traffic_pattern", "f4_prefix", "f13_prefix",
                "common_prefix", "e1_user_hit_rate"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    (out / "traffic_pattern" / "metadata.json").write_text(
        json.dumps({"bin_size_s": 60, "block_size": GOLDEN_BLOCK_SIZE}),
        encoding="utf-8",
    )
    with (out / "traffic_pattern" / "volume.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = _csv.writer(f); w.writerow(["bin_start_s", "request_count"])
        # 3 bins; only bin@120 has count high enough to clear the p90 line.
        w.writerow([0, 1]); w.writerow([60, 1]); w.writerow([120, 100])

    (out / "f4_prefix" / "metadata.json").write_text(json.dumps({
        "ideal_overall_hit_ratio": 0.50,
        "block_size": GOLDEN_BLOCK_SIZE,
        "total_blocks_sum": 200, "hit_blocks_sum": 100,
        "hit_definition": "content_prefix_reuse_blocks",
    }), encoding="utf-8")

    (out / "f13_prefix" / "metadata.json").write_text(json.dumps({
        "single_turn_request_count": 5,
        "event_definition": "content_prefix_reuse",
    }), encoding="utf-8")
    with (out / "f13_prefix" / "cdf_series.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = _csv.writer(f); w.writerow(["reuse_time_seconds", "cdf"])
        w.writerow([30.0, 0.5]); w.writerow([300.0, 1.0])

    (out / "common_prefix" / "metadata.json").write_text(json.dumps({
        "prefix_length_blocks": 50, "prefix_length_chars": 800,
    }), encoding="utf-8")
    with (out / "common_prefix" / "coverage_profile.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = _csv.writer(f); w.writerow(["position", "block_id", "count", "coverage_pct"])
        w.writerow([0, "model_block_a", 50, 99.0])
        w.writerow([1, "model_block_b", 50, 99.0])

    with (out / "e1_user_hit_rate" / f"user_hit_bs{GOLDEN_BLOCK_SIZE}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = _csv.writer(f)
        w.writerow(["rank", "hit_rate", "prefix_reuse_blocks", "total_blocks", "request_count"])
        w.writerow([1, 0.40, 4, 10, 3])
        w.writerow([2, 0.60, 6, 10, 5])
    return out


# ---------------------------------------------------------------------------
# Primary golden test
# ---------------------------------------------------------------------------

def test_e2e_synthetic_golden_values(tmp_path: Path, builder, capsys) -> None:
    """End-to-end run with a controlled fixture; assert exact computed values."""
    registry = _write_registry(tmp_path)
    _write_raw_jsonl(tmp_path)
    _seed_model_outputs(tmp_path)

    rc = builder.main([
        "--model", GOLDEN_MODEL,
        "--app", GOLDEN_APP_ID,
        "--registry", str(registry),
        "--data-root", str(tmp_path / "data"),
        "--outputs-root", str(tmp_path / "outputs"),
    ])
    assert rc == 0

    app_dir = tmp_path / "outputs" / GOLDEN_MODEL / "apps" / GOLDEN_APP_ID
    report_path = app_dir / "report.json"
    html_path = app_dir / "report.html"
    filtered_path = app_dir / "filtered_requests.jsonl"
    assert report_path.is_file()
    assert html_path.is_file()
    assert filtered_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))

    # ------- envelope + scope -------
    assert report["schema_version"] == "1.3"
    scope = report["scope"]
    assert scope["kind"] == "app"
    assert scope["model_id"] == GOLDEN_MODEL
    assert scope["app_id"] == GOLDEN_APP_ID
    assert scope["product_name"] == GOLDEN_PRODUCT
    assert scope["declared_model"] == GOLDEN_DECLARED
    assert len(scope["app_history"]) == 1
    assert scope["app_history"][0]["business_purpose"] == "生产"

    # ------- meta -------
    meta = report["meta"]
    assert meta["model_id"] == GOLDEN_MODEL
    assert meta["app_id"] == GOLDEN_APP_ID
    assert meta["block_size"] == GOLDEN_BLOCK_SIZE
    # 3 matching rows kept by the filter; raw file has 5 total.
    assert meta["total_requests"] == 3
    assert meta["app_filter_stats"]["kept_count"] == 3
    assert meta["app_filter_stats"]["total_lines"] == 5
    assert meta["time_range"]["start_s"] == pytest.approx(0.0)
    assert meta["time_range"]["end_s"] == pytest.approx(120.0)

    # ------- section 1 — exact F4 values for 3 identical requests -------
    s1 = report["section_1_ideal_hit"]
    app_f4 = s1["app_f4"]
    # Each request → 9 full blocks; r1: 0 hits; r2: 9 hits; r3: 9 hits.
    assert app_f4["total_blocks_sum"] == 3 * GOLDEN_BLOCKS_PER_REQUEST
    assert app_f4["hit_blocks_sum"] == 2 * GOLDEN_BLOCKS_PER_REQUEST
    assert app_f4["ideal_hit_ratio"] == pytest.approx(18 / 27)
    assert app_f4["total_requests"] == 3
    # Cross-app baseline picked up from synthetic e1 csv (2 users).
    assert s1["model_baseline"]["ideal_hit_ratio"] == 0.50
    assert s1["user_hit_distribution"]["stats"]["user_count"] == 2

    # ------- section 2 — traffic timing/volume + peak alignment -------
    s2 = report["section_2_traffic"]
    at = s2["app_traffic"]
    assert at["total_requests"] == 3
    assert at["bin_size_s"] == 60
    # bin_start_s aligned at multiples of 60 s.
    assert at["volume_series"] == [[0, 1], [60, 1], [120, 1]]
    pa = s2["peak_alignment"]
    # Model bin counts [1, 1, 100]; p90 is well above 1, peak set = {120}.
    assert pa["model_total_bins"] == 3
    assert pa["model_peak_bins"] == 1
    # APP has 1 of its 3 requests in bin@120.
    assert pa["app_requests_in_peak_bins"] == 1
    assert pa["peak_alignment_ratio"] == pytest.approx(1 / 3)

    # ------- section 3 — F13 reuse times -------
    s3 = report["section_3_locality"]
    f13 = s3["app_f13"]
    # 9 blocks × 2 reuse-bearing requests (r2, r3) = 18 events; all 60 s.
    assert f13["reuse_event_count"] == 2 * GOLDEN_BLOCKS_PER_REQUEST
    assert f13["single_turn_request_count"] == 3
    assert f13["stats_seconds"]["p50"] == pytest.approx(60.0)
    assert f13["stats_seconds"]["p95"] == pytest.approx(60.0)

    # ------- section 4 — consensus + overlap -------
    s4 = report["section_4_content"]
    ac = s4["app_consensus"]
    assert ac["prefix_length_blocks"] == GOLDEN_BLOCKS_PER_REQUEST
    assert ac["prefix_length_chars"] == GOLDEN_BLOCKS_PER_REQUEST * GOLDEN_BLOCK_SIZE
    assert ac["min_count_threshold"] == 2
    # v1.3: count → freq. Each consensus block has freq == 3 (all 3 share).
    assert all(b["freq"] == 3 for b in ac["consensus_blocks"])
    # Overlap: APP block_ids vs synthetic model_block_a/b → 0 shared.
    overlap = s4["model_overlap"]
    assert overlap["model_unique_block_count"] == 2
    assert overlap["shared_block_count"] == 0

    # ------- relative_position — declared/model match heuristic -------
    rp = report["relative_position"]
    assert rp is not None
    dmc = rp["declared_model_consistency"]
    assert dmc["is_consistent"] is True
    assert dmc["matched_declared_models"] == [GOLDEN_DECLARED]

    # ------- HTML envelope checks -------
    html = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert html.endswith("</body></html>\n")
    assert GOLDEN_PRODUCT in html
    assert GOLDEN_APP_ID in html
    assert "kind: <b>app</b>" in html
    # At least one base64-embedded chart present (Section B volume curve).
    assert "data:image/png;base64" in html


def test_e2e_unregistered_app_acceptance_criterion(
    tmp_path: Path, builder,
) -> None:
    """Plan §9 acceptance #3: unregistered APP still produces a complete
    report with the warning banner visible in HTML."""
    _write_raw_jsonl(tmp_path)
    _seed_model_outputs(tmp_path)
    rc = builder.main([
        "--model", GOLDEN_MODEL,
        "--app", "com.app.NEVER_REGISTERED",
        "--registry", str(tmp_path / "no-such-registry.csv"),
        "--data-root", str(tmp_path / "data"),
        "--outputs-root", str(tmp_path / "outputs"),
    ])
    assert rc == 0
    app_dir = (tmp_path / "outputs" / GOLDEN_MODEL / "apps" /
               "com.app.NEVER_REGISTERED")
    report = json.loads((app_dir / "report.json").read_text(encoding="utf-8"))
    assert report["scope"]["product_name"] == "<unregistered>"
    assert report["scope"]["app_history"] == []
    # No matching rows → app sections degrade gracefully, but envelope intact.
    assert report["section_1_ideal_hit"] is not None
    assert report["section_1_ideal_hit"]["app_f4"] is None  # no matching records
    html = (app_dir / "report.html").read_text(encoding="utf-8")
    assert "未在最新会议申请记录中找到" in html


# ---------------------------------------------------------------------------
# Opportunistic — real synthetic_demo trace if available locally
# ---------------------------------------------------------------------------

_REAL_SYNTHETIC_JSONL = REPO_ROOT / "data" / "internal" / "synthetic_demo" / "requests.jsonl"
_REAL_SYNTHETIC_OUTPUTS = REPO_ROOT / "outputs" / "maas" / "synthetic_demo"


@pytest.mark.skipif(
    not _REAL_SYNTHETIC_JSONL.is_file()
    or not (_REAL_SYNTHETIC_OUTPUTS / "f4_prefix" / "metadata.json").is_file(),
    reason="data/internal/synthetic_demo/ + outputs/maas/synthetic_demo/ not present "
           "(this is normal in CI; data is gitignored).",
)
def test_e2e_against_real_synthetic_demo(tmp_path: Path, builder) -> None:
    """If the local environment has the real synthetic_demo trace + outputs,
    run the full pipeline against it. This validates plan §9 acceptance #2
    (build_app_report produces an html in well under the 5-minute budget)."""
    # Pick a known user_id from the dataset.
    sample_uid: str | None = None
    with _REAL_SYNTHETIC_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uid = json.loads(line).get("user_id")
            if uid:
                sample_uid = str(uid)
                break
    assert sample_uid is not None, "synthetic_demo jsonl had no usable user_id"

    rc = builder.main([
        "--model", "synthetic_demo",
        "--app", sample_uid,
        "--registry", str(tmp_path / "no-such-registry.csv"),
        "--data-root", str(REPO_ROOT / "data" / "internal"),
        "--outputs-root", str(tmp_path / "outputs"),
    ])
    assert rc == 0
    app_dir = (tmp_path / "outputs" / "synthetic_demo" / "apps" / sample_uid)
    assert (app_dir / "report.json").is_file()
    assert (app_dir / "report.html").is_file()
    assert (app_dir / "filtered_requests.jsonl").is_file()
