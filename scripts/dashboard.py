#!/usr/bin/env python3
"""Streamlit dashboard for phase-1 prefix-cache analysis reports.

This is the Step-9 skeleton — sidebar with model selector + section
placeholders. Steps 10–14 wire each section to the report.json content.

Run
---
    pip install -e .[ui]
    streamlit run scripts/dashboard.py

Per ``可视化.md §1`` decisions:
* Local developer self-use only — no auth, no refresh button, no live
  reload.
* Dashboard reads ``outputs/maas/<model>/report.json`` exclusively. All
  computation happens upstream in :mod:`block_prefix_analyzer.report_builder`.
* Department dropdown is disabled in phase 1 (Phase 2 will populate it
  from the user_id × model_id mapping CSV).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "maas"


# ---------------------------------------------------------------------------
# Pure helpers (kept out of streamlit-render code so they're unit-testable).
# ---------------------------------------------------------------------------

def discover_reports(outputs_root: Path) -> list[str]:
    """Return slugs of every model directory containing ``report.json``."""
    if not outputs_root.exists():
        return []
    return sorted(
        d.name for d in outputs_root.iterdir()
        if d.is_dir() and (d / "report.json").is_file()
    )


def load_report(outputs_root: Path, model_id: str) -> dict[str, Any] | None:
    """Load and return ``report.json`` for ``model_id``, or None on failure."""
    path = outputs_root / model_id / "report.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------

def _render_sidebar(outputs_root: Path) -> str | None:
    st.sidebar.title("Prefix Cache Dashboard")
    st.sidebar.caption("Phase 1 — model-level analysis")

    models = discover_reports(outputs_root)
    if not models:
        st.sidebar.error(
            f"No report.json found under `{outputs_root}`.\n\n"
            "Run `python scripts/build_model_report.py --all` first."
        )
        return None

    selected = st.sidebar.selectbox("模型 (model_id)", models, index=0)

    # Phase 2 placeholder — disabled.
    st.sidebar.divider()
    st.sidebar.selectbox(
        "部门 (department)",
        ["— Phase 2 待开放 —"],
        index=0,
        disabled=True,
        help="Phase 2 待开放（需业务侧 user_id × model_id 映射 CSV）。",
    )

    return selected


def _render_meta(report: dict[str, Any]) -> None:
    meta = report.get("meta") or {}
    scope = report.get("scope") or {}

    cols = st.columns(4)
    cols[0].metric("Model", scope.get("model_id", "—"))
    cols[1].metric("Block size", str(meta.get("block_size") or "—"))
    cols[2].metric("Total requests",
                   f"{meta.get('total_requests'):,}" if meta.get("total_requests") else "—")
    cols[3].metric("Total users",
                   str(meta.get("total_users") or "—"))

    time_range = meta.get("time_range") or {}
    duration_h = time_range.get("duration_h")
    duration_str = f"{duration_h} h" if duration_h is not None else "—"

    st.caption(
        f"generated_at: `{meta.get('generated_at', '—')}`  ·  "
        f"data_version: `{meta.get('data_version', '—')}`  ·  "
        f"window duration: {duration_str}  ·  "
        f"input_file: `{meta.get('input_file', '—')}`"
    )


def _render_placeholder(header: str, body: str) -> None:
    st.header(header)
    st.info(body)


def main() -> None:
    st.set_page_config(
        page_title="Prefix Cache Dashboard",
        layout="wide",
    )

    selected = _render_sidebar(DEFAULT_OUTPUTS_ROOT)
    if selected is None:
        st.title("Prefix Cache Dashboard")
        st.warning("没有可用的 report.json — 请先运行 build_model_report。")
        return

    report = load_report(DEFAULT_OUTPUTS_ROOT, selected)
    if report is None:
        st.title("Prefix Cache Dashboard")
        st.error(f"加载 {selected}/report.json 失败（文件缺失或 JSON 损坏）。")
        return

    st.title(f"{selected} — Prefix Cache Analysis")
    _render_meta(report)
    st.divider()

    _render_placeholder(
        "1. 理想命中率 (Ideal Hit Rate)",
        "Step 10 will wire: F4 大数字卡片 + block_size sweep 折线 + "
        "user hit histogram + reuse_rank histogram。",
    )
    _render_placeholder(
        "2. 流量业务模式 (Traffic Pattern)",
        "Step 11 will wire: interval 分位 + volume timeseries + "
        "write rate + working set + F9/F10 会话结构。",
    )
    _render_placeholder(
        "3. KV cache 时间局部性 (Locality)",
        "Step 12 will wire: F13 / F14 / reuse_distance 三张 CDF + 分位表。",
    )
    _render_placeholder(
        "4. 可复用内容 (Content)",
        "Step 13 will wire: common_prefix top-N 共识块 + content_type_guess。",
    )
    _render_placeholder(
        "5. 优化建议 (Recommendations)",
        "Step 14 will wire: P0/P1/P2 + Warning 卡片，按优先级分组。",
    )


if __name__ == "__main__":
    main()
