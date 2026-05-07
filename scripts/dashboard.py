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

import pandas as pd
import streamlit as st


SECTION_1_CALIBER_NOTE = (
    "⚠️ Block-size sweep 来自 e1_user_hit_rate（4 档），F4 大数字使用配置的主 "
    "block_size。两者口径不同，请勿直接对比绝对数值。"
)
HISTOGRAM_BINS = 20


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


def histogram_frame(values: pd.Series, bins: int = HISTOGRAM_BINS) -> pd.DataFrame:
    """Bin a numeric series into ``bins`` equal-width buckets.

    Returns a DataFrame indexed by the bucket's left edge (formatted to 4
    decimals so float labels stay readable in ``st.bar_chart``). Empty or
    constant inputs return an empty DataFrame so the caller can branch on
    ``df.empty``.
    """
    cleaned = pd.to_numeric(values, errors="coerce").dropna()
    if cleaned.empty:
        return pd.DataFrame(columns=["count"])
    if cleaned.min() == cleaned.max():
        # cut() refuses a degenerate range; treat as a single-bucket case.
        return pd.DataFrame({"count": [len(cleaned)]},
                            index=[f"{cleaned.iloc[0]:.4f}"])
    binned = pd.cut(cleaned, bins=bins).value_counts().sort_index()
    return pd.DataFrame(
        {"count": binned.values},
        index=[f"{interval.left:.4f}" for interval in binned.index],
    )


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


def _render_section_1(model_dir: Path, report: dict[str, Any]) -> None:
    st.header("1. 理想命中率 (Ideal Hit Rate)")
    s1 = report.get("section_1_ideal_hit")
    if not s1:
        st.info("Section 1 未生成 — F4 / e1 / reuse_rank 都没跑。")
        return

    # ---- F4 big-number card ----
    f4 = s1.get("f4_overall")
    if f4 and f4.get("ideal_hit_ratio") is not None:
        cols = st.columns(3)
        cols[0].metric(
            "Ideal hit ratio (F4)",
            f"{float(f4['ideal_hit_ratio']):.2%}",
        )
        cols[1].metric("Main block_size", str(f4.get("block_size") or "—"))
        cols[2].metric(
            "Hit definition",
            f4.get("hit_definition") or "—",
            help=("content_prefix_reuse = 无限容量 vLLM APC 等价命中数。"
                  "content_block_reuse = 任意位置 reuse，并非 vLLM hit。"),
        )
    else:
        st.caption("F4 未跑 — 大数字卡片无数据。")

    st.divider()

    # ---- block_size sweep ----
    sweep = s1.get("block_size_sweep")
    st.subheader("Block-size sensitivity")
    if not sweep:
        st.caption("e1_user_hit_rate 未跑 — 跳过 sweep。")
    elif sweep.get("sweep_available"):
        df = pd.DataFrame({
            "block_size": sweep["block_sizes"],
            "micro_hit_rate": sweep["micro_hit_rate"],
        }).set_index("block_size")
        st.line_chart(df, y="micro_hit_rate")
        st.caption(SECTION_1_CALIBER_NOTE)
    else:
        bsz = sweep["block_sizes"][0] if sweep.get("block_sizes") else "—"
        rate = sweep["micro_hit_rate"][0] if sweep.get("micro_hit_rate") else None
        rate_str = f"{float(rate):.2%}" if rate is not None else "—"
        st.info(
            f"e1_user_hit_rate 仅跑了单档 block_size={bsz}，"
            f"micro_hit_rate={rate_str}。\n\n"
            "需要 sweep 视图时，把 YAML 改为 `block_sizes: 16,32,64,128` 重跑。"
        )

    st.divider()

    # ---- per-user hit-rate distribution ----
    uhd = s1.get("user_hit_distribution")
    st.subheader("Per-user hit-rate distribution")
    if not uhd or not uhd.get("stats"):
        st.caption("user_hit_distribution 不可用。")
    else:
        stats = uhd["stats"]
        cols = st.columns(4)
        cols[0].metric("p50", f"{float(stats.get('p50', 0)):.2%}")
        cols[1].metric("p80", f"{float(stats.get('p80', 0)):.2%}")
        cols[2].metric("max", f"{float(stats.get('max', 0)):.2%}")
        cols[3].metric("Users", str(stats.get("user_count", "—")))

        csv_rel = uhd.get("csv_path")
        if csv_rel:
            csv_path = model_dir / csv_rel
            if csv_path.is_file():
                try:
                    df = pd.read_csv(csv_path)
                    col = next((c for c in ("hit_rate", "ideal_hit_rate",
                                             "prefix_hit_rate")
                                if c in df.columns), None)
                    if col is None:
                        st.caption(
                            f"未找到 hit_rate 列；CSV 列名: {list(df.columns)}"
                        )
                    else:
                        chart_df = histogram_frame(df[col])
                        if chart_df.empty:
                            st.caption("hit_rate 列为空。")
                        else:
                            st.bar_chart(chart_df)
                except Exception as exc:  # noqa: BLE001 — surface read errors
                    st.caption(f"读取 {csv_rel} 失败: {exc}")
            else:
                st.caption(f"CSV 缺失: `{csv_rel}`")

    st.divider()

    # ---- per-request reuse-rank distribution ----
    rr = s1.get("reuse_rank_distribution")
    st.subheader("Per-request reuse-rank distribution")
    if not rr:
        st.caption("reuse_rank 未跑 — 跳过分布图。")
    else:
        stats = rr.get("stats") or {}
        if stats:
            cols = st.columns(5)
            cols[0].metric("p50", f"{float(stats.get('p50', 0)):.0f}")
            cols[1].metric("p80", f"{float(stats.get('p80', 0)):.0f}")
            cols[2].metric("p95", f"{float(stats.get('p95', 0)):.0f}")
            cols[3].metric("mean", f"{float(stats.get('mean', 0)):.1f}")
            cols[4].metric("max", f"{float(stats.get('max', 0)):.0f}")

        summary = rr.get("summary") or {}
        if summary:
            reuse_rate = summary.get("reuse_rate")
            rate_str = (f"{float(reuse_rate):.2%}"
                        if reuse_rate is not None else "—")
            st.caption(
                f"requests with any reuse: {summary.get('requests_with_any_reuse')} "
                f"/ {summary.get('total_requests')}  ·  reuse rate: {rate_str}"
            )

        csv_rel = rr.get("csv_path")
        if csv_rel:
            csv_path = model_dir / csv_rel
            if csv_path.is_file():
                try:
                    df = pd.read_csv(csv_path)
                    col = "content_prefix_reuse_blocks"
                    if col not in df.columns:
                        st.caption(
                            f"未找到 {col} 列；CSV 列名: {list(df.columns)}"
                        )
                    else:
                        chart_df = histogram_frame(df[col])
                        if chart_df.empty:
                            st.caption(f"{col} 列为空。")
                        else:
                            st.bar_chart(chart_df)
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"读取 {csv_rel} 失败: {exc}")
            else:
                st.caption(f"CSV 缺失: `{csv_rel}`")


def _read_csv_safely(model_dir: Path, csv_rel: str | None) -> pd.DataFrame | None:
    """Read a CSV under ``model_dir/csv_rel``; return None on missing/error."""
    if not csv_rel:
        return None
    path = model_dir / csv_rel
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001 — caller handles missing/None
        return None


def _render_section_2(model_dir: Path, report: dict[str, Any]) -> None:
    st.header("2. 流量业务模式 (Traffic Pattern)")
    s2 = report.get("section_2_traffic")
    if not s2:
        st.info("Section 2 未生成 — traffic_pattern / F9 / F10 都没跑。")
        return

    # ---- Request interval percentiles ----
    intervals = s2.get("request_interval_seconds")
    st.subheader("Inter-arrival intervals (s)")
    if not intervals:
        st.caption("traffic_pattern.interval_percentiles 缺失。")
    else:
        cols = st.columns(4)
        cols[0].metric("p50", f"{float(intervals.get('p50', 0)):.3f}")
        cols[1].metric("p75", f"{float(intervals.get('p75', 0)):.3f}")
        cols[2].metric("p80", f"{float(intervals.get('p80', 0)):.3f}")
        cols[3].metric("p95", f"{float(intervals.get('p95', 0)):.3f}")
        if float(intervals.get("p50", 1)) == 0:
            st.warning(
                "request_interval_p50 = 0s — 同秒并发主导。理想命中率上限"
                "可能虚高，请结合 W-SAME-SECOND 警告解读。"
            )

    st.divider()

    # ---- Volume timeseries ----
    rvt = s2.get("request_volume_timeseries") or {}
    st.subheader("Request volume over time")
    bin_size_s = rvt.get("bin_size_s")
    if bin_size_s:
        st.caption(f"Bin size: {bin_size_s}s")
    df = _read_csv_safely(model_dir, rvt.get("csv_path"))
    if df is None or "bin_start_s" not in df.columns or "request_count" not in df.columns:
        st.caption("volume.csv 不可用或列不正确。")
    elif df.empty:
        st.caption("volume.csv 为空。")
    else:
        st.line_chart(df.set_index("bin_start_s")[["request_count"]])

    st.divider()

    # ---- Block write-rate timeseries ----
    bwr = s2.get("block_write_rate") or {}
    st.subheader("New unique blocks per second")
    total_unique = bwr.get("total_unique_blocks")
    if total_unique is not None:
        st.caption(f"Total unique blocks across the trace: {total_unique:,}")
    df = _read_csv_safely(model_dir, bwr.get("csv_path"))
    if df is None or "second" not in df.columns or "new_unique_blocks" not in df.columns:
        st.caption("write_rate.csv 不可用或列不正确。")
    elif df.empty:
        st.caption("write_rate.csv 为空。")
    else:
        st.line_chart(df.set_index("second")[["new_unique_blocks"]])

    st.divider()

    # ---- Working set bar chart ----
    ws = s2.get("working_set") or {}
    st.subheader("Working set (unique blocks within leading window)")
    windows = ws.get("windows_min") or []
    blocks = ws.get("unique_blocks") or []
    if not windows or len(windows) != len(blocks):
        st.caption("working_set 数据缺失或长度不匹配。")
    else:
        ws_df = pd.DataFrame(
            {"unique_blocks": blocks},
            index=[f"{w} min" for w in windows],
        )
        st.bar_chart(ws_df)
        st.caption(
            "Window 含义：``[t_min, t_min + W*60)`` 起点对齐窗口；表示"
            "「拉满该窗口内全部 reuse 所需的最小 KV cache 容量」。"
        )

    st.divider()

    # ---- Session structure (F9 + F10) ----
    sess = s2.get("session_structure") or {}
    st.subheader("Session structure (F9 / F10)")
    if not sess:
        st.caption("session_structure 不可用 — F9/F10 都没跑。")
    else:
        _render_f9(model_dir, sess.get("f9_turn_count_cdf"))
        _render_f10(model_dir, sess.get("f10_user_turn_stats"))


def _render_f9(model_dir: Path, f9: dict[str, Any] | None) -> None:
    st.markdown("**F9 — turn-count CDF**")
    if not f9:
        st.caption("F9 未跑。")
        return
    cols = st.columns(5)
    cols[0].metric("Total sessions", f"{int(f9.get('total_sessions') or 0):,}")
    cols[1].metric("Single-turn", f"{int(f9.get('single_turn_sessions') or 0):,}")
    cols[2].metric("Multi-turn",  f"{int(f9.get('multi_turn_sessions') or 0):,}")
    cols[3].metric("Max turns", str(f9.get("max_turns") or "—"))
    mean_turns = f9.get("mean_turns")
    cols[4].metric(
        "Mean turns",
        f"{float(mean_turns):.2f}" if mean_turns is not None else "—",
    )

    df = _read_csv_safely(model_dir, f9.get("cdf_csv"))
    if df is None or "turn_count" not in df.columns or "cumulative_fraction" not in df.columns:
        st.caption("f9_cdf.csv 不可用或列不正确。")
    elif df.empty:
        st.caption("f9_cdf.csv 为空。")
    else:
        st.line_chart(df.set_index("turn_count")[["cumulative_fraction"]])


def events_to_cdf(values: pd.Series) -> pd.DataFrame:
    """Empirical CDF DataFrame for a numeric series.

    Returns a DataFrame indexed by the *sorted distinct* values with one
    ``cdf`` column. Duplicate input values collapse to the highest cdf at
    that value (i.e. ``cdf[v] = P[X ≤ v]``).
    """
    cleaned = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if cleaned.empty:
        return pd.DataFrame(columns=["cdf"])
    n = len(cleaned)
    ranks = pd.Series(range(1, n + 1), index=cleaned.values, dtype="float64") / n
    deduped = ranks.groupby(level=0).max()
    return pd.DataFrame({"cdf": deduped.values}, index=deduped.index)


def _render_cdf_chart(df: pd.DataFrame) -> None:
    """Render an F13/F14 CDF chart from a cdf_series.csv DataFrame.

    Honours the ``request_type`` column when present (one line per type);
    otherwise plots a single ``cdf`` line vs ``reuse_time_seconds``.
    """
    if "reuse_time_seconds" not in df.columns or "cdf" not in df.columns:
        st.caption(f"CDF csv 缺少必要列；列名: {list(df.columns)}")
        return
    if df.empty:
        st.caption("CDF csv 为空。")
        return
    if "request_type" in df.columns and df["request_type"].nunique() > 1:
        wide = df.pivot_table(
            index="reuse_time_seconds",
            columns="request_type",
            values="cdf",
            aggfunc="max",
        ).sort_index()
        st.line_chart(wide)
    else:
        st.line_chart(
            df.set_index("reuse_time_seconds")[["cdf"]].sort_index()
        )


def _render_stats_strip(stats: dict[str, Any] | None,
                         keys: list[str], unit: str = "") -> None:
    if not stats:
        st.caption("分位数据缺失。")
        return
    cols = st.columns(len(keys))
    for col, k in zip(cols, keys):
        v = stats.get(k)
        col.metric(k, f"{float(v):.2f}{unit}" if v is not None else "—")


def _render_section_3(model_dir: Path, report: dict[str, Any]) -> None:
    st.header("3. KV cache 时间局部性 (Locality)")
    s3 = report.get("section_3_locality")
    if not s3:
        st.info("Section 3 未生成 — F13 / F14 / reuse_distance 都没跑。")
        return

    # ---- F13 single-turn reuse-time CDF ----
    f13 = s3.get("f13_single_turn")
    st.subheader("F13 — single-turn reuse-time CDF")
    if not f13:
        st.caption("F13 未跑。")
    else:
        st.caption(f"input_definition: `{f13.get('input_definition', '—')}`")
        cnt = f13.get("single_turn_request_count")
        if cnt is not None:
            st.caption(f"single_turn_request_count: {int(cnt):,}")
        _render_stats_strip(f13.get("stats_seconds"),
                            ["p50", "p75", "p80", "p95"], unit=" s")
        df = _read_csv_safely(model_dir, f13.get("cdf_csv"))
        if df is None:
            st.caption(f"cdf_series.csv 不可用: `{f13.get('cdf_csv')}`")
        else:
            _render_cdf_chart(df)

    st.divider()

    # ---- F14 multi-turn reuse-time CDF ----
    f14 = s3.get("f14_multi_turn")
    st.subheader("F14 — multi-turn reuse-time CDF")
    if not f14:
        st.caption("F14 未跑。")
    else:
        st.caption(f"input_definition: `{f14.get('input_definition', '—')}`")
        _render_stats_strip(f14.get("stats_seconds"),
                            ["p50", "p75", "p80", "p95"], unit=" s")
        df = _read_csv_safely(model_dir, f14.get("cdf_csv"))
        if df is None:
            st.caption(f"cdf_series.csv 不可用: `{f14.get('cdf_csv')}`")
        else:
            _render_cdf_chart(df)

    st.divider()

    # ---- reuse_distance: cache pressure indicator ----
    rd = s3.get("reuse_distance")
    st.subheader("Reuse distance — cache pressure indicator")
    if not rd:
        st.caption("reuse_distance 未跑。")
        return

    st.caption(
        rd.get("purpose")
        or "两个 reuse 事件之间插入的 unique block 数；超过 cache 容量时会"
           "触发 LRU 早淘汰。"
    )

    cap = rd.get("available_cache_blocks")
    evicted = rd.get("evicted_under_lru")
    frac = rd.get("evicted_fraction")
    cap_cols = st.columns(3)
    cap_cols[0].metric(
        "Available cache blocks",
        f"{int(cap):,}" if cap is not None else "未配置",
    )
    cap_cols[1].metric(
        "Evicted under LRU",
        f"{int(evicted):,}" if evicted is not None else "—",
    )
    cap_cols[2].metric(
        "Evicted fraction",
        f"{float(frac):.2%}" if frac is not None else "—",
    )
    if cap is None:
        st.caption(
            "ℹ️ `available_cache_blocks` 未配置 — 在 reuse_distance YAML "
            "里填 vLLM 启动日志的 `num_gpu_blocks` 后重跑可得到精确淘汰估计。"
        )

    st.markdown("**Reuse distance percentiles (blocks)**")
    _render_stats_strip(rd.get("stats_blocks"),
                        ["p25", "p50", "p80", "p95"], unit="")
    st.markdown("**Reuse time percentiles for reference (s)**")
    _render_stats_strip(rd.get("reuse_time_stats"),
                        ["p50", "p80", "p95"], unit=" s")

    df = _read_csv_safely(model_dir, rd.get("events_csv"))
    if df is None or "reuse_distance_blocks" not in df.columns:
        st.caption(f"events.csv 不可用: `{rd.get('events_csv')}`")
        return
    cdf_df = events_to_cdf(df["reuse_distance_blocks"])
    if cdf_df.empty:
        st.caption("reuse_distance events 为空。")
    else:
        cdf_df.index.name = "reuse_distance_blocks"
        st.line_chart(cdf_df)


def _render_f10(model_dir: Path, f10: dict[str, Any] | None) -> None:
    st.markdown("**F10 — per-user turn statistics**")
    if not f10:
        st.caption("F10 未跑。")
        return
    cols = st.columns(4)
    cols[0].metric("Users", str(f10.get("total_users") or "—"))
    mean_overall = f10.get("mean_turns_overall")
    std_overall = f10.get("std_turns_overall")
    cols[1].metric(
        "Mean turns / user",
        f"{float(mean_overall):.2f}" if mean_overall is not None else "—",
    )
    cols[2].metric(
        "Std turns / user",
        f"{float(std_overall):.2f}" if std_overall is not None else "—",
    )
    top10 = f10.get("lorenz_top10_pct_share_of_turns")
    cols[3].metric(
        "Top-10% turn share",
        f"{float(top10):.2%}" if top10 is not None else "—",
        help="Top 10% 用户的轮次占总轮次的比例（Lorenz 曲线右尾）。"
             "> 60% 时配合 F9 mean_turns > 3 触发 R-MULTI-TENANT 建议。",
    )

    df = _read_csv_safely(model_dir, f10.get("csv_path"))
    if df is None or "rank" not in df.columns or "cumulative_fraction" not in df.columns:
        st.caption("f10_mean_turns.csv 不可用或列不正确。")
    elif df.empty:
        st.caption("f10_mean_turns.csv 为空。")
    else:
        st.line_chart(df.set_index("rank")[["cumulative_fraction"]])
        st.caption(
            "Lorenz 曲线（按 mean_turns 升序）：横轴用户排名，纵轴累计 turn 占比。"
            "曲线越下凹 → 头部租户越主导。"
        )


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

    model_dir = DEFAULT_OUTPUTS_ROOT / selected
    _render_section_1(model_dir, report)
    st.divider()

    _render_section_2(model_dir, report)
    st.divider()

    _render_section_3(model_dir, report)
    st.divider()

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
