#!/usr/bin/env python3
"""Render a self-contained HTML report from a built v1.1 report.json.

The output is a single file (``outputs/maas/<MODEL>/report.html``) with all
images base64-embedded — no external CSS, JS, or asset directory. Open it
directly in any browser, mail it, drop it in OBS, etc.

Why static instead of Streamlit
-------------------------------
The Streamlit dashboard re-runs the entire script on every interaction,
which is painful when the dashboard is served through a slow proxy
(e.g. ModelArts JupyterLab). Most real reporting needs are read-only —
we just want the page once. Static HTML eliminates server round-trips,
caches forever in the browser, and travels cleanly as a deliverable.

Streamlit (``scripts/dashboard.py``) is kept for development / interactive
exploration; this script is the default operations path.

Usage
-----
    # Single model
    python scripts/render_static_report.py --model qwen_v3_5_27b_64k

    # All models with report.json present
    python scripts/render_static_report.py --all

    # Custom outputs root (e.g. staging tree)
    python scripts/render_static_report.py --model demo --outputs-root /tmp/x
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# matplotlib for the 3 traffic_pattern charts that no analysis pre-renders.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "maas"


# ---------------------------------------------------------------------------
# Helpers — embed PNGs and generate inline charts
# ---------------------------------------------------------------------------

def _png_to_data_uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    raw = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _read_csv(model_dir: Path, csv_rel: str | None) -> pd.DataFrame | None:
    if not csv_rel:
        return None
    p = model_dir / csv_rel
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None


def _img(src: str | None, alt: str, *, max_width: str = "100%") -> str:
    if not src:
        return f"<p class='muted'>[{html.escape(alt)} 图缺失]</p>"
    return (f"<img src='{src}' alt='{html.escape(alt)}' "
            f"style='max-width:{max_width};height:auto;display:block;"
            "margin:0.5rem 0;border:1px solid #e5e7eb;border-radius:4px'/>")


def _generate_traffic_charts(
    model_dir: Path, traffic: dict[str, Any]
) -> tuple[str | None, str | None, str | None]:
    """Render 3 inline matplotlib charts for the traffic section."""
    volume_uri = write_rate_uri = working_set_uri = None

    # Volume timeseries
    df = _read_csv(model_dir, "traffic_pattern/volume.csv")
    if df is not None and {"bin_start_s", "request_count"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.plot(df["bin_start_s"] / 60, df["request_count"], lw=1.2)
        ax.set_xlabel("time (min, since window start)")
        ax.set_ylabel("requests / bin")
        bin_size = traffic.get("request_volume_timeseries", {}).get("bin_size_s", 60)
        ax.set_title(f"Request volume (bin = {bin_size}s)")
        ax.grid(True, alpha=0.3)
        volume_uri = _fig_to_data_uri(fig)

    # Write-rate timeseries
    df = _read_csv(model_dir, "traffic_pattern/write_rate.csv")
    if df is not None and {"second", "new_unique_blocks"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.plot(df["second"] / 60, df["new_unique_blocks"], lw=0.9, color="#dc2626")
        ax.set_xlabel("time (min, since window start)")
        ax.set_ylabel("new unique blocks / s")
        ax.set_title("Block write rate (first-appearance per second)")
        ax.grid(True, alpha=0.3)
        write_rate_uri = _fig_to_data_uri(fig)

    # Working set bar
    ws = traffic.get("working_set", {})
    windows = ws.get("windows_min") or []
    blocks = ws.get("unique_blocks") or []
    if windows and len(windows) == len(blocks):
        fig, ax = plt.subplots(figsize=(5, 2.6))
        labels = [f"{w} min" for w in windows]
        ax.bar(labels, blocks, color="#2563eb")
        ax.set_ylabel("unique blocks")
        ax.set_title("Working set (leading window from t_min)")
        for i, v in enumerate(blocks):
            ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        working_set_uri = _fig_to_data_uri(fig)

    return volume_uri, write_rate_uri, working_set_uri


# ---------------------------------------------------------------------------
# HTML helpers — small primitives reused by every section
# ---------------------------------------------------------------------------

def _escape(s: Any) -> str:
    return html.escape(str(s)) if s is not None else "—"


def _fmt_pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(v: Any, fmt: str = ",.0f") -> str:
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "—"


def _metric(label: str, value: str, hint: str = "") -> str:
    h = (f"<div class='hint'>{html.escape(hint)}</div>"
         if hint else "")
    return (f"<div class='metric'><div class='label'>{html.escape(label)}</div>"
            f"<div class='value'>{value}</div>{h}</div>")


def _metric_strip(items: list[tuple[str, str]]) -> str:
    cells = "".join(_metric(k, v) for k, v in items)
    return f"<div class='metric-strip'>{cells}</div>"


# ---------------------------------------------------------------------------
# Section renderers — each returns an HTML fragment string
# ---------------------------------------------------------------------------

def _section_1(model_dir: Path, report: dict) -> str:
    s1 = report.get("section_1_ideal_hit") or {}
    if not s1:
        return ("<section><h2>1. 理想命中率 (Ideal Hit Rate)</h2>"
                "<p class='muted'>未生成 — F4 / e1 / reuse_rank 都没跑。</p></section>")

    out = ["<section><h2>1. 理想命中率 (Ideal Hit Rate)</h2>"]
    f4 = s1.get("f4_overall") or {}
    if f4.get("ideal_hit_ratio") is not None:
        out.append(_metric_strip([
            ("Ideal hit ratio (F4)", _fmt_pct(f4.get("ideal_hit_ratio"))),
            ("Main block_size", _escape(f4.get("block_size"))),
            ("Hit definition", _escape(f4.get("hit_definition"))),
        ]))

    out.append("<h3>F4 — 理想命中率随时间变化</h3>")
    out.append(_img(_png_to_data_uri(model_dir / "f4_prefix" / "plot.png"), "F4"))

    sweep = s1.get("block_size_sweep") or {}
    if sweep.get("sweep_available"):
        out.append("<h3>Block-size 敏感度（4 档 sweep）</h3>")
        out.append(_img(_png_to_data_uri(model_dir / "e1_user_hit_rate" / "plot.png"),
                        "e1 sweep"))
        out.append("<p class='caliber'>⚠️ Block-size sweep 来自 e1_user_hit_rate"
                   "（4 档），F4 大数字使用配置的主 block_size。两者口径不同。</p>")
    elif sweep:
        bsz = sweep.get("block_sizes", [128])[0] if sweep.get("block_sizes") else "—"
        rate = (sweep.get("micro_hit_rate") or [None])[0]
        rate_str = _fmt_pct(rate) if rate is not None else "—"
        out.append("<h3>Block-size 敏感度</h3>")
        out.append(f"<p class='muted'>e1_user_hit_rate 仅跑了单档 block_size={bsz}，"
                   f"micro_hit_rate={rate_str}。如需 sweep 视图改 YAML 重跑。</p>")

    uhd = s1.get("user_hit_distribution") or {}
    if uhd.get("stats"):
        st_ = uhd["stats"]
        out.append("<h3>用户级 hit rate 分布</h3>")
        out.append(_metric_strip([
            ("p50", _fmt_pct(st_.get("p50"))),
            ("p80", _fmt_pct(st_.get("p80"))),
            ("max", _fmt_pct(st_.get("max"))),
            ("Users", _escape(st_.get("user_count"))),
        ]))
        out.append(_img(_png_to_data_uri(model_dir / "e1b_skewness" / "hit_contribution.png"),
                        "e1b hit contribution"))

    rr = s1.get("reuse_rank_distribution") or {}
    rr_stats = rr.get("stats") or {}
    if rr_stats:
        out.append("<h3>每请求 reuse-rank 分布</h3>")
        out.append(_metric_strip([
            ("p50", _fmt_num(rr_stats.get("p50"))),
            ("p80", _fmt_num(rr_stats.get("p80"))),
            ("p95", _fmt_num(rr_stats.get("p95"))),
            ("mean", _fmt_num(rr_stats.get("mean"), ",.1f")),
            ("max", _fmt_num(rr_stats.get("max"))),
        ]))
        out.append(_img(_png_to_data_uri(model_dir / "reuse_rank" / "reuse_rank.png"),
                        "reuse_rank"))

    out.append("</section>")
    return "\n".join(out)


def _section_2(model_dir: Path, report: dict) -> str:
    s2 = report.get("section_2_traffic") or {}
    if not s2:
        return ("<section><h2>2. 流量业务模式 (Traffic Pattern)</h2>"
                "<p class='muted'>未生成 — traffic_pattern / F9 / F10 都没跑。</p></section>")

    out = ["<section><h2>2. 流量业务模式 (Traffic Pattern)</h2>"]

    iv = s2.get("request_interval_seconds") or {}
    if iv:
        out.append("<h3>请求间隔分位数（秒）</h3>")
        out.append(_metric_strip([
            ("p50", _fmt_num(iv.get("p50"), ",.3f")),
            ("p75", _fmt_num(iv.get("p75"), ",.3f")),
            ("p80", _fmt_num(iv.get("p80"), ",.3f")),
            ("p95", _fmt_num(iv.get("p95"), ",.3f")),
        ]))
        if float(iv.get("p50") or 0) == 0:
            out.append("<p class='warning'>⚠️ request_interval_p50 = 0s — "
                       "理想命中率上限可能虚高（同秒并发主导）。</p>")

    volume_uri, write_rate_uri, working_set_uri = _generate_traffic_charts(
        model_dir, s2
    )
    out.append("<h3>请求量时序</h3>")
    out.append(_img(volume_uri, "volume timeseries"))
    out.append("<h3>新写入 unique block / 秒</h3>")
    bwr = s2.get("block_write_rate") or {}
    if bwr.get("total_unique_blocks") is not None:
        out.append(f"<p class='muted'>整窗口 unique blocks: "
                   f"{_fmt_num(bwr['total_unique_blocks'])}</p>")
    out.append(_img(write_rate_uri, "write rate timeseries"))
    out.append("<h3>工作集（leading window）</h3>")
    out.append(_img(working_set_uri, "working set bar", max_width="500px"))

    sess = s2.get("session_structure") or {}
    if sess:
        out.append("<h3>会话结构 (F9 / F10)</h3>")
        f9 = sess.get("f9_turn_count_cdf") or {}
        if f9:
            out.append("<h4>F9 — 会话轮数 CDF</h4>")
            out.append(_metric_strip([
                ("Total sessions", _fmt_num(f9.get("total_sessions"))),
                ("Single-turn", _fmt_num(f9.get("single_turn_sessions"))),
                ("Multi-turn", _fmt_num(f9.get("multi_turn_sessions"))),
                ("Max turns", _escape(f9.get("max_turns"))),
                ("Mean turns", _fmt_num(f9.get("mean_turns"), ",.2f")),
            ]))
            out.append(_img(_png_to_data_uri(model_dir / "f9_agent" / "plot.png"),
                            "F9 plot"))
        f10 = sess.get("f10_user_turn_stats") or {}
        if f10:
            out.append("<h4>F10 — 用户级轮数统计 (Lorenz)</h4>")
            top10 = f10.get("lorenz_top10_pct_share_of_turns")
            out.append(_metric_strip([
                ("Users", _escape(f10.get("total_users"))),
                ("Mean turns / user",
                 _fmt_num(f10.get("mean_turns_overall"), ",.2f")),
                ("Std turns / user",
                 _fmt_num(f10.get("std_turns_overall"), ",.2f")),
                ("Top-10% turn share", _fmt_pct(top10)),
            ]))
            out.append(_img(_png_to_data_uri(model_dir / "f10_agent" / "plot.png"),
                            "F10 plot"))

    out.append("</section>")
    return "\n".join(out)


def _section_3(model_dir: Path, report: dict) -> str:
    s3 = report.get("section_3_locality") or {}
    if not s3:
        return ("<section><h2>3. KV cache 时间局部性 (Locality)</h2>"
                "<p class='muted'>未生成 — F13 / F14 / reuse_distance 都没跑。</p></section>")

    out = ["<section><h2>3. KV cache 时间局部性 (Locality)</h2>"]

    f13 = s3.get("f13_single_turn") or {}
    if f13:
        out.append("<h3>F13 — 单轮 reuse-time CDF</h3>")
        out.append(f"<p class='muted'>input_definition: <code>"
                   f"{_escape(f13.get('input_definition'))}</code></p>")
        st_ = f13.get("stats_seconds") or {}
        if st_:
            out.append(_metric_strip([
                ("p50 (s)", _fmt_num(st_.get("p50"), ",.2f")),
                ("p75 (s)", _fmt_num(st_.get("p75"), ",.2f")),
                ("p80 (s)", _fmt_num(st_.get("p80"), ",.2f")),
                ("p95 (s)", _fmt_num(st_.get("p95"), ",.2f")),
            ]))
        out.append(_img(_png_to_data_uri(model_dir / "f13_prefix" / "plot.png"),
                        "F13 CDF"))

    f14 = s3.get("f14_multi_turn") or {}
    if f14:
        out.append("<h3>F14 — 多轮 reuse-time CDF</h3>")
        st_ = f14.get("stats_seconds") or {}
        if st_:
            out.append(_metric_strip([
                ("p50 (s)", _fmt_num(st_.get("p50"), ",.2f")),
                ("p75 (s)", _fmt_num(st_.get("p75"), ",.2f")),
                ("p80 (s)", _fmt_num(st_.get("p80"), ",.2f")),
                ("p95 (s)", _fmt_num(st_.get("p95"), ",.2f")),
            ]))
        out.append(_img(_png_to_data_uri(model_dir / "f14_prefix" / "plot.png"),
                        "F14 CDF"))

    rd = s3.get("reuse_distance") or {}
    if rd:
        out.append("<h3>Reuse distance — cache 压力指标</h3>")
        out.append(f"<p class='muted'>{_escape(rd.get('purpose'))}</p>")
        cap = rd.get("available_cache_blocks")
        evicted = rd.get("evicted_under_lru")
        frac = rd.get("evicted_fraction")
        out.append(_metric_strip([
            ("Available cache blocks",
             _fmt_num(cap) if cap is not None else "未配置"),
            ("Evicted under LRU",
             _fmt_num(evicted) if evicted is not None else "—"),
            ("Evicted fraction", _fmt_pct(frac)),
        ]))
        sb = rd.get("stats_blocks") or {}
        if sb:
            out.append("<h4>Reuse distance 分位（blocks）</h4>")
            out.append(_metric_strip([
                ("p25", _fmt_num(sb.get("p25"))),
                ("p50", _fmt_num(sb.get("p50"))),
                ("p80", _fmt_num(sb.get("p80"))),
                ("p95", _fmt_num(sb.get("p95"))),
            ]))
        rt = rd.get("reuse_time_stats") or {}
        if rt:
            out.append("<h4>Reuse time 分位（参考，秒）</h4>")
            out.append(_metric_strip([
                ("p50", _fmt_num(rt.get("p50"), ",.2f")),
                ("p80", _fmt_num(rt.get("p80"), ",.2f")),
                ("p95", _fmt_num(rt.get("p95"), ",.2f")),
            ]))
        out.append(_img(
            _png_to_data_uri(model_dir / "reuse_distance" / "reuse_distance_cdf.png"),
            "reuse_distance CDF"))

    out.append("</section>")
    return "\n".join(out)


CONTENT_TYPE_EMOJI: dict[str, str] = {
    "json_schema": "🧩", "agent_tool_prompt": "🔧", "system_prompt": "🧭",
    "rag_template": "📚", "code": "💻", "qa_template": "❓",
    "long_document": "📄", "other": "·",
}


def _section_4(model_dir: Path, report: dict) -> str:
    s4 = report.get("section_4_content") or {}
    if not s4:
        return ("<section><h2>4. 可复用内容 (Content)</h2>"
                "<p class='muted'>未生成 — common_prefix 没跑。</p></section>")

    out = ["<section><h2>4. 可复用内容 (Content)</h2>"]
    out.append(_metric_strip([
        ("Prefix length",
         f"{_fmt_num(s4.get('prefix_length_blocks'))} blocks"),
        ("Prefix chars", _fmt_num(s4.get('prefix_length_chars'))),
        ("Mean coverage", _fmt_pct((s4.get("mean_coverage_pct") or 0) / 100)),
        ("min_count", _escape(s4.get("min_count_threshold"))),
    ]))
    out.append(_img(_png_to_data_uri(model_dir / "common_prefix" / "coverage_plot.png"),
                    "common_prefix coverage"))

    blocks = s4.get("consensus_blocks") or []
    if blocks:
        out.append(f"<h3>Top {len(blocks)} consensus blocks</h3>")
        rows = []
        for b in blocks:
            ctype = b.get("content_type_guess") or "other"
            emoji = CONTENT_TYPE_EMOJI.get(ctype, "·")
            preview = (b.get("text_preview") or "").replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "…"
            rows.append(
                "<tr>"
                f"<td>{_escape(b.get('rank'))}</td>"
                f"<td>{_escape(b.get('position'))}</td>"
                f"<td>{_fmt_num(b.get('count'))}</td>"
                f"<td>{_fmt_num(b.get('coverage_pct'), ',.2f')}%</td>"
                f"<td>{emoji} {_escape(ctype)}</td>"
                f"<td><code>{_escape(preview)}</code></td>"
                "</tr>"
            )
        out.append(
            "<table class='consensus'><thead><tr>"
            "<th>rank</th><th>position</th><th>count</th><th>coverage</th>"
            "<th>type</th><th>text_preview</th>"
            "</tr></thead><tbody>"
            + "\n".join(rows)
            + "</tbody></table>"
        )

    preview = s4.get("decoded_text_preview") or ""
    if preview:
        out.append("<h3>Decoded prefix text (前 500 字符)</h3>")
        out.append(f"<pre class='preview'>{_escape(preview)}</pre>")

    out.append("</section>")
    return "\n".join(out)


PRIORITY_BADGE = {"P0": "🔴 P0", "P1": "🟠 P1", "P2": "🟡 P2", None: "⚠️ Warning"}
PRIORITY_TITLE = {
    "P0": "🔴 P0 — 立即处理", "P1": "🟠 P1 — 中期规划",
    "P2": "🟡 P2 — 战略调整", "warning": "⚠️ 数据质量警告",
}


def _section_5(report: dict) -> str:
    recs = report.get("section_5_recommendations") or []
    out = ["<section><h2>5. 优化建议 (Recommendations)</h2>"]
    if not recs:
        out.append("<p class='success'>✅ 当前没有规则触发 — 数据看起来稳定，"
                   "或上游分析输入不完整。</p></section>")
        return "\n".join(out)

    n_rec = sum(1 for r in recs if r.get("type") != "warning")
    n_warn = sum(1 for r in recs if r.get("type") == "warning")
    out.append(f"<p class='muted'>共 {n_rec} 条建议 + {n_warn} 条警告</p>")

    groups: dict[str, list[dict]] = {"P0": [], "P1": [], "P2": [], "warning": []}
    for r in recs:
        if r.get("type") == "warning":
            groups["warning"].append(r)
        else:
            p = r.get("priority")
            groups[p if p in ("P0", "P1", "P2") else "P2"].append(r)

    for key in ("P0", "P1", "P2", "warning"):
        items = groups[key]
        if not items:
            continue
        out.append(f"<h3>{html.escape(PRIORITY_TITLE[key])}</h3>")
        for r in items:
            badge = PRIORITY_BADGE.get(r.get("priority"), "·")
            ev_html = "".join(f"<li>{_escape(e)}</li>"
                              for e in (r.get("evidence") or []))
            out.append(
                "<div class='card'>"
                f"<div class='card-head'>"
                f"<span class='badge'>{html.escape(badge)}</span>"
                f"<code class='rule-id'>{_escape(r.get('rule_id'))}</code>"
                f"<span class='conf'>confidence: <b>"
                f"{_escape(r.get('confidence'))}</b></span>"
                "</div>"
                f"<div class='conclusion'><b>结论：</b>"
                f"{_escape(r.get('conclusion'))}</div>"
                + (f"<div><b>证据：</b><ul>{ev_html}</ul></div>"
                   if ev_html else "")
                + (f"<div><b>建议行动：</b>{_escape(r.get('action'))}</div>"
                   if r.get("action") else "")
                + "</div>"
            )

    out.append("</section>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
       "Microsoft YaHei", sans-serif; margin: 0; padding: 1.5rem 2rem;
       max-width: 1100px; margin-left: auto; margin-right: auto;
       color: #111827; line-height: 1.55; background: #fafafa; }
h1 { font-size: 1.6rem; margin: 0 0 0.3rem; }
h2 { font-size: 1.25rem; margin: 2rem 0 0.6rem; padding-bottom: 0.3rem;
     border-bottom: 2px solid #e5e7eb; }
h3 { font-size: 1.05rem; margin: 1.4rem 0 0.4rem; color: #1f2937; }
h4 { font-size: 0.95rem; margin: 1rem 0 0.3rem; color: #374151; }
header .meta { color: #6b7280; font-size: 0.88rem; }
.metric-strip { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.6rem 0; }
.metric { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
          padding: 0.55rem 0.9rem; min-width: 120px; flex: 1 1 120px; }
.metric .label { font-size: 0.78rem; color: #6b7280; }
.metric .value { font-size: 1.15rem; font-weight: 600; color: #111827; }
.metric .hint  { font-size: 0.7rem;  color: #9ca3af; margin-top: 0.1rem; }
.muted { color: #6b7280; font-size: 0.88rem; }
.warning { background: #fef3c7; border-left: 3px solid #f59e0b;
           padding: 0.5rem 0.9rem; border-radius: 4px; }
.caliber { background: #fefce8; padding: 0.5rem 0.9rem; border-radius: 4px;
           font-size: 0.88rem; }
.success { background: #d1fae5; padding: 0.7rem 0.9rem; border-radius: 4px; }
section { background: #fff; padding: 1rem 1.4rem 1.4rem; border-radius: 8px;
          border: 1px solid #e5e7eb; margin: 1rem 0; }
table.consensus { width: 100%; border-collapse: collapse; font-size: 0.85rem;
                  margin: 0.6rem 0; }
table.consensus th, table.consensus td { border-bottom: 1px solid #e5e7eb;
                                          padding: 0.4rem 0.6rem; text-align: left;
                                          vertical-align: top; }
table.consensus th { background: #f9fafb; font-weight: 600; }
table.consensus code { background: #f3f4f6; padding: 0.1rem 0.3rem;
                        border-radius: 3px; font-size: 0.78rem;
                        word-break: break-all; }
pre.preview { background: #f9fafb; border: 1px solid #e5e7eb; padding: 0.7rem;
              border-radius: 4px; max-height: 240px; overflow: auto;
              font-size: 0.78rem; white-space: pre-wrap; word-break: break-all; }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
        padding: 0.8rem 1rem; margin: 0.5rem 0; }
.card-head { display: flex; gap: 0.7rem; align-items: center; flex-wrap: wrap;
             margin-bottom: 0.4rem; padding-bottom: 0.4rem;
             border-bottom: 1px dashed #e5e7eb; }
.card .badge { font-weight: 600; }
.card .rule-id { background: #f3f4f6; padding: 0.1rem 0.4rem; border-radius: 3px;
                  font-size: 0.85rem; }
.card .conf { color: #6b7280; font-size: 0.85rem; }
.card .conclusion { margin: 0.3rem 0; }
.card ul { margin: 0.2rem 0; padding-left: 1.2rem; }
footer { color: #9ca3af; font-size: 0.78rem; text-align: center; margin: 2rem 0; }
"""


def _render_html(model_id: str, model_dir: Path, report: dict) -> str:
    meta = report.get("meta") or {}
    scope = report.get("scope") or {}
    time_range = meta.get("time_range") or {}

    header_meta = (
        f"block_size: <b>{_escape(meta.get('block_size'))}</b> · "
        f"requests: <b>{_fmt_num(meta.get('total_requests'))}</b> · "
        f"users: <b>{_escape(meta.get('total_users'))}</b> · "
        f"window: <b>{_escape(time_range.get('duration_h'))} h</b> · "
        f"generated_at: <code>{_escape(meta.get('generated_at'))}</code> · "
        f"data_version: <code>{_escape(meta.get('data_version'))}</code>"
    )
    rendered_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    sections = [
        _section_1(model_dir, report),
        _section_2(model_dir, report),
        _section_3(model_dir, report),
        _section_4(model_dir, report),
        _section_5(report),
    ]

    return (
        "<!DOCTYPE html>\n"
        "<html lang='zh-Hans'>\n"
        "<head>\n"
        f"<meta charset='utf-8'><title>{_escape(model_id)} — Prefix Cache Report</title>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<header><h1>{_escape(model_id)} — Prefix Cache Analysis</h1>"
        f"<div class='meta'>{header_meta}</div></header>\n"
        + "\n".join(sections)
        + "\n<footer>静态报告由 <code>scripts/render_static_report.py</code> 生成 "
        f"· rendered_at: <code>{_escape(rendered_at)}</code> "
        f"· schema: {_escape(report.get('schema_version'))}</footer>\n"
        "</body></html>\n"
    )


def render_one(outputs_root: Path, model_id: str) -> Path:
    model_dir = outputs_root / model_id
    report_path = model_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"report.json not found for {model_id}: {report_path}. "
            "Run build_model_report (or run_dashboard_pipeline.sh) first."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    html_str = _render_html(model_id, model_dir, report)
    out_path = model_dir / "report.html"
    out_path.write_text(html_str, encoding="utf-8")
    return out_path


def discover_reports(outputs_root: Path) -> list[str]:
    if not outputs_root.exists():
        return []
    return sorted(
        d.name for d in outputs_root.iterdir()
        if d.is_dir() and (d / "report.json").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render outputs/maas/<MODEL>/report.json → report.html (single file)"
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="Model slug")
    g.add_argument("--all", action="store_true",
                   help="Render every model with a report.json")
    parser.add_argument("--outputs-root", default=str(DEFAULT_OUTPUTS_ROOT),
                       help=f"Root of per-model outputs/ trees "
                            f"[default: {DEFAULT_OUTPUTS_ROOT}]")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)

    if args.all:
        models = discover_reports(outputs_root)
        if not models:
            print(f"[ERROR] No report.json under {outputs_root}", file=sys.stderr)
            sys.exit(1)
    else:
        models = [args.model]

    failures: list[tuple[str, str]] = []
    for slug in models:
        try:
            out = render_one(outputs_root, slug)
            print(f"  [OK] {out}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {slug}: {exc}", file=sys.stderr)
            failures.append((slug, str(exc)))

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
