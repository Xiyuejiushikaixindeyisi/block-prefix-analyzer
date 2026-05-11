"""Recommendation rules for phase-1 reports.

Each rule is a pure function ``(report: dict) -> Recommendation | None``.
A rule returns ``None`` when its input data is missing OR when the
trigger condition is not met. Triggers and outputs are frozen by the
user-confirmed ``docs/可视化.md §5`` table.

Rule catalog (priority then alphabetical):

  R-PIN-CHAIN        P0 high      pin core chain when reuse rate is mid &
                                  prefix is long & dominant
  R-CACHE-PRESSURE   P1 high      reuse_distance p80 exceeds available cache
  R-CAPACITY-FIRST   P1 high      high reuse + large working set → capacity
                                  experiment, not eviction tuning
  R-BATCH-TTL        P1 medium    very short reuse times → batch-level TTL
  R-LONG-TTL         P1 medium    p80 reuse_time > 30s → TTL-LRU with long base
  R-MULTI-TENANT     P2 medium    multi-turn-deep + tenant-skewed turn share
                                  → tenant partitioning
  R-LOW-CEILING      P2 high      ideal_hit < 0.30 → don't optimise eviction;
                                  pivot to tenant isolation / RAG cache
  W-SAME-SECOND      Warning      request_interval_p50 == 0 (sub-second
                                  bursts ⇒ ideal_hit may be inflated)
  W-REUSE-ZERO       Warning      reuse_time_p50 == 0 (intra-batch, needs
                                  validation)

All evidence strings are human-readable and round numbers to 4 sig figs.
"""
from __future__ import annotations

from typing import Any, Callable

from block_prefix_analyzer.recommendation.engine import Recommendation


# ---------------------------------------------------------------------------
# Safe nested-dict access
# ---------------------------------------------------------------------------

def _get(report: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = report
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _working_set_at(report: dict, w_min: int) -> int | None:
    ws = _get(report, "section_2_traffic", "working_set")
    if not isinstance(ws, dict):
        return None
    windows = ws.get("windows_min") or []
    blocks = ws.get("unique_blocks") or []
    for i, w in enumerate(windows):
        if w == w_min and i < len(blocks):
            try:
                return int(blocks[i])
            except (TypeError, ValueError):
                return None
    return None


def _ideal_hit(report: dict) -> float | None:
    val = _get(report, "section_1_ideal_hit", "f4_overall", "ideal_hit_ratio")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Rules — each ``rule_<id>(report)`` returns Recommendation or None.
# ---------------------------------------------------------------------------

def rule_pin_chain(report: dict) -> Recommendation | None:
    """R-PIN-CHAIN: pin a long, dominant common prefix when hit ratio is mid."""
    prefix_len = _get(report, "section_4_content", "prefix_length_blocks")
    consensus = _get(report, "section_4_content", "consensus_blocks") or []
    head_cov = consensus[0].get("global_coverage_pct") if consensus else None
    ideal_hit = _ideal_hit(report)

    if prefix_len is None or head_cov is None or ideal_hit is None:
        return None
    if not (prefix_len >= 32 and head_cov >= 60.0 and 0.40 <= ideal_hit <= 0.70):
        return None

    return Recommendation(
        rule_id="R-PIN-CHAIN",
        type="recommendation",
        priority="P0",
        confidence="high",
        conclusion=f"建议对 {prefix_len} 个核心 chain block 做 pin 保护。",
        evidence=(
            f"F4 ideal_hit_ratio = {ideal_hit:.4f}",
            f"common_prefix prefix_length_blocks = {prefix_len}",
            f"top-1 consensus block coverage = {head_cov:.1f}%",
        ),
        action="在 cache manager 实现 pinned block set，初期灰度一个实例。",
    )


def rule_capacity_first(report: dict) -> Recommendation | None:
    """R-CAPACITY-FIRST: high ceiling + big working set → grow cache, not algorithm."""
    ideal_hit = _ideal_hit(report)
    ws_60 = _working_set_at(report, 60)

    if ideal_hit is None or ws_60 is None:
        return None
    if not (ideal_hit >= 0.70 and ws_60 > 100_000):
        return None

    return Recommendation(
        rule_id="R-CAPACITY-FIRST",
        type="recommendation",
        priority="P1",
        confidence="high",
        conclusion="理想命中率高但 60-min 工作集大；优先做容量梯度实验，再谈淘汰算法。",
        evidence=(
            f"F4 ideal_hit_ratio = {ideal_hit:.4f}",
            f"working_set @ 60min = {ws_60:,} unique blocks",
        ),
        action="按 1×/2×/4× 当前 KV cache 容量做对比实验，定位拐点。",
    )


def rule_low_ceiling(report: dict) -> Recommendation | None:
    """R-LOW-CEILING: when even infinite cache yields <30% hit, eviction tuning is futile."""
    ideal_hit = _ideal_hit(report)
    if ideal_hit is None or ideal_hit >= 0.30:
        return None

    return Recommendation(
        rule_id="R-LOW-CEILING",
        type="recommendation",
        priority="P2",
        confidence="high",
        conclusion=f"理想命中率仅 {ideal_hit:.4f} (<30%)；不建议优先优化 cache 淘汰。",
        evidence=(
            f"F4 ideal_hit_ratio = {ideal_hit:.4f}",
            "无限容量假设下命中率仍偏低，瓶颈在内容相似度而非 cache 大小。",
        ),
        action="将工程精力投向租户隔离、RAG-side cache、Prompt 模板治理。",
    )


def rule_batch_ttl(report: dict) -> Recommendation | None:
    """R-BATCH-TTL: very short reuse times → batch-window TTL protection."""
    p50 = _get(report, "section_3_locality", "f13_single_turn", "stats_seconds", "p50")
    p80 = _get(report, "section_3_locality", "f13_single_turn", "stats_seconds", "p80")
    ideal_hit = _ideal_hit(report)
    if p50 is None or p80 is None or ideal_hit is None:
        return None
    if not (p50 <= 1.0 and p80 <= 5.0 and ideal_hit >= 0.40):
        return None

    return Recommendation(
        rule_id="R-BATCH-TTL",
        type="recommendation",
        priority="P1",
        confidence="medium",
        conclusion="reuse 高度集中在亚秒级；建议 batch 级短 TTL 保护，避免误淘汰。",
        evidence=(
            f"F13 reuse_time p50 = {p50}s, p80 = {p80}s",
            f"F4 ideal_hit_ratio = {ideal_hit:.4f}",
        ),
        action="为相邻 batch 内的请求组配置 ≥ p80 的最小驻留时间。",
    )


def rule_long_ttl(report: dict) -> Recommendation | None:
    """R-LONG-TTL: long-tail reuse times → TTL-LRU with base TTL anchored at p80."""
    p80 = _get(report, "section_3_locality", "f13_single_turn", "stats_seconds", "p80")
    ideal_hit = _ideal_hit(report)
    if p80 is None or ideal_hit is None:
        return None
    if not (p80 > 30.0 and ideal_hit > 0.50):
        return None

    return Recommendation(
        rule_id="R-LONG-TTL",
        type="recommendation",
        priority="P1",
        confidence="medium",
        conclusion=f"reuse 长尾显著（p80 = {p80}s）；尝试 TTL-LRU，base_ttl 锚定 p80。",
        evidence=(
            f"F13 reuse_time p80 = {p80}s",
            f"F4 ideal_hit_ratio = {ideal_hit:.4f}",
        ),
        action=f"灰度 TTL-LRU，base_ttl = {p80}s，对比命中率与淘汰率。",
    )


def rule_multi_tenant(report: dict) -> Recommendation | None:
    """R-MULTI-TENANT: top-10% tenants own >60% of turns AND mean_turns > 3."""
    f9_mean = _get(report, "section_2_traffic", "session_structure",
                   "f9_turn_count_cdf", "mean_turns")
    top10_share = _get(report, "section_2_traffic", "session_structure",
                       "f10_user_turn_stats", "lorenz_top10_pct_share_of_turns")
    if f9_mean is None or top10_share is None:
        return None
    if not (top10_share > 0.60 and f9_mean > 3.0):
        return None

    return Recommendation(
        rule_id="R-MULTI-TENANT",
        type="recommendation",
        priority="P2",
        confidence="medium",
        conclusion="多轮深度 + 头部租户主导轮次；存在 chain 资源竞争，建议租户分区。",
        evidence=(
            f"F9 mean_turns = {f9_mean}",
            f"F10 top-10% 用户贡献 turns 占比 = {top10_share:.4f}",
        ),
        action="按租户做 cache 子池或路由分区，隔离头部长会话租户。",
    )


def rule_cache_pressure(report: dict) -> Recommendation | None:
    """R-CACHE-PRESSURE: reuse_distance p80 exceeds available cache."""
    p80 = _get(report, "section_3_locality", "reuse_distance", "stats_blocks", "p80")
    cap = _get(report, "section_3_locality", "reuse_distance", "available_cache_blocks")
    if p80 is None:
        return None

    fired_capacity = cap is not None and p80 > cap
    fired_fallback = cap is None and p80 > 50_000
    if not (fired_capacity or fired_fallback):
        return None

    if fired_capacity:
        evidence = (
            f"reuse_distance p80 = {p80} blocks",
            f"available_cache_blocks = {cap}",
            f"p80 / capacity = {p80 / cap:.2f}× — eviction faster than reuse.",
        )
        action = ("扩容（短期）或评估分级存储 / 路由策略（长期）；同时观察 "
                  "evicted_under_lru 比例。")
    else:
        evidence = (
            f"reuse_distance p80 = {p80} blocks",
            "available_cache_blocks 未配置，按 50K 兜底阈值判定。",
        )
        action = "先在 reuse_distance YAML 填入真实 num_gpu_blocks，再评估扩容/分级。"

    return Recommendation(
        rule_id="R-CACHE-PRESSURE",
        type="recommendation",
        priority="P1",
        confidence="high",
        conclusion="reuse 之间插入的 unique block 数超过可用容量，存在 LRU 早淘汰风险。",
        evidence=evidence,
        action=action,
    )


def rule_warn_same_second(report: dict) -> Recommendation | None:
    """W-SAME-SECOND: request_interval_p50 == 0 ⇒ ideal_hit may be inflated."""
    p50 = _get(report, "section_2_traffic", "request_interval_seconds", "p50")
    if p50 is None or p50 != 0:
        return None
    return Recommendation(
        rule_id="W-SAME-SECOND",
        type="warning",
        priority=None,
        confidence="medium",
        conclusion="request_interval_p50 = 0s；理想命中率可能因同秒并发被高估。",
        evidence=("section_2_traffic.request_interval_seconds.p50 == 0.0",),
        action="下一阶段补毫秒级 timestamp 或 batch_id 验证；本报告数值上限解读需谨慎。",
    )


def rule_warn_reuse_zero(report: dict) -> Recommendation | None:
    """W-REUSE-ZERO: reuse_time p50 == 0 ⇒ reuse mostly intra-batch."""
    p50 = _get(report, "section_3_locality", "f13_single_turn", "stats_seconds", "p50")
    if p50 is None:
        # Fallback to reuse_distance.reuse_time_stats if F13 absent.
        p50 = _get(report, "section_3_locality", "reuse_distance",
                   "reuse_time_stats", "p50")
    if p50 is None or p50 != 0:
        return None
    return Recommendation(
        rule_id="W-REUSE-ZERO",
        type="warning",
        priority=None,
        confidence="medium",
        conclusion="reuse_time_p50 = 0s；reuse 多发生在同一秒内，需 batch 级验证。",
        evidence=("F13 stats_seconds.p50 (or reuse_distance.reuse_time_stats.p50) == 0",),
        action="结合 batch_id 重新评估 reuse 是否跨 batch，避免高估稳定 reuse 占比。",
    )


# ---------------------------------------------------------------------------
# Public registry — engine imports this list.
# ---------------------------------------------------------------------------

ALL_RULES: tuple[Callable[[dict], Recommendation | None], ...] = (
    rule_pin_chain,
    rule_capacity_first,
    rule_low_ceiling,
    rule_batch_ttl,
    rule_long_ttl,
    rule_multi_tenant,
    rule_cache_pressure,
    rule_warn_same_second,
    rule_warn_reuse_zero,
)
