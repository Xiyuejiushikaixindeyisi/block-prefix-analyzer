"""Phase-1 recommendation engine.

Each rule is a pure function ``(report: dict) -> Recommendation | None`` that
inspects a v1.1 report dict (output of :func:`report_builder.assemble_report`)
and either returns a :class:`Recommendation` or ``None`` (no fire).

The engine collects all rule results, drops ``None`` returns, and emits a
list sorted by priority then rule_id. Warnings (priority ``None``) sort
last; otherwise ``P0 < P1 < P2``.
"""
from block_prefix_analyzer.recommendation.engine import (
    Recommendation,
    get_registry,
    run_all_rules,
    sort_recommendations,
)

__all__ = ["Recommendation", "get_registry", "run_all_rules", "sort_recommendations"]
