"""Recommendation engine — collects rule outputs and sorts them.

The :class:`Recommendation` dataclass is the single output type for every
rule. The engine is intentionally minimal: it imports the rule registry,
runs each rule, drops ``None`` returns, and sorts the survivors.

Sort order
----------
Recommendations sort by ``(priority_rank, rule_id)``:

    P0=0  P1=1  P2=2  warning(None)=3

So warnings always appear after recommendations. Within the same priority,
sort is alphabetical by ``rule_id`` for stability.

Output dict shape (matches ``可视化.md §3 section_5_recommendations``)
---------------------------------------------------------------------
``Recommendation.to_dict()`` produces::

    {
      "priority":   "P0" | "P1" | "P2" | None,
      "confidence": "high" | "medium" | "low",
      "type":       "recommendation" | "warning",
      "conclusion": "...",
      "evidence":   ["...", "..."],
      "action":     "...",
      "rule_id":    "R-PIN-CHAIN"          # debug aid; UI may hide
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


_PRIORITY_RANK: dict[str | None, int] = {
    "P0": 0,
    "P1": 1,
    "P2": 2,
    None: 3,
}


@dataclass(frozen=True)
class Recommendation:
    """One row in section_5_recommendations.

    ``priority`` is ``None`` exactly when ``type == "warning"``.
    """
    rule_id: str
    type: str                  # "recommendation" | "warning"
    priority: str | None       # "P0" | "P1" | "P2" | None
    confidence: str            # "high" | "medium" | "low"
    conclusion: str
    evidence: tuple[str, ...]
    action: str

    def to_dict(self) -> dict:
        return {
            "priority": self.priority,
            "confidence": self.confidence,
            "type": self.type,
            "conclusion": self.conclusion,
            "evidence": list(self.evidence),
            "action": self.action,
            "rule_id": self.rule_id,
        }


# Imported lazily to avoid an import cycle (``rules.py`` imports
# ``Recommendation`` from this module).

def get_registry() -> tuple[Callable[[dict], Recommendation | None], ...]:
    """Return the immutable tuple of registered rule functions."""
    from block_prefix_analyzer.recommendation.rules import ALL_RULES
    return ALL_RULES


def sort_recommendations(items: list[Recommendation]) -> list[Recommendation]:
    """Return a new list sorted by (priority_rank, rule_id)."""
    return sorted(items, key=lambda r: (_PRIORITY_RANK.get(r.priority, 99), r.rule_id))


def run_all_rules(report: dict) -> list[Recommendation]:
    """Run every registered rule against ``report`` and return survivors.

    Rule functions that raise are NOT silenced — bugs should surface during
    development. Use try/except at the caller boundary if you need to
    tolerate buggy rules in production.
    """
    out: list[Recommendation] = []
    for rule in get_registry():
        result = rule(report)
        if result is not None:
            out.append(result)
    return sort_recommendations(out)
