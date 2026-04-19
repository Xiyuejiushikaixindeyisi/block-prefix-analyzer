"""Tests for :mod:`block_prefix_analyzer.metrics`.

Real suite lands in Step 5 of IMPLEMENTATION_PLAN.md.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.metrics import (
    block_level_reusable_ratio,
    prefix_aware_ideal_hit_ratio,
)


def test_metric_stubs_raise_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        prefix_aware_ideal_hit_ratio([])
    with pytest.raises(NotImplementedError):
        block_level_reusable_ratio([])


@pytest.mark.skip(reason="Implemented in Step 5 — see IMPLEMENTATION_PLAN.md")
def test_metric_values_placeholder() -> None:  # pragma: no cover
    raise AssertionError("placeholder")
