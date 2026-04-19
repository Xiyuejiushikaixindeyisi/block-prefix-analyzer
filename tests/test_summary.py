"""Tests for :mod:`block_prefix_analyzer.reports.summary`.

Real suite lands in Step 6 of IMPLEMENTATION_PLAN.md.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.reports.summary import format_summary


def test_summary_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        format_summary([])


@pytest.mark.skip(reason="Implemented in Step 6 — see IMPLEMENTATION_PLAN.md")
def test_summary_output_placeholder() -> None:  # pragma: no cover
    raise AssertionError("placeholder")
