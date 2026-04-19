"""Tests for :mod:`block_prefix_analyzer.io.jsonl_loader`.

Real suite lands in Step 3 of IMPLEMENTATION_PLAN.md.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.io.jsonl_loader import load_jsonl


def test_loader_is_not_yet_implemented(tmp_path) -> None:
    dummy = tmp_path / "trace.jsonl"
    dummy.write_text("", encoding="utf-8")
    with pytest.raises(NotImplementedError):
        load_jsonl(dummy)


@pytest.mark.skip(reason="Implemented in Step 3 — see IMPLEMENTATION_PLAN.md")
def test_loader_assigns_arrival_index_placeholder() -> None:  # pragma: no cover
    raise AssertionError("placeholder")
