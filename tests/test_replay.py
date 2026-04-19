"""Tests for :mod:`block_prefix_analyzer.replay`.

The real suite lands in Step 4 of IMPLEMENTATION_PLAN.md. Until then we
only pin the stub's "not yet implemented" behaviour so regressions show up.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.index.trie import TrieIndex
from block_prefix_analyzer.replay import replay


def test_replay_is_not_yet_implemented() -> None:
    with pytest.raises(NotImplementedError):
        # Consume the iterator; NotImplementedError may be raised eagerly
        # (from the function body) or on first iteration — either is fine
        # as long as it is raised before the first result is produced.
        list(replay([], index_factory=TrieIndex))


@pytest.mark.skip(reason="Implemented in Step 4 — see IMPLEMENTATION_PLAN.md")
def test_no_self_hit_placeholder() -> None:  # pragma: no cover
    raise AssertionError("placeholder")
