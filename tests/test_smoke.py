"""Smoke test: the package imports and exposes a version string."""
from __future__ import annotations

import block_prefix_analyzer as bpa


def test_package_exposes_version() -> None:
    assert isinstance(bpa.__version__, str)
    assert bpa.__version__.count(".") >= 2


def test_public_submodules_importable() -> None:
    # Import each public submodule so syntax / circular-import regressions
    # are caught by the smoke test even while most modules are stubs.
    import block_prefix_analyzer.types  # noqa: F401
    import block_prefix_analyzer.replay  # noqa: F401
    import block_prefix_analyzer.metrics  # noqa: F401
    import block_prefix_analyzer.index  # noqa: F401
    import block_prefix_analyzer.index.base  # noqa: F401
    import block_prefix_analyzer.index.trie  # noqa: F401
    import block_prefix_analyzer.io  # noqa: F401
    import block_prefix_analyzer.io.jsonl_loader  # noqa: F401
    import block_prefix_analyzer.reports  # noqa: F401
    import block_prefix_analyzer.reports.summary  # noqa: F401
