"""Trace loaders.

Loaders are the boundary between raw trace files and the canonical
:class:`~block_prefix_analyzer.types.RequestRecord`. They are the **only**
component allowed to assign ``arrival_index``.
"""
