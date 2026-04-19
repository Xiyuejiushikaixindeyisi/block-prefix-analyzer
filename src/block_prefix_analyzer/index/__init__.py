"""Prefix index implementations.

Only the abstract :class:`~block_prefix_analyzer.index.base.PrefixIndex`
protocol is part of the stable surface; concrete implementations may be
swapped (simple trie today, radix-compressed trie later) without touching
callers.
"""

from .base import PrefixIndex

__all__ = ["PrefixIndex"]
