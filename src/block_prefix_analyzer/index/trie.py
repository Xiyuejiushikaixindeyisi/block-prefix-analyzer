"""Uncompressed trie-based :class:`PrefixIndex` implementation.

Design notes
------------
* Nodes are plain Python ``dict`` objects mapping ``BlockId -> child dict``.
  This makes the structure trivially introspectable from tests and avoids
  any dependency on dataclasses or external libraries.
* Equality is Python's ``==``; the trie never coerces block ids. Mixing
  ``int`` and ``str`` keys across different inserted sequences is allowed but
  will never produce a cross-type match, which is the expected behaviour.
* Duplicate block ids within one inserted sequence are legal and simply
  descend deeper into the same subtree.
* Insertion is idempotent: inserting a sequence twice has the same effect as
  inserting it once.
* ``longest_prefix_match`` returns the number of leading elements of the
  query that match a **path in the trie** — not necessarily a complete
  sequence. In other words, if ``[a, b, c]`` was inserted, the query
  ``[a, b]`` returns 2 (the trie path extends beyond the query end, and
  ``a → b`` is a valid trie path).
* No eviction, no capacity limit, no path compression. This is intentional
  for V1; V3 may add radix compression as a separate implementation behind
  the same :class:`~block_prefix_analyzer.index.base.PrefixIndex` interface.

Complexity
----------
* ``insert``: O(L) time, O(L) space per new path (shared prefixes reuse nodes).
* ``longest_prefix_match``: O(L) time, O(1) extra space.
"""
from __future__ import annotations

from collections.abc import Sequence

from ..types import BlockId


class TrieIndex:
    """Uncompressed trie storing seen block-id sequences.

    Satisfies the :class:`~block_prefix_analyzer.index.base.PrefixIndex`
    protocol; swap it for a radix-compressed implementation in V3 by
    replacing the factory passed to the replay engine.
    """

    def __init__(self) -> None:
        # Root node: {block_id: child_node, ...}
        # A "node" is itself a dict with the same structure.
        self._root: dict[BlockId, dict] = {}

    # ------------------------------------------------------------------
    # PrefixIndex protocol
    # ------------------------------------------------------------------

    def longest_prefix_match(self, block_ids: Sequence[BlockId]) -> int:
        """Return the length of the longest prefix of ``block_ids`` that
        matches a path in the trie.

        Returns ``0`` when the trie is empty, when ``block_ids`` is empty,
        or when the first element is not in the trie.
        """
        node = self._root
        length = 0
        for bid in block_ids:
            if bid not in node:
                break
            node = node[bid]
            length += 1
        return length

    def insert(self, block_ids: Sequence[BlockId]) -> None:
        """Insert ``block_ids`` into the trie.

        Inserting an empty sequence is a no-op. Inserting a sequence that is
        already present (or is a prefix of an existing one) is idempotent.
        """
        node = self._root
        for bid in block_ids:
            if bid not in node:
                node[bid] = {}
            node = node[bid]
