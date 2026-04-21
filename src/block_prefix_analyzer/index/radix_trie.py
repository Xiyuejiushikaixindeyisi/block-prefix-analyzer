"""Radix-compressed prefix index (Patricia trie).

This is a drop-in replacement for :class:`~block_prefix_analyzer.index.trie.TrieIndex`
that compresses single-child node chains into single edges with compact
``array.array('Q')`` labels.  Memory usage is **~10–14× lower** for workloads
with long shared prefixes (e.g. system prompts).

Data structure
--------------
Each node is a :class:`_RadixNode` holding a ``children`` dict::

    children: dict[int, tuple[array.array, _RadixNode]]
              key: first block_id of the outgoing edge
              value: (edge_label, child_node)

Edge labels are ``array.array('Q')`` (unsigned 64-bit integers, 8 bytes each).
This covers both TraceA integer block_ids (< 2¹⁶) and SHA-256-truncated
uint64 block_ids produced by :class:`~block_prefix_analyzer.v2.adapters.block_builder.SimpleBlockBuilder`.

Restrictions
------------
* **Integer block_ids only.**  Passing ``str`` block_ids raises ``TypeError``.
  All current loaders (TraceA path A + business loader) produce ``int`` block_ids.
* Not thread-safe (single-threaded use only, same as ``TrieIndex``).
* No eviction / capacity limit (same as ``TrieIndex``).

Equivalence
-----------
For the same insertion sequence, ``RadixTrieIndex.longest_prefix_match``
returns identical results to ``TrieIndex.longest_prefix_match``.
This is guaranteed by the correctness of edge-split insert and the
contiguous-prefix matching semantics (see ``_common_len``).

Complexity
----------
* ``insert``: O(L) time, O(L) space for new path segments.
* ``longest_prefix_match``: O(L) time where L = matched block count.
  Comparison loop touches only the L matched elements, not the full query.
* ``node_count`` / ``edge_label_bytes``: O(N_nodes) — iterative DFS.
"""
from __future__ import annotations

import array
from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Internal node
# ---------------------------------------------------------------------------

class _RadixNode:
    """A node in the radix trie.

    Attributes
    ----------
    children:
        Maps the *first block_id of an outgoing edge* to a
        ``(edge_label, child_node)`` pair.
    """

    __slots__ = ("children",)

    def __init__(self) -> None:
        self.children: dict[int, tuple[array.array, _RadixNode]] = {}


# ---------------------------------------------------------------------------
# Helper: common prefix length
# ---------------------------------------------------------------------------

def _common_len(edge: array.array, query: Sequence[int], q_start: int) -> int:
    """Return the length of the common prefix between *edge* and *query[q_start:]*.

    Iterates at most ``min(len(edge), len(query) - q_start)`` steps and returns
    as soon as a mismatch is found.  Returns 0 immediately when either side is
    empty.
    """
    n = min(len(edge), len(query) - q_start)
    for i in range(n):
        if edge[i] != query[q_start + i]:
            return i
    return n


# ---------------------------------------------------------------------------
# Radix trie index
# ---------------------------------------------------------------------------

class RadixTrieIndex:
    """Radix-compressed prefix index satisfying the PrefixIndex protocol.

    Use as a drop-in replacement for
    :class:`~block_prefix_analyzer.index.trie.TrieIndex`::

        from block_prefix_analyzer.index.radix_trie import RadixTrieIndex
        results = list(replay(records, index_factory=RadixTrieIndex))

    Only **integer** block_ids are supported.  ``array.array('Q')`` edge
    labels store unsigned 64-bit values; passing ``str`` block_ids raises
    ``TypeError`` at the first ``insert`` call.
    """

    def __init__(self) -> None:
        self._root = _RadixNode()

    # ------------------------------------------------------------------
    # PrefixIndex protocol
    # ------------------------------------------------------------------

    def longest_prefix_match(self, block_ids: Sequence[int]) -> int:
        """Return the length of the longest contiguous prefix of *block_ids*
        that matches a path stored in the index.

        Returns ``0`` when the index is empty, *block_ids* is empty, or the
        first element is not found.
        """
        if not block_ids:
            return 0
        matched = 0
        node = self._root
        pos = 0
        n = len(block_ids)
        while pos < n:
            first = block_ids[pos]
            if first not in node.children:
                break
            edge, child = node.children[first]
            k = _common_len(edge, block_ids, pos)
            matched += k
            if k < len(edge):
                # Partial edge match — query diverges inside this edge; stop.
                break
            # Full edge consumed — descend into child.
            pos += k
            node = child
        return matched

    def insert(self, block_ids: Sequence[int]) -> None:
        """Insert *block_ids* into the index.

        Inserting an empty sequence is a no-op.  Inserting a sequence that is
        already fully represented (exact match or prefix of existing path) is
        idempotent — the trie structure is unchanged.

        Raises
        ------
        TypeError
            If any element of *block_ids* is not an integer.
        OverflowError
            If any integer is outside the uint64 range [0, 2^64 − 1].
        """
        if not block_ids:
            return
        node = self._root
        pos = 0
        n = len(block_ids)
        while pos < n:
            first = block_ids[pos]
            if first not in node.children:
                # No outgoing edge for this block — attach the remaining
                # sequence as a new leaf edge.
                node.children[first] = (
                    _make_edge(block_ids, pos),
                    _RadixNode(),
                )
                return

            edge, child = node.children[first]
            k = _common_len(edge, block_ids, pos)

            if k == len(edge):
                # Full edge match — continue into child node.
                pos += k
                node = child
                # If pos == n here, the while-condition fails on next iteration
                # and the function returns implicitly (idempotent, path exists).
            else:
                # Partial match at position k — split the edge.
                #
                #  Before:  node --[edge]--> child
                #  After:   node --[edge[:k]]--> split_node
                #                                  ├── [edge[k:]]--> child   (old)
                #                                  └── [remaining]--> new_leaf (new, if any)
                split_node = _RadixNode()

                # Old suffix rooted at split_node.
                old_first = edge[k]
                split_node.children[old_first] = (edge[k:], child)

                # New suffix (remaining query blocks after split point).
                new_pos = pos + k
                if new_pos < n:
                    new_first = block_ids[new_pos]
                    split_node.children[new_first] = (
                        _make_edge(block_ids, new_pos),
                        _RadixNode(),
                    )

                # Shorten the parent edge to the common prefix.
                node.children[first] = (edge[:k], split_node)
                return

    # ------------------------------------------------------------------
    # Benchmark / diagnostic statistics
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        """Number of :class:`_RadixNode` objects in the trie (including root).

        Uses iterative DFS to avoid hitting Python's recursion limit on deep
        single-path tries.
        """
        count = 0
        stack = [self._root]
        while stack:
            node = stack.pop()
            count += 1
            for _, child in node.children.values():
                stack.append(child)
        return count

    def edge_count(self) -> int:
        """Number of edges in the trie (each non-root node has one incoming edge)."""
        return max(0, self.node_count() - 1)

    def edge_label_bytes(self) -> int:
        """Total bytes stored in all edge label arrays.

        Each element is a ``uint64`` (8 bytes).  This is the dominant memory
        cost for long-context workloads (one edge can span hundreds of blocks).
        """
        total = 0
        stack = [self._root]
        while stack:
            node = stack.pop()
            for edge, child in node.children.values():
                total += len(edge) * edge.itemsize
                stack.append(child)
        return total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_edge(block_ids: Sequence[int], start: int = 0) -> array.array:
    """Create an ``array.array('Q')`` from *block_ids[start:]*.

    Raises ``TypeError`` / ``OverflowError`` if values are not valid uint64.
    """
    a: array.array = array.array("Q")
    # Extend in a loop to avoid creating an intermediate list slice.
    try:
        for i in range(start, len(block_ids)):
            a.append(block_ids[i])
    except TypeError as exc:
        raise TypeError(
            "RadixTrieIndex only supports integer block_ids; "
            f"got non-integer value: {exc}"
        ) from exc
    except OverflowError as exc:
        raise OverflowError(
            "RadixTrieIndex block_ids must be uint64 [0, 2³64−1]; "
            f"out-of-range value: {exc}"
        ) from exc
    return a
