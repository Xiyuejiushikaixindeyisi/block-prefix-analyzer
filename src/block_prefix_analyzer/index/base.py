"""Abstract interface for block-sequence prefix indexes.

Keeping this interface minimal is deliberate. The V1 implementation is a
simple trie; a future V3 may introduce a radix-compressed tree. As long as
they both satisfy this protocol, the replay engine does not need to change.

Non-goals (intentionally excluded from the interface)
-----------------------------------------------------
* Random lookup by request id. The replay engine does not need it and it
  encourages misuse (see PROJECT_SPEC.md §3.3).
* Bulk batch operations. V1 is single-record at a time; optimisations come
  after correctness.
* Deletion / eviction. V1 assumes infinite capacity.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..types import BlockId


@runtime_checkable
class PrefixIndex(Protocol):
    """Minimal prefix index protocol.

    Implementations must be deterministic: for the same insertion order and
    the same query, they must return the same value on every invocation.
    """

    def longest_prefix_match(self, block_ids: Sequence[BlockId]) -> int:
        """Return the length of the longest prefix of ``block_ids`` that
        matches some previously-inserted sequence (or prefix of one).

        Returns ``0`` when no prefix matches, including when the index is
        empty or when ``block_ids`` is empty.
        """
        ...

    def insert(self, block_ids: Sequence[BlockId]) -> None:
        """Insert ``block_ids`` into the index.

        Inserting an empty sequence is a no-op. Inserting the same sequence
        twice is idempotent — the index records *which prefixes have been
        seen*, not how often.
        """
        ...
