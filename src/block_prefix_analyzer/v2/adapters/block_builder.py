"""Block builder: split token IDs into fixed-size blocks and hash each block.

The :class:`SimpleBlockBuilder` implements the standard KV-cache block
allocation model:

* Token IDs are grouped into non-overlapping fixed-size blocks.
* Each **complete** block is hashed to a single ``BlockId`` (``int``).
* The last block is **incomplete** if ``len(token_ids) % block_size != 0``.
  Incomplete blocks are excluded from ``block_ids``; their token count is
  reported as ``leftover_token_count``.

This "exclude incomplete last block" rule matches the KV-cache semantic:
only fully-filled blocks occupy a cache slot.

Hash function
-------------
The default hash is SHA-256 over the block contents, truncated to 8 bytes
and interpreted as a little-endian unsigned 64-bit integer.  The payload is
the comma-joined decimal representation of the token IDs (``"1,2,3"``), which
is safe for any integer range and produces a stable, collision-resistant ID.

To replace the hash function::

    def my_hash(token_ids: list[int]) -> int:
        ...  # return int

    builder = SimpleBlockBuilder(block_size=16, hash_fn=my_hash)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from block_prefix_analyzer.types import BlockId


def _sha256_block_default(token_ids: list[int]) -> int:
    """Hash a block of token IDs via SHA-256 (first 8 bytes, little-endian)."""
    payload = ",".join(map(str, token_ids)).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass
class BlockBuildResult:
    """Output of :meth:`SimpleBlockBuilder.build`.

    Attributes
    ----------
    block_ids:
        Hashed IDs for each **complete** block.  Empty if fewer tokens than
        one block.
    leftover_token_count:
        Number of tokens in the final incomplete block (0 if perfectly full).
    """

    block_ids: list[BlockId]
    leftover_token_count: int


class SimpleBlockBuilder:
    """Split token IDs into fixed-size blocks and hash each complete block.

    Parameters
    ----------
    block_size:
        Number of tokens per block.  Must be positive.
    hash_fn:
        Callable mapping ``list[int]`` → ``BlockId``.  Defaults to
        :func:`_sha256_block_default`.
    """

    def __init__(
        self,
        block_size: int = 16,
        hash_fn: Callable[[list[int]], BlockId] = _sha256_block_default,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be a positive integer, got {block_size}")
        self.block_size = block_size
        self._hash_fn = hash_fn

    def build(self, token_ids: list[int]) -> BlockBuildResult:
        """Convert token IDs to block IDs.

        Parameters
        ----------
        token_ids:
            Non-negative integer token IDs produced by a tokenizer.

        Returns
        -------
        BlockBuildResult
            Complete block IDs and the leftover token count.
        """
        block_ids: list[BlockId] = []
        n = len(token_ids)
        i = 0
        while i + self.block_size <= n:
            block_ids.append(self._hash_fn(token_ids[i : i + self.block_size]))
            i += self.block_size
        return BlockBuildResult(
            block_ids=block_ids,
            leftover_token_count=n - i,
        )
