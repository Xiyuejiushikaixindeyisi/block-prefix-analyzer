"""SipHash-based block builder stub (pending real-alignment implementation).

Layer-3 alignment status: PENDING
-----------------------------------
vLLM hashes KV-cache blocks using MurmurHash3 with **prefix chaining**:
each block's hash depends on the hash of the preceding block, not just its
own token content.  This differs from the current ``SimpleBlockBuilder``
which hashes each block independently with SHA-256.

Chained-hash semantic (from vLLM source)::

    block_hash[0] = mmh3.hash(bytes(block_tokens[0]), seed=0, signed=False)
    block_hash[i] = mmh3.hash(bytes(block_tokens[i]),
                               block_hash[i-1], signed=False)

To enable this adapter:

    pip install mmh3

Once ``mmh3`` is available, ``ChainedBlockBuilder`` will produce block IDs
that are aligned with vLLM's prefix-cache key space.

Until then, tests that import ``ChainedBlockBuilder`` with a real hash
function should be guarded with ``pytest.importorskip("mmh3")``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from block_prefix_analyzer.types import BlockId
from block_prefix_analyzer.v2.adapters.block_builder import BlockBuildResult


def _mmh3_chained_hash(token_ids: list[int], prev_hash: int) -> int:
    """MurmurHash3 with prefix chaining — requires ``mmh3``."""
    try:
        import mmh3  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "ChainedBlockBuilder with mmh3 requires: pip install mmh3"
        ) from exc
    return mmh3.hash(bytes(token_ids), prev_hash, signed=False)


class ChainedBlockBuilder:
    """Block builder that chains block hashes as in vLLM's prefix cache.

    Each block's ID depends on all preceding block content, so two blocks
    with the same token content but different prefixes will get different IDs.
    This matches vLLM's cache-key semantics exactly.

    Parameters
    ----------
    block_size:
        Number of tokens per block.  Must be positive.
    hash_fn:
        ``(token_ids: list[int], prev_hash: int) -> int``.
        Defaults to :func:`_mmh3_chained_hash`; override for testing without
        the ``mmh3`` dependency.
    initial_hash:
        Seed for the first block's hash.

        **Analysis default (``0``)**: Any fixed, consistent seed produces
        correct *relative* prefix-reuse statistics within a single dataset.
        Using ``0`` is fine for measuring hit rates.

        **Cross-system alignment**: vLLM's actual ``NONE_HASH`` is *not* ``0``.
        It is derived from ``PYTHONHASHSEED`` if set, otherwise from
        ``os.urandom(32)`` — a per-process random value, not a fixed constant.
        To reproduce the exact block IDs of a specific vLLM deployment,
        obtain its ``NONE_HASH`` value and pass it here.  Without that value,
        the block IDs computed by this class will differ from vLLM's, even
        though the prefix-reuse analysis results (hit rates) remain accurate.
    """

    def __init__(
        self,
        block_size: int = 16,
        hash_fn: Callable[[list[int], int], int] = _mmh3_chained_hash,
        initial_hash: int = 0,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size
        self._hash_fn = hash_fn
        self._initial_hash = initial_hash

    def build(self, token_ids: list[int]) -> BlockBuildResult:
        """Convert token IDs to chained block IDs."""
        block_ids: list[BlockId] = []
        n = len(token_ids)
        prev_hash = self._initial_hash
        i = 0
        while i + self.block_size <= n:
            block = token_ids[i : i + self.block_size]
            h = self._hash_fn(block, prev_hash)
            block_ids.append(h)
            prev_hash = h
            i += self.block_size
        return BlockBuildResult(
            block_ids=block_ids,
            leftover_token_count=n - i,
        )
