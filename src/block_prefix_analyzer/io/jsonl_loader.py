"""JSONL trace loader — V1 skeleton.

Expected JSON schema per line (V1)::

    {
      "request_id": "req-0001",
      "timestamp": 1713500000,
      "block_ids": [101, 102, 103],
      "token_count": 512,           // optional
      "block_size": 128,            // optional
      "metadata": {"department": "x"} // optional
    }

Loader contract (frozen for V1)
-------------------------------
* ``arrival_index`` is assigned by this loader in file-read order, starting
  from 0. Any ``arrival_index`` field present in the input is ignored.
* The returned list is sorted by
  :func:`~block_prefix_analyzer.types.ordering_key`. Two records with the
  same timestamp keep their relative file order.
* Missing required fields (``request_id``, ``timestamp``, ``block_ids``)
  raise :class:`ValueError` with the line number.

Implementation lands in Step 3.
"""
from __future__ import annotations

from pathlib import Path

from ..types import RequestRecord


def load_jsonl(path: str | Path) -> list[RequestRecord]:
    """Load JSONL trace at ``path`` and return sorted :class:`RequestRecord`s.

    V1 skeleton — implemented in Step 3.
    """
    raise NotImplementedError("implemented in Step 3; see IMPLEMENTATION_PLAN.md")
