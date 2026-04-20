"""V2 raw request schema: input objects before any processing.

These types represent the external interface of the V2 pipeline.
They are deliberately separate from :class:`~block_prefix_analyzer.types.RequestRecord`
so that each layer has a single responsibility:

* ``RawRequest`` — what the caller provides (messages, metadata, session fields)
* ``RequestRecord`` — what the V1 core consumes (block IDs, timestamps)

The conversion happens inside :func:`~block_prefix_analyzer.v2.pipeline.build_block_records_from_raw_requests`.

Session / category fields (``parent_request_id``, ``session_id``, ``category``,
``turn``) are included now to avoid a schema break later, but are not consumed
by any V1 or V2-min logic.  They are forwarded to ``RequestRecord.metadata``
for future use (F13 / F14 / F15 analytics in V2+).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
"""Valid roles for a conversation message."""


@dataclass
class Message:
    """A single turn in a conversation.

    Attributes
    ----------
    role:
        One of ``"system"``, ``"user"``, or ``"assistant"``.
    content:
        The text content of the message.  May be empty.
    """

    role: Role
    content: str


@dataclass
class RawRequest:
    """Raw inference request before template rendering or tokenization.

    This is the stable external input schema for V2.  Callers construct
    ``RawRequest`` objects and pass them to the V2 pipeline; the pipeline
    handles all further processing steps.

    Invariants enforced by the normalizer (not here):
    * ``request_id`` must be non-empty after stripping whitespace.
    * ``timestamp`` must be a numeric type (``int`` or ``float``), not ``bool``.
    * Each message's ``role`` must be one of ``"system"``, ``"user"``, ``"assistant"``.

    Attributes
    ----------
    request_id:
        Stable identifier for this request.  Used as a display label only.
    timestamp:
        Arrival time; unit is caller's responsibility.  Passed through to
        ``RequestRecord.timestamp`` unchanged after float conversion.
    messages:
        Ordered list of conversation turns.  May be empty (produces an empty
        block sequence, excluded from hit-rate denominators in V1).
    parent_request_id:
        ID of the preceding turn's request for multi-turn sessions.
        ``None`` for root (single-turn or first-turn) requests.  V2+ only.
    session_id:
        Opaque session identifier grouping related requests.  V2+ only.
    category:
        Request category label (e.g. ``"text-1"``, ``"search-2"``).  V2+ only.
    turn:
        Conversation turn number, 1-indexed.  V2+ only.
    metadata:
        Free-form annotations forwarded to ``RequestRecord.metadata``.
    """

    request_id: str
    timestamp: int | float
    messages: list[Message]
    parent_request_id: str | None = None
    session_id: str | None = None
    category: str | None = None
    turn: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
