"""Request normalization: validate and canonicalize a RawRequest.

Normalization is a thin validation + type-coercion layer.  It does not
perform any rendering, tokenization, or block-building.  Its only job is to
guarantee that downstream adapters receive well-typed, well-formed input.

Rules
-----
* ``request_id``: stripped of leading/trailing whitespace; must be non-empty.
* ``timestamp``: coerced to ``float``; ``bool`` is rejected.
* ``messages``: each ``role`` must be ``"system"``, ``"user"``, or ``"assistant"``.
* Optional session fields (``parent_request_id``, ``session_id``, ``category``,
  ``turn``) are forwarded unchanged.
* ``metadata``: shallow-copied so the normalizer's output is independent of
  the original ``RawRequest``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from block_prefix_analyzer.v2.schema import Message, RawRequest

_VALID_ROLES = frozenset({"system", "user", "assistant"})


class NormalizationError(ValueError):
    """Raised when a :class:`RawRequest` cannot be normalized.

    The message includes enough context to identify the offending field.
    """


@dataclass
class NormalizedRequest:
    """Canonicalized request ready for the rendering/tokenization adapters.

    All fields are validated and type-coerced versions of the originals.
    This object is internal to the V2 pipeline; callers interact with
    :class:`~block_prefix_analyzer.v2.schema.RawRequest` and
    :class:`~block_prefix_analyzer.types.RequestRecord`.
    """

    request_id: str
    timestamp: float
    messages: list[Message]
    parent_request_id: str | None
    session_id: str | None
    category: str | None
    turn: int | None
    metadata: dict[str, Any]


def normalize(raw: RawRequest) -> NormalizedRequest:
    """Validate and canonicalize a :class:`RawRequest`.

    Parameters
    ----------
    raw:
        The raw request from the caller.

    Returns
    -------
    NormalizedRequest
        A validated, type-coerced copy of the input.

    Raises
    ------
    NormalizationError
        If any field fails validation.
    """
    request_id = raw.request_id.strip()
    if not request_id:
        raise NormalizationError(
            "request_id must be non-empty after stripping whitespace"
        )

    if isinstance(raw.timestamp, bool) or not isinstance(raw.timestamp, (int, float)):
        raise NormalizationError(
            f"timestamp must be int or float, got {type(raw.timestamp).__name__!r}"
        )

    for i, msg in enumerate(raw.messages):
        if msg.role not in _VALID_ROLES:
            raise NormalizationError(
                f"messages[{i}].role must be one of {sorted(_VALID_ROLES)}, "
                f"got {msg.role!r}"
            )

    return NormalizedRequest(
        request_id=request_id,
        timestamp=float(raw.timestamp),
        messages=list(raw.messages),
        parent_request_id=raw.parent_request_id,
        session_id=raw.session_id,
        category=raw.category,
        turn=raw.turn,
        metadata=dict(raw.metadata),
    )
