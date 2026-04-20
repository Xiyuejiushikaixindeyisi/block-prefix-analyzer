"""Tests for V2 request normalizer (normalizer.py).

Verifies validation rules and canonicalization behavior.
All tests use hand-constructed RawRequest objects; no I/O or network.
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.v2.normalizer import NormalizationError, normalize
from block_prefix_analyzer.v2.schema import Message, RawRequest


def _req(**kwargs) -> RawRequest:
    defaults = dict(
        request_id="r1",
        timestamp=0.0,
        messages=[Message(role="user", content="hi")],
    )
    defaults.update(kwargs)
    return RawRequest(**defaults)


# ---------------------------------------------------------------------------
# request_id validation
# ---------------------------------------------------------------------------

def test_normalize_strips_whitespace_from_request_id() -> None:
    norm = normalize(_req(request_id="  r1  "))
    assert norm.request_id == "r1"


def test_normalize_empty_request_id_raises() -> None:
    with pytest.raises(NormalizationError, match="request_id"):
        normalize(_req(request_id=""))


def test_normalize_whitespace_only_request_id_raises() -> None:
    with pytest.raises(NormalizationError, match="request_id"):
        normalize(_req(request_id="   "))


# ---------------------------------------------------------------------------
# timestamp validation
# ---------------------------------------------------------------------------

def test_normalize_int_timestamp_converted_to_float() -> None:
    norm = normalize(_req(timestamp=100))
    assert isinstance(norm.timestamp, float)
    assert norm.timestamp == 100.0


def test_normalize_float_timestamp_preserved() -> None:
    norm = normalize(_req(timestamp=61.114))
    assert abs(norm.timestamp - 61.114) < 1e-12


def test_normalize_bool_timestamp_raises() -> None:
    with pytest.raises(NormalizationError, match="timestamp"):
        normalize(_req(timestamp=True))


def test_normalize_string_timestamp_raises() -> None:
    with pytest.raises(NormalizationError, match="timestamp"):
        normalize(_req(timestamp="1.0"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# message role validation
# ---------------------------------------------------------------------------

def test_normalize_valid_roles_accepted() -> None:
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="usr"),
        Message(role="assistant", content="ast"),
    ]
    norm = normalize(_req(messages=msgs))
    assert len(norm.messages) == 3


def test_normalize_invalid_role_raises() -> None:
    bad_msgs = [Message(role="human", content="hi")]  # type: ignore[arg-type]
    with pytest.raises(NormalizationError, match="role"):
        normalize(_req(messages=bad_msgs))


def test_normalize_invalid_role_error_includes_index() -> None:
    msgs = [
        Message(role="user", content="ok"),
        Message(role="bot", content="bad"),  # type: ignore[arg-type]
    ]
    with pytest.raises(NormalizationError, match=r"messages\[1\]"):
        normalize(_req(messages=msgs))


# ---------------------------------------------------------------------------
# Optional session fields
# ---------------------------------------------------------------------------

def test_normalize_session_fields_forwarded() -> None:
    req = _req(
        parent_request_id="r0",
        session_id="sess-1",
        category="text-1",
        turn=1,
    )
    norm = normalize(req)
    assert norm.parent_request_id == "r0"
    assert norm.session_id == "sess-1"
    assert norm.category == "text-1"
    assert norm.turn == 1


def test_normalize_optional_fields_none_by_default() -> None:
    norm = normalize(_req())
    assert norm.parent_request_id is None
    assert norm.session_id is None
    assert norm.category is None
    assert norm.turn is None


# ---------------------------------------------------------------------------
# metadata isolation
# ---------------------------------------------------------------------------

def test_normalize_metadata_is_shallow_copied() -> None:
    meta = {"key": "value"}
    norm = normalize(_req(metadata=meta))
    norm.metadata["key"] = "changed"
    assert meta["key"] == "value"  # original not mutated


def test_normalize_metadata_preserved() -> None:
    norm = normalize(_req(metadata={"time_unit": "s", "source": "test"}))
    assert norm.metadata["time_unit"] == "s"
    assert norm.metadata["source"] == "test"


# ---------------------------------------------------------------------------
# Empty messages
# ---------------------------------------------------------------------------

def test_normalize_empty_messages_allowed() -> None:
    norm = normalize(_req(messages=[]))
    assert norm.messages == []
