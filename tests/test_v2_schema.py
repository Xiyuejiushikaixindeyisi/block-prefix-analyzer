"""Tests for V2 raw request schema (schema.py).

Verifies that RawRequest and Message can be constructed correctly and that
all optional fields have the expected defaults.  No validation logic lives
here — that belongs in the normalizer.
"""
from __future__ import annotations

from block_prefix_analyzer.v2.schema import Message, RawRequest


def test_message_stores_role_and_content() -> None:
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_message_empty_content_allowed() -> None:
    msg = Message(role="system", content="")
    assert msg.content == ""


def test_raw_request_required_fields() -> None:
    req = RawRequest(
        request_id="r1",
        timestamp=1.0,
        messages=[Message(role="user", content="hi")],
    )
    assert req.request_id == "r1"
    assert req.timestamp == 1.0
    assert len(req.messages) == 1


def test_raw_request_optional_fields_default_to_none() -> None:
    req = RawRequest(request_id="r1", timestamp=0.0, messages=[])
    assert req.parent_request_id is None
    assert req.session_id is None
    assert req.category is None
    assert req.turn is None


def test_raw_request_metadata_defaults_to_empty_dict() -> None:
    req = RawRequest(request_id="r1", timestamp=0.0, messages=[])
    assert req.metadata == {}


def test_raw_request_metadata_not_shared_across_instances() -> None:
    r1 = RawRequest(request_id="r1", timestamp=0.0, messages=[])
    r2 = RawRequest(request_id="r2", timestamp=0.0, messages=[])
    r1.metadata["key"] = "value"
    assert "key" not in r2.metadata


def test_raw_request_accepts_int_timestamp() -> None:
    req = RawRequest(request_id="r1", timestamp=100, messages=[])
    assert req.timestamp == 100


def test_raw_request_accepts_float_timestamp() -> None:
    req = RawRequest(request_id="r1", timestamp=61.114, messages=[])
    assert req.timestamp == 61.114


def test_raw_request_session_fields_can_be_set() -> None:
    req = RawRequest(
        request_id="r1",
        timestamp=0.0,
        messages=[],
        parent_request_id="r0",
        session_id="session-42",
        category="text-2",
        turn=2,
    )
    assert req.parent_request_id == "r0"
    assert req.session_id == "session-42"
    assert req.category == "text-2"
    assert req.turn == 2


def test_raw_request_messages_can_be_empty() -> None:
    req = RawRequest(request_id="r1", timestamp=0.0, messages=[])
    assert req.messages == []


def test_raw_request_multi_turn_messages() -> None:
    msgs = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
        Message(role="user", content="How are you?"),
    ]
    req = RawRequest(request_id="r1", timestamp=0.0, messages=msgs)
    assert len(req.messages) == 4
    assert req.messages[0].role == "system"
    assert req.messages[-1].role == "user"
