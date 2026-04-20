"""Tests for V2 chat template adapters (adapters/chat_template.py).

All tests are deterministic and require no external dependencies.
"""
from __future__ import annotations

from block_prefix_analyzer.v2.adapters.chat_template import (
    ChatTemplateAdapter,
    MinimalChatTemplate,
)
from block_prefix_analyzer.v2.schema import Message


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_minimal_template_satisfies_protocol() -> None:
    assert isinstance(MinimalChatTemplate(), ChatTemplateAdapter)


# ---------------------------------------------------------------------------
# MinimalChatTemplate rendering
# ---------------------------------------------------------------------------

def test_render_single_user_message() -> None:
    tpl = MinimalChatTemplate()
    msgs = [Message(role="user", content="Hello")]
    rendered = tpl.render(msgs)
    assert "<|user|>" in rendered
    assert "Hello" in rendered
    assert rendered.endswith("<|assistant|>")


def test_render_system_user_messages() -> None:
    tpl = MinimalChatTemplate()
    msgs = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="What is 2+2?"),
    ]
    rendered = tpl.render(msgs)
    assert "<|system|>" in rendered
    assert "You are helpful." in rendered
    assert "<|user|>" in rendered
    assert "What is 2+2?" in rendered
    assert rendered.endswith("<|assistant|>")


def test_render_role_order_preserved() -> None:
    tpl = MinimalChatTemplate()
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="usr"),
        Message(role="assistant", content="ast"),
        Message(role="user", content="usr2"),
    ]
    rendered = tpl.render(msgs)
    pos_sys = rendered.index("<|system|>")
    pos_usr = rendered.index("<|user|>")
    pos_ast = rendered.index("<|assistant|>")
    assert pos_sys < pos_usr < pos_ast


def test_render_empty_messages_returns_assistant_marker_only() -> None:
    tpl = MinimalChatTemplate()
    rendered = tpl.render([])
    assert rendered == "<|assistant|>"


def test_render_is_deterministic() -> None:
    tpl = MinimalChatTemplate()
    msgs = [Message(role="user", content="hi")]
    assert tpl.render(msgs) == tpl.render(msgs)


def test_render_empty_content_message() -> None:
    tpl = MinimalChatTemplate()
    msgs = [Message(role="system", content="")]
    rendered = tpl.render(msgs)
    assert "<|system|>" in rendered
    assert rendered.endswith("<|assistant|>")


def test_name_returns_string() -> None:
    tpl = MinimalChatTemplate()
    assert isinstance(tpl.name(), str)
    assert len(tpl.name()) > 0


def test_name_is_stable() -> None:
    tpl = MinimalChatTemplate()
    assert tpl.name() == tpl.name()


# ---------------------------------------------------------------------------
# Custom template conformance (structural subtyping check)
# ---------------------------------------------------------------------------

def test_custom_template_satisfies_protocol() -> None:
    class CustomTemplate:
        def render(self, messages: list[Message]) -> str:
            return " ".join(m.content for m in messages)

        def name(self) -> str:
            return "custom"

    assert isinstance(CustomTemplate(), ChatTemplateAdapter)
