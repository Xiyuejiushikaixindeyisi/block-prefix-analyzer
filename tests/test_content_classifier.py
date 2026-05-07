"""Tests for :mod:`block_prefix_analyzer.analysis.content_classifier`.

Coverage matrix
---------------
Per-category golden inputs (8 categories) plus priority-cascade tests that
verify higher-priority rules win when multiple match. Edge cases:

* Empty / whitespace-only input → "other".
* system_prompt requires BOTH prefix match AND length > 200.
* qa_template requires BOTH a Q-marker AND an A-marker.
* long_document only fires when no higher rule matched.
* Code keyword "def " must not match the substring "definition".
"""
from __future__ import annotations

import pytest

from block_prefix_analyzer.analysis.content_classifier import (
    CATEGORIES,
    CODE_PUNCT_DENSITY_THRESHOLD,
    LONG_DOCUMENT_MIN_LEN,
    SYSTEM_PROMPT_MIN_LEN,
    classify_content,
)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_categories_constant():
    assert CATEGORIES == (
        "json_schema",
        "agent_tool_prompt",
        "system_prompt",
        "rag_template",
        "code",
        "qa_template",
        "long_document",
        "other",
    )


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t  \n"])
def test_empty_or_whitespace_is_other(text):
    assert classify_content(text) == "other"


# ---------------------------------------------------------------------------
# Per-category goldens (priority high → low)
# ---------------------------------------------------------------------------

def test_json_schema_golden():
    text = '{"type": "object", "properties": {"foo": {"type": "string"}}}'
    assert classify_content(text) == "json_schema"


def test_json_schema_required_only_match():
    text = '... "required": ["a", "b"] ...'
    assert classify_content(text) == "json_schema"


def test_agent_tool_via_type_function():
    text = '{"type": "function", "function": {"name": "lookup_weather"}}'
    assert classify_content(text) == "agent_tool_prompt"


def test_agent_tool_via_name_description_parameters_triple():
    """No "type":"function" but the three-key triple is present."""
    text = (
        '{"name": "send_email", "description": "Send an email to a user", '
        '"parameters": {}}'
    )
    assert classify_content(text) == "agent_tool_prompt"


def test_system_prompt_chinese_prefix_long():
    text = "你是一个" + ("专业的代码审查助手，请仔细分析以下代码并提出改进建议。" * 10)
    assert len(text) > SYSTEM_PROMPT_MIN_LEN
    assert classify_content(text) == "system_prompt"


def test_system_prompt_english_prefix_long():
    text = "You are a helpful coding assistant." + (" Please analyse the snippet below." * 20)
    assert len(text) > SYSTEM_PROMPT_MIN_LEN
    assert classify_content(text) == "system_prompt"


def test_system_prompt_too_short_falls_through():
    text = "你是一个助手"
    assert len(text) < SYSTEM_PROMPT_MIN_LEN
    # No higher rule matches → falls all the way down. Short enough → "other".
    assert classify_content(text) == "other"


def test_system_prompt_prefix_with_leading_whitespace_still_matches():
    text = "   \n  你是一个助手" + ("，请回答下面的问题。" * 30)
    assert len(text) > SYSTEM_PROMPT_MIN_LEN
    assert classify_content(text) == "system_prompt"


def test_rag_template_chinese_keyword():
    text = "请根据以下参考资料回答用户的问题。回答应当忠实于资料原文。"
    assert classify_content(text) == "rag_template"


def test_rag_template_english_keyword():
    text = "Use the following Context: to answer the user. Be concise."
    assert classify_content(text) == "rag_template"


def test_code_python_def_keyword():
    text = "def compute_total(items):\n    return sum(items)"
    assert classify_content(text) == "code"


def test_code_punctuation_density_dominant():
    # Non-whitespace: heavy braces/parens; no code keyword required.
    text = "obj = {a: (b[0] = c[1]); d = (e<f>); g={h;};};"
    assert classify_content(text) == "code"


def test_code_keyword_must_have_trailing_space():
    """'definition' contains 'def' but is not a code keyword match."""
    text = (
        "The official definition of a transformer is a neural network "
        "architecture that uses attention. " * 15
    )
    # No QA / RAG / schema markers; > 1000 chars → long_document.
    assert len(text) > LONG_DOCUMENT_MIN_LEN
    assert classify_content(text) == "long_document"


def test_qa_template_chinese():
    text = "问题：什么是 KV cache？\n答案：KV cache 是大模型推理时缓存历史注意力计算结果的机制。"
    assert classify_content(text) == "qa_template"


def test_qa_template_english():
    text = "Q: What is prefix caching?\nA: It is a reuse strategy for shared prompts."
    assert classify_content(text) == "qa_template"


def test_qa_template_requires_both_markers():
    """A Q-only or A-only snippet must NOT classify as qa_template."""
    q_only = "Q: what is X? followed by an essay-style explanation"
    a_only = "Answer: lorem ipsum dolor"
    assert classify_content(q_only) != "qa_template"
    assert classify_content(a_only) != "qa_template"


def test_long_document_fallback_chinese():
    text = "这是一段非常长的中文文档" * 100
    assert len(text) > LONG_DOCUMENT_MIN_LEN
    assert classify_content(text) == "long_document"


def test_other_when_short_and_no_signal():
    text = "短文本，无任何标识符。"
    assert classify_content(text) == "other"


# ---------------------------------------------------------------------------
# Priority-cascade interactions
# ---------------------------------------------------------------------------

def test_json_schema_wins_over_agent_tool_when_both_match():
    """Tool definitions often include inline JSON Schema; spec orders
    json_schema (priority 1) above agent_tool_prompt (priority 2)."""
    text = (
        '{"type": "function", "name": "x", "description": "y", '
        '"parameters": {"type": "object", "properties": {}}}'
    )
    # Both rules would fire; json_schema must win by §4 priority order.
    assert classify_content(text) == "json_schema"


def test_system_prompt_wins_over_rag():
    text = (
        "You are a knowledge assistant. Use the following Context: to answer. "
        + "Pad pad pad. " * 50
    )
    assert classify_content(text) == "system_prompt"


def test_rag_wins_over_code_when_both_match():
    """A RAG template that incidentally has dense punctuation still classifies
    as rag_template because priority 4 < priority 5."""
    text = "Context: [a, b, c]; (d, e, f); {g, h, i}; <j>; <k>; <l>"
    # Punct density > 8% would trigger code, but rag_template fires first.
    assert classify_content(text) == "rag_template"


def test_code_wins_over_qa_when_both_match():
    text = "def foo():\n    return 1\nQ: what?\nA: that"
    assert classify_content(text) == "code"


def test_long_document_only_when_no_higher_rule_matches():
    # Long, but starts with "你是" → system_prompt wins, not long_document.
    text = "你是一个助手。" + ("详细解答用户问题。" * 200)
    assert len(text) > LONG_DOCUMENT_MIN_LEN
    assert classify_content(text) == "system_prompt"


# ---------------------------------------------------------------------------
# Tunable thresholds exposed
# ---------------------------------------------------------------------------

def test_threshold_constants_match_spec():
    assert SYSTEM_PROMPT_MIN_LEN == 200
    assert LONG_DOCUMENT_MIN_LEN == 1000
    assert CODE_PUNCT_DENSITY_THRESHOLD == pytest.approx(0.08)
