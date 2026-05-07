"""Rule-based content classifier for KV-cache analysis.

Maps a decoded text snippet (typically a common_prefix consensus block) to
one of 8 fixed labels. Priority order is frozen in
``可视化.md §4`` and matches the user-confirmed spec; the first matching rule
wins (no scoring, no aggregation).

Output labels (priority high → low)
-----------------------------------
    1. json_schema         JSON Schema structural features present
    2. agent_tool_prompt   Tool / function definition (OpenAI-style or
                           name+description+parameters triple)
    3. system_prompt       Starts with "你是" / "You are" / "Your task"
                           AND length > 200 chars
    4. rag_template        Retrieval-augmentation marker keyword present
    5. code                Punctuation density > 8% over non-whitespace
                           chars OR matches one of six code keywords
    6. qa_template         Both a Q-marker AND an A-marker present
    7. long_document       Length > 1000 chars and nothing above matched
    8. other               Fallback

The classifier is intentionally coarse — the goal is "which KV-cache
optimisation strategy applies", not full content type detection. Pure
stdlib (re), no transformers / sklearn / langdetect / fasttext.
"""
from __future__ import annotations

import re
from typing import Final

# A wider type alias would be nicer, but Python typing.Literal across 3.10/3.11
# is verbose and the labels are stable enough for plain str.
ContentType = str

CATEGORIES: Final[tuple[str, ...]] = (
    "json_schema",
    "agent_tool_prompt",
    "system_prompt",
    "rag_template",
    "code",
    "qa_template",
    "long_document",
    "other",
)

SYSTEM_PROMPT_MIN_LEN: Final[int] = 200
LONG_DOCUMENT_MIN_LEN: Final[int] = 1000
CODE_PUNCT_DENSITY_THRESHOLD: Final[float] = 0.08

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_RE_JSON_SCHEMA = re.compile(
    r'"type"\s*:\s*"object"|"properties"\s*:|"required"\s*:'
)
_RE_AGENT_TOOL_TYPE_FUNCTION = re.compile(r'"type"\s*:\s*"function"')
_RE_AGENT_TOOL_NAME = re.compile(r'"name"\s*:')
_RE_AGENT_TOOL_DESCRIPTION = re.compile(r'"description"\s*:')
_RE_AGENT_TOOL_PARAMETERS = re.compile(r'"parameters"\s*:')

_RE_SYSTEM_PROMPT_PREFIX = re.compile(
    r'(你是|You are|Your task)', re.IGNORECASE
)

_RE_RAG = re.compile(
    r'参考资料|上下文|Context:|Reference:|Documents:', re.IGNORECASE
)

# Six code keywords as listed in the spec; require trailing whitespace so
# "definition" does not match "def".
_RE_CODE_KEYWORD = re.compile(
    r'\b(def|class|function|import|public|private)\s'
)
_CODE_PUNCT_CHARS: Final[frozenset[str]] = frozenset("{};()[]=<>")

_RE_QA_QUESTION = re.compile(r'问题|Q:|Question:', re.IGNORECASE)
_RE_QA_ANSWER = re.compile(r'答案|A:|Answer:', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Per-rule predicates (kept individually for unit testability)
# ---------------------------------------------------------------------------

def _is_json_schema(text: str) -> bool:
    return _RE_JSON_SCHEMA.search(text) is not None


def _is_agent_tool(text: str) -> bool:
    if _RE_AGENT_TOOL_TYPE_FUNCTION.search(text):
        return True
    return (
        _RE_AGENT_TOOL_NAME.search(text) is not None
        and _RE_AGENT_TOOL_DESCRIPTION.search(text) is not None
        and _RE_AGENT_TOOL_PARAMETERS.search(text) is not None
    )


def _is_system_prompt(text: str) -> bool:
    if len(text) <= SYSTEM_PROMPT_MIN_LEN:
        return False
    return _RE_SYSTEM_PROMPT_PREFIX.match(text.lstrip()) is not None


def _is_rag_template(text: str) -> bool:
    return _RE_RAG.search(text) is not None


def _is_code(text: str) -> bool:
    non_ws = [c for c in text if not c.isspace()]
    if non_ws:
        punct_count = sum(1 for c in non_ws if c in _CODE_PUNCT_CHARS)
        if punct_count / len(non_ws) > CODE_PUNCT_DENSITY_THRESHOLD:
            return True
    return _RE_CODE_KEYWORD.search(text) is not None


def _is_qa_template(text: str) -> bool:
    return (
        _RE_QA_QUESTION.search(text) is not None
        and _RE_QA_ANSWER.search(text) is not None
    )


def _is_long_document(text: str) -> bool:
    return len(text) > LONG_DOCUMENT_MIN_LEN


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def classify_content(text: str) -> ContentType:
    """Return the first matching label among :data:`CATEGORIES`.

    Empty or whitespace-only text returns ``"other"`` immediately.
    """
    if not text or not text.strip():
        return "other"

    if _is_json_schema(text):
        return "json_schema"
    if _is_agent_tool(text):
        return "agent_tool_prompt"
    if _is_system_prompt(text):
        return "system_prompt"
    if _is_rag_template(text):
        return "rag_template"
    if _is_code(text):
        return "code"
    if _is_qa_template(text):
        return "qa_template"
    if _is_long_document(text):
        return "long_document"
    return "other"
