"""Tests for V2 tokenizer adapters (adapters/tokenizer.py).

All tests are deterministic and require no external dependencies.
"""
from __future__ import annotations

from block_prefix_analyzer.v2.adapters.tokenizer import CharTokenizer, TokenizerAdapter


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_char_tokenizer_satisfies_protocol() -> None:
    assert isinstance(CharTokenizer(), TokenizerAdapter)


# ---------------------------------------------------------------------------
# CharTokenizer encoding
# ---------------------------------------------------------------------------

def test_encode_empty_string_returns_empty_list() -> None:
    tok = CharTokenizer()
    assert tok.encode("") == []


def test_encode_single_char() -> None:
    tok = CharTokenizer()
    assert tok.encode("A") == [65]  # ord("A") == 65


def test_encode_ascii_string() -> None:
    tok = CharTokenizer()
    result = tok.encode("Hi")
    assert result == [72, 105]  # ord("H")=72, ord("i")=105


def test_encode_length_equals_char_count() -> None:
    tok = CharTokenizer()
    text = "hello world"
    assert len(tok.encode(text)) == len(text)


def test_encode_is_deterministic() -> None:
    tok = CharTokenizer()
    text = "test string"
    assert tok.encode(text) == tok.encode(text)


def test_encode_unicode_returns_code_points() -> None:
    tok = CharTokenizer()
    # "中" is U+4E2D = 20013
    result = tok.encode("中")
    assert result == [0x4E2D]


def test_encode_all_values_nonnegative() -> None:
    tok = CharTokenizer()
    for token_id in tok.encode("Hello, 世界! 🎉"):
        assert token_id >= 0


def test_encode_max_unicode_fits_in_uint32() -> None:
    tok = CharTokenizer()
    # Maximum Unicode code point is U+10FFFF = 1_114_111 < 2^32
    result = tok.encode("\U0010FFFF")
    assert result == [0x10FFFF]
    assert result[0] < 2**32


def test_name_returns_string() -> None:
    tok = CharTokenizer()
    assert isinstance(tok.name(), str)
    assert len(tok.name()) > 0


# ---------------------------------------------------------------------------
# Custom tokenizer structural subtyping
# ---------------------------------------------------------------------------

def test_custom_tokenizer_satisfies_protocol() -> None:
    class WordTokenizer:
        _vocab: dict[str, int] = {}
        _next_id: int = 0

        def encode(self, text: str) -> list[int]:
            ids = []
            for word in text.split():
                if word not in self._vocab:
                    self._vocab[word] = self._next_id
                    self._next_id += 1
                ids.append(self._vocab[word])
            return ids

        def name(self) -> str:
            return "word_tokenizer"

    assert isinstance(WordTokenizer(), TokenizerAdapter)
