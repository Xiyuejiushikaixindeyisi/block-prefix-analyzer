"""Tokenizer adapter protocol and character-level implementation.

The :class:`TokenizerAdapter` protocol defines what the V2 pipeline expects
from a tokenizer: encode a string to a list of non-negative integers.

The bundled :class:`CharTokenizer` maps each character to its Unicode code
point (``ord(c)``).  It is deterministic, zero-dependency, and produces
one token per character.  It is NOT aligned with any real tokenizer and
must not be used for framework-correctness comparisons.

To plug in a HuggingFace tokenizer::

    from transformers import AutoTokenizer

    class HFTokenizerAdapter:
        def __init__(self, model_name: str) -> None:
            self._tok = AutoTokenizer.from_pretrained(model_name)

        def encode(self, text: str) -> list[int]:
            return self._tok.encode(text, add_special_tokens=False)

        def name(self) -> str:
            return self._tok.name_or_path

    records = build_block_records_from_raw_requests(
        requests,
        tokenizer=HFTokenizerAdapter("Qwen/Qwen2-7B"),
    )
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Protocol for text tokenizers.

    An adapter encodes a rendered prompt string into a list of integer token
    IDs.  Token IDs must be non-negative integers (they will be hashed by the
    block builder).
    """

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` to a list of non-negative integer token IDs."""
        ...

    def name(self) -> str:
        """Human-readable identifier for this tokenizer (stored in metadata)."""
        ...


class CharTokenizer:
    """Map each character to its Unicode code point.

    ``encode("Hi")`` → ``[72, 105]``

    Properties:
    * Deterministic across all Python versions (``ord()`` is stable).
    * One token per character.
    * Max token ID is 1,114,111 (``0x10FFFF``), fits in uint32.
    * NOT aligned with any model tokenizer.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def name(self) -> str:
        return "char_tokenizer"
