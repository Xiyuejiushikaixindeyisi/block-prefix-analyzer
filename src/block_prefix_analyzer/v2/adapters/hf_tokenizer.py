"""HuggingFace tokenizer adapter (optional dependency).

Layer-2 alignment status: PENDING
-----------------------------------
This adapter wraps ``transformers.AutoTokenizer`` to provide real token IDs
from a named model checkpoint.  It requires:

    pip install transformers

and network access to download the model vocabulary on first use.

When ``transformers`` is not installed, importing this module succeeds but
constructing ``HFTokenizerAdapter`` raises ``ImportError``.  Tests that
require this adapter should guard with ``pytest.importorskip("transformers")``.

Target configuration
---------------------
The intended target is Qwen2-7B-Instruct (or any Qwen2-* variant).
Once connected, token IDs produced by this adapter will match vLLM's
tokenization exactly, enabling Layer-2 verified golden fixtures.
"""
from __future__ import annotations


class HFTokenizerAdapter:
    """Wraps ``transformers.AutoTokenizer`` for real tokenization.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model identifier or local path, e.g.
        ``"Qwen/Qwen2-7B-Instruct"``.
    add_special_tokens:
        Passed to ``tokenizer.encode()``.  Default ``False`` matches the
        inference-time convention where the chat template has already
        rendered all special tokens as text.

    Raises
    ------
    ImportError
        If ``transformers`` is not installed.
    OSError
        If the model cannot be loaded (network unavailable, wrong name, etc.).
    """

    def __init__(
        self,
        model_name_or_path: str,
        add_special_tokens: bool = False,
    ) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "HFTokenizerAdapter requires 'transformers'. "
                "Install with: pip install transformers"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model_name = model_name_or_path
        self._add_special_tokens = add_special_tokens

    def encode(self, text: str) -> list[int]:
        """Return token IDs for ``text`` using the loaded tokenizer."""
        return self._tokenizer.encode(text, add_special_tokens=self._add_special_tokens)

    def name(self) -> str:
        return f"hf_tokenizer:{self._model_name}"
