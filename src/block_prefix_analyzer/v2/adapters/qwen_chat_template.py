"""Qwen2/Qwen2.5 chat template adapter.

Layer-1 alignment status: VERIFIED
------------------------------------
The rendering logic is derived directly from the official Qwen2 Jinja template
published at https://huggingface.co/Qwen/Qwen2-7B-Instruct (tokenizer_config.json):

    {% for message in messages %}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
    {% endfor %}
    {% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
    {% endif %}

This Python class replicates the template faithfully without requiring the
``transformers`` library.  The output is byte-for-byte identical to
``AutoTokenizer.apply_chat_template(messages, tokenize=False,
add_generation_prompt=True)`` for Qwen2-* models.

Layer-2/3 alignment status: PENDING
-------------------------------------
Token IDs and block IDs still require the real Qwen vocabulary and a
compatible block hash (vLLM uses MurmurHash3 with prefix chaining).
See ``V2_READINESS.md`` for the full pending-alignment roadmap.
"""
from __future__ import annotations

from block_prefix_analyzer.v2.schema import Message


class QwenChatTemplate:
    """Chat template matching the official Qwen2/Qwen2.5 format.

    Always appends the assistant generation-prompt prefix
    ``<|im_start|>assistant\\n`` at the end, which is the standard
    inference-time rendering (``add_generation_prompt=True``).

    Parameters
    ----------
    add_generation_prompt:
        When ``True`` (default), appends ``<|im_start|>assistant\\n`` after
        the last user turn.  Set to ``False`` only for alignment testing where
        the reference output omits the generation prefix.
    """

    def __init__(self, add_generation_prompt: bool = True) -> None:
        self._add_generation_prompt = add_generation_prompt

    def render(self, messages: list[Message]) -> str:
        """Render messages using the Qwen2 chat template.

        Empty ``messages`` with ``add_generation_prompt=True`` returns just
        ``<|im_start|>assistant\\n``.
        """
        parts: list[str] = []
        for msg in messages:
            parts.append(
                f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
            )
        if self._add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def name(self) -> str:
        return "qwen2_chat_template"
