"""Chat template adapter protocol and minimal implementation.

The :class:`ChatTemplateAdapter` protocol defines the single interface that
the V2 pipeline expects from a template renderer.  Any object with a
``render(messages) -> str`` and a ``name() -> str`` method satisfies it.

The bundled :class:`MinimalChatTemplate` is a deterministic placeholder that
formats messages with ``<|role|>`` markers.  It is NOT aligned with any
specific model's tokenizer or chat template and should not be used for
framework-correctness comparisons.  Its purpose is to make the pipeline
runnable end-to-end while a model-specific adapter is being developed.

To plug in a different template (e.g. vLLM's Jinja-based template):

    class VllmChatTemplate:
        def render(self, messages: list[Message]) -> str:
            ...  # call vllm's template engine
        def name(self) -> str:
            return "vllm_qwen"

    records = build_block_records_from_raw_requests(
        requests,
        chat_template=VllmChatTemplate(),
    )
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from block_prefix_analyzer.v2.schema import Message


@runtime_checkable
class ChatTemplateAdapter(Protocol):
    """Protocol for chat template renderers.

    An adapter converts a list of :class:`~block_prefix_analyzer.v2.schema.Message`
    objects into a single string that will be tokenized by the next stage.

    The rendered output is stored in ``RequestRecord.metadata["v2_rendered_prompt"]``
    when ``include_debug_metadata=True`` is passed to the pipeline.
    """

    def render(self, messages: list[Message]) -> str:
        """Render messages to a single prompt string."""
        ...

    def name(self) -> str:
        """Human-readable identifier for this template (stored in metadata)."""
        ...


class MinimalChatTemplate:
    """Deterministic placeholder template using ``<|role|>`` markers.

    Format produced::

        <|system|>
        {system content}
        <|user|>
        {user content}
        <|assistant|>

    The trailing ``<|assistant|>`` marker signals the start of the model's
    response.  Empty messages produce the marker only (no blank line).

    This template is NOT aligned with Qwen, Llama, or any other model.
    It exists solely to make the V2 pipeline runnable without external deps.
    """

    def render(self, messages: list[Message]) -> str:
        parts: list[str] = []
        for msg in messages:
            parts.append(f"<|{msg.role}|>\n{msg.content}" if msg.content else f"<|{msg.role}|>")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def name(self) -> str:
        return "minimal_chat_template"
