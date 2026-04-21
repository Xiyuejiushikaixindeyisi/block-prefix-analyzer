"""V2 pipeline entry point: raw requests → V1 RequestRecord list.

This module is the single composition point for the V2 chain::

    RawRequest
      → normalize()            (NormalizedRequest)
      → ChatTemplateAdapter    (rendered prompt str)
      → TokenizerAdapter       (token IDs list[int])
      → SimpleBlockBuilder     (BlockBuildResult)
      → RequestRecord          (V1-compatible)

The output is a list of :class:`~block_prefix_analyzer.types.RequestRecord`
objects sorted by ``(timestamp, arrival_index)`` and ready for the V1 chain::

    records = build_block_records_from_raw_requests(raw_requests)
    results = list(replay(records))          # V1 replay — unchanged
    summary = compute_metrics(results)       # V1 metrics — unchanged
    print(format_summary(summary))           # V1 reports — unchanged

Design decisions
----------------
* V1 modules (``replay``, ``metrics``, ``reports``) are NOT modified.
* Adapters are injected; defaults run with no external dependencies.
* ``arrival_index`` is assigned in input-list order before ``sort_records``,
  exactly as the JSONL loader does.
* Debug metadata (rendered prompt, token count, leftover tokens) is stored in
  ``RequestRecord.metadata`` and can be disabled for production use.
"""
from __future__ import annotations

from block_prefix_analyzer.types import RequestRecord, sort_records
from block_prefix_analyzer.v2.adapters.block_builder import SimpleBlockBuilder
from block_prefix_analyzer.v2.adapters.chat_template import (
    ChatTemplateAdapter,
    MinimalChatTemplate,
)
from block_prefix_analyzer.v2.adapters.tokenizer import CharTokenizer, TokenizerAdapter
from block_prefix_analyzer.v2.normalizer import normalize
from block_prefix_analyzer.v2.schema import RawRequest


def build_block_records_from_raw_requests(
    raw_requests: list[RawRequest],
    *,
    block_size: int | None = None,
    chat_template: ChatTemplateAdapter | None = None,
    tokenizer: TokenizerAdapter | None = None,
    block_builder: SimpleBlockBuilder | None = None,
    include_debug_metadata: bool = True,
) -> list[RequestRecord]:
    """Convert raw requests to V1-compatible ``RequestRecord`` objects.

    Applies the full V2 chain (normalize → render → tokenize → block-build)
    to each request and returns a list ready for :func:`~block_prefix_analyzer.replay.replay`.

    Parameters
    ----------
    raw_requests:
        Input requests in any order.  ``arrival_index`` is assigned by
        input-list position; ``sort_records`` then sorts by
        ``(timestamp, arrival_index)``.
    block_size:
        Number of tokens per block.  **Must be specified** if ``block_builder``
        is not provided.  Should match the vLLM deployment's ``block_size``
        (typically 16).  Raises ``ValueError`` when both ``block_size`` and
        ``block_builder`` are absent.

        When only ``block_size`` is given, a :class:`~block_prefix_analyzer.v2.adapters.block_builder.SimpleBlockBuilder`
        is created internally.  For vLLM-aligned chained-hash analysis, pass
        ``block_builder=ChainedBlockBuilder(block_size=<N>)`` explicitly.
    chat_template:
        Template adapter.  Defaults to :class:`~block_prefix_analyzer.v2.adapters.chat_template.MinimalChatTemplate`.
    tokenizer:
        Tokenizer adapter.  Defaults to :class:`~block_prefix_analyzer.v2.adapters.tokenizer.CharTokenizer`.
    block_builder:
        Block builder.  When provided, ``block_size`` is ignored.
    include_debug_metadata:
        When ``True`` (default), stores ``v2_rendered_prompt``,
        ``v2_token_count``, ``v2_leftover_tokens``, ``v2_tokenizer``, and
        ``v2_chat_template`` in each ``RequestRecord.metadata``.

    Returns
    -------
    list[RequestRecord]
        Sorted by ``(timestamp, arrival_index)``, ready for V1 replay.

    Raises
    ------
    ValueError
        When neither ``block_size`` nor ``block_builder`` is provided.
    """
    _template: ChatTemplateAdapter = chat_template if chat_template is not None else MinimalChatTemplate()
    _tokenizer: TokenizerAdapter = tokenizer if tokenizer is not None else CharTokenizer()

    if block_builder is not None:
        _builder: SimpleBlockBuilder = block_builder
    elif block_size is not None:
        _builder = SimpleBlockBuilder(block_size=block_size)
    else:
        raise ValueError(
            "block_size must be specified explicitly when block_builder is not provided. "
            "Pass block_size=<N> (e.g. block_size=16 for the default vLLM deployment). "
            "For vLLM-aligned chained-hash analysis, pass "
            "block_builder=ChainedBlockBuilder(block_size=<N>) instead."
        )

    records: list[RequestRecord] = []
    for arrival_index, raw in enumerate(raw_requests):
        norm = normalize(raw)
        rendered = _template.render(norm.messages)
        token_ids = _tokenizer.encode(rendered)
        result = _builder.build(token_ids)

        meta: dict = dict(norm.metadata)
        # Forward V2+ session/category fields into metadata
        if norm.parent_request_id is not None:
            meta["parent_request_id"] = norm.parent_request_id
        if norm.session_id is not None:
            meta["session_id"] = norm.session_id
        if norm.category is not None:
            meta["category"] = norm.category
        if norm.turn is not None:
            meta["turn"] = norm.turn
        # Debug metadata (can be disabled for production)
        if include_debug_metadata:
            meta["v2_rendered_prompt"] = rendered
            meta["v2_token_count"] = len(token_ids)
            meta["v2_leftover_tokens"] = result.leftover_token_count
            meta["v2_tokenizer"] = _tokenizer.name()
            meta["v2_chat_template"] = _template.name()

        records.append(RequestRecord(
            request_id=norm.request_id,
            timestamp=norm.timestamp,
            arrival_index=arrival_index,
            block_ids=result.block_ids,
            token_count=len(token_ids),
            block_size=_builder.block_size,
            metadata=meta,
        ))

    return sort_records(records)
