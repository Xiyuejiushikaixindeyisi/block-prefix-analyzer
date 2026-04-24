"""Business dataset JSONL loader.

Reads a JSONL file where each line must contain:

    user_id     – tenant / user identifier
    request_id  – unique request identifier
    timestamp   – arrival time (seconds; see note below)
    raw_prompt  – the already-rendered prompt text

Converts each record to a :class:`~block_prefix_analyzer.types.RequestRecord`
via the V2 pipeline using :class:`CharTokenizer` (one character → one token)
and :class:`~block_prefix_analyzer.v2.adapters.block_builder.SimpleBlockBuilder`
(SHA-256 per block). No external tokenizer or hash library is required.

Tokenizer note
--------------
``CharTokenizer`` maps each Unicode character to its code-point integer.
One character ≠ one subword token, so block counts differ from token-level
analysis. The hashes are internally consistent across the entire dataset and
fully support prefix-reuse analysis; they are NOT aligned with any vLLM
deployment's actual KV-cache block IDs.

Block-size guidance
-------------------
* ``block_size=128`` — primary target (vLLM-Ascend default).
* ``block_size=16 / 32 / 64`` — secondary, for finer-grained analysis.
Call this function once per desired block_size; each call is independent.

Single-turn / Agent splitting
------------------------------
This loader returns ALL records without splitting. Phase-3 work will add
:func:`~block_prefix_analyzer.analysis.request_classifier.filter_single_turn`.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from block_prefix_analyzer.types import RequestRecord
from block_prefix_analyzer.v2.adapters.block_builder import SimpleBlockBuilder
from block_prefix_analyzer.v2.adapters.tokenizer import CharTokenizer
from block_prefix_analyzer.v2.pipeline import build_block_records_from_raw_requests
from block_prefix_analyzer.v2.schema import Message, RawRequest

# ---------------------------------------------------------------------------
# Internal chat-template: pass raw_prompt through unchanged
# ---------------------------------------------------------------------------

class _RawPromptTemplate:
    """Return the first user-role message content verbatim.

    ``raw_prompt`` values are fully rendered strings. Adding chat-template
    markers (``<|user|>`` etc.) would prepend identical bytes to every
    request, creating an artificial shared-prefix segment and inflating the
    first block's hit rate.  This template avoids that distortion.
    """

    def render(self, messages: list[Message]) -> str:
        for msg in messages:
            if msg.role == "user":
                return msg.content
        return ""

    def name(self) -> str:
        return "raw_prompt_passthrough"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: tuple[str, ...] = ("user_id", "request_id", "timestamp", "raw_prompt")
_DEFAULT_WARN_THRESHOLD: int = 30_000_000


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_business_jsonl(
    path: str | Path,
    *,
    block_size: int | None = None,
    block_builder: SimpleBlockBuilder | None = None,
    field_map: dict[str, str] | None = None,
    include_debug_metadata: bool = False,
    warn_memory_threshold: int = _DEFAULT_WARN_THRESHOLD,
    block_registry: dict[int, str] | None = None,
) -> list[RequestRecord]:
    """Load a business-dataset JSONL file and return ``RequestRecord`` objects.

    Parameters
    ----------
    path:
        Path to the JSONL file. Each non-empty line must be a JSON object.
    block_size:
        Characters (via ``CharTokenizer``) per block. **Must be provided**
        unless ``block_builder`` is given.
        Recommended values: ``128`` (vLLM-Ascend primary), ``16``, ``32``, ``64``.
    block_builder:
        Pre-constructed :class:`SimpleBlockBuilder`. When given, ``block_size``
        is ignored.
    field_map:
        Renames source fields to the canonical names expected by this loader.
        Keys are canonical names; values are the actual source field names.
        Example — if the file uses ``"prompt"`` instead of ``"raw_prompt"``::

            field_map={"raw_prompt": "prompt"}

        Fields not listed use their canonical names unchanged.
    include_debug_metadata:
        Store ``v2_rendered_prompt``, ``v2_token_count``, ``v2_leftover_tokens``,
        ``v2_tokenizer``, and ``v2_chat_template`` in each record's metadata.
        Defaults to ``False`` to reduce memory usage for large datasets.
        Set to ``True`` when :func:`~block_prefix_analyzer.analysis.request_classifier.classify_requests`
        is needed (keyword detection reads ``v2_rendered_prompt``).
    warn_memory_threshold:
        Emit a :class:`ResourceWarning` when total block count exceeds this
        value. Pass ``warn_memory_threshold=0`` to suppress.
    block_registry:
        If provided, this dict is populated in-place with
        ``{block_id: text_slice}`` mappings, where ``text_slice`` is the
        exact ``raw_prompt`` substring that was hashed to produce that
        ``block_id``. Existing entries are **not** overwritten (first-seen
        wins). Pass an empty dict ``{}`` and inspect it after the call.
        Used by :func:`~block_prefix_analyzer.analysis.block_text_decoder.decode_ngram_rows`
        to convert block IDs back to human-readable text.

    Returns
    -------
    list[RequestRecord]
        Sorted by ``(timestamp, arrival_index)``. Every record has
        ``metadata["user_id"]`` set to the tenant identifier.

    Raises
    ------
    ValueError
        If neither ``block_size`` nor ``block_builder`` is given, if a line
        contains invalid JSON, or if a required field is absent.
    """
    if block_size is None and block_builder is None:
        raise ValueError(
            "block_size must be specified. "
            "Typical values: block_size=128 (vLLM-Ascend), 16, 32, or 64. "
            "For a custom builder pass block_builder=SimpleBlockBuilder(block_size=N)."
        )

    # Build the effective field map (canonical → source)
    effective: dict[str, str] = {f: f for f in _REQUIRED_FIELDS}
    if field_map:
        effective.update(field_map)

    # Agent-specific optional fields passed through to metadata unchanged.
    _AGENT_PASSTHROUGH_FIELDS = ("chat_id", "turn_index")

    raw_requests: list[RawRequest] = []
    user_ids: list[str] = []           # parallel to raw_requests; indexed by arrival_index
    raw_prompts: list[str] = []        # parallel to raw_requests; used for block_registry
    agent_extras: list[dict] = []      # parallel to raw_requests; Agent passthrough fields

    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc

            try:
                user_id    = str(obj[effective["user_id"]])
                request_id = str(obj[effective["request_id"]])
                timestamp  = float(obj[effective["timestamp"]])
                raw_prompt = str(obj[effective["raw_prompt"]])
            except KeyError as exc:
                raise ValueError(
                    f"{path}:{lineno}: missing field {exc}. "
                    f"Fields present: {sorted(obj)}. "
                    f"Use field_map= to remap field names."
                ) from exc

            raw_requests.append(RawRequest(
                request_id=request_id,
                timestamp=timestamp,
                messages=[Message(role="user", content=raw_prompt)],
            ))
            user_ids.append(user_id)
            raw_prompts.append(raw_prompt)
            agent_extras.append(
                {k: obj[k] for k in _AGENT_PASSTHROUGH_FIELDS if k in obj}
            )

    if not raw_requests:
        return []

    records = build_block_records_from_raw_requests(
        raw_requests,
        block_size=block_size,
        block_builder=block_builder,
        chat_template=_RawPromptTemplate(),
        tokenizer=CharTokenizer(),
        include_debug_metadata=include_debug_metadata,
    )

    # build_block_records_from_raw_requests assigns arrival_index by input-list
    # position before sorting, so record.arrival_index indexes into user_ids / raw_prompts.
    effective_bs: int = block_builder.block_size if block_builder is not None else block_size  # type: ignore[assignment]
    for record in records:
        record.metadata["user_id"] = user_ids[record.arrival_index]
        record.metadata.update(agent_extras[record.arrival_index])
        if block_registry is not None:
            rp = raw_prompts[record.arrival_index]
            for i, bid in enumerate(record.block_ids):
                if bid not in block_registry:
                    block_registry[bid] = rp[i * effective_bs : (i + 1) * effective_bs]

    total_blocks = sum(len(r.block_ids) for r in records)
    if warn_memory_threshold > 0 and total_blocks > warn_memory_threshold:
        warnings.warn(
            f"load_business_jsonl: total block count {total_blocks:,} exceeds "
            f"threshold {warn_memory_threshold:,}. "
            f"TrieIndex memory may be large; consider slicing by time window "
            f"(e.g. 24 h) before calling replay(). "
            f"Pass warn_memory_threshold=0 to suppress.",
            ResourceWarning,
            stacklevel=2,
        )

    return records
