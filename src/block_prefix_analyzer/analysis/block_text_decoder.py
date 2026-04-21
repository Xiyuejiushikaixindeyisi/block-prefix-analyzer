"""Decode block IDs back to the original text slices they represent.

Intended use: after running ``build_top_ngrams()`` you have a list of
:class:`~block_prefix_analyzer.analysis.top_ngrams.NgramRow` objects whose
``blocks`` fields contain opaque integer IDs.  Load a dataset with a
``block_registry`` dict (via
:func:`~block_prefix_analyzer.io.business_loader.load_business_jsonl`) to
map each ID back to the exact text that was hashed to produce it, then call
:func:`decode_ngram_rows` to produce human-readable rows.

``BlockRegistry`` is a plain ``dict[int, str]`` — block_id → text slice.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from block_prefix_analyzer.analysis.top_ngrams import NgramRow

if TYPE_CHECKING:
    pass

BlockRegistry = dict[int, str]

_MISSING_PLACEHOLDER = "<MISSING:{id}>"


@dataclass
class DecodedNgramRow:
    rank: int
    count: int
    pct: float
    length: int
    text: str          # concatenated text of all blocks, possibly truncated
    truncated: bool    # True when text was cut to max_chars
    blocks: tuple[int, ...]  # original block IDs (for cross-referencing)


def decode_ngram_rows(
    rows: list[NgramRow],
    block_registry: BlockRegistry,
    max_chars: int = 300,
) -> list[DecodedNgramRow]:
    """Convert NgramRow list to DecodedNgramRow list with human-readable text.

    Parameters
    ----------
    rows:
        Output from :func:`~block_prefix_analyzer.analysis.top_ngrams.build_top_ngrams`.
    block_registry:
        ``{block_id: text_slice}`` mapping built by
        :func:`~block_prefix_analyzer.io.business_loader.load_business_jsonl`
        with ``block_registry={}``.
    max_chars:
        Maximum characters in the ``text`` field. Text is truncated with an
        ellipsis suffix when the concatenated content exceeds this limit.
        Pass ``0`` to disable truncation.

    Returns
    -------
    list[DecodedNgramRow]
        Same order and length as ``rows``.
    """
    result: list[DecodedNgramRow] = []
    for row in rows:
        parts: list[str] = []
        for bid in row.blocks:
            if bid in block_registry:
                parts.append(block_registry[bid])
            else:
                parts.append(_MISSING_PLACEHOLDER.format(id=bid))
        full_text = "".join(parts)
        truncated = False
        if max_chars > 0 and len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "…"
            truncated = True
        result.append(DecodedNgramRow(
            rank=row.rank,
            count=row.count,
            pct=row.pct,
            length=len(row.blocks),
            text=full_text,
            truncated=truncated,
            blocks=row.blocks,
        ))
    return result


def format_decoded_table(rows: list[DecodedNgramRow], title: str) -> str:
    lines = [title, "=" * len(title)]
    for r in rows:
        trunc_mark = " [truncated]" if r.truncated else ""
        lines.append(
            f"Rank {r.rank}  count={r.count:,}  pct={r.pct*100:.1f}%  "
            f"len={r.length} blocks{trunc_mark}"
        )
        lines.append(f"  Text: {r.text!r}")
        lines.append("")
    return "\n".join(lines)


def save_decoded_csv(rows: list[DecodedNgramRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "count", "pct_of_population", "length", "truncated", "text", "blocks"])
        for r in rows:
            w.writerow([
                r.rank, r.count, round(r.pct, 6), r.length, r.truncated,
                r.text,
                " ".join(str(b) for b in r.blocks),
            ])
