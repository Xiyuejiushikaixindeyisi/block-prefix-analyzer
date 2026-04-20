"""V2 alignment diagnostics: lightweight diff helpers for locating mismatches.

These helpers are used by alignment tests to produce human-readable failure
messages that pinpoint *which layer* of the V2 pipeline diverges from a
golden fixture.  They do not modify any state; all functions return strings.

Layer hierarchy (for interpreting output)::

    Layer 1 — chat_template.render(messages)     → rendered_prompt (str)
    Layer 2 — tokenizer.encode(rendered_prompt)  → token_ids (list[int])
    Layer 3 — block_builder.build(token_ids)     → block_ids + leftover

A mismatch at Layer 1 cascades through Layers 2 and 3, so always start
diagnosis from the outermost failing layer.

Standard library only; no matplotlib, no pandas.
"""
from __future__ import annotations

import difflib


# ---------------------------------------------------------------------------
# Text-level diff (Layer 1: rendered prompt)
# ---------------------------------------------------------------------------

def diff_text(expected: str, actual: str, label: str = "rendered_prompt") -> str:
    """Return a unified diff between two strings, or a short summary if equal.

    Useful for diagnosing chat-template mismatches.  Characters that differ
    only in whitespace/newlines are highlighted in the diff output.
    """
    if expected == actual:
        return f"[{label}] OK — strings are identical ({len(actual)} chars)"

    exp_lines = expected.splitlines(keepends=True)
    act_lines = actual.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        exp_lines, act_lines,
        fromfile=f"{label}:expected",
        tofile=f"{label}:actual",
        lineterm="",
    ))
    header = (
        f"[{label}] MISMATCH — expected {len(expected)} chars, got {len(actual)} chars\n"
        f"  First divergence at char index: {_first_char_diff(expected, actual)}\n"
    )
    return header + "\n".join(diff)


def _first_char_diff(a: str, b: str) -> int | str:
    """Return the index of the first differing character, or 'N/A'."""
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    return len(a) if len(a) != len(b) else "N/A"


# ---------------------------------------------------------------------------
# Token-level diff (Layer 2: token IDs)
# ---------------------------------------------------------------------------

def diff_token_ids(
    expected: list[int],
    actual: list[int],
    label: str = "token_ids",
    context: int = 3,
) -> str:
    """Summarise mismatches between two token ID sequences.

    Reports:
    - Whether lengths differ
    - Up to ``context`` mismatched positions (index, expected_id, actual_id)
    - The first mismatch index
    """
    if expected == actual:
        return f"[{label}] OK — {len(actual)} token IDs match exactly"

    lines = [
        f"[{label}] MISMATCH — expected {len(expected)} tokens, got {len(actual)} tokens"
    ]
    mismatches: list[tuple[int, int | str, int | str]] = []
    max_len = max(len(expected), len(actual))
    for i in range(max_len):
        exp_tok = expected[i] if i < len(expected) else "<missing>"
        act_tok = actual[i] if i < len(actual) else "<missing>"
        if exp_tok != act_tok:
            mismatches.append((i, exp_tok, act_tok))
        if len(mismatches) >= context:
            break

    for pos, exp_t, act_t in mismatches:
        lines.append(f"  pos {pos:4d}: expected {exp_t!r:>8}  actual {act_t!r:>8}")

    total_mismatches = sum(
        1 for i in range(max_len)
        if (expected[i] if i < len(expected) else None) != (actual[i] if i < len(actual) else None)
    )
    if total_mismatches > context:
        lines.append(f"  ... and {total_mismatches - context} more mismatches (showing first {context})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block-level diff (Layer 3: block IDs and leftover)
# ---------------------------------------------------------------------------

def diff_block_ids(
    expected: list[int],
    actual: list[int],
    label: str = "block_ids",
) -> str:
    """Summarise mismatches between two block ID sequences.

    Reports whether block counts differ and which block positions have
    different hash values.  A hash mismatch at block N means the token
    content of that block changed, which in turn means the rendered prompt
    or tokenization changed.
    """
    if expected == actual:
        return f"[{label}] OK — {len(actual)} block IDs match exactly"

    lines = [
        f"[{label}] MISMATCH — expected {len(expected)} blocks, got {len(actual)} blocks"
    ]
    max_len = max(len(expected), len(actual))
    for i in range(max_len):
        exp_b = expected[i] if i < len(expected) else "<missing>"
        act_b = actual[i] if i < len(actual) else "<missing>"
        if exp_b != act_b:
            lines.append(f"  block {i}: expected {exp_b!r}  actual {act_b!r}")
    return "\n".join(lines)


def diff_leftover(
    expected_leftover: int,
    actual_leftover: int,
    label: str = "leftover_token_count",
) -> str:
    """Check whether leftover token counts match."""
    if expected_leftover == actual_leftover:
        return f"[{label}] OK — {actual_leftover} leftover tokens"
    return (
        f"[{label}] MISMATCH — expected {expected_leftover} leftover tokens, "
        f"got {actual_leftover}"
    )


# ---------------------------------------------------------------------------
# Per-request alignment report
# ---------------------------------------------------------------------------

def alignment_report(
    fixture_name: str,
    *,
    rendered_prompt_result: str,
    token_ids_result: str,
    block_ids_result: str,
    leftover_result: str,
) -> str:
    """Combine per-layer diagnostic strings into a single printable report.

    Each ``*_result`` string should come from one of the diff functions above.
    The report groups results by layer so failures are easy to locate.
    """
    sep = "-" * 60
    lines = [
        sep,
        f"V2 Alignment Report — fixture: {fixture_name!r}",
        sep,
        "  Layer 1 (chat_template):",
        f"    {rendered_prompt_result}",
        "  Layer 2 (tokenizer):",
        f"    {token_ids_result}",
        "  Layer 3a (block_ids):",
        f"    {block_ids_result}",
        "  Layer 3b (leftover):",
        f"    {leftover_result}",
        sep,
    ]
    return "\n".join(lines)
