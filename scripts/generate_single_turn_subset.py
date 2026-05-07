#!/usr/bin/env python3
"""Filter a multi-turn JSONL trace to its single-turn subset.

Streams the input JSONL line by line and keeps only records whose
``turn_index == 0`` (i.e. the first turn of every chat). The line is written
verbatim — no JSON re-encoding — so all original fields are preserved
exactly.

Why this exists
---------------
Phase-1 visualisation defines "single-turn" as ``turn_index == 0`` (the first
turn of any chat, regardless of total chat length). F13 input must therefore
be pre-filtered with this rule. F14 continues to consume the full file and
applies its own internal multi-turn filter.

Missing-field policy
--------------------
A record with no ``turn_index`` key is treated as ``turn_index == 0`` and
kept. This matches the user's reference one-liner
(``obj.get('turn_index', 0) == 0``) and tolerates older datasets converted
before the agent-CSV pipeline added the field.

Usage
-----
    python scripts/generate_single_turn_subset.py \\
        --input  data/internal/<model>/requests.jsonl \\
        --output data/internal/<model>/requests_single_turn.jsonl

Streaming design: O(1) memory regardless of input size.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def filter_single_turn(input_path: Path, output_path: Path,
                       progress_every: int = 100_000) -> tuple[int, int]:
    """Stream-filter ``input_path`` to ``output_path``, keeping turn_index == 0.

    Returns a ``(read, kept)`` tuple. Records whose ``turn_index`` field is
    missing are kept (treated as 0). Records whose JSON cannot be parsed are
    skipped with a warning.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    read = 0
    kept = 0
    skipped_invalid_json = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            read += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] line {line_num}: invalid JSON ({exc}); skipping.",
                    file=sys.stderr,
                )
                skipped_invalid_json += 1
                continue

            if obj.get("turn_index", 0) == 0:
                # Write the original line verbatim to preserve byte-exact
                # field order, formatting, and any custom keys.
                if not line.endswith("\n"):
                    line += "\n"
                fout.write(line)
                kept += 1

            if read % progress_every == 0:
                print(f"  ... {read:,} read, {kept:,} kept")

    msg = f"\n[DONE] {kept:,} / {read:,} records kept (turn_index == 0)"
    if skipped_invalid_json:
        msg += f"  [{skipped_invalid_json} invalid-JSON lines skipped]"
    print(msg)
    return read, kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep only turn_index == 0 records from a multi-turn JSONL trace."
    )
    parser.add_argument("--input",  required=True,
                        help="Path to input JSONL (e.g. data/internal/<model>/requests.jsonl)")
    parser.add_argument("--output", required=True,
                        help="Path to output JSONL (e.g. data/internal/<model>/requests_single_turn.jsonl)")
    parser.add_argument("--progress-every", type=int, default=100_000, metavar="N",
                        help="Print progress every N input rows [default: 100000]")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Filtering {input_path} -> {output_path}")
    filter_single_turn(input_path, output_path, progress_every=args.progress_every)


if __name__ == "__main__":
    main()
