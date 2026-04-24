#!/usr/bin/env python3
"""Convert Agent system CSV data to multi-turn JSONL for block-prefix-analyzer.

Agent data difference from single-turn MaaS data
-------------------------------------------------
In Agent CSV, the original request_id is a SESSION identifier (chat_id): one
agent task may trigger hundreds of LLM calls sharing the same request_id.
This script reconstructs turn ordering within each session and emits one
JSONL record per turn with a unique request_id = "{chat_id}_{turn_index}".

Output JSONL fields (one JSON object per line)
----------------------------------------------
    user_id      – tenant / user identifier
    request_id   – "{chat_id}_{turn_index}"  (unique per turn)
    timestamp    – arrival time (float, same unit as source)
    raw_prompt   – prompt text
    chat_id      – original session identifier (for F9/F10 session analysis)
    turn_index   – 0-based position within session, ordered by timestamp

Session ordering rule
---------------------
Within each chat_id, turns are ordered by (timestamp, file_arrival_order).
Duplicate timestamps within the same chat_id are broken by the row's position
in the CSV (stable sort).

Usage
-----
    python scripts/convert_agent_csv_to_jsonl.py \\
        --input  data/internal/<model>/raw/<file>.csv \\
        --output data/internal/<model>/requests.jsonl \\
        --col-chat-id 0 --col-user-id 1 --col-raw-prompt 2 --col-timestamp 3 \\
        --has-header --encoding utf-8-sig

Column defaults match the standard Agent CSV export format:
    col 0: request_id  (→ chat_id)
    col 1: user_id
    col 2: raw_prompt
    col 3: timestamp

Memory note
-----------
All rows are buffered in memory to sort within sessions. For a 500 MB CSV with
average 50 K-char prompts this requires ~500 MB RAM. Safe on lab machines.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Timestamp normalisation (shared with convert_csv_to_jsonl.py)
# ---------------------------------------------------------------------------

def _parse_timestamp(raw: str, row_num: int) -> float:
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    print(
        f"[WARN] row {row_num}: cannot parse timestamp '{raw}'; defaulting to 0.0",
        file=sys.stderr,
    )
    return 0.0


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    input_path: Path,
    output_path: Path,
    col_chat_id: int = 0,
    col_user_id: int = 1,
    col_raw_prompt: int = 2,
    col_timestamp: int = 3,
    has_header: bool = False,
    encoding: str = "utf-8",
    progress_every: int = 10_000,
) -> None:
    """Read Agent CSV, reconstruct turn order, write multi-turn JSONL."""
    csv.field_size_limit(10 * 1024 * 1024)  # allow up to 10 MB per field
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_cols = max(col_chat_id, col_user_id, col_raw_prompt, col_timestamp)

    # ---- Pass 1: buffer all rows grouped by chat_id ----
    # Each entry: (timestamp, file_order, user_id, raw_prompt)
    sessions: dict[str, list[tuple[float, int, str, str]]] = defaultdict(list)
    skipped = 0
    total_rows = 0

    print(f"[Pass 1] Reading {input_path} ...")
    with input_path.open(encoding=encoding, errors="replace", newline="") as fin:
        reader = csv.reader(fin)
        if has_header:
            try:
                header = next(reader)
                print(f"  Skipping header: {header[:5]}")
            except StopIteration:
                print("[WARN] CSV file appears empty.", file=sys.stderr)
                return

        for row_num, row in enumerate(reader, start=2 if has_header else 1):
            if len(row) <= required_cols:
                print(
                    f"[WARN] row {row_num}: only {len(row)} columns, "
                    f"need {required_cols + 1}; skipping.",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            chat_id   = row[col_chat_id].strip()
            user_id   = row[col_user_id].strip()
            raw_prompt = row[col_raw_prompt]
            timestamp  = _parse_timestamp(row[col_timestamp], row_num)
            sessions[chat_id].append((timestamp, total_rows, user_id, raw_prompt))
            total_rows += 1

            if total_rows % progress_every == 0:
                print(f"  ... {total_rows:,} rows read  ({len(sessions):,} sessions so far)")

    print(f"  {total_rows:,} rows read, {len(sessions):,} sessions found"
          + (f"  ({skipped} rows skipped)" if skipped else ""))

    # ---- Pass 2: sort within sessions, assign turn_index, write JSONL ----
    print(f"[Pass 2] Sorting turns and writing {output_path} ...")
    written = 0
    multi_turn_sessions = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for chat_id, turns in sessions.items():
            # Sort by (timestamp, file_arrival_order) — stable within duplicates
            turns.sort(key=lambda t: (t[0], t[1]))
            if len(turns) > 1:
                multi_turn_sessions += 1

            for turn_index, (timestamp, _, user_id, raw_prompt) in enumerate(turns):
                record = {
                    "user_id":    user_id,
                    "request_id": f"{chat_id}_{turn_index}",
                    "timestamp":  timestamp,
                    "raw_prompt": raw_prompt,
                    "chat_id":    chat_id,
                    "turn_index": turn_index,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    total_sessions = len(sessions)
    single_turn = total_sessions - multi_turn_sessions
    print(
        f"\n[DONE] {written:,} records written to {output_path}\n"
        f"  Sessions total   : {total_sessions:,}\n"
        f"  Single-turn      : {single_turn:,}  "
        f"({single_turn / total_sessions * 100:.1f}%)\n"
        f"  Multi-turn       : {multi_turn_sessions:,}  "
        f"({multi_turn_sessions / total_sessions * 100:.1f}%)\n"
        f"  Avg turns/session: {written / total_sessions:.2f}"
        + (f"\n  Rows skipped     : {skipped}" if skipped else "")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Agent CSV to multi-turn JSONL for block-prefix-analyzer"
    )
    parser.add_argument("--input",  required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--col-chat-id",    type=int, default=0, metavar="N",
                        help="0-indexed column for the session ID (original request_id → chat_id) "
                             "[default: 0]")
    parser.add_argument("--col-user-id",    type=int, default=1, metavar="N",
                        help="0-indexed column for user_id / tenant_id [default: 1]")
    parser.add_argument("--col-raw-prompt", type=int, default=2, metavar="N",
                        help="0-indexed column for prompt text [default: 2]")
    parser.add_argument("--col-timestamp",  type=int, default=3, metavar="N",
                        help="0-indexed column for timestamp [default: 3]")
    parser.add_argument("--has-header", action="store_true",
                        help="Skip the first row as a header")
    parser.add_argument("--encoding", default="utf-8",
                        help="Input file encoding [default: utf-8]")
    parser.add_argument("--progress-every", type=int, default=10_000, metavar="N",
                        help="Print progress every N rows [default: 10000]")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {input_path} → {output_path}")
    print(f"  chat_id col: {args.col_chat_id}  user_id col: {args.col_user_id}  "
          f"prompt col: {args.col_raw_prompt}  timestamp col: {args.col_timestamp}")
    convert(
        input_path,
        output_path,
        col_chat_id    = args.col_chat_id,
        col_user_id    = args.col_user_id,
        col_raw_prompt = args.col_raw_prompt,
        col_timestamp  = args.col_timestamp,
        has_header     = args.has_header,
        encoding       = args.encoding,
        progress_every = args.progress_every,
    )


if __name__ == "__main__":
    main()
