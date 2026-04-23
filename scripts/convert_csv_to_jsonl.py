#!/usr/bin/env python3
"""Convert MaaS system CSV data to JSONL format for block-prefix-analyzer.

Two modes:

Mode A — direct fields (new internal datasets)
  CSV already contains {user_id, request_id, timestamp, raw_prompt} columns.
  Use --col-raw-prompt to specify the raw_prompt column index.

  python scripts/convert_csv_to_jsonl.py \\
      --input  data/internal/qwen_v3_32b_8k/raw/data.csv \\
      --output data/internal/qwen_v3_32b_8k/requests.jsonl \\
      --col-user-id 0 --col-request-id 1 --col-timestamp 2 --col-raw-prompt 3

Mode B — request_params JSON (legacy MaaS CSV with embedded API body)
  CSV column 2 is a JSON string {"messages": [...], ...}; raw_prompt is
  extracted by concatenating all message contents.

  python scripts/convert_csv_to_jsonl.py \\
      --input  data/internal/deepseek_v3_671b_8k/raw/data.csv \\
      --output data/internal/deepseek_v3_671b_8k/requests.jsonl

Common options:
  --has-header        Skip the first row as a header
  --encoding ENCODING Input file encoding (default: utf-8)
  --progress-every N  Print progress every N rows (default: 10000)

Streaming design: processes one CSV row at a time; safe for 500 MB+ files.
Output JSONL line (one per request):
    {"user_id": "...", "request_id": "...", "timestamp": 1700000000.0, "raw_prompt": "..."}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

def _extract_raw_prompt(request_params_str: str, row_num: int) -> str:
    """Parse request_params JSON and concatenate messages into a flat string.

    Supports two common layouts:
      {"messages": [{"role": "...", "content": "..."}, ...], ...}
      {"prompt": "..."}  (fallback if messages key is absent)

    Returns empty string on parse failure (row is still written with empty prompt).
    """
    try:
        params: dict = json.loads(request_params_str)
    except json.JSONDecodeError as exc:
        print(
            f"[WARN] row {row_num}: request_params is not valid JSON ({exc}); "
            "raw_prompt set to empty string.",
            file=sys.stderr,
        )
        return ""

    messages = params.get("messages")
    if messages is not None:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # multimodal: extract text parts only
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            parts.append(f"{role}: {content}" if role else str(content))
        return "\n\n".join(parts)

    # Fallback: plain "prompt" field
    prompt = params.get("prompt")
    if prompt is not None:
        return str(prompt)

    # Last resort: dump everything (minus the messages key)
    print(
        f"[WARN] row {row_num}: no 'messages' or 'prompt' key found in request_params; "
        "raw_prompt set to full JSON string.",
        file=sys.stderr,
    )
    return request_params_str


# ---------------------------------------------------------------------------
# Timestamp normalisation
# ---------------------------------------------------------------------------

def _parse_timestamp(raw: str, row_num: int) -> float:
    """Return a Unix float from either a numeric string or ISO-8601 string."""
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    # Try ISO-8601 (e.g. "2024-01-15T10:30:00Z" or "2024-01-15 10:30:00")
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
# Main conversion logic
# ---------------------------------------------------------------------------

def convert(
    input_path: Path,
    output_path: Path,
    col_request_id: int = 0,
    col_tenant_id: int = 1,
    col_params: int | None = 2,       # Mode B: JSON request_params column
    col_timestamp: int = 3,
    col_raw_prompt: int | None = None, # Mode A: direct raw_prompt column
    has_header: bool = False,
    encoding: str = "utf-8",
    progress_every: int = 10_000,
) -> None:
    """Convert CSV to JSONL.

    Mode A (col_raw_prompt is not None): read raw_prompt directly from that column.
    Mode B (col_raw_prompt is None):     extract raw_prompt from col_params JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    direct_mode = col_raw_prompt is not None
    if direct_mode:
        required_cols = max(col_request_id, col_tenant_id, col_timestamp, col_raw_prompt)
    else:
        assert col_params is not None
        required_cols = max(col_request_id, col_tenant_id, col_params, col_timestamp)

    written = 0
    skipped = 0

    with (
        input_path.open(encoding=encoding, errors="replace", newline="") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        reader = csv.reader(fin)
        if has_header:
            try:
                header = next(reader)
                print(f"[INFO] Skipping header row: {header[:5]}")
            except StopIteration:
                print("[WARN] CSV file appears empty.", file=sys.stderr)
                return

        for row_num, row in enumerate(reader, start=2 if has_header else 1):
            if len(row) <= required_cols:
                print(
                    f"[WARN] row {row_num}: only {len(row)} columns, need {required_cols + 1}; skipping.",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            request_id = row[col_request_id].strip()
            tenant_id  = row[col_tenant_id].strip()
            ts_str     = row[col_timestamp]

            if direct_mode:
                raw_prompt = row[col_raw_prompt]
            else:
                raw_prompt = _extract_raw_prompt(row[col_params], row_num)  # type: ignore[index]
            timestamp  = _parse_timestamp(ts_str, row_num)

            record = {
                "user_id":     tenant_id,
                "request_id":  request_id,
                "timestamp":   timestamp,
                "raw_prompt":  raw_prompt,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if written % progress_every == 0:
                print(f"  ... {written:,} rows written  (row {row_num:,} in CSV)")

    print(
        f"\n[DONE] {written:,} records written to {output_path}"
        + (f"  ({skipped} rows skipped)" if skipped else "")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MaaS CSV to JSONL for block-prefix-analyzer"
    )
    parser.add_argument("--input",  required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--col-user-id",    type=int, default=0, metavar="N",
                        help="0-indexed column for user_id / tenant_id (default: 0)")
    parser.add_argument("--col-request-id", type=int, default=1, metavar="N",
                        help="0-indexed column for request_id (default: 1)")
    parser.add_argument("--col-timestamp",  type=int, default=2, metavar="N",
                        help="0-indexed column for timestamp (default: 2)")
    parser.add_argument("--col-raw-prompt", type=int, default=3, metavar="N",
                        help="0-indexed column for raw_prompt text [Mode A, default: 3]. "
                             "Use --col-params instead to parse a request_params JSON column.")
    parser.add_argument("--col-params",     type=int, default=None, metavar="N",
                        help="0-indexed column for request_params JSON [Mode B]. "
                             "When set, overrides --col-raw-prompt and extracts raw_prompt "
                             "from messages[] inside the JSON.")
    parser.add_argument("--has-header", action="store_true",
                        help="Skip the first row as a header")
    parser.add_argument("--encoding", default="utf-8",
                        help="Input file encoding (default: utf-8)")
    parser.add_argument("--progress-every", type=int, default=10_000, metavar="N",
                        help="Print progress every N rows (default: 10000)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Mode B (--col-params) overrides Mode A (--col-raw-prompt)
    if args.col_params is not None:
        col_raw_prompt = None
        col_params     = args.col_params
        print(f"[Mode B] Extracting raw_prompt from request_params JSON in column {col_params}")
    else:
        col_raw_prompt = args.col_raw_prompt
        col_params     = None
        print(f"[Mode A] Reading raw_prompt directly from column {col_raw_prompt}")

    print(f"Converting {input_path} → {output_path}")
    convert(
        input_path,
        output_path,
        col_request_id  = args.col_request_id,
        col_tenant_id   = args.col_user_id,
        col_params      = col_params,
        col_timestamp   = args.col_timestamp,
        col_raw_prompt  = col_raw_prompt,
        has_header      = args.has_header,
        encoding        = args.encoding,
        progress_every  = args.progress_every,
    )


if __name__ == "__main__":
    main()
