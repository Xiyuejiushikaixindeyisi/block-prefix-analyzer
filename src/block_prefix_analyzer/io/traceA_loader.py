"""TraceA-format JSONL loader.

Maps external trace field names to the internal RequestRecord model:
  chat_id         → request_id  (str)
  hash_ids        → block_ids   (list[int])
  timestamp       → timestamp
  parent_chat_id  → metadata["parent_chat_id"]  (V2+)
  type            → metadata["type"]             (V2+)
  turn            → metadata["turn"]             (V2+)
  input_length    → token_count (optional)
  output_length   → metadata["output_length"]

The core loader (jsonl_loader.py) is not modified; this module is the
thin adapter for TraceA-specific field names.
"""
from __future__ import annotations

import json
from pathlib import Path

from block_prefix_analyzer.io.jsonl_loader import LoadError
from block_prefix_analyzer.types import RequestRecord, sort_records

_TRACEА_EXTRA_FIELDS = ("parent_chat_id", "type", "turn", "output_length")


def load_traceA_jsonl(path: Path | str) -> list[RequestRecord]:
    """Load a TraceA JSONL file and return records sorted by (timestamp, arrival_index).

    Required trace fields: chat_id, timestamp, hash_ids.
    Optional: parent_chat_id, type, turn, input_length, output_length.
    """
    path = Path(path)
    records: list[RequestRecord] = []
    arrival_index = 0

    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LoadError(lineno, f"invalid JSON: {exc}") from exc

            for field in ("chat_id", "timestamp", "hash_ids"):
                if field not in obj:
                    raise LoadError(lineno, f"missing required field '{field}'")

            ts = obj["timestamp"]
            if isinstance(ts, bool) or not isinstance(ts, (int, float)):
                raise LoadError(lineno, "'timestamp' must be a number")

            hash_ids = obj["hash_ids"]
            if not isinstance(hash_ids, list):
                raise LoadError(lineno, "'hash_ids' must be a list")

            metadata: dict = {}
            for key in _TRACEА_EXTRA_FIELDS:
                if key in obj:
                    metadata[key] = obj[key]

            records.append(RequestRecord(
                request_id=str(obj["chat_id"]),
                timestamp=ts,
                arrival_index=arrival_index,
                block_ids=hash_ids,
                token_count=obj.get("input_length"),
                metadata=metadata,
            ))
            arrival_index += 1

    return sort_records(records)
