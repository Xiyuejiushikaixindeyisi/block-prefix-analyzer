"""F9 analysis — session turn-count CDF.

Two session reconstruction paths are supported:

Path A — TraceA (parent_chat_id chain)
    ``reconstruct_sessions(records)``
    * Root : parent_chat_id absent OR < 0
    * Child: parent_chat_id >= 0, points to parent's integer chat_id
    Used for: TraceA public dataset.

Path B — Agent JSONL (direct chat_id grouping)
    ``reconstruct_sessions_by_chat_id(records)``
    * Groups all records sharing metadata["chat_id"] into one session.
    * Sorted within each session by (timestamp, arrival_index).
    Used for: Agent JSONL produced by convert_agent_csv_to_jsonl.py.

``compute_f9_series`` auto-detects the path: if any record carries
``metadata["chat_id"]``, Path B is used; otherwise Path A.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from block_prefix_analyzer.types import RequestRecord, sort_records


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

@dataclass
class F9CdfRow:
    turn_count: int
    session_count: int        # sessions with exactly this many turns
    cumulative_fraction: float


@dataclass
class F9Series:
    total_sessions: int
    total_requests: int
    single_turn_sessions: int
    multi_turn_sessions: int
    max_turns: int
    mean_turns: float
    cdf_rows: list[F9CdfRow]
    session_size_dist: dict[int, int]  # turn_count → #sessions


# ---------------------------------------------------------------------------
# Session reconstruction
# ---------------------------------------------------------------------------

def reconstruct_sessions(
    records: list[RequestRecord],
) -> dict[str, list[RequestRecord]]:
    """Group records into sessions via parent_chat_id chains.

    Returns
    -------
    dict[root_request_id, list[RequestRecord]]
        Each value is sorted by (timestamp, arrival_index).
    Records whose parent_chat_id is absent or non-integer are treated as roots.
    """
    id_to_record: dict[str, RequestRecord] = {r.request_id: r for r in records}

    # parent request_id (str) → list of child request_ids (str)
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []

    for r in records:
        pid = r.metadata.get("parent_chat_id")
        if pid is None:
            roots.append(r.request_id)
            continue
        try:
            pid_int = int(pid)
        except (ValueError, TypeError):
            roots.append(r.request_id)
            continue
        if pid_int < 0:
            roots.append(r.request_id)
        else:
            children[str(pid_int)].append(r.request_id)

    sessions: dict[str, list[RequestRecord]] = {}
    for root_id in roots:
        session_recs: list[RequestRecord] = []
        queue: deque[str] = deque([root_id])
        while queue:
            cur = queue.popleft()
            rec = id_to_record.get(cur)
            if rec is not None:
                session_recs.append(rec)
            queue.extend(children.get(cur, []))
        sessions[root_id] = sort_records(session_recs)

    return sessions


# ---------------------------------------------------------------------------
# Path B: Agent direct chat_id grouping
# ---------------------------------------------------------------------------

def reconstruct_sessions_by_chat_id(
    records: list[RequestRecord],
) -> dict[str, list[RequestRecord]]:
    """Group records into sessions by metadata["chat_id"] (Agent path).

    Returns
    -------
    dict[chat_id, list[RequestRecord]]
        Each value is sorted by (timestamp, arrival_index).
    Records without metadata["chat_id"] are each treated as their own session.
    """
    buckets: dict[str, list[RequestRecord]] = defaultdict(list)
    for r in records:
        cid = r.metadata.get("chat_id") or r.request_id
        buckets[str(cid)].append(r)
    return {cid: sort_records(recs) for cid, recs in buckets.items()}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_f9_series(records: list[RequestRecord]) -> F9Series:
    """Compute F9 session turn-count distribution and CDF.

    Auto-detects session reconstruction path:
    - Path B (Agent): used when any record has metadata["chat_id"].
    - Path A (TraceA): falls back to parent_chat_id chain reconstruction.
    """
    has_chat_id = any(r.metadata.get("chat_id") is not None for r in records)
    if has_chat_id:
        sessions = reconstruct_sessions_by_chat_id(records)
    else:
        sessions = reconstruct_sessions(records)

    size_counter: Counter[int] = Counter()
    for sess_recs in sessions.values():
        size_counter[len(sess_recs)] += 1

    total_sessions = sum(size_counter.values())
    total_requests = sum(k * v for k, v in size_counter.items())
    single_turn = size_counter.get(1, 0)
    multi_turn = total_sessions - single_turn
    max_turns = max(size_counter.keys()) if size_counter else 0
    mean_turns = total_requests / total_sessions if total_sessions > 0 else 0.0

    cdf_rows: list[F9CdfRow] = []
    cumulative = 0
    for turn_count in sorted(size_counter.keys()):
        cumulative += size_counter[turn_count]
        cdf_rows.append(F9CdfRow(
            turn_count=turn_count,
            session_count=size_counter[turn_count],
            cumulative_fraction=cumulative / total_sessions,
        ))

    return F9Series(
        total_sessions=total_sessions,
        total_requests=total_requests,
        single_turn_sessions=single_turn,
        multi_turn_sessions=multi_turn,
        max_turns=max_turns,
        mean_turns=mean_turns,
        cdf_rows=cdf_rows,
        session_size_dist=dict(size_counter),
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_f9_csv(series: F9Series, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["turn_count", "session_count", "cumulative_fraction"])
        for row in series.cdf_rows:
            w.writerow([row.turn_count, row.session_count,
                        f"{row.cumulative_fraction:.6f}"])


def save_f9_metadata_json(
    series: F9Series,
    path: Path,
    *,
    trace_name: str,
    input_file: str,
    note: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = series.total_sessions
    meta = {
        "figure": "F9",
        "trace_name": trace_name,
        "input_file": input_file,
        "total_sessions": total,
        "total_requests": series.total_requests,
        "single_turn_sessions": series.single_turn_sessions,
        "single_turn_fraction": round(series.single_turn_sessions / total, 6) if total else 0.0,
        "multi_turn_sessions": series.multi_turn_sessions,
        "multi_turn_fraction": round(series.multi_turn_sessions / total, 6) if total else 0.0,
        "max_turns": series.max_turns,
        "mean_turns": round(series.mean_turns, 4),
        "session_size_distribution": series.session_size_dist,
        "note": note,
        "analysis_path": "TraceA replay (Path A) — parent_chat_id chain reconstruction",
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
