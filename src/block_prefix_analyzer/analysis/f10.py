"""F10 analysis — per-user session turn-count mean and standard deviation.

Builds on F9 session reconstruction. For each user (identified by
metadata["user_id"]):

  mean_turns   = mean of session sizes across all that user's sessions
  std_turns    = population std dev of session sizes (0.0 for single-session users)

Outputs two Lorenz-style series (one per metric) sorted ascending so the
plotting layer can draw dual-Y-axis bar+cumulation figures matching the paper.

Lorenz cumulation definition
----------------------------
Users are ranked by their metric value (ascending).
cumulative_fraction[i] = sum(metric[0..i]) / sum(metric[all])

This is a strict Lorenz curve: it shows how much of the total metric mass
is concentrated in the highest-value users.

Analysis path: TraceA replay (Path A) or Agent JSONL —
uses metadata["user_id"] and metadata["parent_chat_id"].
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from block_prefix_analyzer.analysis.f9 import (
    reconstruct_sessions,
    reconstruct_sessions_by_chat_id,
)
from block_prefix_analyzer.types import RequestRecord


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

@dataclass
class F10UserRow:
    """Per-user statistics, used in both Plot-A and Plot-B."""
    user_id: str
    session_count: int
    mean_turns: float
    std_turns: float              # 0.0 when session_count == 1
    total_turns: int              # sum of all session sizes for this user


@dataclass
class F10LorenzRow:
    """One rank-sorted entry for the dual-Y-axis figure."""
    rank: int                     # 1-indexed, ascending by metric value
    user_id: str
    metric_value: float
    cumulative_fraction: float    # Lorenz cumulative fraction of total metric


@dataclass
class F10Series:
    total_users: int
    total_sessions: int
    users_with_single_session: int  # std_turns == 0 (only 1 session)

    # All users, unsorted
    user_rows: list[F10UserRow]

    # Plot A: sorted ascending by mean_turns
    mean_lorenz: list[F10LorenzRow]
    mean_min: float
    mean_max: float
    mean_overall: float           # grand mean across all users

    # Plot B: sorted ascending by std_turns
    std_lorenz: list[F10LorenzRow]
    std_max: float
    std_overall: float            # mean of per-user std values


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _lorenz_rows(
    user_rows: list[F10UserRow],
    metric: str,        # "mean_turns" or "std_turns"
) -> list[F10LorenzRow]:
    """Sort users by metric ascending and compute Lorenz cumulative fractions."""
    sorted_rows = sorted(user_rows, key=lambda r: getattr(r, metric))
    total = sum(getattr(r, metric) for r in sorted_rows)

    result: list[F10LorenzRow] = []
    cumsum = 0.0
    for rank, row in enumerate(sorted_rows, start=1):
        cumsum += getattr(row, metric)
        result.append(F10LorenzRow(
            rank=rank,
            user_id=row.user_id,
            metric_value=getattr(row, metric),
            cumulative_fraction=cumsum / total if total > 0 else 0.0,
        ))
    return result


def compute_f10_series(records: list[RequestRecord]) -> F10Series:
    """Compute per-user mean/std of session turn counts.

    Parameters
    ----------
    records:
        All records.  Each must have ``metadata["user_id"]`` set.
        Records without ``user_id`` are grouped under ``"__unknown__"``.
    """
    has_chat_id = any(r.metadata.get("chat_id") is not None for r in records)
    if has_chat_id:
        sessions = reconstruct_sessions_by_chat_id(records)
    else:
        sessions = reconstruct_sessions(records)

    # Extract user_id from the first record of each session.
    # Works for both Path A (root_id == request_id) and Path B (root_id == chat_id).
    user_to_sizes: dict[str, list[int]] = defaultdict(list)

    for sess_recs in sessions.values():
        first = sess_recs[0] if sess_recs else None
        uid = (first.metadata.get("user_id") or "__unknown__") if first else "__unknown__"
        user_to_sizes[uid].append(len(sess_recs))

    user_rows: list[F10UserRow] = []
    for uid, sizes in user_to_sizes.items():
        mean_t = statistics.mean(sizes)
        std_t = statistics.stdev(sizes) if len(sizes) > 1 else 0.0
        user_rows.append(F10UserRow(
            user_id=uid,
            session_count=len(sizes),
            mean_turns=mean_t,
            std_turns=std_t,
            total_turns=sum(sizes),
        ))

    total_users = len(user_rows)
    total_sessions = sum(r.session_count for r in user_rows)
    single_session_users = sum(1 for r in user_rows if r.session_count == 1)

    mean_lorenz = _lorenz_rows(user_rows, "mean_turns")
    std_lorenz  = _lorenz_rows(user_rows, "std_turns")

    all_means = [r.mean_turns for r in user_rows]
    all_stds  = [r.std_turns  for r in user_rows]

    return F10Series(
        total_users=total_users,
        total_sessions=total_sessions,
        users_with_single_session=single_session_users,
        user_rows=user_rows,
        mean_lorenz=mean_lorenz,
        mean_min=min(all_means),
        mean_max=max(all_means),
        mean_overall=statistics.mean(all_means),
        std_lorenz=std_lorenz,
        std_max=max(all_stds),
        std_overall=statistics.mean(all_stds),
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_f10_mean_csv(series: F10Series, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "user_id", "mean_turns", "cumulative_fraction"])
        for row in series.mean_lorenz:
            w.writerow([row.rank, row.user_id,
                        f"{row.metric_value:.4f}", f"{row.cumulative_fraction:.6f}"])


def save_f10_std_csv(series: F10Series, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "user_id", "std_turns", "cumulative_fraction"])
        for row in series.std_lorenz:
            w.writerow([row.rank, row.user_id,
                        f"{row.metric_value:.4f}", f"{row.cumulative_fraction:.6f}"])


def save_f10_metadata_json(
    series: F10Series,
    path: Path,
    *,
    trace_name: str,
    input_file: str,
    note: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "figure": "F10",
        "trace_name": trace_name,
        "input_file": input_file,
        "total_users": series.total_users,
        "total_sessions": series.total_sessions,
        "users_with_single_session": series.users_with_single_session,
        "users_with_multi_session": series.total_users - series.users_with_single_session,
        "mean_turns_overall": round(series.mean_overall, 4),
        "mean_turns_min": round(series.mean_min, 4),
        "mean_turns_max": round(series.mean_max, 4),
        "std_turns_overall": round(series.std_overall, 4),
        "std_turns_max": round(series.std_max, 4),
        "note": note,
        "analysis_path": "parent_chat_id session reconstruction + metadata[user_id] grouping",
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
