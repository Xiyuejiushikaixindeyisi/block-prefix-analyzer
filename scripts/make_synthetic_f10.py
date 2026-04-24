#!/usr/bin/env python3
"""Generate synthetic multi-turn JSONL with user_id for F10 development.

Creates a realistic user-session distribution:
  - Casual users  (~55%): 1-3 sessions, mostly 1-2 turns each
  - Regular users (~30%): 3-10 sessions, 1-5 turns each
  - Power users   (~15%): 10-40 sessions, 2-12 turns each

Output format: TraceA-compatible JSONL with an extra user_id field.
Usage:
    python scripts/make_synthetic_f10.py
    python scripts/make_synthetic_f10.py --seed 42 --users 300 \
        --output data/synthetic/f10_synthetic.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _poisson_clipped(rng: random.Random, lam: float, lo: int, hi: int) -> int:
    import math
    # Simple rejection sampling for small ranges
    for _ in range(1000):
        # Use normal approximation for larger lambda
        v = int(rng.gauss(lam, lam ** 0.5) + 0.5)
        if lo <= v <= hi:
            return v
    return lo


def generate(
    n_users: int = 300,
    seed: int = 42,
    output_path: Path = Path("data/synthetic/f10_synthetic.jsonl"),
) -> None:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chat_id = 0
    timestamp = 0.0
    records: list[dict] = []

    for u_idx in range(n_users):
        user_id = f"user_{u_idx:04d}"

        # Tier determines session count and turn distribution
        tier_roll = rng.random()
        if tier_roll < 0.55:       # casual
            n_sessions = rng.randint(1, 3)
            turn_lam, turn_max = 1.5, 4
        elif tier_roll < 0.85:     # regular
            n_sessions = rng.randint(3, 10)
            turn_lam, turn_max = 2.5, 8
        else:                       # power user
            n_sessions = rng.randint(10, 40)
            turn_lam, turn_max = 5.0, 15

        for _ in range(n_sessions):
            n_turns = _poisson_clipped(rng, turn_lam, 1, turn_max)
            session_root_id = chat_id
            prev_id = -1

            for t in range(n_turns):
                # Generate a short hash_ids list (F10 doesn't need real blocks)
                hash_ids = [rng.getrandbits(32) for _ in range(rng.randint(4, 20))]
                record = {
                    "chat_id":        chat_id,
                    "parent_chat_id": prev_id if t > 0 else -1,
                    "timestamp":      round(timestamp, 3),
                    "user_id":        user_id,
                    "input_length":   len(hash_ids) * 16,
                    "hash_ids":       hash_ids,
                    "type":           "text",
                    "turn":           t + 1,
                }
                records.append(record)
                prev_id = chat_id
                chat_id += 1
                timestamp += rng.uniform(5.0, 120.0)

            # Gap between sessions for same user
            timestamp += rng.uniform(300.0, 3600.0)

        # Gap between users
        timestamp += rng.uniform(60.0, 600.0)

    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print summary
    from collections import Counter, defaultdict
    children: dict[int, list[int]] = defaultdict(list)
    id_map = {r["chat_id"]: r for r in records}
    roots = []
    for r in records:
        if r["parent_chat_id"] < 0:
            roots.append(r["chat_id"])
        else:
            children[r["parent_chat_id"]].append(r["chat_id"])

    session_sizes: list[int] = []
    for root in roots:
        size = 0
        stack = [root]
        while stack:
            cur = stack.pop()
            size += 1
            stack.extend(children.get(cur, []))
        session_sizes.append(size)

    size_counter = Counter(session_sizes)
    user_sessions: dict[str, list[int]] = defaultdict(list)
    for root in roots:
        uid = id_map[root]["user_id"]
        size = 0
        stack = [root]
        while stack:
            cur = stack.pop()
            size += 1
            stack.extend(children.get(cur, []))
        user_sessions[uid].append(size)

    import statistics
    means = [statistics.mean(v) for v in user_sessions.values()]
    stds  = [statistics.stdev(v) if len(v) > 1 else 0.0 for v in user_sessions.values()]

    print(f"Generated {len(records):,} records  →  {output_path}")
    print(f"  Users         : {n_users}")
    print(f"  Total sessions: {len(session_sizes):,}")
    print(f"  Mean turns/session: {sum(session_sizes)/len(session_sizes):.2f}")
    print(f"  Max turns        : {max(session_sizes)}")
    print(f"  Per-user mean turns — min={min(means):.1f}  "
          f"median={sorted(means)[len(means)//2]:.1f}  max={max(means):.1f}")
    print(f"  Per-user std turns  — min={min(stds):.2f}  "
          f"median={sorted(stds)[len(stds)//2]:.2f}   max={max(stds):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--users",  type=int, default=300)
    parser.add_argument("--output", default="data/synthetic/f10_synthetic.jsonl")
    args = parser.parse_args()
    generate(n_users=args.users, seed=args.seed,
             output_path=Path(args.output))


if __name__ == "__main__":
    main()
