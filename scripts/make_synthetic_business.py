#!/usr/bin/env python3
"""Generate a synthetic business-dataset JSONL for smoke-testing Phase-2 scripts.

Output: data/synthetic/business_synthetic.jsonl

Three reuse patterns are embedded so that every Phase-2 analysis module sees
meaningful signal even without a real dataset:

  A  System-prompt prefix sharing — 8 users × 5 requests.
     All requests begin with a 512-char shared system prompt followed by a
     unique 200-char query.  Expected: strong prefix reuse (4 shared blocks at
     block_size=128; 32 shared blocks at block_size=16).

  B  Hot-prompt repeat — 20 different users each send the same 256-char query.
     Expected: moderate prefix reuse (2 shared blocks at block_size=128).

  C  Cold start — 10 users × 4 unique 160-char prompts; no content sharing.
     Expected: cold misses only (warm-up baseline).

Total: 100 requests, 7 200-second window (~2 hours).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT = Path(__file__).parent.parent / "data" / "synthetic" / "business_synthetic.jsonl"

# ---------------------------------------------------------------------------
# Shared text fixtures (deterministic, char-level tokens via CharTokenizer)
# ---------------------------------------------------------------------------

_SYS_BASE = (
    "You are a helpful AI assistant for an enterprise software platform. "
    "Your primary responsibilities include answering technical questions, "
    "summarizing documents, and assisting with data analysis tasks. "
    "Always provide clear, accurate, and actionable responses. "
    "Maintain confidentiality and adhere to company policies at all times. "
    "When uncertain, acknowledge limitations and suggest appropriate external resources. "
    "Your responses should be concise yet comprehensive, tailored to professional contexts."
)
# Pad to exactly 512 chars → 4 complete blocks at block_size=128
SYSTEM_PROMPT: str = (_SYS_BASE + " " * 512)[:512]
assert len(SYSTEM_PROMPT) == 512

_HOT_BASE = (
    "What are the best practices for optimizing database query performance in a "
    "high-traffic enterprise application? Please explain indexing strategies, "
    "query plan analysis, connection pooling, and caching approaches in detail."
)
# Pad to exactly 256 chars → 2 complete blocks at block_size=128
HOT_PROMPT: str = (_HOT_BASE + " " * 256)[:256]
assert len(HOT_PROMPT) == 256

# 200-char unique query suffixes for Pattern A (system-prompt users)
_QUERIES_A: list[str] = [
    "How do I implement OAuth 2.0 token refresh in a Python REST client safely?",
    "Explain the tradeoffs between optimistic and pessimistic database locking.",
    "What monitoring metrics should I collect for a Kubernetes deployment?",
    "Describe the differences between eventual and strong consistency models.",
    "How should I structure error handling in an async Python FastAPI service?",
    "What are common pitfalls when migrating from PostgreSQL 13 to PostgreSQL 16?",
    "Summarize the key principles of domain-driven design for a microservice arch.",
    "How do distributed tracing systems like Jaeger and Zipkin compare?",
    "What is the correct way to implement idempotent API endpoints in REST?",
    "Explain blue-green deployment strategy and its advantages over rolling updates.",
    "How should I manage secrets in a containerized production environment safely?",
    "What are the performance implications of using ORM versus raw SQL queries?",
    "Describe strategies for handling back-pressure in a message-driven system.",
    "How do I design a rate-limiting middleware for a multi-tenant API gateway?",
    "Explain the difference between horizontal and vertical scaling in cloud infra.",
    "What is circuit-breaker pattern and when should I use it in microservices?",
    "How do I implement distributed locking using Redis in a clustered deployment?",
    "Describe the CAP theorem and its practical implications for system design.",
    "What are best practices for structuring CI/CD pipelines in a monorepo setup?",
    "How should I choose between gRPC and REST for inter-service communication?",
    "Explain the concept of eventual consistency and how saga pattern addresses it.",
    "What strategies exist for zero-downtime database schema migrations in prod?",
    "How do I benchmark and profile Python async code in production environments?",
    "Describe the differences between CQRS and traditional CRUD architectures.",
    "What are common causes of memory leaks in long-running Python services?",
    "How should I configure connection pool sizes for optimal throughput?",
    "Explain the role of service meshes like Istio in a microservice ecosystem.",
    "What are the tradeoffs between event sourcing and traditional state storage?",
    "How do I implement proper request tracing across multiple microservices?",
    "Describe strategies for managing configuration drift in a cloud environment.",
    "What are the security implications of using JWTs for session management?",
    "How do I implement proper graceful shutdown in a containerized service?",
    "Explain the differences between pub-sub and message-queue messaging patterns.",
    "What metrics indicate that a service needs horizontal scaling adjustments?",
    "How should I structure multi-tenant data isolation in a shared database?",
    "Describe approaches for implementing full-text search in a SaaS platform.",
    "What are the key considerations when choosing a time-series database?",
    "How do I ensure data consistency across multiple microservice boundaries?",
    "Explain the concept of backfill in a data pipeline and its common issues.",
    "What are the best practices for API versioning in a long-lived service?",
]
assert len(_QUERIES_A) >= 40

# Pad each query to exactly 200 chars for clean block boundaries
QUERIES_A: list[str] = [
    (q + " " * 200)[:200] for q in _QUERIES_A[:40]
]

# 160-char unique cold prompts for Pattern C
def _cold_prompt(i: int) -> str:
    base = f"Cold query number {i:04d}: describe the architectural component labeled X{i:04d} in the system design documentation, focusing on its interface and integration points with upstream services."
    return (base + " " * 160)[:160]


# ---------------------------------------------------------------------------
# Request generation
# ---------------------------------------------------------------------------

def _make_record(
    user_id: str,
    request_id: str,
    timestamp: float,
    raw_prompt: str,
) -> dict:
    return {
        "user_id": user_id,
        "request_id": request_id,
        "timestamp": timestamp,
        "raw_prompt": raw_prompt,
    }


def generate() -> list[dict]:
    records: list[dict] = []

    # --- Pattern A: 8 users × 5 requests, shared 512-char system prompt ------
    # Interleave users at 90-second intervals; user k starts at k*90 seconds.
    query_idx = 0
    for user_idx in range(8):
        uid = f"u{user_idx + 1:03d}"
        base_t = user_idx * 90
        for req_idx in range(5):
            t = base_t + req_idx * 720  # 12-minute interval between same-user requests
            rid = f"A-{uid}-{req_idx:02d}"
            prompt = SYSTEM_PROMPT + QUERIES_A[query_idx]
            records.append(_make_record(uid, rid, float(t), prompt))
            query_idx += 1

    # --- Pattern B: 20 users, same 256-char hot prompt ------------------------
    # One request each, spread 300–6900 s
    for user_idx in range(20):
        uid = f"u{user_idx + 21:03d}"
        t = 300.0 + user_idx * 330.0
        rid = f"B-{uid}-00"
        records.append(_make_record(uid, rid, t, HOT_PROMPT))

    # --- Pattern C: 10 users × 4 cold prompts (no sharing) -------------------
    cold_idx = 0
    for user_idx in range(10):
        uid = f"u{user_idx + 41:03d}"
        base_t = 150.0 + user_idx * 60.0
        for req_idx in range(4):
            t = base_t + req_idx * 1800.0
            rid = f"C-{uid}-{req_idx:02d}"
            prompt = _cold_prompt(cold_idx)
            records.append(_make_record(uid, rid, t, prompt))
            cold_idx += 1

    # Sort by timestamp so the file is naturally ordered
    records.sort(key=lambda r: (r["timestamp"], r["request_id"]))
    return records


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    records = generate()
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written {len(records)} records → {OUTPUT}")
    print(f"  Time range : 0 – {max(r['timestamp'] for r in records):.0f} s "
          f"({max(r['timestamp'] for r in records) / 60:.1f} min)")
    print(f"  Users      : {len({r['user_id'] for r in records})}")
    print(f"  Pattern A  : {sum(1 for r in records if r['request_id'].startswith('A-'))} "
          f"requests (512-char system prompt + 200-char query)")
    print(f"  Pattern B  : {sum(1 for r in records if r['request_id'].startswith('B-'))} "
          f"requests (256-char hot prompt, shared)")
    print(f"  Pattern C  : {sum(1 for r in records if r['request_id'].startswith('C-'))} "
          f"requests (160-char cold unique prompts)")


if __name__ == "__main__":
    main()
