"""
Retrieval evaluation harness for Paper Pilot.

Measures ranking quality of the ACTUAL retrieval pipeline against a labelled
gold set. No LLM generation and no LLM judging -- all metrics are arithmetic
over ranked chunk IDs, so scoring costs zero API calls.

Gold chunks are resolved at runtime from section-heading regexes against real
chunk rows, so the harness survives re-ingestion (no hardcoded UUIDs).

Ablation configs:
    vector_only    dense pgvector search only
    hybrid         dense + Postgres FTS fused with RRF (k=60)
    hybrid_rerank  hybrid -> cross-encoder rerank (top 20 -> top 5)
    full           hybrid_rerank + multi-turn query rewriting

Usage (inside the backend container or with DATABASE_URL set):
    uv run python scripts/eval_retrieval.py
    uv run python scripts/eval_retrieval.py --top-k 5 --configs vector_only,hybrid
    uv run python scripts/eval_retrieval.py --dump-results results.json
"""

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import _get_session_factory
from app.services.embedder import embed_query
from app.services.query_rewriter import should_rewrite, rewrite_query
from app.services.reranker import rerank
from app.services.retriever import (
    _LEXICAL_SELECT,
    _VECTOR_SELECT,
    _build_tsquery,
    _rows_to_results,
    _rrf_fuse,
    ChunkResult,
)

_HERE = Path(__file__).resolve().parent
GOLD_SET_PATH = _HERE.parent / "evals" / "gold_set.json"

_EVAL_TOP_K = 20          # depth of the candidate list we score
_RERANK_CANDIDATES = 20   # matches production retrieve(top_k=20) -> rerank(5)
_RECALL_KS = (1, 3, 5, 10)
_PRECISION_KS = (3, 5, 10)

# Populated during the run: qid -> chunk rows, qid -> gold chunk ids.
chunks_cache: dict[str, list[dict]] = {}
GOLD_BY_QID: dict[str, set[str]] = {}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class EvalQuestion:
    qid: str
    paper_key: str
    query: str
    anchors: list[str]
    history: list[tuple[str, str]] = field(default_factory=list)
    paper_id: str | None = None
    gold_chunk_ids: set[str] = field(default_factory=set)
    status: str = "pending"


def load_gold_set() -> tuple[list[EvalQuestion], list[EvalQuestion], dict]:
    if not GOLD_SET_PATH.exists():
        raise SystemExit(f"Gold set not found at {GOLD_SET_PATH}")

    raw = json.loads(GOLD_SET_PATH.read_text())
    papers = raw["papers"]

    def build(rec: dict, with_history: bool) -> EvalQuestion:
        return EvalQuestion(
            qid=rec["id"],
            paper_key=rec["paper"],
            query=rec["question"],
            anchors=rec["anchors"],
            history=[tuple(h) for h in rec.get("history", [])] if with_history else [],
        )

    singles = [build(r, False) for r in raw["questions"]]
    followups = [build(r, True) for r in raw.get("followups", [])]
    return singles, followups, papers


# ---------------------------------------------------------------------------
# Gold resolution: map section regexes -> real chunk UUIDs
# ---------------------------------------------------------------------------
_PAPER_SQL = """
    SELECT p.id::text, p.title, p.status, COUNT(c.id) AS chunk_count
    FROM papers p
    LEFT JOIN chunks c ON c.paper_id = p.id
    GROUP BY p.id, p.title, p.status
    ORDER BY p.created_at DESC
"""


async def fetch_papers(db: AsyncSession) -> list[dict]:
    rows = (await db.execute(text(_PAPER_SQL))).mappings().all()
    return [dict(r) for r in rows]


async def fetch_chunks(db: AsyncSession, paper_id: str) -> list[dict]:
    rows = (
        await db.execute(
            text(
                """
                SELECT c.id::text AS chunk_id,
                       c.metadata ->> 'section_heading' AS heading,
                       c.content
                FROM chunks c
                WHERE c.paper_id = CAST(:pid AS uuid)
                ORDER BY (c.metadata ->> 'chunk_index')::int
                """
            ),
            {"pid": paper_id},
        )
    ).mappings().all()
    return [dict(r) for r in rows]


def resolve_paper(papers: list[dict], key: str, pattern: str) -> dict | None:
    rx = re.compile(pattern, re.IGNORECASE)
    candidates = [p for p in papers if rx.search(p["title"] or "")]
    if not candidates:
        return None
    # Prefer the most chunk-rich match: re-ingesting creates duplicates.
    return max(candidates, key=lambda p: p["chunk_count"] or 0)


def resolve_gold_chunks(question: EvalQuestion, chunks: list[dict]) -> set[str]:
    """A chunk is gold if its CONTENT contains any anchor phrase.

    Content-based (not heading-based) because the PyMuPDF heading detector is
    unreliable on real arXiv layouts -- e.g. "Attention Is All You Need"
    ingests as a single 'Document Content' section, and BERT produces numeric
    garbage headings.
    """
    gold: set[str] = set()
    needles = [a.lower() for a in question.anchors]
    for ch in chunks:
        hay = (ch["content"] or "").lower()
        if any(n in hay for n in needles):
            gold.add(ch["chunk_id"])
    return gold


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------
async def _vector_only(db: AsyncSession, question: EvalQuestion, top_k: int) -> list[str]:
    query_vec = await asyncio.to_thread(embed_query, question.query)
    if not query_vec:
        return []
    stmt = text(
        _VECTOR_SELECT.format(
            filter_clause="AND c.paper_id::text = ANY(:paper_ids)"
        )
    )
    rows = (
        await db.execute(
            stmt,
            {
                "query_vec": str(query_vec),
                "threshold": 0.0,
                "top_k": top_k,
                "paper_ids": [question.paper_id],
            },
        )
    ).fetchall()
    return [c.chunk_id for c in _rows_to_results(rows)]


async def _hybrid(db: AsyncSession, question: EvalQuestion, top_k: int) -> list[str]:
    query_vec = await asyncio.to_thread(embed_query, question.query)
    vec_results: list[ChunkResult] = []
    if query_vec:
        stmt = text(_VECTOR_SELECT.format(filter_clause="AND c.paper_id::text = ANY(:paper_ids)"))
        rows = (
            await db.execute(
                stmt,
                {
                    "query_vec": str(query_vec),
                    "threshold": 0.0,
                    "top_k": top_k,
                    "paper_ids": [question.paper_id],
                },
            )
        ).fetchall()
        vec_results = _rows_to_results(rows)

    lex_results: list[ChunkResult] = []
    tsq = _build_tsquery(question.query)
    if tsq:
        lex_stmt = text(_LEXICAL_SELECT.format(filter_clause="AND c.paper_id::text = ANY(:paper_ids)"))
        lex_rows = (
            await db.execute(
                lex_stmt, {"tsq": tsq, "top_k": top_k, "paper_ids": [question.paper_id]}
            )
        ).fetchall()
        lex_results = _rows_to_results(lex_rows)

    if not vec_results and not lex_results:
        return []
    return [c.chunk_id for c in _rrf_fuse(vec_results, lex_results, top_k)]


async def run_config(
    db: AsyncSession,
    question: EvalQuestion,
    config: str,
    top_k: int,
) -> list[str]:
    """Return ranked chunk IDs for one config."""
    if config == "vector_only":
        return await _vector_only(db, question, top_k)

    if config == "hybrid":
        return await _hybrid(db, question, top_k)

    if config in ("hybrid_rerank", "full"):
        candidates = await _hybrid(db, question, _RERANK_CANDIDATES)
        if not candidates:
            return []

        # Rehydrate ChunkResults so the reranker has document text to score.
        by_id = {c["chunk_id"]: c for c in chunks_cache.get(question.qid, [])}
        docs = [
            ChunkResult(
                chunk_id=cid,
                paper_id=question.paper_id or "",
                section_heading=by_id.get(cid, {}).get("heading") or "",
                content=by_id.get(cid, {}).get("content") or "",
                score=0.0,
                metadata={},
            )
            for cid in candidates
        ]
        docs = [d for d in docs if d.content]
        if not docs:
            return candidates[:top_k]

        reranked = await rerank(question.query, docs, top_k=top_k)
        return [c.chunk_id for c in reranked]

    raise ValueError(f"unknown config: {config}")


# ---------------------------------------------------------------------------
# Metrics (pure arithmetic -- no LLM)
# ---------------------------------------------------------------------------
def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = len(gold.intersection(ranked[:k]))
    return hits / len(gold)


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold or not ranked:
        return 0.0
    hits = len(gold.intersection(ranked[:k]))
    return hits / min(k, len(ranked))


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for i, cid in enumerate(ranked):
        if cid in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ranked[:k]) if cid in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(per_question: dict[str, list[str]]) -> dict:
    """per_question: qid -> ranked ids. Returns mean metrics."""
    if not per_question:
        return {}

    recalls = {f"recall@{k}": [] for k in _RECALL_KS}
    precisions = {f"precision@{k}": [] for k in _PRECISION_KS}
    rrs: list[float] = []
    ndcgs: list[float] = []

    gold_map: dict[str, set[str]] = {}
    for qid, ranked in per_question.items():
        gold = GOLD_BY_QID.get(qid, set())
        gold_map[qid] = gold
        if not gold:
            continue
        for k in _RECALL_KS:
            recalls[f"recall@{k}"].append(recall_at_k(ranked, gold, k))
        for k in _PRECISION_KS:
            precisions[f"precision@{k}"].append(precision_at_k(ranked, gold, k))
        rrs.append(reciprocal_rank(ranked, gold))
        ndcgs.append(ndcg_at_k(ranked, gold, 10))

    out = {}
    for name, vals in {**recalls, **precisions}.items():
        out[name] = round(statistics.mean(vals), 4) if vals else 0.0
    out["mrr"] = round(statistics.mean(rrs), 4) if rrs else 0.0
    out["ndcg@10"] = round(statistics.mean(ndcgs), 4) if ndcgs else 0.0
    out["n_questions"] = len(rrs)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieval eval for Paper Pilot")
    ap.add_argument("--top-k", type=int, default=_EVAL_TOP_K)
    ap.add_argument(
        "--configs",
        default="vector_only,hybrid,hybrid_rerank",
        help="comma-separated: vector_only,hybrid,hybrid_rerank,full",
    )
    ap.add_argument("--dump-results", default=None, help="write JSON results here")
    ap.add_argument("--include-followups", action="store_true")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    singles, followups, paper_patterns = load_gold_set()
    questions = list(singles) + (followups if args.include_followups else [])

    factory = _get_session_factory()
    results: dict = {"configs": {}, "per_question": {}}
    skipped: list[EvalQuestion] = []

    async with factory() as db:
        papers = await fetch_papers(db)
        if not papers:
            print("No papers in database. Upload PDFs first.")
            return 1

        print(f"Found {len(papers)} paper(s) in DB:")
        for p in papers:
            print(f"  - {p['title'][:60]:60s} chunks={p['chunk_count']:5d} status={p['status']}")

        # Resolve papers + gold chunks
        chunk_counts: dict[str, int] = {}
        paper_chunks: dict[str, list[dict]] = {}
        for q in questions:
            paper = resolve_paper(papers, q.paper_key, paper_patterns[q.paper_key])
            if paper is None:
                q.status = "no_paper"
                skipped.append(q)
                continue
            if paper["status"] != "ready":
                q.status = "not_ready"
                skipped.append(q)
                continue

            q.paper_id = paper["id"]
            if paper["id"] not in paper_chunks:
                paper_chunks[paper["id"]] = await fetch_chunks(db, paper["id"])
            chunks = paper_chunks[paper["id"]]
            chunks_cache[q.qid] = chunks
            chunk_counts[paper["id"]] = len(chunks)

            gold = resolve_gold_chunks(q, chunks)
            if not gold:
                q.status = "no_gold_match"
                skipped.append(q)
                continue
            q.gold_chunk_ids = gold
            q.status = "ok"
            GOLD_BY_QID[q.qid] = gold

        ready = [q for q in questions if q.status == "ok"]
        print(f"\nResolved {len(ready)}/{len(questions)} questions to gold chunks.")
        if skipped:
            print(f"Skipped {len(skipped)}:")
            for q in skipped:
                print(f"  - {q.qid:12s} ({q.status})  {q.query[:52]}")

        if not ready:
            print("\nNo usable questions. Ingest the PDFs and check section headings above.")
            return 1

        print(f"\nCorpus size per paper: {chunk_counts}")
        print(f"Top-k for eval: {args.top_k}\n")

        # Run each config
        for config in configs:
            print(f"=== config: {config} ===")
            per_q: dict[str, list[str]] = {}
            for i, q in enumerate(ready, 1):
                if config == "full" and q.history and should_rewrite(q.query, q.history):
                    rewritten = await rewrite_query(q.query, q.history)
                    if rewritten != q.query:
                        q.query_for_run = rewritten  # type: ignore[attr-defined]
                run_query = getattr(q, "query_for_run", q.query)

                if config == "full":
                    ranked = await _run_full(db, q, run_query, args.top_k)
                else:
                    ranked = await run_config(db, q, config, args.top_k)

                per_q[q.qid] = ranked
                gold_n = len(q.gold_chunk_ids)
                hit = "HIT " if set(ranked[:5]) & q.gold_chunk_ids else "MISS"
                print(f"  [{i:2d}/{len(ready)}] {q.qid:12s} {hit} got={len(ranked):3d} gold={gold_n:3d} | {q.query[:58]}")

            metrics = aggregate(per_q)
            results["configs"][config] = metrics
            results["per_question"][config] = per_q
            print(f"  -> " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
            print()

    # Report
    print("\n" + "=" * 78)
    print("RETRIEVAL ABLATION RESULTS")
    print("=" * 78)
    metric_order = ["recall@1", "recall@3", "recall@5", "recall@10", "precision@3", "precision@5", "mrr", "ndcg@10"]
    header = f"{'config':16s}" + "".join(f"{m:>13s}" for m in metric_order)
    print(header)
    print("-" * len(header))
    for config in configs:
        m = results["configs"].get(config, {})
        row = f"{config:16s}" + "".join(f"{m.get(k, 0.0):>13.4f}" for k in metric_order)
        print(row)
    print("=" * 78)
    print(f"Questions scored: {results['configs'][configs[0]].get('n_questions', 0)}")
    print("All metrics are deterministic ranking measures -- no LLM judging used.")

    if args.dump_results:
        out = Path(args.dump_results)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {out}")

    return 0


async def _run_full(db: AsyncSession, q: EvalQuestion, query: str, top_k: int) -> list[str]:
    saved = q.query
    q.query = query
    try:
        return await run_config(db, q, "hybrid_rerank", top_k)
    finally:
        q.query = saved


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))