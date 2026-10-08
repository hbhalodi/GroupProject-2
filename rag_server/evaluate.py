"""Evaluate retrieval: vector-only baseline vs. Fusion Retrieval.

Owner: Member C (results go in the report's RAG section)

Each question in eval_questions.json lists the doc file(s) that answer it.
A question counts as a "hit" when one of the top-k chunks comes from an
expected file. We report:
    Hit@k  share of questions with a correct chunk in the top k
    MRR    mean reciprocal rank of the first correct chunk (1.0 = always first)

Usage:
    python -m rag_server.evaluate                 # both retrievers, k=5
    python -m rag_server.evaluate --alphas 0.3 0.5 0.7   # also sweep fusion weights
    python -m rag_server.evaluate --show-misses   # print questions each method missed

TODO(Member C): review/extend the questions (aim for 25+), including some
with exact identifiers and some purely conceptual, so the report can explain
where fusion helps and where it doesn't.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_server import settings
from rag_server.retriever import FusionRetriever, VectorRetriever

QUESTIONS = Path(__file__).with_name("eval_questions.json")


def first_correct_rank(hits, expected: list[str]) -> int | None:
    for rank, hit in enumerate(hits, 1):
        source = hit.metadata.get("source", "")
        if any(source == e or source.startswith(e) for e in expected):
            return rank
    return None


def evaluate(retriever, questions: list[dict], k: int) -> dict:
    ranks = []
    for q in questions:
        ranks.append(first_correct_rank(retriever.search(q["question"], k), q["expected"]))
    hits = sum(r is not None for r in ranks)
    return {
        "hit_at_k": hits / len(questions),
        "mrr": sum(1 / r for r in ranks if r) / len(questions),
        "ranks": ranks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare baseline vs fusion retrieval")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alphas", type=float, nargs="*", default=[settings.FUSION_ALPHA])
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument("--out", type=Path, help="save results as JSON")
    args = parser.parse_args(argv)

    from rag_server.store import VectorStore

    store = VectorStore()
    if store.count() == 0:
        print("Index is empty. Run `python -m rag_server.ingest` first.", file=sys.stderr)
        return 1
    questions = json.loads(args.questions.read_text(encoding="utf-8"))

    retrievers = [("vector (baseline)", VectorRetriever(store))]
    for alpha in args.alphas:
        retrievers.append((f"fusion alpha={alpha}", FusionRetriever(store, alpha, settings.CANDIDATE_POOL)))

    results = {}
    print(f"\n{len(questions)} questions, k={args.k}\n")
    print(f"| {'method':<22} | Hit@{args.k:<3} | MRR   |")
    print(f"|{'-' * 24}|{'-' * 8}|{'-' * 7}|")
    for label, retriever in retrievers:
        r = evaluate(retriever, questions, args.k)
        results[label] = r
        print(f"| {label:<22} | {r['hit_at_k']:.2f}   | {r['mrr']:.2f}  |")

    if args.show_misses:
        for label, r in results.items():
            missed = [q["question"] for q, rank in zip(questions, r["ranks"]) if rank is None]
            print(f"\nMissed by {label}: {len(missed)}")
            for m in missed:
                print(f"  - {m}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
