"""检索评测：Hit@k 与 MRR，对比 unigram / bigram / RRF 融合三种方案。"""

from __future__ import annotations

import json
from pathlib import Path

from .retrieve import Retriever

MODES = [("bm25", "unigram BM25"), ("bm25_bi", "bigram BM25"), ("hybrid", "RRF 融合")]


def first_hit_rank(hits: list[dict], expect: list[str]) -> int | None:
    """返回第一个同时包含全部关键词的块的名次（从 1 开始）。"""
    for rank, h in enumerate(hits, 1):
        if all(word in h["text"] for word in expect):
            return rank
    return None


def run(retriever: Retriever, questions_path: Path, topk: int = 5) -> dict:
    questions = json.loads(questions_path.read_text(encoding="utf-8-sig"))["questions"]

    report = {}
    for mode, label in MODES:
        ranks = []
        for q in questions:
            hits = retriever.search(q["q"], topk=topk, mode=mode)
            ranks.append(first_hit_rank(hits, q["expect"]))
        n = len(ranks)
        report[mode] = {
            "label": label,
            "hit@1": sum(1 for r in ranks if r and r <= 1) / n,
            "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
            "hit@5": sum(1 for r in ranks if r and r <= 5) / n,
            "mrr": sum(1.0 / r for r in ranks if r) / n,
            "ranks": ranks,
        }
    return report


def print_report(report: dict, questions: list[dict]) -> None:
    modes = list(report)
    print(f"\n{'问题':<30}" + "".join(f"{report[m]['label']:>14}" for m in modes))
    print("-" * (30 + 14 * len(modes)))
    for i, q in enumerate(questions):
        cells = "".join(
            f"{(report[m]['ranks'][i] and '#' + str(report[m]['ranks'][i])) or 'miss':>14}"
            for m in modes
        )
        print(f"{q['q'][:26]:<30}{cells}")

    print("-" * (30 + 14 * len(modes)))
    for m in modes:
        r = report[m]
        print(
            f"{r['label']:<14} Hit@1 {r['hit@1']:.0%}   Hit@3 {r['hit@3']:.0%}   "
            f"Hit@5 {r['hit@5']:.0%}   MRR {r['mrr']:.3f}"
        )

