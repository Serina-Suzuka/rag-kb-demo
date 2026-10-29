"""检索层：字符级 BM25 + 可选的 RRF 融合。

中文没有分词时，字符 unigram 比 bigram 更稳：短问句里 bigram 太稀疏，
一个字的差异就会导致匹配不上。评测结果见 README。
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

CJK = r"[\u4e00-\u9fff]"
TOKEN_RE = re.compile(rf"{CJK}+|[a-zA-Z][a-zA-Z0-9_+#./-]*|\d+(?:\.\d+)?")


def _segments(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def unigrams(text: str) -> list[str]:
    """中文按单字切，英文/数字按整词切。"""
    out: list[str] = []
    for seg in _segments(text):
        if re.fullmatch(rf"{CJK}+", seg):
            out.extend(seg)
        else:
            out.append(seg)
    return out


def bigrams(text: str) -> list[str]:
    """中文按字 bigram 切，英文/数字按整词切（作为对照方案）。"""
    out: list[str] = []
    for seg in _segments(text):
        if re.fullmatch(rf"{CJK}+", seg):
            out.extend([seg] if len(seg) == 1 else [seg[i : i + 2] for i in range(len(seg) - 1)])
        else:
            out.append(seg)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 2.0, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=float)
        self.avg_len = float(self.doc_len.mean()) if self.n else 1.0
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {
            t: math.log((self.n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()
        }

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=float)
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


class Tfidf:
    """bigram TF-IDF，只用于融合方案的对照实验。"""

    def __init__(self, docs: list[list[str]]):
        self.n = len(docs)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log((self.n + 1) / (f + 1)) + 1.0 for t, f in df.items()}
        self.vectors = [self._vec(Counter(d)) for d in docs]
        self.norms = np.array(
            [math.sqrt(sum(w * w for w in v.values())) or 1.0 for v in self.vectors]
        )

    def _vec(self, tf: Counter) -> dict[str, float]:
        return {
            t: (1 + math.log(f)) * self.idf.get(t, 1.0)
            for t, f in tf.items()
            if t in self.idf
        }

    def scores(self, query: list[str]) -> np.ndarray:
        q = self._vec(Counter(query))
        qnorm = math.sqrt(sum(w * w for w in q.values())) or 1.0
        out = np.zeros(self.n, dtype=float)
        for i, v in enumerate(self.vectors):
            if not v:
                continue
            dot = sum(w * v.get(t, 0.0) for t, w in q.items())
            out[i] = dot / (qnorm * self.norms[i])
        return out


def rrf(rank_lists: list[list[int]], k: int = 10) -> dict[int, float]:
    """Reciprocal Rank Fusion：只融合名次，避免两路分数量纲不一致。"""
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks, 1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25([unigrams(c["text"]) for c in chunks])
        self.bm25_bi = BM25([bigrams(c["text"]) for c in chunks])
        self.tfidf_bi = Tfidf([bigrams(c["text"]) for c in chunks])

    def search(self, query: str, topk: int = 5, mode: str = "bm25") -> list[dict]:
        """mode: bm25（默认，字 unigram）/ bm25_bi（对照）/ hybrid（RRF 融合）。"""
        if mode == "bm25_bi":
            order = list(np.argsort(-self.bm25_bi.scores(bigrams(query))))
        elif mode == "hybrid":
            uni_rank = list(np.argsort(-self.bm25.scores(unigrams(query))))
            tf_rank = list(np.argsort(-self.tfidf_bi.scores(bigrams(query))))
            fused = rrf([uni_rank, tf_rank])
            order = [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])]
        else:
            order = list(np.argsort(-self.bm25.scores(unigrams(query))))

        results = []
        for idx in order[:topk]:
            item = dict(self.chunks[idx])
            item["score"] = float(self.bm25.scores(unigrams(query))[idx])
            results.append(item)
        return results
