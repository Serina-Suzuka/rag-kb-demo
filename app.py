"""命令行入口：

    python app.py ingest            # 解析 data/ 下的文档，建索引
    python app.py ask "问题"        # 检索 + 回答
    python app.py eval              # 跑评测集，输出 Hit@k / MRR
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rag.evaluate import print_report, run
from rag.ingest import build_index, load_chunks
from rag.llm import answer
from rag.retrieve import Retriever

BASE = Path(__file__).resolve().parent
INDEX = BASE / "index" / "chunks.json"


def _load_retriever() -> Retriever:
    if not INDEX.exists():
        raise SystemExit("还没有索引，先运行：python app.py ingest")
    return Retriever(load_chunks(INDEX))


def cmd_ingest(args: argparse.Namespace) -> None:
    print(f"扫描 {args.data} ...")
    build_index(BASE / args.data, INDEX)


def cmd_ask(args: argparse.Namespace) -> None:
    retriever = _load_retriever()
    t0 = time.perf_counter()
    hits = retriever.search(args.question, topk=args.topk, mode=args.mode)
    cost = (time.perf_counter() - t0) * 1000

    print(f"\n问题：{args.question}")
    print(f"检索：{args.mode} · {len(retriever.chunks)} 个块 · {cost:.0f} ms\n")
    for i, h in enumerate(hits, 1):
        head = h["text"].replace("\n", " ")[:96]
        print(f"  [{i}] p.{h['page']:<3} score={h['score']:.3f}  {head}...")

    if args.show_context:
        print("\n--- 送入模型的上下文 ---")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] p.{h['page']} {h['text']}\n")

    print("\n回答：")
    result = answer(
        args.question, hits, backend=args.backend, model=args.model, stream=args.stream
    )
    if isinstance(result, tuple) and result[0] == "ollama" and args.stream:
        for piece in result[1]:
            print(piece, end="", flush=True)
        print()
    else:
        backend, text = result
        print(text)
        if backend == "mock":
            print("\n提示：装好 Ollama 后加 --backend ollama 即可用本地模型生成。")


def cmd_eval(args: argparse.Namespace) -> None:
    retriever = _load_retriever()
    report = run(retriever, BASE / args.questions, topk=args.topk)
    import json

    questions = json.loads((BASE / args.questions).read_text(encoding="utf-8-sig"))["questions"]
    print_report(report, questions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地文档问答（RAG）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="建索引")
    p.add_argument("--data", default="data", help="文档目录，默认 data/")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="提问")
    p.add_argument("question")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--mode", choices=["bm25", "bm25_bi", "hybrid"], default="bm25")
    p.add_argument("--backend", choices=["auto", "ollama", "mock"], default="auto")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--stream", action="store_true", help="流式输出（需 Ollama）")
    p.add_argument("--show-context", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="检索评测")
    p.add_argument("--questions", default="eval/questions.json")
    p.add_argument("--topk", type=int, default=5)
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())


