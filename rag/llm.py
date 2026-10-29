"""回答生成：默认接本地 Ollama；连不上时退回离线抽取式回答，保证链路可跑。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
SYSTEM_PROMPT = (
    "你是文档问答助手。只依据给出的资料回答，不要编造；"
    "资料里没有的信息就直说“资料中没有提到”。"
    "回答尽量控制在 120 字内，并在句末用 [页码] 标注依据。"
)


def build_prompt(question: str, hits: list[dict]) -> list[dict]:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}]（第 {h['page']} 页）{h['text']}")
    context = "\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
    ]


def ollama_chat(messages: list[dict], model: str, stream: bool = False, timeout: int = 180):
    payload = json.dumps(
        {"model": model, "messages": messages, "stream": stream,
         "options": {"temperature": 0.2}}
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )

    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "").strip()

    def gen():
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if not line.strip():
                    continue
                obj = json.loads(line.decode("utf-8"))
                piece = obj.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if obj.get("done"):
                    break

    return gen()


def _best_sentence(text: str, question: str) -> str:
    """离线模式下挑出与问题重合度最高的句子，作为摘录式回答。"""
    import re

    q = set(question)
    best, best_score = "", -1
    for sent in re.split(r"(?<=[。！？；])", text):
        sent = sent.strip()
        if len(sent) < 6:
            continue
        score = len(q & set(sent))
        if score > best_score:
            best, best_score = sent, score
    return best or text[:80]


def mock_answer(question: str, hits: list[dict]) -> str:
    """离线模式：抽取与问题最相关的原句，验证“检索→拼上下文”链路。"""
    if not hits:
        return "没有检索到相关内容。"
    lines = [f"（离线模式，未调用大模型）与「{question}」最相关的原文："]
    for h in hits[:2]:
        sentence = _best_sentence(h["text"], question)
        lines.append(f"- 第 {h['page']} 页：{sentence}")
    return "\n".join(lines)


def answer(question: str, hits: list[dict], backend: str = "auto",
           model: str = "qwen2.5:7b", stream: bool = False):
    """backend: auto / ollama / mock。auto 先试 Ollama，失败则降级。"""
    messages = build_prompt(question, hits)

    if backend in {"auto", "ollama"}:
        try:
            out = ollama_chat(messages, model=model, stream=stream)
            return out if stream else ("ollama", out)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            if backend == "ollama":
                raise SystemExit(
                    f"连不上 Ollama（{exc}）。先运行 `ollama serve` 并 `ollama pull {model}`。"
                ) from exc
            return ("mock", mock_answer(question, hits) + "\n（未检测到 Ollama，已自动降级）")

    return ("mock", mock_answer(question, hits))
