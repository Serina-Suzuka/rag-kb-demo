"""把 PDF / Markdown / 文本切成带页码的小块，供检索使用。"""

from __future__ import annotations

import json
import re
from pathlib import Path

SENT_END = "。！？；!?;\n"
CHUNK_SIZE = 360
OVERLAP_SENTENCES = 1


def _clean(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)  # 去掉单独成行的页码
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def read_pdf(path: Path) -> list[tuple[int, str]]:
    """返回 [(页码, 文本)]，页码从 1 开始。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("缺少依赖：pip install pypdf") from exc

    pages = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, 1):
        pages.append((i, _clean(page.extract_text() or "")))
    return pages


def read_plain(path: Path) -> list[tuple[int, str]]:
    return [(1, _clean(path.read_text(encoding="utf-8", errors="ignore")))]


def split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句，保留标点。"""
    out, buf = [], ""
    for ch in text:
        buf += ch
        if ch in SENT_END:
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def split_long(sentence: str, size: int) -> list[str]:
    if len(sentence) <= size:
        return [sentence]
    return [sentence[i : i + size] for i in range(0, len(sentence), size)]


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / len(text)


def chunk_page(text: str, page: int, doc: str, size: int = CHUNK_SIZE) -> list[dict]:
    """按句子拼接成约 size 字的块，块之间重叠 1 句，避免答案被切断。"""
    sentences: list[str] = []
    for sent in split_sentences(text):
        sentences.extend(split_long(sent, size))

    chunks, buf = [], []
    length = 0
    for sent in sentences:
        if length + len(sent) > size and buf:
            chunks.append(buf)
            buf = buf[-OVERLAP_SENTENCES:]
            length = sum(len(s) for s in buf)
        buf.append(sent)
        length += len(sent)
    if buf:
        chunks.append(buf)

    results = []
    for i, group in enumerate(chunks):
        body = "".join(group).strip()
        if len(body) < 20:  # 丢掉目录行、页码之类的碎片
            continue
        if _cjk_ratio(body) < 0.25:  # 丢掉代码附录，中文占比太低，检索时只会添乱
            continue
        results.append(
            {"doc": doc, "page": page, "chunk": i, "text": body}
        )
    return results


def build_index(data_dir: Path, out_path: Path) -> list[dict]:
    files = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".md", ".txt"}
    )
    if not files:
        raise SystemExit(f"{data_dir} 里没有可索引的文件（pdf/md/txt）")

    all_chunks: list[dict] = []
    for path in files:
        pages = read_pdf(path) if path.suffix.lower() == ".pdf" else read_plain(path)
        for page, text in pages:
            if not text:
                continue
            all_chunks.extend(chunk_page(text, page, path.name))
        print(f"  {path.name}: {len(pages)} 页")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"索引完成：{len(all_chunks)} 个块 -> {out_path}")
    return all_chunks


def load_chunks(index_path: Path) -> list[dict]:
    return json.loads(index_path.read_text(encoding="utf-8"))
