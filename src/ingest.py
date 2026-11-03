"""data/raw 의 PDF·docx 를 조/문항 단위 청크로 나눠 data/index 에 색인한다(embeddings.npy + chunks.json).

사용: python -m src.ingest        (매번 색인 파일을 새로 쓴다)
"""
import json
import re
import time

import numpy as np
import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.config import EMBED_BATCH, EMBED_MODEL, INDEX_DIR, OLLAMA_CLIENT, RAW_DIR

ARTICLE = re.compile(r"^(제\d+조(?=\s*\()|Q\d+(?=\.)|양식 \d+(?=\.)|안내(?=\.))")  # 줄바꿈된 본문의 "제11조 제4항" 오인 방지
CHAPTER = re.compile(r"^(제\d+장|부칙)(\s|$)")
FOOTER = re.compile(r"·.*-\s*\d+\s*-$")
EFFECTIVE = re.compile(r"시행\s*(\d{4}-\d{2}-\d{2})")


def pdf_items(path):
    """-> [(kind, text, page)]. kind: chapter/article/body. 표는 행 단위(셀 ' | ')로 복원한다."""
    items, in_toc = [], False
    for pno, page in enumerate(pymupdf.open(path), 1):
        tables = page.find_tables().tables
        parts = []
        for b in page.get_text("blocks", sort=True):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if b[6] != 0 or any(t.bbox[0] <= cx <= t.bbox[2] and t.bbox[1] <= cy <= t.bbox[3] for t in tables):
                continue
            parts.append((b[1], b[4].rstrip("\n").splitlines()))
        for t in tables:
            parts.append((t.bbox[1], [" | ".join((c or "").replace("\n", " ") for c in r) for r in t.extract()]))
        lines = [ln.strip() for _, ls in sorted(parts, key=lambda p: p[0]) for ln in ls if ln.strip()]
        lines = [ln for ln in lines if not FOOTER.search(ln)]
        if lines and lines[0] == "목차":
            in_toc = True
        if in_toc:  # 목차 쪽: 제목 줄만 늘어선 쪽은 건너뛴다
            if all(ln == "목차" or ARTICLE.match(ln) or CHAPTER.match(ln) for ln in lines):
                continue
            in_toc = False
        for ln in lines:
            kind = "article" if ARTICLE.match(ln) else "chapter" if CHAPTER.match(ln) else "body"
            items.append((kind, ln, pno))
    return items


def docx_items(path):
    d = Document(path)
    items = []
    for el in d.element.body.iterchildren():
        if el.tag.endswith("}p"):
            p = Paragraph(el, d)
            if p.text.strip():
                kind = {"Heading 1": "chapter", "Heading 2": "article"}.get(p.style.name, "body")
                items.append((kind, p.text.strip(), 1))
        elif el.tag.endswith("}tbl"):
            for row in Table(el, d).rows:
                items.append(("body", " | ".join(c.text.strip() for c in row.cells), 1))
    return items


def chunks_of(path):
    items = pdf_items(path) if path.suffix == ".pdf" else docx_items(path)
    doc = path.stem
    effective = next((m.group(1) for _, t, _ in items if (m := EFFECTIVE.search(t))), "")
    out, chapter, cur = [], "", None
    for kind, text, page in items:
        if kind == "chapter":
            chapter = text
        elif kind == "article":
            cur = {"doc": doc, "chapter": chapter, "heading": text, "page": page, "body": [],
                   "article": ARTICLE.match(text).group(1)}
            out.append(cur)
        elif cur is not None:
            cur["body"].append(text)
    for c in out:
        c["title"] = c["heading"]
        c["effective"] = effective
        # 검색·답변 근거로 그대로 쓰므로 문서명·장·조 제목을 본문 앞에 붙인다
        c["text"] = f"[{doc} · {c['chapter'] or '본문'}]\n{c['heading']}\n" + "\n".join(c["body"])
        c["kind"] = "history" if "개정 이력" in c["heading"] else "regular"
    return out


def main():
    chunks = []
    for path in sorted(RAW_DIR.glob("*.*")):
        cs = chunks_of(path)
        print(f"{path.name}: 청크 {len(cs)}개")
        chunks += cs
    lens = [len(c["text"]) for c in chunks]
    print(f"합계 {len(chunks)}개, 글자 수 최소/평균/최대 = {min(lens)}/{sum(lens) // len(lens)}/{max(lens)}")

    records = [{"id": f"{c['doc']}:{c['article']}", "text": c["text"],
                "meta": {k: c[k] for k in ("doc", "article", "title", "chapter", "effective", "page", "kind")}}
               for c in chunks]
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"청크 id 중복: {sorted({i for i in ids if ids.count(i) > 1})}")
    t0 = time.time()
    embs = []
    for i in range(0, len(chunks), EMBED_BATCH):
        embs += OLLAMA_CLIENT.embed(model=EMBED_MODEL, input=[c["text"] for c in chunks[i:i + EMBED_BATCH]])["embeddings"]
        print(f"  임베딩 {min(i + EMBED_BATCH, len(chunks))}/{len(chunks)}  ({time.time() - t0:.0f}s)")
    emb = np.asarray(embs, dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)  # 검색 때 내적 = 코사인 유사도
    INDEX_DIR.mkdir(parents=True, exist_ok=True)  # 임베딩이 모두 끝난 뒤에만 써서 실패 시 기존 색인을 남긴다
    np.save(INDEX_DIR / "embeddings.npy", emb)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"색인 완료: {len(records)}개, {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
