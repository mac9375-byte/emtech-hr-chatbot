"""평가 질문셋의 정답 근거가 생성된 문서(data/raw)에 실제로 있는지 대조한다.

- in_scope/followup: gold_doc 의 gold_article 본문에 모든 keywords 가 있어야 한다.
- no_evidence: absent_keywords 가 어느 문서에도 없어야 한다.
사용: python eval/check_gold.py   (실패가 있으면 종료 코드 1)
"""
import json
import re
import sys
from pathlib import Path

import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
HEAD = re.compile(r"^(제\d+조|Q\d+|양식 \d+)[.\s(]")


def lines_of(path):
    if path.suffix == ".pdf":
        for page in pymupdf.open(path):
            yield from page.get_text().splitlines()
        return
    d = Document(path)
    for el in d.element.body.iterchildren():
        if el.tag.endswith("}p"):
            yield Paragraph(el, d).text
        elif el.tag.endswith("}tbl"):
            for row in Table(el, d).rows:
                yield " ".join(c.text for c in row.cells)


def sections(path):
    """-> {'제14조': 본문, ...}. 목차·쪽번호 줄이 섞여도 키워드 검사에는 영향이 없다."""
    out, key = {}, None
    for line in lines_of(path):
        m = HEAD.match(line.strip())
        if m:
            key = m.group(1)
        if key:
            out[key] = out.get(key, "") + line
    return {k: re.sub(r"\s+", "", v) for k, v in out.items()}


def main():
    docs = {p.stem: sections(p) for p in sorted((ROOT / "data" / "raw").glob("*.*"))}
    alltext = "".join("".join(s.values()) for s in docs.values())
    qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    fails = 0
    for q in qs:
        errs = []
        if "gold_doc" in q:
            # 정답 조항과 gold_also(같은 사실이 실린 다른 조항) 모두 키워드를 갖고 있어야 한다. 키워드는 '|' 로 동의어 허용
            for ref in [f"{q['gold_doc']}:{q['gold_article']}", *q.get("gold_also", [])]:
                doc, art = ref.split(":", 1)
                body = docs.get(doc, {}).get(art)
                if body is None:
                    errs.append(f"조항 없음: {ref}")
                else:
                    errs += [f"키워드 없음({ref}): {k}" for k in q["keywords"]
                             if not any(re.sub(r"\s+", "", alt) in body for alt in k.split("|"))]
        for k in q.get("absent_keywords", []):
            if re.sub(r"\s+", "", k) in alltext:
                errs.append(f"있으면 안 되는 키워드가 문서에 있음: {k}")
        fails += bool(errs)
        print(("FAIL " if errs else "ok   ") + q["id"], "; ".join(errs))
    print(f"\n문서 {len(docs)}개, 조항/문항 {sum(map(len, docs.values()))}개, 질문 {len(qs)}개, 실패 {fails}개")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
