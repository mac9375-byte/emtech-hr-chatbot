"""청킹 검증: 원본(data/src)의 조/문항 목록과 본문이 data/raw 에서 추출한 청크와 일치하는지 본다.
사용: python -m unittest discover -s tests -v
"""
import re
import unittest
from pathlib import Path

from src.config import RAW_DIR
from src.ingest import chunks_of

SRC = Path(__file__).resolve().parent.parent / "data" / "src"


def norm(s):
    """공백과 표 구분자만 없앤 뒤 비교한다. 줄바꿈 위치나 ' | ' 표기 차이는 무시."""
    return re.sub(r"[\s|]+", "", s)


def source_articles(txt):
    """원본 txt -> (raw 파일 경로, effective, [(제목, 본문)])."""
    text = txt.read_text(encoding="utf-8")
    title = re.search(r"^@title (.+)$", text, re.M).group(1).replace(" ", "")
    fmt = re.search(r"^@format (.+)$", text, re.M).group(1).strip()
    effective = re.search(r"^@effective (.+)$", text, re.M).group(1).strip()
    arts, cur = [], None
    for line in text.splitlines():
        if line.startswith("### "):
            cur = (line[4:].strip(), [])
            arts.append(cur)
        elif line.startswith(("## ", "@")) or not line.strip():
            continue
        elif cur:
            cur[1].append(line)
    return RAW_DIR / f"{title}.{fmt}", effective, [(h, "\n".join(b)) for h, b in arts]


class IngestTest(unittest.TestCase):
    def test_chunks_match_source(self):
        srcs = sorted(SRC.glob("*.txt"))
        self.assertEqual(len(srcs), 5)
        for txt in srcs:
            raw, effective, arts = source_articles(txt)
            with self.subTest(doc=raw.name):
                chunks = chunks_of(raw)
                self.assertEqual([c["heading"] for c in chunks], [h for h, _ in arts], "조/문항 목록이 원본과 다름")
                for c, (h, body) in zip(chunks, arts):
                    self.assertEqual(norm("\n".join(c["body"])), norm(body), f"본문이 원본과 다름(잘림/누락/혼입): {h}")
                    self.assertEqual(c["effective"], effective)

    def test_table_rows_kept_together(self):
        """경조휴가 표가 다음 쪽에 걸려도 행 단위로 복원돼야 한다."""
        chunk = next(c for c in chunks_of(RAW_DIR / "복무규정.pdf") if c["article"] == "제19조")
        self.assertIn("본인 결혼 | 5일", chunk["text"])
        self.assertIn("조부모, 외조부모, 형제자매 사망 | 3일", chunk["text"])

    def test_history_chunks_flagged(self):
        hist = [c for f in RAW_DIR.glob("*.pdf") for c in chunks_of(f) if c["kind"] == "history"]
        self.assertEqual(len(hist), 3)  # 취업규칙, 복무규정, 복리후생규정


if __name__ == "__main__":
    unittest.main()
