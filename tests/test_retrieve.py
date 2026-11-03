"""검색 회귀 테스트. 색인(data/index)과 Ollama(bge-m3)가 필요하다.
사용: python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.retrieve import _index, search, tokens


class TokensTest(unittest.TestCase):
    def test_empty_index_is_reported_before_bm25(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "chunks.json").write_text("[]", encoding="utf-8")
            with patch("src.retrieve.INDEX_DIR", Path(d)):
                with self.assertRaisesRegex(ValueError, "空|빈|비어"):
                    _index.__wrapped__()

    def test_missing_index_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as d, patch("src.retrieve.INDEX_DIR", Path(d)):
            with self.assertRaises(FileNotFoundError):
                _index.__wrapped__()

    def test_particle_variants_share_bigrams(self):
        self.assertEqual(tokens("연차를"), ["연차", "차를"])
        self.assertIn("연차", tokens("연차는"))

    def test_short_and_mixed_words(self):
        self.assertEqual(tokens("5일 EM-HR"), ["5일", "EM", "HR"])


class SearchTest(unittest.TestCase):
    def ids(self, q, k=5):
        return [r["id"] for r in search(q, k=k)]

    def test_natural_question_finds_article(self):
        self.assertIn("복무규정:제26조", self.ids("리프레시 휴가는 몇 년마다 며칠을 받나요?"))

    def test_exact_term_found_by_bm25_side(self):
        """벡터 단독은 '시말서'를 6위로 놓쳤다. 하이브리드는 상위 3위 안이어야 한다."""
        self.assertIn("취업규칙:제36조", self.ids("시말서", k=3))

    def test_result_shape(self):
        r = search("월급은 며칠에 나오나요?")[0]
        self.assertEqual(set(r), {"id", "text", "meta", "distance", "rrf"})
        self.assertTrue(0 <= r["distance"] < 2 if r["distance"] is not None else True)


if __name__ == "__main__":
    unittest.main()
