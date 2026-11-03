"""키워드만 맞고 첨부 서류를 잘못 나열한 답변도 실패해야 한다."""
import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from eval.eval_generation import judge, rejudge, run
from eval.eval_retrieval import main as retrieval_main, query_of


class EvaluationTest(unittest.TestCase):
    def test_missing_policy_refusal_synonym_passes_without_sources(self):
        from src.generate import REFUSE
        answer = ("규정에서 주 4일 근무제에 대한 내용이 명시되어 있지 않습니다. "
                  "따라서 주 4일 근무제를 시행하는지 여부는 확인할 수 없습니다. "
                  "HR팀에 문의해 주세요. [근거1]")
        ok, errors, shown, sources = judge({"id": "N03", "type": "no_evidence"}, self.snapshot(), answer)
        self.assertTrue(ok)
        self.assertEqual(errors, [])
        self.assertEqual(shown, REFUSE)
        self.assertEqual(sources, [])

    def test_retrieval_topic_change_keeps_standalone_question(self):
        q = {"question": "월급은 언제 나오나요?", "history": ["연차는 며칠인가요?"]}
        self.assertEqual(query_of(q), [q["question"], "연차는 며칠인가요? 월급은 언제 나오나요?"])

    def test_retrieval_uses_only_immediate_previous_question(self):
        q = {"question": "그건 며칠인가요?", "history": ["급여는 언제?", "리프레시 휴가는?"]}
        self.assertEqual(query_of(q), [q["question"], "리프레시 휴가는? 그건 며칠인가요?"])

    def test_retrieval_summary_counts_and_distance_match_runtime(self):
        qs = [self.question(), {**self.question(), "id": "F01", "type": "followup",
                               "question": "그건 며칠인가요?", "history": ["연차는?"]}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "eval").mkdir()
            (root / "eval" / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in qs), encoding="utf-8")
            output = io.StringIO()
            with patch("eval.eval_retrieval.ROOT", root), patch("eval.eval_retrieval.sys.argv", ["eval_retrieval"]), \
                    patch("eval.eval_retrieval._index", return_value=(None, None, [], [], [])), \
                    patch("eval.eval_retrieval.search", return_value=[{"id": "취업규칙:제8조"}]), \
                    patch("eval.eval_retrieval.top_distance", return_value=0.2) as distance, redirect_stdout(output):
                retrieval_main()
            self.assertIn("규정 내 1 + 후속 1", output.getvalue())
            self.assertEqual([call.args[0] for call in distance.call_args_list], [q["question"] for q in qs])

    def snapshot(self):
        return {"route": "answer", "question": "수습기간은?", "prev": None,
                "chunks": [{"id": "취업규칙:제8조", "text": "수습기간은 3개월이다.",
                            "meta": {"doc": "취업규칙", "title": "제8조", "kind": "regular"}}]}

    def question(self):
        return {"id": "Q01", "type": "in_scope", "question": "수습기간은?",
                "gold_doc": "취업규칙", "gold_article": "제8조", "keywords": ["3개월"]}

    def test_run_saves_generation_context(self):
        prep = self.snapshot()
        with tempfile.TemporaryDirectory() as tmp, patch("eval.eval_generation.RESULTS", Path(tmp)), \
                patch("eval.eval_generation.prepare", return_value=prep), \
                patch("eval.eval_generation.generate", return_value=iter(["3개월입니다. [근거1]"])):
            rows = run("test", [self.question()])
            saved = json.loads((Path(tmp) / "test.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(rows[0]["prep"], prep)
        self.assertEqual(saved["prep"], prep)

    def test_rejudge_uses_saved_context_without_search(self):
        row = {"id": "Q01", "type": "in_scope", "route": "answer", "answer": "3개월입니다.",
               "raw_answer": "3개월입니다. [근거1]", "prep": self.snapshot(),
               "t_first_token": 1, "t_prepare": 1, "t_total": 2}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            with patch("eval.eval_generation.prepare") as prepare, patch("eval.eval_generation.summarize") as summarize:
                rejudge(path, [self.question()])
            prepare.assert_not_called()
            result = summarize.call_args.args[1][0]
        self.assertTrue(result["ok"])
        self.assertEqual(result["sources"], ["취업규칙:제8조"])

    def test_legacy_numbered_citations_require_regeneration(self):
        row = {"id": "Q01", "answer": "3개월입니다. [근거1]"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            with patch("eval.eval_generation.prepare") as prepare:
                with self.assertRaisesRegex(ValueError, "재생성"):
                    rejudge(path, [self.question()])
            prepare.assert_not_called()

    def test_form_fields_are_not_attachment_documents(self):
        question = {"type": "in_scope", "gold_doc": "신청양식모음", "gold_article": "양식 3",
                    "keywords": ["가족관계증명서"], "forbidden_keywords": ["성명", "사번", "분할 사용 여부"]}
        answer = "성명, 사번, 분할 사용 여부와 가족관계증명서를 첨부해야 합니다."
        with patch("eval.eval_generation.finalize", return_value=(answer, [{"id": "신청양식모음:양식 3"}])):
            ok, errors, _, _ = judge(question, {"route": "answer"}, answer)
        self.assertFalse(ok)
        self.assertTrue(any("부정확한 항목" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
