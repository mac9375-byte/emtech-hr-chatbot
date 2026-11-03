"""UI 회귀 검증. 색인/모델은 mock, 로그는 임시 디렉터리만 사용한다."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

from src import config, generate, retrieve


class AppTestCase(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "events.jsonl"
        self.enterContext(patch.object(config, "LOG_PATH", self.log, create=True))
        self.index = self.enterContext(patch.object(retrieve, "_index"))
        self.warm = self.enterContext(patch.object(generate, "warm_up"))
        self.at =AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=10)

    def test_invalid_draft_is_never_rendered_before_finalization(self):
        chunk = {"id": "취업규칙:제8조", "text": "수습기간은 3개월이다.",
                 "meta": {"doc": "취업규칙", "title": "제8조", "kind": "regular"}}
        self.enterContext(patch.object(generate, "prepare", return_value={"route": "answer", "chunks": [chunk]}))
        self.enterContext(patch.object(generate, "generate", return_value=iter([
            "전 직원에게 매월 100만원을 지급합니다.", " [근거99]"])))
        rendered = []
        original = DeltaGenerator.markdown

        def trace_markdown(element, body, *args, **kwargs):
            rendered.append(str(body))
            return original(element, body, *args, **kwargs)

        self.enterContext(patch.object(DeltaGenerator, "markdown", trace_markdown))
        self.at.run()
        self.at.chat_input[0].set_value("복지 정책 알려줘").run()
        self.assertEqual(len(self.at.exception), 0)
        self.assertFalse(any("100만원" in body for body in rendered))
        self.assertEqual(self.at.chat_message[1].markdown[0].value, generate.REFUSE)
        self.assertEqual(len(self.at.expander), 0)

    def test_startup_and_streaming_sources_feedback_without_content_logging(self):
        chunk = {"id": "취업규칙:제8조", "text": "수습기간은 3개월이다.",
                 "meta": {"doc": "취업규칙", "title": "제8조 (수습기간)", "kind": "regular"}}
        prep = {"route": "answer", "distance": 0.2, "chunks": [chunk],
                "sources": [{"id": chunk["id"], "label": "취업규칙 제8조", "text": chunk["text"]}]}
        self.enterContext(patch.object(generate, "prepare", return_value=prep))
        stream = self.enterContext(patch.object(generate, "generate", return_value=iter([
            "수습기간은 ", "3개월입니다. [근거1]"])))
        self.at.run()
        self.assertEqual(len(self.at.exception), 0)
        self.at.chat_input[0].set_value("홍길동의 수습기간은?").run()
        self.assertEqual(len(self.at.exception), 0)
        self.assertEqual(self.at.chat_message[1].markdown[0].value, "수습기간은 3개월입니다.")
        self.assertIn("취업규칙 제8조", self.at.expander[0].label)
        stream.assert_called_once()
        feedback = self.at.get("feedback")[0]
        feedback.set_value(1).run()
        events = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(events[-1]["value"], "up")
        self.assertNotIn("홍길동", self.log.read_text(encoding="utf-8"))
        self.assertNotIn("question", events[0])
        self.assertNotIn("answer", events[0])

    def test_prepare_connection_failure_has_friendly_message(self):
        self.enterContext(patch.object(generate, "prepare", side_effect=ConnectionError("secret internal detail")))
        self.at.run()
        self.at.chat_input[0].set_value("연차 규정 알려줘").run()
        self.assertEqual(len(self.at.exception), 0)
        content = self.at.chat_message[1].markdown[0].value
        self.assertIn("Ollama", content)
        self.assertNotIn("secret", content)

    def test_stream_failure_does_not_save_partial_answer_or_sources(self):
        def failed_stream(prep):
            yield "미완성 답변"
            raise ConnectionError("secret internal detail")
        self.enterContext(patch.object(generate, "prepare", return_value={"route": "answer"}))
        self.enterContext(patch.object(generate, "generate", side_effect=failed_stream))
        self.at.run()
        self.at.chat_input[0].set_value("수습기간 알려줘").run()
        self.assertEqual(len(self.at.exception), 0)
        self.assertNotIn("미완성", self.at.chat_message[1].markdown[0].value)
        self.assertEqual(len(self.at.expander), 0)

    def test_refusal_skips_generation_and_reset_clears_history(self):
        prepare = self.enterContext(patch.object(generate, "prepare", return_value={
            "route": "refer_hr", "message": "HR팀에 문의해 주세요."}))
        stream = self.enterContext(patch.object(generate, "generate"))
        self.at.run()
        self.at.chat_input[0].set_value("내 남은 연차 알려줘").run()
        self.at.chat_input[0].set_value("그럼 급여는?").run()
        self.assertEqual(len(self.at.exception), 0)
        prepare.assert_called_with("그럼 급여는?", ["내 남은 연차 알려줘"])
        stream.assert_not_called()
        self.assertEqual(len(self.at.expander), 0)
        self.at.button[0].click().run()
        self.assertEqual(len(self.at.chat_message), 0)

    def test_missing_index_has_friendly_startup_message(self):
        self.index.side_effect = FileNotFoundError("secret database details")
        self.at.run()
        self.assertEqual(len(self.at.exception), 0)
        self.assertIn("검색 색인", self.at.error[0].value)
        self.assertNotIn("secret", self.at.error[0].value)
        self.assertEqual(len(self.at.chat_input), 0)

    def test_empty_index_has_friendly_startup_message(self):
        self.index.side_effect = ValueError("secret empty corpus details")
        self.at.run()
        self.assertEqual(len(self.at.exception), 0)
        self.assertIn("검색 색인", self.at.error[0].value)
        self.assertNotIn("secret", self.at.error[0].value)
        self.assertEqual(len(self.at.chat_input), 0)

    def test_startup_warms_models(self):
        self.at.run()
        self.warm.assert_called_once()

    def test_warm_up_failure_does_not_block_startup(self):
        self.warm.side_effect = ConnectionError("secret internal detail")
        self.at.run()
        self.assertEqual(len(self.at.exception), 0)
        self.assertEqual(len(self.at.error), 0)
        self.assertEqual(len(self.at.chat_input), 1)


if __name__ == "__main__":
    unittest.main()
