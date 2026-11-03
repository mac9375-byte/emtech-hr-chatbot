"""경로 판정 규칙(개인·법률)과 근거 수집 테스트. build_context 는 색인과 Ollama(bge-m3)가 필요하다."""
import unittest
from unittest.mock import patch

from src.config import EMBED_MODEL, GEN_MODEL, KEEP_ALIVE
from src.generate import LEGAL, PERSONAL, REFUSE, bad_citations, build_context, cited_ids, finalize, generate, prepare, warm_up
from src.retrieve import embed, faq_refs


class RouteRulesTest(unittest.TestCase):
    def test_personal_salary_amount_and_named_balance_skip_search(self):
        for q in ["내 급여가 얼마인지 알려줘", "저의 월급이 얼마나 되는지 알려줘", "김철수의 남은 연차 조회해줘"]:
            with self.subTest(question=q), patch("src.generate.top_distance", return_value=0.6) as distance:
                self.assertEqual(prepare(q)["route"], "refer_hr")
                distance.assert_not_called()

    def test_salary_policy_amount_is_not_personal_lookup(self):
        for q in ["급여가 얼마인지 어떻게 산정하나요?", "내 급여 산정 규정 알려줘", "직원의 연차 조회 절차는 무엇인가요?"]:
            self.assertFalse(PERSONAL.search(q), q)

    def test_personal_data_questions(self):
        for q in ["제 연차가 며칠 남았는지 알려주세요.", "내 월급 얼마 받는지 확인해줘", "제 남은 휴가 알려줘"]:
            self.assertTrue(PERSONAL.search(q), q)

    def test_policy_questions_with_first_person_not_personal(self):
        for q in ["제 연차는 며칠인가요?", "제가 입사 1년차인데 연차는 며칠이에요?", "연차는 며칠 남았을 때 촉진하나요?",
                  "7일 이내 연차를 쓰고 남은 일수는 어떻게 되나요?",  # '이내 연차'의 '내'를 1인칭으로 오인하지 않기
                  "입사 1년 미만 직원은 연차를 최대 며칠까지 받을 수 있나요?"]:
            self.assertFalse(PERSONAL.search(q), q)

    def test_legal_judgement(self):
        for q in ["이게 정당한 해고인지 판단해 주세요.", "부당해고로 소송할 수 있나요?", "이거 불법 아닌가요?"]:
            self.assertTrue(LEGAL.search(q), q)

    def test_informational_dismissal_question_not_legal(self):
        self.assertFalse(LEGAL.search("해고하려면 며칠 전에 예고해야 하나요?"))


class FaqRefsTest(unittest.TestCase):
    def test_multiple_docs_and_articles(self):
        t = "답변입니다. (참조: 복무규정 제16조, 제17조, 복리후생 규정 제7조)"
        self.assertEqual(faq_refs(t), {("복무규정", "제16조"), ("복무규정", "제17조"), ("복리후생규정", "제7조")})

    def test_no_ref(self):
        self.assertEqual(faq_refs("참조 없음"), set())


class CitationTest(unittest.TestCase):
    def test_each_named_document_requires_an_article(self):
        chunks = [{"id": "취업규칙:제8조"}]
        for citation in ["취업규칙 제8조, 복무규정", "취업규칙, 복무규정 제8조", "취업규칙 제8조, 복무규정 제999항"]:
            with self.subTest(citation=citation):
                self.assertTrue(bad_citations(f"답. (근거: {citation})", chunks))

    def test_paragraph_requires_preceding_article(self):
        chunks = [{"id": "취업규칙:제8조"}]
        self.assertTrue(bad_citations("답. (근거: 제999항, 취업규칙 제8조)", chunks))
        self.assertFalse(bad_citations("답. (근거: 취업규칙 제8조 제2항)", chunks))

    def test_article_before_document_is_invalid_even_with_valid_citation(self):
        chunks = [{"id": "취업규칙:제8조"}]
        self.assertTrue(bad_citations("답. (근거: 제99조, 취업규칙 제8조)", chunks))

    def test_malformed_numbered_citation_across_newline_is_invalid(self):
        chunks = [{"id": "취업규칙:제8조"}]
        self.assertTrue(bad_citations("답. [근거1][근거X\n]", chunks))

    def test_malformed_citation_mixed_with_valid_citation_is_flagged(self):
        chunks = [{"id": "취업규칙:제8조"}]
        for text in ["답. [근거1][근거X]", "답. [근거1][근거2", "답. (근거: 취업규칙 제8조) (근거: 존재하지않는 문서)",
                     "답. (근거: 취업규칙 제8조, 존재하지않는 문서)"]:
            self.assertTrue(bad_citations(text, chunks), text)

    def test_document_name_does_not_carry_between_citations(self):
        self.assertEqual(cited_ids("(근거: 취업규칙 제8조) (근거: 제9조)"), {"취업규칙:제8조"})

    def test_cited_ids_multiple_docs_and_paragraph_suffix(self):
        t = "답변입니다. (근거: 복무규정 제16조, 제17조, 복리후생 규정 제7조 제2항)"
        self.assertEqual(cited_ids(t), {"복무규정:제16조", "복무규정:제17조", "복리후생규정:제7조"})

    def test_form_citation_and_fullwidth_parens(self):
        self.assertEqual(cited_ids("좋습니다.（근거: 신청양식모음 양식 3）"), {"신청양식모음:양식 3"})

    def test_no_citation(self):
        self.assertEqual(cited_ids("규정에서 찾을 수 없습니다."), set())

    def test_wrong_document_name_is_flagged(self):
        """실제 사례(Q14): 근거는 취업규칙 제8조인데 모델이 '복무규정 제8조'라고 씀."""
        chunks = [{"id": "취업규칙:제8조"}, {"id": "취업규칙:제9조"}]
        self.assertEqual(bad_citations("100% 지급됩니다. (근거: 복무규정 제8조)", chunks), ["복무규정:제8조"])
        self.assertEqual(bad_citations("100% 지급됩니다. (근거: 취업규칙 제8조)", chunks), [])


class FinalizeTest(unittest.TestCase):
    def test_absent_policy_refusal_discards_following_claim(self):
        from src.generate import REFUSE
        for prefix in ["규정에 없습니다.", "문서에 해당 내용이 없습니다.", "근거가 없습니다."]:
            with self.subTest(prefix=prefix):
                self.assertEqual(finalize(self.prep(), prefix + " 하지만 매월 100만원을 지급합니다. [근거1]"), (REFUSE, []))

    def test_supported_negative_policy_with_no_extra_allowance_is_preserved(self):
        answer = "추가 수당 규정은 없습니다. [근거1]"
        self.assertEqual(finalize(self.prep(), answer)[0], "추가 수당 규정은 없습니다.")

    def test_explicit_missing_policy_synonym_is_canonical_refusal(self):
        from src.generate import REFUSE
        answer = ("규정에서 주 4일 근무제에 대한 내용이 명시되어 있지 않습니다. "
                  "따라서 시행하는지 여부는 확인할 수 없습니다. HR팀에 문의해 주세요. [근거1][근거2]")
        self.assertEqual(finalize(self.prep(), answer), (REFUSE, []))

    def test_negative_policy_statement_remains_supported_answer(self):
        answer = "무급휴직 기간에는 급여가 지급되지 않습니다. [근거1]"
        text, sources = finalize(self.prep(), answer)
        self.assertEqual(text, "무급휴직 기간에는 급여가 지급되지 않습니다.")
        self.assertEqual([source["id"] for source in sources], ["취업규칙:제8조"])

    def prep(self):
        def chunk(cid, doc, title):
            return {"id": cid, "text": "수습기간은 3개월로 한다.", "meta": {"doc": doc, "title": title, "kind": "regular"}}
        return {"chunks": [chunk("취업규칙:제8조", "취업규칙", "제8조 (수습기간)"), chunk("취업규칙:제9조", "취업규칙", "제9조 (수습 평가)"),
                           chunk("인사FAQ:Q3", "인사FAQ", "Q3. 수습")]}

    def test_valid_citation_selects_that_source_and_is_stripped(self):
        text, src = finalize(self.prep(), "3개월입니다. (근거: 취업규칙 제9조)")
        self.assertEqual(text, "3개월입니다.")
        self.assertEqual([s["id"] for s in src], ["취업규칙:제9조"])

    def test_wrong_document_is_refused_without_invented_source(self):
        """실제 사례(Q14): 잘못된 문서 인용은 출처를 보정하지 않고 거부한다."""
        _, src = finalize(self.prep(), "100% 지급됩니다. (근거: 복무규정 제8조)")
        self.assertEqual(src, [])

    def test_no_citation_is_refused_without_invented_source(self):
        _, src = finalize(self.prep(), "3개월입니다.")
        self.assertEqual(src, [])

    def test_numbered_citation_selects_exact_context(self):
        text, src = finalize(self.prep(), "3개월입니다. [근거2]")
        self.assertEqual(text, "3개월입니다.")
        self.assertEqual([s["id"] for s in src], ["취업규칙:제9조"])

    def test_invalid_or_empty_citation_answer_is_refused(self):
        for answer in ("답. [근거1][근거99]", "답. (근거: 취업규칙 제8조, 복무규정 제99조)", "[근거1]"):
            _, src = finalize(self.prep(), answer)
            self.assertEqual(src, [], answer)
        self.assertEqual(bad_citations("답. [근거99]", self.prep()["chunks"]), ["[근거99]"])

    def test_refusal_has_no_sources_even_with_stray_citation(self):
        text, src = finalize(self.prep(), "규정에서 찾을 수 없습니다. HR팀에 문의해 주세요. (근거: 복무규정 제35조)")
        self.assertEqual(src, [])
        self.assertNotIn("근거:", text)

    def test_refusal_discards_unsupported_following_claim(self):
        from src.generate import REFUSE
        text, src = finalize(self.prep(), "규정에서 찾을 수 없습니다. 하지만 매월 100만원을 지급합니다. [근거1]")
        self.assertEqual(text, REFUSE)
        self.assertEqual(src, [])

    def test_faq_is_not_shown_as_source(self):
        _, src = finalize(self.prep(), "답. (근거: 인사FAQ Q3)")
        self.assertEqual(src, [])


class ContextTest(unittest.TestCase):
    def test_direct_regulation_outranks_faq_references(self):
        """재현: 2026-10-27 Q22 '야근하면 저녁 밥값 나와요?'. 상위 FAQ 2개가 인용한 규정 3개가
        근거 3자리를 먼저 채워, 5위로 직접 검색된 정답 복리후생규정 제3조가 밀려났다."""
        hits = [{"id": "인사FAQ:Q21", "text": "(참조: 취업규칙 제25조)"},
                {"id": "인사FAQ:Q17", "text": "(참조: 복무규정 제9조, 제10조)"},
                {"id": "복무규정:제12조", "text": ""},
                {"id": "복리후생규정:제4조", "text": ""},
                {"id": "복리후생규정:제3조", "text": ""}]
        with patch("src.generate.search", return_value=hits),              patch("src.generate.get_chunk", side_effect=lambda cid: {"id": cid, "text": ""}):
            ids = [c["id"] for c in build_context("야근하면 저녁 밥값 나와요?")]
        self.assertIn("복리후생규정:제3조", ids)

    def test_faq_reference_fills_remaining_room(self):
        """직접 검색된 규정이 적으면 FAQ 인용 규정이 남은 자리를 채운다(병가 Q25 유형)."""
        faq = {"id": "인사FAQ:Q1", "text": "(참조: 복무규정 제24조)"}
        regs = [{"id": "복무규정:제1조", "text": ""}]
        with patch("src.generate.search", return_value=[faq, *regs]),              patch("src.generate.get_chunk", side_effect=lambda cid: {"id": cid, "text": ""}):
            ids = [c["id"] for c in build_context("아파서 쉬면 월급 나와요?")]
        self.assertEqual(ids, ["복무규정:제1조", "복무규정:제24조", "인사FAQ:Q1"])

    def test_colloquial_illness_context_finds_paid_sick_leave(self):
        self.assertIn("복무규정:제24조", [c["id"] for c in build_context("아파서 쉬면 월급 나와요?")])
    def test_faq_hit_pulls_in_cited_regulation(self):
        ids = [c["id"] for c in build_context("리프레시 휴가는 몇 년마다 며칠을 받나요?")]
        self.assertIn("복무규정:제26조", ids)
        self.assertLessEqual(len(ids), 5)

    def test_regulations_come_before_faq(self):
        ids = [c["id"] for c in build_context("월급은 매달 며칠에 나오나요?")]
        firsts_faq = next((n for n, i in enumerate(ids) if i.startswith("인사FAQ:")), len(ids))
        self.assertTrue(all(not i.startswith("인사FAQ:") for i in ids[:firsts_faq]))
        self.assertTrue(all(i.startswith("인사FAQ:") for i in ids[firsts_faq:]))


class NumberGroundingTest(unittest.TestCase):
    """답변의 숫자가 인용한 근거에 없으면 출처를 숫자가 실제로 있는 규정으로 바꾸고, 근거 어디에도 없으면 거부한다.
    재현: 2026-10-21 앱에서 '경조휴가 본인 결혼 5일' 답의 출처가 복리후생규정 제7조(경조금)로 표시됨."""

    def chunk(self, cid, title, text):
        doc = cid.split(":")[0]
        return {"id": cid, "text": f"[{doc}]\n{title}\n{text}", "meta": {"doc": doc, "title": title, "kind": "regular"}}

    def prep(self, question="경조휴가는 본인 결혼 시 며칠인가요?"):
        return {"question": question, "chunks": [
            self.chunk("복리후생규정:제7조", "제7조 (경조금)", "|구분|금액|\n|본인 결혼|50만 원|화환 제공|\n|자녀 결혼|30만 원||"),
            self.chunk("복무규정:제19조", "제19조 (경조휴가)", "|구분|일수|\n|본인 결혼|5일|\n|자녀 결혼|1일|"),
            self.chunk("복무규정:제15조", "제15조 (연차 사용)", "연차를 3일 이상 연속하여 사용하는 경우 사용하려는 날의 7일 전까지 신청한다.")]}

    def test_wrong_cited_number_moves_source_to_chunk_with_number(self):
        text, sources = finalize(self.prep(), "경조휴가는 본인 결혼 시 5일입니다. [근거1]")
        self.assertEqual(text, "경조휴가는 본인 결혼 시 5일입니다.")
        self.assertEqual([s["id"] for s in sources], ["복무규정:제19조"])

    def test_number_absent_from_all_evidence_is_refused(self):
        self.assertEqual(finalize(self.prep(), "본인 결혼 시 경조휴가는 6일입니다. [근거2]")[0], REFUSE)

    def test_number_boundary_does_not_match_inside_larger_number(self):
        # '5' 는 '50만 원' 이나 '15일' 안의 5 로 인정하지 않는다
        prep = self.prep()
        prep["chunks"] = prep["chunks"][:1]
        self.assertEqual(finalize(prep, "본인 결혼 시 5일입니다. [근거1]")[0], REFUSE)

    def test_correct_citation_is_kept(self):
        text, sources = finalize(self.prep(), "본인 결혼 경조금은 50만 원입니다. [근거1]")
        self.assertEqual([s["id"] for s in sources], ["복리후생규정:제7조"])

    def test_numbers_from_question_and_article_refs_are_ignored(self):
        prep = self.prep("연차를 3일 이상 붙여 쓰면 언제 신청하나요?")
        text, sources = finalize(prep, "3일 이상이면 제15조에 따라 7일 전까지 신청합니다. [근거3]")
        self.assertEqual([s["id"] for s in sources], ["복무규정:제15조"])

    def test_answer_without_numbers_keeps_citation(self):
        text, sources = finalize(self.prep(), "경조휴가는 증빙서류를 제출해야 합니다. [근거2]")
        self.assertEqual([s["id"] for s in sources], ["복무규정:제19조"])


class KeepAliveTest(unittest.TestCase):
    """모델이 기본 5분 뒤 내려가 다음 질문에서 다시 로딩(약 10초)하지 않도록 keep_alive 를 넘긴다. Ollama 는 mock."""

    def test_generate_passes_keep_alive(self):
        with patch("src.generate.OLLAMA_CLIENT") as client:
            client.chat.return_value = iter([{"message": {"content": "답"}}])
            list(generate({"chunks": [], "prev": None, "question": "질문"}))
        self.assertEqual(client.chat.call_args.kwargs["keep_alive"], KEEP_ALIVE)

    def test_embed_passes_keep_alive(self):
        embed.cache_clear()
        self.addCleanup(embed.cache_clear)
        with patch("src.retrieve.OLLAMA_CLIENT") as client:
            client.embed.return_value = {"embeddings": [[0.1]]}
            embed("keep-alive 시험")
        self.assertEqual(client.embed.call_args.kwargs["keep_alive"], KEEP_ALIVE)

    def test_warm_up_loads_generation_and_embedding_models(self):
        with patch("src.generate.OLLAMA_CLIENT") as client:
            warm_up()
        self.assertEqual(client.generate.call_args.kwargs["model"], GEN_MODEL)
        self.assertEqual(client.generate.call_args.kwargs["keep_alive"], KEEP_ALIVE)
        self.assertEqual(client.embed.call_args.kwargs["model"], EMBED_MODEL)
        self.assertEqual(client.embed.call_args.kwargs["keep_alive"], KEEP_ALIVE)


if __name__ == "__main__":
    unittest.main()
