# HR Chatbot (RAG)

## 개요
한국어 인사 규정을 근거로 답하는 로컬 RAG 챗봇. 일반 직원이 규정·절차를 묻는 내부 시연용 프로토타입이며, 사내 서버로 이전 가능한 구조로 만든다.
- 목적/범위 결정은 `PLAN.md`가 기준이다. 계획·실행·평가 기록은 `PLAN.md` 한 곳에 이어 쓴다.
- 문서는 제가 만든 **가상 샘플**(회사명·수치 모두 가상)이다. 실제 사내 문서가 아니다.
- 범위 밖(이번 단계 제외): 개인 데이터 조회(내 잔여 연차), 인증, HR 담당자 기능, HWP 파싱.

## 기술 스택
- Python 3.13, `venv` + `pip` (`uv` 미사용). 3.14는 wheel 미지원 위험으로 쓰지 않는다.
- LLM/임베딩: Ollama 로컬. 임베딩 `bge-m3`, 생성 `gemma3:4b`(2026-10-21 qwen2.5:7b에서 교체, 속도 우선. 비교 기록은 PLAN.md).
- 벡터 검색: `numpy` 전수 코사인(`data/index/embeddings.npy` + `chunks.json`). `chromadb`는 grpc DLL이 Windows 스마트 앱 컨트롤에 차단되어 2026-10-07 제거. 키워드 검색: `rank_bm25`. 검색은 벡터+BM25 하이브리드.
- 파싱: `pymupdf`(PDF), `python-docx`(docx). UI: `streamlit`.
- 프레임워크(LangChain/LlamaIndex) 없이 위 라이브러리만 직접 조립한다.
- 개발 환경: Windows 11, CPU 전용(i5-1335U, RAM 15.7GB, GPU 없음), 동시 사용자 1~3명 기준.

## 디렉터리 구조 (필요할 때만 파일을 추가한다)
```
CLAUDE.md, PLAN.md, requirements.txt
data/src/        가상 샘플 문서 원본(txt, 수정은 여기서). make_samples.py 가 raw 로 렌더링
data/raw/        생성된 샘플 문서 (PDF 3, docx 2). RAG 입력은 이쪽
data/index/      검색 색인(embeddings.npy, chunks.json) (재생성 가능, 커밋 제외)
eval/            평가 질문셋(questions.jsonl), check_gold.py(정답 근거 검증), 채점 스크립트
src/             ingest.py(파싱·청킹·색인), retrieve.py, generate.py, config.py
app.py           Streamlit 진입점
logs/            운영 지표·피드백 JSONL (원문 미저장)
tests/
```
`hr_chatbot.html`은 Streamlit 빌드 산출물(껍데기)이다. 소스가 아니므로 수정하지 않는다.

## 명령 (프로젝트 루트, Windows)
```
.venv\Scripts\python.exe -m pip install -r requirements.txt   # 환경
.venv\Scripts\python.exe data\make_samples.py                 # 샘플 문서 재생성 (data/src -> data/raw)
.venv\Scripts\python.exe -m src.ingest                        # 파싱·청킹·색인 (data/index 를 매번 새로 만듦)
.venv\Scripts\python.exe -m unittest discover -s tests -v     # 청킹 테스트
.venv\Scripts\python.exe eval\check_gold.py                   # 질문셋 정답 근거 검증
.venv\Scripts\python.exe -m eval.eval_retrieval [--strict] [-v]  # 검색 평가 (Ollama 필요, --strict: FAQ 인용 미인정)
```
테스트(`tests/test_retrieve.py`)는 색인과 Ollama(bge-m3)가 필요하다. 문서를 바꾸면 `src.ingest`를 다시 실행한다.
한글 출력이 깨지면 `PYTHONIOENCODING=utf-8` 을 붙인다.

## 코딩 규칙
- 최소 구현(Ponytail). 요청 없는 추상화, "나중을 위한" 코드, 불필요한 의존성을 만들지 않는다. 의도적 단순화는 `ponytail: <한계>, <업그레이드 경로>` 주석으로 남긴다.
- 청킹은 규정 구조(제N조/항/표) 단위. 각 청크에 문서명·조 번호·시행일 메타데이터를 붙인다.
- 설정값(모델명, 청크 크기, top-k, 거부 기준치)은 `src/config.py` 한 곳에서 관리한다.
- 답변에는 항상 출처(문서명·조항)를 붙인다.
- 변경 후 관련 테스트와 평가 스크립트를 실행한다. 버그는 재현 테스트부터 만든다.

## 답변 정책
1. 답변마다 출처(문서명, 조항, 페이지)를 표시한다.
2. 검색 점수가 기준치 미만이면 LLM을 호출하지 않고 "규정에서 찾을 수 없습니다. HR팀에 문의하세요"를 반환한다. 프롬프트에도 같은 지시를 병행한다.
3. 법률 해석과 개인별 판단은 답하지 않고 HR팀으로 안내한다.
4. 문서 밖 일반 질문(날씨, 잡담)은 거절한다.
5. 멀티턴: 질문 단독과 직전 질문 결합을 함께 검색한다.

## Constraints (방어 조건)
- **외부 API 호출 금지.** 문서·질문을 외부로 보내는 코드(OpenAI, Anthropic API 등)를 넣지 않는다. 모델은 Ollama 로컬만 쓴다.
- **로그에 사용자 식별 정보를 저장하지 않는다.** 질문·답변 원문은 저장하지 않고 처리 지표·피드백만 저장한다.
- **근거 없는 답변 금지.** 검색 근거가 없거나 점수 미달이면 생성하지 않는다.
- 색인(`data/index/`)과 `logs/`는 산출물이므로 코드와 섞어 커밋하지 않는다.
- 같은 오류가 2회 발생하면 여기에 방어 조건을 추가한다.
- 모델을 새로 받기 전(수 GB 다운로드)에는 사용자에게 모델명과 용량을 알리고 승인을 받는다.
