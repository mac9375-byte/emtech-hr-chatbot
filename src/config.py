"""프로젝트 설정 한 곳. 모델명·경로·기준치는 여기서만 바꾼다."""
from pathlib import Path
import ollama

# 환경변수 OLLAMA_HOST와 HTTP 프록시로 문서가 외부에 전송되지 않도록 로컬 주소를 고정한다.
OLLAMA_CLIENT = ollama.Client(host="http://127.0.0.1:11434", trust_env=False)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"
LOG_PATH = ROOT / "logs" / "events.jsonl"

EMBED_MODEL = "bge-m3"
EMBED_BATCH = 16

TOP_K = 5             # 답변에 쓰는 청크 수
CANDIDATES = 20       # 벡터/BM25 각각에서 뽑는 후보 수
RRF_K = 60            # Reciprocal Rank Fusion 상수

GEN_MODEL = "gemma3:4b"    # 2026-10-21 qwen2.5:7b 대비 체감 21s vs 36s, 규정 내 23/28 vs 20/28 (PLAN.md)
REFUSE_DISTANCE = 0.50     # 벡터 1위 거리가 이 값보다 크면 LLM 없이 거부. 정상 질문 33개 최대 0.40, 범위 밖 최소 0.59
                           # (0.35 는 구어체 정상 질문 4개를 오거부해 폐기). 도메인 근접 근거 없음은 프롬프트가 거른다
CONTEXT_MAX = 3            # LLM 에 넣는 근거 청크 상한. CPU prefill 이 초당 ~15토큰이라 프롬프트 길이가 곧 지연이다.
                           # 정답 조항 포함률 33문항 중 5개=32, 3개=32, 2개=31 이라 3으로 줄임
KEEP_ALIVE = "30m"         # Ollama 기본 5분이면 쉬었다 온 질문마다 모델을 다시 올려 약 10초가 더 든다. 대신 메모리를 계속 쓴다
                           # (gemma3:4b 약 3.5GB + bge-m3 약 1.2GB). ponytail: 고정 30분, 상시 시연 서버면 -1(계속 유지)로 바꾼다
NUM_CTX = 4096
MAX_ANSWER_TOKENS = 400
