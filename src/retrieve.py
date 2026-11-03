"""하이브리드 검색: 벡터(bge-m3, numpy 전수 코사인) + BM25(글자 2-gram)를 RRF 로 합친다."""
import json
import re
from functools import cache, lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import CANDIDATES, EMBED_MODEL, INDEX_DIR, KEEP_ALIVE, OLLAMA_CLIENT, RRF_K, TOP_K


def tokens(text):
    """한국어 조사 변형(연차를/연차는)에 강하도록 단어를 글자 2-gram 으로 쪼갠다.
    ponytail: 형태소 분석기 없음, 필요하면 kiwipiepy 등으로 교체."""
    out = []
    for w in re.findall(r"[가-힣A-Za-z0-9]+", text):
        out += [w[i:i + 2] for i in range(len(w) - 1)] if len(w) > 1 else [w]
    return out


@cache
def _index():
    """색인이 없으면 FileNotFoundError, 비어 있으면 ValueError."""
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("검색 색인이 비어 있습니다. 문서를 색인해 주세요.")
    emb = np.load(INDEX_DIR / "embeddings.npy")  # ingest 가 행마다 L2 정규화해 저장
    docs = [c["text"] for c in chunks]
    return emb, BM25Okapi([tokens(d) for d in docs]), [c["id"] for c in chunks], docs, [c["meta"] for c in chunks]


@lru_cache(maxsize=256)
def embed(query):
    return OLLAMA_CLIENT.embed(model=EMBED_MODEL, input=[query], keep_alive=KEEP_ALIVE)["embeddings"][0]


def _nearest(query, n):
    """-> [(id, 코사인 거리)] 가까운 순 n개. 거리 = 1 - 코사인 유사도(Chroma cosine 과 같은 정의).
    ponytail: 전수 비교(청크 1만 개도 수 ms). 수십만 개를 넘으면 hnswlib/faiss 같은 근사 색인으로 교체."""
    emb, _, ids, _, _ = _index()
    v = np.asarray(embed(query), dtype=np.float32)
    dist = 1 - emb @ (v / np.linalg.norm(v))
    return [(ids[i], float(dist[i])) for i in np.argsort(dist, kind="stable")[:n]]


def search(query, k=TOP_K, mode="hybrid"):
    """-> [{id, text, meta, distance, rrf}] 상위 k개. distance 는 벡터 코사인 거리(없으면 None).
    query: 문자열 또는 문자열 목록(여러 검색어의 순위를 RRF 로 함께 합친다. 멀티턴 후속 질문용).
    mode: vector | bm25 | hybrid (평가 비교용, 실서비스는 hybrid)."""
    _, bm25, ids, docs, metas = _index()
    pos = {i: n for n, i in enumerate(ids)}
    ranks, dist = [], {}
    for q in [query] if isinstance(query, str) else query:
        # ponytail: 개인의 짧은 병가 표현만 보충. 표현이 늘면 문서 기반 용어 사전으로 교체.
        if re.search(r"(?:아파서|아프면|몸이 아파)\s*쉬", q) and not re.search(r"가족|부모|자녀|아이|돌보", q):
            q = f"{q} 병가"
        if mode in ("vector", "hybrid"):
            near = _nearest(q, CANDIDATES)
            ranks.append([i for i, _ in near])
            for i, d in near:
                dist[i] = min(dist.get(i, d), d)
        if mode in ("bm25", "hybrid"):
            scores = bm25.get_scores(tokens(q))
            ranks.append([ids[n] for n in sorted(range(len(ids)), key=lambda n: -scores[n])[:CANDIDATES]])
    fused = {}
    for lst in ranks:
        for rank, i in enumerate(lst, 1):
            fused[i] = fused.get(i, 0) + 1 / (RRF_K + rank)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{"id": i, "text": docs[pos[i]], "meta": metas[pos[i]], "distance": dist.get(i), "rrf": fused[i]} for i in top]


def faq_refs(text):
    """FAQ 청크의 '(참조: 복무규정 제14조, 제15조, 복리후생 규정 제7조)' -> {(문서, 조), ...}"""
    m = re.search(r"\(참조:([^)]*)\)", text)
    refs, doc = set(), None
    for tok in re.finditer(r"(취업규칙|복무규정|복리후생 규정)|(제\d+조)", m.group(1) if m else ""):
        if tok.group(1):
            doc = tok.group(1).replace(" ", "")
        else:
            refs.add((doc, tok.group(2)))
    return refs


def get_chunk(chunk_id):
    """id -> {id, text, meta}. 없으면 None."""
    _, _, ids, docs, metas = _index()
    if chunk_id not in ids:
        return None
    n = ids.index(chunk_id)
    return {"id": chunk_id, "text": docs[n], "meta": metas[n]}


def top_distance(query):
    """벡터 검색 1위의 코사인 거리(작을수록 관련). 답변 거부 판정에 쓴다. 목록이면 가장 작은 값."""
    return min(_nearest(q, 1)[0][1] for q in ([query] if isinstance(query, str) else query))
