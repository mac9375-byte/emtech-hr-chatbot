"""검색 평가: 질문마다 정답 조항이 상위 k개 안에 드는지(hit@k)와 MRR 을 여러 설정으로 비교한다.

정답 판정: 정답 조항 청크, 또는 그 조항을 "(참조: ...)" 로 인용한 인사FAQ 청크(사용자에게는 둘 다 맞는 답).
실서비스와 같은 질문 단독 + 직전 질문 결합 검색을 사용한다.
사용: python -m eval.eval_retrieval [--strict] [-v]   (--strict: FAQ 인용을 정답으로 인정하지 않음, -v: 질문별 순위)
"""
import json
import sys
from pathlib import Path

from src.generate import queries_of
from src.retrieve import _index, faq_refs, search, top_distance

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ["vector", "bm25", "hybrid"]


def accepted_ids(q, ids, docs, strict=False):
    """strict=True 면 정답 조항 청크만 인정(FAQ 인용 제외)."""
    ok = {f"{q['gold_doc']}:{q['gold_article']}"}
    for i, d in zip(ids, docs):
        if not strict and i.startswith("인사FAQ:") and (q["gold_doc"], q["gold_article"]) in faq_refs(d):
            ok.add(i)
    return ok


def query_of(q):
    return queries_of(q["question"], q.get("history", []))


def main():
    _, _, ids, docs, _ = _index()
    qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = [q for q in qs if "gold_doc" in q]
    verbose, strict = "-v" in sys.argv, "--strict" in sys.argv
    summary = []
    for name in CONFIGS:
        ranks = []
        for q in gold:
            res = [r["id"] for r in search(query_of(q), k=20, mode=name)]
            ok = accepted_ids(q, ids, docs, strict)
            rank = next((n for n, i in enumerate(res, 1) if i in ok), None)
            ranks.append(rank)
            if verbose:
                print(f"  [{name}] {q['id']} rank={rank} gold={q['gold_doc']}:{q['gold_article']} top3={res[:3]}")
        hit = lambda k: sum(1 for r in ranks if r and r <= k) / len(ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        summary.append((name, hit(1), hit(3), hit(5), mrr, [q["id"] for q, r in zip(gold, ranks) if not r or r > 5]))
    in_scope = sum(q["type"] == "in_scope" for q in gold)
    followup = sum(q["type"] == "followup" for q in gold)
    print(f"\n정답 질문 {len(gold)}개 (규정 내 {in_scope} + 후속 {followup}), 정답 기준: {'(a) 규정 조항만' if strict else '(b) 조항 또는 인용 FAQ'}")
    print(f"{'설정':<22}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>7}   hit@5 실패")
    for name, h1, h3, h5, mrr, miss in summary:
        print(f"{name:<22}{h1:>7.0%}{h3:>7.0%}{h5:>7.0%}{mrr:>7.2f}   {miss}")

    print("\n벡터 1위 거리 (거부 기준치 참고: 작을수록 관련)")
    for typ in ("in_scope", "followup", "no_evidence", "out_of_scope", "legal_personal"):
        ds = sorted((top_distance(q["question"]), q["id"]) for q in qs if q["type"] == typ)
        print(f"  {typ:<15} " + "  ".join(f"{d:.2f}({i})" for d, i in ds))


if __name__ == "__main__":
    main()
