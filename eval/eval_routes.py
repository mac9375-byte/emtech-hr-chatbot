"""경로 판정 평가(LLM 생성 없음): 질문이 올바른 경로(answer / refuse / refer_hr)로 가는지, 거부 기준치를 훑어 본다.
후속 질문은 (질문 단독 + 이전 질문 결합) 이중 검색으로 정답 조항을 찾는지도 출력한다.
사용: python -m eval.eval_routes
"""
import json
from pathlib import Path

from eval.eval_retrieval import accepted_ids
from src.config import REFUSE_DISTANCE
from src.generate import prepare
from src.retrieve import _index, search

ROOT = Path(__file__).resolve().parent.parent
EXPECT = {"in_scope": "answer", "followup": "answer", "no_evidence": "refuse", "out_of_scope": "refuse", "legal_personal": "refer_hr"}


def main():
    qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    _, _, ids, docs, _ = _index()
    rows = [(q, prepare(q["question"], q.get("history", []))) for q in qs]

    print(f"거부 기준치: {REFUSE_DISTANCE}\n")
    wrong = [(q, p) for q, p in rows if p["route"] != EXPECT[q["type"]]]
    for q, p in wrong:
        d = p.get("distance")
        print(f"  오판 {q['id']} ({q['type']}): 기대 {EXPECT[q['type']]}, 실제 {p['route']}, 거리 {'-' if d is None else f'{d:.2f}'}  '{q['question']}'")
    print(f"경로 판정: {len(rows) - len(wrong)}/{len(rows)} 정답\n")

    pos = [(p["distance"], q["id"]) for q, p in rows if q["type"] in ("in_scope", "followup") and "distance" in p]
    neg = [(p["distance"], q["id"]) for q, p in rows if q["type"] in ("no_evidence", "out_of_scope") and "distance" in p]
    print(f"정상 질문 거리: 최대 {max(pos)[0]:.2f}({max(pos)[1]}), 근거 없음·범위 밖 거리: 최소 {min(neg)[0]:.2f}({min(neg)[1]})")
    print("기준치 훑기:  기준치  정상 오거부  거부 누락")
    for t in (0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60):
        fr = [i for d, i in pos if d > t]
        leak = [i for d, i in neg if d <= t]
        print(f"            {t:.2f}   {len(fr):>2}/{len(pos)} {str(fr) if fr else '':<22} {len(leak)}/{len(neg)} {leak if leak else ''}")

    print("\n후속 질문 이중 검색 결과 (순위: 정답 기준 b, 상위 20 안)")
    for q, p in rows:
        if q["type"] != "followup":
            continue
        queries = p.get("queries", [])
        res = [r["id"] for r in search(queries, k=20)] if queries else []
        ok = accepted_ids(q, ids, docs)
        rank = next((n for n, i in enumerate(res, 1) if i in ok), None)
        d = p.get("distance")
        print(f"  {q['id']} 순위 {rank} 거리 {'-' if d is None else f'{d:.2f}'} | {queries}")


if __name__ == "__main__":
    main()
