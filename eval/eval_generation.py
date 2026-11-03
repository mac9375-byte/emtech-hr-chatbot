"""생성 평가: 모델별로 질문셋 전체를 실제 파이프라인(prepare -> generate)에 통과시켜 정확도·출처·거부·지연을 잰다.

판정(자동, 정답 키워드 기준이라 보수적. 실패는 결과 파일로 직접 확인할 것):
- in_scope/followup : answer 경로 + 정답 키워드 전부 포함 + 금지 키워드 없음 + finalize 후 정답 조항이 출처에 있음 + 거부 문구 아님
- no_evidence, out_of_scope : refuse 경로이거나, answer 경로에서 finalize 후 화면 표시 답변이 거부임(모델 거부·무효 인용·무인용 포함)
- legal_personal    : refer_hr 경로
결과: eval/results/<모델><tag>.jsonl (모델명 ':'는 '_'로 바꿈. 질문마다 생성 당시 근거를 함께 저장)
사용: python -m eval.eval_generation --models qwen2.5:7b,exaone3.5:7.8b [--only Q01,F04] [--limit 10] [--tag _v2]
     python -m eval.eval_generation --rejudge eval/results/<파일>.jsonl   (저장된 답변을 현재 판정 기준으로 재채점)
"""
import json
import re
import sys
import time
from pathlib import Path

from src.config import GEN_MODEL
from src.generate import NUMBERED_CITE, bad_citations, finalize, generate, prepare

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
EXPECT = {"in_scope": "answer", "followup": "answer", "no_evidence": "refuse", "out_of_scope": "refuse", "legal_personal": "refer_hr"}


def norm(s):
    return re.sub(r"[\s,]+", "", s)


def judge(q, prep, text):
    """모델 원문 text 를 finalize() 로 화면에 보일 답변·출처로 바꾼 뒤 채점한다.
    -> (통과 여부, 실패 사유 목록, 표시 답변, 표시 출처 id 목록). 유효한 인용이 없으면 finalize가 거부한다."""
    typ, route = q["type"], prep["route"]
    shown, sources = finalize(prep, text) if route == "answer" else (text, [])
    src_ids = [s["id"] for s in sources]
    if typ in ("no_evidence", "out_of_scope"):  # 거리 게이트가 못 거르는 경계 사례는 LLM 이 거부해야 한다
        ok = route == "refuse" or (route == "answer" and "찾을수없" in norm(shown))
        return ok, [] if ok else ["근거 없는/범위 밖 질문에 답을 지어냄"], shown, src_ids
    if route != EXPECT[typ]:
        return False, [f"경로 {route} (기대 {EXPECT[typ]})"], shown, src_ids
    if typ not in ("in_scope", "followup"):
        return True, [], shown, src_ids
    errs = []
    # 키워드는 '|' 로 동의어를 줄 수 있다(예: "무급|지급되지"). 하나라도 있으면 통과
    errs += [f"키워드 없음: {k}" for k in q["keywords"] if not any(norm(alt) in norm(shown) for alt in k.split("|"))]
    errs += [f"부정확한 항목: {k}" for k in q.get("forbidden_keywords", []) if norm(k) in norm(shown)]
    golds = {f"{q['gold_doc']}:{q['gold_article']}", *q.get("gold_also", [])}  # gold_also: 같은 사실이 실린 다른 조항(check_gold 가 검증)
    if not golds & set(src_ids):
        errs.append("정답 조항이 출처에 없음")
    if "찾을수없" in norm(shown):
        errs.append("근거가 있는데 거부함")
    return not errs, errs, shown, src_ids


def run(model, qs, tag=""):
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{model.replace(':', '_')}{tag}.jsonl"  # 태그가 없으면 같은 이름의 이전 결과를 덮어쓴다
    out.write_text("", encoding="utf-8")
    rows = []
    for n, q in enumerate(qs, 1):
        t0 = time.time()
        prep = prepare(q["question"], q.get("history", []))
        t_prep = time.time() - t0
        text, t_first = "", None
        if prep["route"] == "answer":
            t1 = time.time()
            for part in generate(prep, model=model):
                if t_first is None and part.strip():
                    t_first = time.time() - t1
                text += part
        else:
            text = prep["message"]
        ok, errs, shown, src_ids = judge(q, prep, text)
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "route": prep["route"], "queries": prep.get("queries"),
               "distance": prep.get("distance"), "ok": ok, "errors": errs, "raw_answer": text, "answer": shown, "prep": prep,
               "sources": src_ids, "bad_citations": bad_citations(text, prep["chunks"]) if prep["route"] == "answer" else [],
               "t_prepare": round(t_prep, 1),
               "t_first_token": None if t_first is None else round(t_first, 1), "t_total": round(time.time() - t0, 1)}
        rows.append(row)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{model}] {n}/{len(qs)} {q['id']} {'OK ' if ok else 'FAIL'} {row['t_total']}s {errs if errs else ''}", flush=True)
    return rows


def summarize(model, rows):
    def rate(types):
        rs = [r for r in rows if r["type"] in types]
        return f"{sum(r['ok'] for r in rs)}/{len(rs)}"
    gen = [r for r in rows if r["t_first_token"] is not None]
    first = sorted(r["t_first_token"] + r["t_prepare"] for r in gen)
    total = sorted(r["t_total"] for r in gen)
    med = lambda xs: xs[len(xs) // 2] if xs else float("nan")
    print(f"\n== {model} ==")
    print(f"  규정 내(in_scope) {rate({'in_scope'})} | 후속(followup) {rate({'followup'})} | 근거 없음 {rate({'no_evidence'})} | "
          f"범위 밖 {rate({'out_of_scope'})} | 개인·법률 {rate({'legal_personal'})}")
    print(f"  모델이 쓴 잘못된 인용: {sum(bool(r.get('bad_citations')) for r in rows)}건")
    print(f"  생성한 답변 {len(gen)}개: 첫 글자까지 중앙 {med(first):.0f}s 최대 {max(first, default=0):.0f}s | 전체 중앙 {med(total):.0f}s 최대 {max(total, default=0):.0f}s")
    for r in rows:
        if not r["ok"]:
            print(f"  FAIL {r['id']} {r['errors']} '{r['question']}'")


def rejudge(path, qs):
    """저장된 생성 당시 근거로 답변을 재채점한다. 근거 번호를 현재 검색 결과로 재해석하지 않는다."""
    by_id = {q["id"]: q for q in qs}
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        q = by_id[r["id"]]
        raw = r.get("raw_answer", r["answer"])
        prep = r.get("prep")
        if prep is None:
            if NUMBERED_CITE.search(raw):
                raise ValueError(f"{r['id']}: 생성 당시 근거 snapshot이 없어 번호 인용을 재채점할 수 없습니다. 평가를 재생성해 주세요.")
            # ponytail: 문서·조항을 직접 쓴 구 결과만 현재 검색으로 재채점. 현재 모델 생성 성능을 증명하지는 않는다.
            prep = prepare(q["question"], q.get("history", []))
        r["ok"], r["errors"], r["answer"], r["sources"] = judge(q, prep, raw)
        r["bad_citations"] = bad_citations(raw, prep["chunks"]) if prep["route"] == "answer" else []
        rows.append(r)
    summarize(Path(path).stem, rows)


def main():
    arg = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    all_qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if arg("--rejudge"):
        return rejudge(arg("--rejudge"), all_qs)
    models = arg("--models", GEN_MODEL).split(",")
    qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if arg("--only"):
        qs = [q for q in qs if q["id"] in arg("--only").split(",")]
    if arg("--limit"):
        qs = qs[:int(arg("--limit"))]
    for m in models:
        summarize(m, run(m, qs, arg("--tag", "")))


if __name__ == "__main__":
    main()
