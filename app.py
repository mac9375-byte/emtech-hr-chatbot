"""이엠텍 HR 챗봇 (Streamlit). 실행: streamlit run app.py"""
import json
import time
import uuid
from datetime import datetime, timezone

import streamlit as st
import ollama
import httpx

from src.config import GEN_MODEL, LOG_PATH
from src.generate import finalize, generate, prepare, warm_up
from src.retrieve import _index

st.set_page_config(page_title="이엠텍 HR 챗봇")


def append_log(event):
    """운영 지표와 피드백만 남긴다. 질문·답변 원문에는 개인 정보가 포함될 수 있어 저장하지 않는다."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


@st.cache_resource(show_spinner="검색 색인을 준비하는 중입니다. 처음 한 번만 걸립니다.")
def boot():
    _index()
    try:
        warm_up()
    except (ConnectionError, ollama.ResponseError, httpx.RequestError):
        pass  # 예열 실패는 시작을 막지 않는다. 질문할 때 연결 오류 안내가 나간다
    return True


def on_feedback(msg_id):
    value = st.session_state.get(f"fb_{msg_id}")
    if value is not None:
        append_log({"type": "feedback", "id": msg_id, "value": "up" if value == 1 else "down"})


def show_sources(sources, title):
    for s in sources:
        with st.expander(f"{title}: {s['label']}"):
            st.text(s["text"])


def render(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return
        show_sources(msg.get("sources", []), "근거")
        st.feedback("thumbs", key=f"fb_{msg['id']}", on_change=on_feedback, args=(msg["id"],))


def answer(question, history):
    """질문 하나를 처리해 화면에 그리고, 저장할 메시지를 돌려준다."""
    t0 = time.time()
    msg = {"role": "assistant", "id": uuid.uuid4().hex[:8], "sources": []}
    prep = {"route": "error"}
    status = st.empty()
    try:
        prep = prepare(question, history)
        if prep["route"] != "answer":  # 개인·법률 안내, 근거 없음 거부: LLM 을 부르지 않는다
            msg["content"] = prep["message"]
        else:
            status.caption("답변을 생성하고 근거를 확인 중입니다. 완료 후 답변과 출처를 표시합니다. 이 PC에서는 30~60초 걸릴 수 있습니다.")
            text = ""
            for part in generate(prep):
                text += part
            msg["content"], msg["sources"] = finalize(prep, text)
    except (ConnectionError, ollama.ResponseError, httpx.RequestError):
        prep = {"route": "error"}
        msg["content"] = "로컬 모델에 연결하지 못했습니다. Ollama 실행 상태와 설치된 모델을 확인한 뒤 다시 질문해 주세요."
    finally:
        status.empty()
    append_log({"type": "answer", "id": msg["id"], "route": prep["route"],
                "distance": prep.get("distance"), "source_count": len(msg["sources"]),
                "model": GEN_MODEL, "seconds": round(time.time() - t0, 1)})
    return msg


try:
    boot()
except (FileNotFoundError, ValueError):
    st.error("검색 색인이 없거나 비어 있습니다. 문서와 색인 준비 상태를 관리자에게 확인해 주세요.")
    st.stop()
st.title("이엠텍 HR 챗봇")
st.caption("인사 규정(취업규칙, 복무규정, 복리후생 규정, FAQ, 신청양식)을 근거로 답합니다. 시연용 가상 문서입니다.")
with st.sidebar:
    st.subheader("안내")
    st.markdown("- 규정과 절차를 안내합니다.\n- 개인 정보(잔여 연차 등) 조회, 법률 해석, 개인별 판단은 하지 않습니다. HR팀에 문의해 주세요.\n"
                "- 처리 시간·오류 여부·피드백만 저장합니다. 질문과 답변 원문은 저장하지 않습니다.")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    render(m)

if question := st.chat_input("인사 규정에 대해 물어보세요. 예: 육아휴직은 언제까지 신청하나요?"):
    history = [m["content"] for m in st.session_state.messages if m["role"] == "user"][-3:]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        msg = answer(question, history)
    st.session_state.messages.append(msg)
    st.rerun()  # 저장된 메시지로 다시 그려서 출처·피드백 위젯을 붙인다
