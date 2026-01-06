import json
import re
import uuid
from typing import Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

from db_mariadb import (
    init_db, create_lesson, load_lesson,
    get_lesson_image, upsert_lesson_image, save_student_response,
    save_student_generated_image, save_copyright_discussion,
    get_copyright_stats, get_recent_discussions
)

from rag_store import get_kb_collection, ask_gpt_text_rag, infer_issue


# -----------------------------
# 1) 페이지 설정
# -----------------------------
st.set_page_config(page_title="AI 윤리교육 콘텐츠 제공 시스템", page_icon="🤖", layout="wide")


# -----------------------------
# 2) DB 초기화
# -----------------------------
try:
    init_db()
except Exception as e:
    st.error(f"⚠️ MariaDB 연결/초기화 실패: {e}")
    st.stop()


# -----------------------------
# 3) OpenAI 클라이언트
# -----------------------------
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

try:
    client = get_openai_client()
except Exception:
    st.error("⚠️ OPENAI_API_KEY 설정 오류")
    st.stop()


# -----------------------------
# 4) 벡터DB(KB) 컬렉션
# -----------------------------
kb = get_kb_collection("kb")


# -----------------------------
# 5) 시스템 페르소나(보조교사)
# -----------------------------
SYSTEM_PERSONA = """
당신은 AI 윤리교육 보조교사 입니다.
대상: 초등학교 5~6학년.

[출력 기본]
- 인사말/잡담 금지. 2~4개 항목 개조식으로만 출력.
- 각 항목은 1문장 이내. 짧고 쉬운 단어 사용.
- 어려운 단어는 괄호로 짧게 풀이.

[학생 피드백 형식]
- 아래 중 하나의 템플릿을 반드시 사용:
  A) 잘한 점 / 위험 요소 / 확인 질문 / 다음 행동
  B) 핵심 판단 / 근거 / 확인 질문 / 다음 행동

[교사용 요청]
- 교사용 요약/설계 요청이면 교사 관점으로 3~6개 항목 개조식.

[JSON 시나리오 생성]
- '시나리오 JSON 생성' 요청이면 JSON 객체만 출력.
- 최상위 키: scenario
- 각 원소 키: story, choice_a, choice_b
- 불필요한 설명/문장/코드블록 금지(순수 JSON).

[안전]
- 개인정보(이름/전화/주소/얼굴 사진 등) 요청, 불법/유해 행위는 거절하고 안전한 대안만 제시.
"""


# -----------------------------
# 6) OpenAI 호출(기본)
# -----------------------------
def ask_gpt_json(prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        data = json.loads(response.choices[0].message.content.strip())
        if "scenario" not in data:
            return {"scenario": []}
        return data
    except Exception:
        return {"scenario": []}


def ask_gpt_text(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "응답 불가."


def generate_image(prompt: str) -> Optional[str]:
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"Minimalist, flat design illustration, educational context: {prompt}",
            size="1024x1024",
            n=1
        )
        return response.data[0].url
    except Exception:
        return None


# -----------------------------
# 7) 간단 PII 방지
# -----------------------------
PHONE_RE = re.compile(r"\b(01[0-9]-?\d{3,4}-?\d{4})\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

def looks_like_pii(text: str) -> bool:
    if not text:
        return False
    return bool(PHONE_RE.search(text) or EMAIL_RE.search(text))


# -----------------------------
# 8) 세션 상태
# -----------------------------
if "scenario" not in st.session_state or not isinstance(st.session_state.scenario, dict):
    st.session_state.scenario = {"scenario": []}

default_keys = {
    "analysis": "",
    "current_step": 0,
    "chat_history": [],
    "topic": "",
    "lesson_id": "",
    "student_key": "",
}
for k, v in default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.student_key:
    st.session_state.student_key = str(uuid.uuid4())


# -----------------------------
# 9) 사이드바
# -----------------------------
st.sidebar.title("🤖 AI 윤리 학습 시스템")

if st.sidebar.button("⚠️ 세션 초기화"):
    st.session_state.clear()
    st.rerun()

mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"])

st.sidebar.divider()
st.sidebar.subheader("수업 불러오기(lesson_id)")
lesson_code = st.sidebar.text_input("수업 코드 입력", value=st.session_state.lesson_id)

if st.sidebar.button("불러오기"):
    loaded = load_lesson(lesson_code.strip())
    if not loaded:
        st.sidebar.error("해당 수업 코드를 찾을 수 없음.")
    else:
        st.session_state.lesson_id = loaded["lesson_id"]
        st.session_state.topic = loaded["topic"]
        st.session_state.scenario = loaded["scenario"]
        st.session_state.analysis = loaded["analysis"]
        st.session_state.current_step = 0
        st.session_state.chat_history = []
        st.sidebar.success("수업 로드 완료.")
        st.rerun()

st.sidebar.divider()
st.sidebar.caption(f"KB 청크 수: {kb.count()} (0이면 kb_ingest.py 실행 필요)")


# =========================================================
# 교사용 모드
# =========================================================
if mode == "👨‍🏫 교사용":
    st.header("🛠️ 수업 생성(콘텐츠 제공 시스템)")

    input_topic = st.text_input("학습 주제 입력", value=st.session_state.topic)

    if st.button("생성 시작"):
        if not input_topic.strip():
            st.warning("주제 필요.")
        else:
            with st.spinner("생성 중..."):
                # 1) 시나리오(JSON) 생성
                s_prompt = (
                    f"시나리오 JSON 생성: 주제 '{input_topic}'의 3단계 딜레마 시나리오를 생성하라. "
                    f"최상위 키 scenario, 내부 키 story, choice_a, choice_b."
                )
                scenario_result = ask_gpt_json(s_prompt)

                # 2) 교사용 분석 요약: RAG 우선(teacher KB), KB가 비어있으면 기본 생성으로 폴백
                if kb.count() > 0:
                    a_prompt = (
                        f"교사용 요청: 주제 '{input_topic}'에 대해\n"
                        f"- 핵심 가치\n- 연계 교과(실과/도덕 중심)\n- 학습 목표\n- 활동 제안(토론/검증/프롬프트)\n- 안전/저작권/개인정보 주의\n"
                        f"를 3~6개 항목 개조식으로 요약."
                    )
                    analysis_text = ask_gpt_text_rag(
                        openai_client=client,
                        collection=kb,
                        system_persona=SYSTEM_PERSONA,
                        user_prompt=a_prompt,
                        audience="teacher",
                        ethical_issue=None,
                        top_k=6
                    )
                else:
                    a_prompt = f"교사용 요청: 주제 '{input_topic}'의 핵심 가치, 교과, 목표를 개조식으로 요약."
                    analysis_text = ask_gpt_text(a_prompt)

                # 3) DB 저장(교사 입력/생성물 영속화)
                try:
                    lesson_id = create_lesson(
                        topic=input_topic,
                        scenario_dict=scenario_result,
                        analysis_text=analysis_text
                    )
                except Exception as e:
                    st.error(f"⚠️ DB 저장 실패: {e}")
                    st.stop()

                # 4) UI 캐시
                st.session_state.topic = input_topic
                st.session_state.scenario = scenario_result
                st.session_state.analysis = analysis_text
                st.session_state.lesson_id = lesson_id
                st.session_state.current_step = 0
                st.success("생성 및 저장 완료(MariaDB).")

    if st.session_state.lesson_id:
        st.subheader("🔑 수업 코드(lesson_id)")
        st.code(st.session_state.lesson_id)

    if st.session_state.analysis:
        st.divider()
        st.subheader("📊 분석 결과")
        st.info(st.session_state.analysis)

    scenario_data = st.session_state.scenario.get("scenario", [])
    if scenario_data:
        st.divider()
        st.subheader("📜 시나리오 미리보기")
        for i, step in enumerate(scenario_data):
            with st.container(border=True):
                st.markdown(f"### 🔹 {i+1}단계")
                st.markdown(f"**📖 상황:** {step.get('story', '')}")
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**🅰️ 선택:** {step.get('choice_a', '')}")
                with col2:
                    st.warning(f"**🅱️ 선택:** {step.get('choice_b', '')}")

    # 교사용: 저작권 토론 집계
    st.divider()
    st.subheader("🧩 저작권(권리) 토론 결과(집계)")

    if not st.session_state.lesson_id:
        st.info("수업 코드(lesson_id)가 있어야 집계를 조회할 수 있습니다.")
    else:
        n_steps = len(st.session_state.scenario.get("scenario", []))
        if n_steps == 0:
            st.info("시나리오가 아직 없습니다. 먼저 수업을 생성하세요.")
        else:
            step_sel = st.selectbox("조회 단계", options=list(range(n_steps)), format_func=lambda x: f"{x+1}단계")
            stats = get_copyright_stats(st.session_state.lesson_id, step_sel)
            if not stats:
                st.warning("아직 제출된 토론 기록이 없습니다.")
            else:
                df = pd.DataFrame(stats, columns=["stance", "count"]).set_index("stance")
                st.bar_chart(df)

                st.subheader("🗂️ 최근 의견 샘플")
                samples = get_recent_discussions(st.session_state.lesson_id, step_sel, limit=10)
                for s in samples:
                    st.markdown(f"- **{s['stance']}**: {s['reasoning']}")


# =========================================================
# 학생용 모드
# =========================================================
else:
    steps = st.session_state.scenario.get("scenario", [])
    if not steps or not st.session_state.lesson_id:
        st.warning("수업 데이터 없음. 사이드바에서 수업 코드를 불러오세요.")
        st.stop()

    lesson_id = st.session_state.lesson_id
    idx = st.session_state.current_step
    total = len(steps)

    if idx >= total:
        st.success("수업 종료.")
        if st.button("처음으로"):
            st.session_state.current_step = 0
            st.session_state.chat_history = []
            st.rerun()
        st.stop()

    data = steps[idx]
    st.progress((idx + 1) / total)
    st.subheader(f"단계 {idx+1}")

    # 단계 공용 이미지(캐시)
    img_url = get_lesson_image(lesson_id, idx)
    if not img_url:
        with st.spinner("이미지 생성..."):
            img_url = generate_image(data.get("story", ""))
            if img_url:
                upsert_lesson_image(lesson_id, idx, img_url)
    if img_url:
        st.image(img_url)

    st.info(data.get("story", "내용 없음"))

    # -------------------------
    # 1) 학생 선택/이유 제출 → RAG 기반 피드백
    # -------------------------
    with st.form(f"form_{idx}"):
        sel = st.radio("선택", [data.get("choice_a", "A"), data.get("choice_b", "B")])
        reason = st.text_area("이유")
        submitted = st.form_submit_button("제출")

    if submitted:
        if not reason.strip():
            st.warning("이유 입력 필요.")
        elif looks_like_pii(reason):
            st.warning("개인정보(전화/이메일 등) 포함 가능. 삭제 후 다시 작성.")
        else:
            user_prompt = (
                f"학생 피드백:\n"
                f"상황: {data.get('story')}\n"
                f"선택: {sel}\n"
                f"이유: {reason}\n"
                f"요구: 초등 5~6학년 수준으로 2~4개 항목 개조식 피드백."
            )

            issue = infer_issue(data.get("story", "") + " " + sel + " " + reason)
            if kb.count() > 0:
                res = ask_gpt_text_rag(
                    openai_client=client,
                    collection=kb,
                    system_persona=SYSTEM_PERSONA,
                    user_prompt=user_prompt,
                    audience="student_56",
                    ethical_issue=issue,
                    top_k=4
                )
            else:
                res = ask_gpt_text(user_prompt)

            st.session_state.chat_history = [
                {"role": "user", "content": f"[{sel}] {reason}"},
                {"role": "assistant", "content": res}
            ]

            save_student_response(
                lesson_id=lesson_id,
                step_index=idx,
                selection=sel,
                reason=reason,
                feedback=res
            )

    # 채팅 출력
    if st.session_state.chat_history:
        st.divider()
        for msg in st.session_state.chat_history:
            role = "assistant" if msg["role"] == "assistant" else "user"
            st.chat_message(role).write(msg["content"])

    # -------------------------
    # 2) 학생 실습: 이미지 생성 + 저작권 토론 + (RAG) 토론 촉진 질문
    # -------------------------
    with st.expander("🧪 이미지 생성 실습 · 저작권(권리) 토론", expanded=False):
        st.caption("주의: 개인정보 입력 금지. 실제 인물 얼굴/딥페이크 요청 금지.")

        lab_prompt = st.text_area(
            "이미지 프롬프트(직접 작성)",
            placeholder="예: '학교 복도에서 로봇이 분리수거를 돕는 장면, 평면 일러스트'",
            key=f"lab_prompt_{idx}"
        )

        gen_key = f"lab_gen_count_{idx}"
        if gen_key not in st.session_state:
            st.session_state[gen_key] = 0

        if st.button("이미지 생성", key=f"lab_gen_btn_{idx}"):
            if st.session_state[gen_key] >= 3:
                st.warning("이 단계에서는 최대 3회 생성 가능.")
            elif not lab_prompt.strip():
                st.warning("프롬프트를 입력하세요.")
            elif looks_like_pii(lab_prompt):
                st.warning("개인정보 포함 가능. 삭제 후 다시 작성.")
            else:
                with st.spinner("이미지 생성 중..."):
                    url = generate_image(lab_prompt.strip())
                if url:
                    st.session_state[f"lab_img_url_{idx}"] = url
                    st.session_state[gen_key] += 1
                else:
                    st.warning("이미지 생성 실패.")

        lab_img_url = st.session_state.get(f"lab_img_url_{idx}")
        if lab_img_url:
            st.image(lab_img_url, caption="학생 생성 이미지(실습)")

            stance = st.radio(
                "이 이미지의 저작권(또는 권리)은 누구에게 있다고 생각?",
                [
                    "학생(프롬프트 작성자)",
                    "AI 서비스 제공자",
                    "누구도 아님/저작권 없음",
                    "학습데이터 원저작자(참고자료 만든 사람)",
                    "공동/기타"
                ],
                key=f"stance_{idx}"
            )

            reasoning = st.text_area(
                "이유(근거) 2~4문장",
                key=f"reasoning_{idx}",
                placeholder="예: 내가 아이디어를 내고 프롬프트를 바꿔가며 결과를 만들었기 때문에..."
            )

            if st.button("토론 기록 제출", key=f"submit_discussion_{idx}"):
                if not reasoning.strip():
                    st.warning("이유를 작성하세요.")
                elif looks_like_pii(reasoning):
                    st.warning("개인정보 포함 가능. 삭제 후 다시 작성.")
                else:
                    image_id = save_student_generated_image(
                        lesson_id=lesson_id,
                        step_index=idx,
                        student_key=st.session_state.student_key,
                        prompt=lab_prompt.strip(),
                        image_url=lab_img_url
                    )
                    save_copyright_discussion(
                        lesson_id=lesson_id,
                        step_index=idx,
                        student_key=st.session_state.student_key,
                        stance=stance,
                        reasoning=reasoning.strip(),
                        image_id=image_id
                    )
                    st.success("제출 완료.")

                    # (RAG) 토론 촉진 질문 생성(저작권 KB 활용)
                    if kb.count() > 0:
                        discuss_prompt = (
                            f"저작권(권리) 토론 촉진:\n"
                            f"학생 입장: {stance}\n"
                            f"학생 근거: {reasoning.strip()}\n"
                            f"요구: (1) 확인 질문 2개 (2) 반대 관점 반론 1개 (3) 다음 활동 1개를 2~4개 항목 개조식으로."
                        )
                        helper = ask_gpt_text_rag(
                            openai_client=client,
                            collection=kb,
                            system_persona=SYSTEM_PERSONA,
                            user_prompt=discuss_prompt,
                            audience="student_56",
                            ethical_issue="copyright",
                            top_k=4
                        )
                        st.session_state[f"copyright_helper_{idx}"] = helper

            # 토론 촉진 질문 표시
            helper_text = st.session_state.get(f"copyright_helper_{idx}")
            if helper_text:
                st.subheader("🧠 토론 촉진 질문(근거 기반)")
                st.info(helper_text)

            # 집계 표시
            stats = get_copyright_stats(lesson_id, idx)
            if stats:
                df = pd.DataFrame(stats, columns=["stance", "count"]).set_index("stance")
                st.subheader("📊 같은 단계 집계(입장별)")
                st.bar_chart(df)

                samples = get_recent_discussions(lesson_id, idx, limit=5)
                if samples:
                    st.subheader("🗣️ 최근 의견 샘플")
                    for s in samples:
                        st.markdown(f"- **{s['stance']}**: {s['reasoning']}")

    st.divider()
    if st.button("다음 단계 >"):
        st.session_state.current_step += 1
        st.session_state.chat_history = []
        st.rerun()
