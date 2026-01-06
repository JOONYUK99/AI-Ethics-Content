import streamlit as st
from openai import OpenAI
import json
import base64
from datetime import datetime

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 윤리 교육", page_icon="🤖", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ API 키 오류: secrets.toml을 확인하세요.")
    st.stop()

# --- 3. 시스템 페르소나 (단답형/건조한 말투) ---
SYSTEM_PERSONA = """
당신은 AI 윤리 튜터입니다.
감정을 배제하고, 질문에 대해 핵심만 '단답형' 혹은 '개조식'으로 대답하세요.
인사말(안녕, 반가워)과 서술어(~입니다, ~해요)를 생략하세요.
예시: "선택 A의 윤리적 문제는 무엇인가?" -> "다수의 이익을 위해 소수를 희생하는 공리주의적 딜레마 발생."
"""

# --- 4. 유틸 / 주요 함수 ---

def _safe_json_load(s: str):
    """JSON 파싱 안정화: 앞뒤 잡문 제거/부분 추출 시도"""
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        # 첫 '{' ~ 마지막 '}' 범위만 재시도
        try:
            a = s.find("{")
            b = s.rfind("}")
            if a != -1 and b != -1 and b > a:
                return json.loads(s[a:b+1])
        except Exception:
            return None
    return None

def ask_gpt_json(prompt):
    """JSON 응답 요청 (오류 발생 시 빈 구조 반환) - scenario 형태"""
    try:
        response = client.chat.completions.create(
            model=st.session_state.text_model,
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        raw = response.choices[0].message.content.strip()
        data = _safe_json_load(raw)
        if not isinstance(data, dict):
            return {"scenario": []}
        if "scenario" not in data or not isinstance(data["scenario"], list):
            return {"scenario": []}
        return data
    except Exception:
        return {"scenario": []}

def ask_gpt_step_json(prompt):
    """단일 step JSON 응답 요청: {story, choice_a, choice_b}"""
    try:
        response = client.chat.completions.create(
            model=st.session_state.text_model,
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        raw = response.choices[0].message.content.strip()
        data = _safe_json_load(raw)
        if not isinstance(data, dict):
            return None
        if not all(k in data for k in ["story", "choice_a", "choice_b"]):
            return None
        return {
            "story": str(data.get("story", "")).strip(),
            "choice_a": str(data.get("choice_a", "")).strip(),
            "choice_b": str(data.get("choice_b", "")).strip(),
        }
    except Exception:
        return None

def ask_gpt_text(prompt):
    """텍스트 응답 요청"""
    try:
        response = client.chat.completions.create(
            model=st.session_state.text_model,
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "응답 불가."

def ask_gpt_feedback_json(story, sel, reason):
    """
    피드백 JSON:
    - tags: [프라이버시, 공정성, 책임, 안전, 투명성, 존엄성, 데이터보호, 편향, 설명가능성] 중 택
    - summary: 학생 이유 1줄 요약
    - feedback: 단답형 피드백
    """
    prompt = f"""
상황: {story}
선택: {sel}
이유: {reason}

출력은 반드시 JSON.
키:
- tags: 문자열 리스트 (최대 3개)
- summary: 1줄 요약
- feedback: 단답형 피드백

tags 후보: 프라이버시, 공정성, 책임, 안전, 투명성, 존엄성, 데이터보호, 편향, 설명가능성
"""
    try:
        response = client.chat.completions.create(
            model=st.session_state.text_model,
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        data = _safe_json_load(raw)
        if not isinstance(data, dict):
            return {"tags": [], "summary": "", "feedback": "응답 불가."}
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).strip() for t in tags if str(t).strip()][:3]
        summary = str(data.get("summary", "")).strip()
        feedback = str(data.get("feedback", "")).strip() or "응답 불가."
        return {"tags": tags, "summary": summary, "feedback": feedback}
    except Exception:
        return {"tags": [], "summary": "", "feedback": "응답 불가."}

@st.cache_data(show_spinner=False)
def generate_image_b64_cached(prompt: str, image_model: str):
    """
    이미지 생성 (b64_json)
    - DALL·E 3 URL은 만료될 수 있어 b64 방식 사용 권장
    """
    try:
        response = client.images.generate(
            model=image_model,
            prompt=f"Minimalist, flat design illustration, educational context: {prompt}",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
        b64 = response.data[0].b64_json
        return b64
    except Exception:
        return None

def b64_to_bytes(b64_str: str):
    try:
        return base64.b64decode(b64_str)
    except Exception:
        return None

def now_kst_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def compute_report(logs):
    """
    logs: list of dict
    리포트:
    - 태그 빈도
    - 단계별 선택 빈도
    """
    tag_counts = {}
    step_choice = {}  # {step: {choice_label: count}}
    for r in logs:
        step = r.get("step")
        choice = r.get("choice")
        tags = r.get("tags", [])
        if isinstance(tags, list):
            for t in tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        if step is not None and choice:
            step_choice.setdefault(step, {})
            step_choice[step][choice] = step_choice[step].get(choice, 0) + 1
    return tag_counts, step_choice


# --- 5. 세션 상태 안전한 초기화 ---

if "text_model" not in st.session_state:
    st.session_state.text_model = "gpt-4o"
if "image_model" not in st.session_state:
    st.session_state.image_model = "dall-e-3"

if "scenario" not in st.session_state or not isinstance(st.session_state.scenario, dict):
    st.session_state.scenario = {"scenario": []}

default_keys = {
    "analysis": "",
    "current_step": 0,
    "chat_history": [],
    "topic": "",
    "tutorial_done": False,
    "tutorial_step": 1,
    "tutorial_img_b64": None,
    "logs": [],  # 학습 로그 누적
    "student_name": "",
    "confirm_student_reset": False,
    "show_images_default": True,
}
for k, v in default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- 6. 사이드바 ---
st.sidebar.title("🤖 AI 윤리 학습")

# 모델/운영 옵션
with st.sidebar.expander("⚙️ 설정", expanded=False):
    st.session_state.text_model = st.selectbox(
        "텍스트 모델",
        options=["gpt-4o", "gpt-4o-mini"],
        index=0 if st.session_state.text_model == "gpt-4o" else 1
    )
    st.session_state.image_model = st.selectbox(
        "이미지 모델",
        options=["dall-e-3"],  # 운영 중 모델 추가 가능
        index=0
    )
    st.session_state.show_images_default = st.checkbox("학생 모드: 이미지 기본 표시", value=st.session_state.show_images_default)

st.sidebar.divider()

# [비상 버튼] 에러가 날 때 누르는 버튼
if st.sidebar.button("⚠️ 앱 전체 초기화(완전 초기화)"):
    st.session_state.clear()
    st.rerun()

mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"])

# 학생 모드 UX: 진행 초기화(확인)
if mode == "🙋‍♂️ 학생용":
    st.sidebar.subheader("🙋‍♂️ 학생 도구")
    st.session_state.student_name = st.sidebar.text_input("이름(선택)", value=st.session_state.student_name)

    if not st.session_state.confirm_student_reset:
        if st.sidebar.button("진행 초기화(학생)"):
            st.session_state.confirm_student_reset = True
            st.rerun()
    else:
        st.sidebar.warning("정말 초기화?")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("초기화 확정", key="confirm_reset"):
                # 학생 진행만 초기화(교사용 생성 데이터는 유지)
                st.session_state.current_step = 0
                st.session_state.tutorial_done = False
                st.session_state.tutorial_step = 1
                st.session_state.tutorial_img_b64 = None
                st.session_state.chat_history = []
                st.session_state.confirm_student_reset = False
                st.rerun()
        with c2:
            if st.button("취소", key="cancel_reset"):
                st.session_state.confirm_student_reset = False
                st.rerun()

    # 로그 다운로드(학생 측)
    if st.session_state.logs:
        st.sidebar.download_button(
            "학습 로그 다운로드(JSON)",
            data=json.dumps(st.session_state.logs, ensure_ascii=False, indent=2),
            file_name="ethics_class_log.json",
            mime="application/json"
        )

# --- 7. 메인 로직 ---

# =========================
# [교사용 모드]
# =========================
if mode == "👨‍🏫 교사용":
    st.header("🛠️ 수업 생성")

    input_topic = st.text_input("주제 입력", value=st.session_state.topic)

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("생성 시작"):
            if not input_topic:
                st.warning("주제 필요.")
            else:
                with st.spinner("데이터 생성 중..."):
                    s_prompt = f"""
주제 '{input_topic}'의 3단계 딜레마 시나리오를 생성.
반드시 JSON으로만 출력.
최상위 키: scenario (리스트, 길이=3)
각 원소 키: story, choice_a, choice_b
조건: 초등 고학년 수준, 과도한 폭력/공포 배제, 선택 A/B는 가치 충돌이 드러나게.
"""
                    result = ask_gpt_json(s_prompt)
                    st.session_state.scenario = result

                    a_prompt = f"주제 '{input_topic}'의 핵심 가치, 교과, 목표를 개조식으로 요약."
                    st.session_state.analysis = ask_gpt_text(a_prompt)

                    st.session_state.topic = input_topic
                    st.session_state.current_step = 0

                    # 이미지 캐시 키 삭제(세션 저장분)
                    keys_to_del = [k for k in st.session_state.keys() if str(k).startswith("img_b64_")]
                    for k in keys_to_del:
                        del st.session_state[k]

                    st.success("생성 완료.")

    with colB:
        if st.session_state.scenario.get("scenario"):
            # 교사용 다운로드: 시나리오 + 분석
            pack = {
                "topic": st.session_state.topic,
                "analysis": st.session_state.analysis,
                "scenario": st.session_state.scenario.get("scenario", []),
            }
            st.download_button(
                "시나리오/분석 다운로드(JSON)",
                data=json.dumps(pack, ensure_ascii=False, indent=2),
                file_name="ethics_class_package.json",
                mime="application/json"
            )

    scenario_data = st.session_state.scenario.get("scenario", [])

    if st.session_state.analysis:
        st.divider()
        st.subheader("📊 분석 결과")
        st.info(st.session_state.analysis)

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

        # ===== 추가: 시나리오 편집/부분 재생성 =====
        st.divider()
        st.subheader("✏️ 시나리오 편집 / 단계별 재생성")

        for i, step in enumerate(scenario_data):
            with st.expander(f"{i+1}단계 편집", expanded=False):
                story_val = st.text_area("상황(story)", value=step.get("story", ""), key=f"edit_story_{i}")
                a_val = st.text_input("선택 A(choice_a)", value=step.get("choice_a", ""), key=f"edit_a_{i}")
                b_val = st.text_input("선택 B(choice_b)", value=step.get("choice_b", ""), key=f"edit_b_{i}")

                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    if st.button("저장", key=f"save_step_{i}"):
                        st.session_state.scenario["scenario"][i] = {
                            "story": story_val,
                            "choice_a": a_val,
                            "choice_b": b_val
                        }
                        st.success("저장 완료.")
                with c2:
                    if st.button("이 단계만 재생성", key=f"regen_step_{i}"):
                        with st.spinner("재생성 중..."):
                            regen_prompt = f"""
주제 '{st.session_state.topic}'의 {i+1}단계 딜레마를 다시 작성.
반드시 JSON으로만 출력.
키: story, choice_a, choice_b
조건: 초등 고학년, 과도한 폭력/공포 배제, 선택 A/B 가치 충돌 명확.
"""
                            new_step = ask_gpt_step_json(regen_prompt)
                            if new_step:
                                st.session_state.scenario["scenario"][i] = new_step
                                # 해당 단계 이미지 캐시(세션 저장분) 삭제
                                imgk = f"img_b64_{i}"
                                if imgk in st.session_state:
                                    del st.session_state[imgk]
                                st.success("재생성 완료.")
                                st.rerun()
                            else:
                                st.error("재생성 실패.")
                with c3:
                    if st.button("분석(가치/목표) 다시 생성", key=f"regen_analysis"):
                        with st.spinner("분석 생성 중..."):
                            a_prompt = f"주제 '{st.session_state.topic}'의 핵심 가치, 교과, 목표를 개조식으로 요약."
                            st.session_state.analysis = ask_gpt_text(a_prompt)
                            st.success("분석 갱신 완료.")
                            st.rerun()

        # ===== 추가: 학습 로그 리포트(세션 기준) =====
        st.divider()
        st.subheader("📈 학습 로그 리포트(현재 세션)")

        if not st.session_state.logs:
            st.caption("아직 학생 제출 로그 없음.")
        else:
            tag_counts, step_choice = compute_report(st.session_state.logs)

            with st.container(border=True):
                st.markdown("#### 태그(가치) 빈도")
                if tag_counts:
                    # Streamlit 기본 차트
                    st.bar_chart(tag_counts)
                else:
                    st.caption("태그 데이터 없음.")

            with st.container(border=True):
                st.markdown("#### 단계별 선택 빈도")
                if step_choice:
                    # 표 형태로 표시
                    rows = []
                    for step_no in sorted(step_choice.keys()):
                        for choice_text, cnt in step_choice[step_no].items():
                            rows.append({"step": step_no, "choice": choice_text, "count": cnt})
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.caption("선택 데이터 없음.")

            st.download_button(
                "학습 로그 다운로드(JSON)",
                data=json.dumps(st.session_state.logs, ensure_ascii=False, indent=2),
                file_name="ethics_class_log.json",
                mime="application/json"
            )

# =========================
# [학생용 모드]
# =========================
elif mode == "🙋‍♂️ 학생용":

   # 튜토리얼
if not st.session_state.tutorial_done:
    st.header("🎒 연습")
    st.progress(st.session_state.tutorial_step / 3)

    # -------------------------
    # 1) 선택 연습
    # -------------------------
    if st.session_state.tutorial_step == 1:
        st.subheader("1. 선택 연습")
        st.caption("목표: 선택 버튼을 눌러보고, 다음 단계로 넘어가기")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("A 선택", key="tutorial_choose_a"):
                st.session_state.tutorial_choice = "A"
                st.toast("선택: A")
                st.session_state.tutorial_step = 2
                st.rerun()

        with c2:
            if st.button("B 선택", key="tutorial_choose_b"):
                st.session_state.tutorial_choice = "B"
                st.toast("선택: B")
                st.session_state.tutorial_step = 2
                st.rerun()

    # -------------------------
    # 2) 입력 연습
    # -------------------------
    elif st.session_state.tutorial_step == 2:
        st.subheader("2. 입력 연습")
        st.caption("목표: 간단한 이유를 입력하고 전송해보기")

        st.write(f"방금 선택: {st.session_state.tutorial_choice or '미선택'}")

        st.session_state.tutorial_reason = st.text_area(
            "이유(연습)",
            value=st.session_state.tutorial_reason,
            placeholder="예: A를 선택한 이유는 ...",
            key="tutorial_reason_area"
        )

        if st.button("전송", key="tutorial_send_reason"):
            if st.session_state.tutorial_reason.strip():
                st.toast("입력 완료")
                st.session_state.tutorial_step = 3
                st.rerun()
            else:
                st.warning("이유 입력 필요.")

    # -------------------------
    # 3) 이미지 생성 테스트
    # -------------------------
    elif st.session_state.tutorial_step == 3:
        st.subheader("3. 이미지 생성 테스트")
        st.caption("목표: 간단한 프롬프트를 입력하고 이미지가 생성되는지 확인")

        st.session_state.tutorial_img_prompt = st.text_input(
            "이미지 프롬프트(연습)",
            value=st.session_state.tutorial_img_prompt,
            placeholder="예: robot teacher in classroom",
            key="tutorial_img_prompt_input"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("이미지 생성", key="tutorial_gen_image"):
                if st.session_state.tutorial_img_prompt.strip():
                    with st.spinner("생성..."):
                        b64 = generate_image_b64_cached(
                            st.session_state.tutorial_img_prompt.strip(),
                            st.session_state.image_model
                        )
                        st.session_state.tutorial_img_b64 = b64
                    if not st.session_state.tutorial_img_b64:
                        st.error("이미지 생성 실패(텍스트만 진행 가능).")
                else:
                    st.warning("프롬프트 입력 필요.")

        with col2:
            if st.button("프롬프트 예시 넣기", key="tutorial_prompt_example"):
                st.session_state.tutorial_img_prompt = "A student discussing AI ethics with a robot tutor"
                st.rerun()

        if st.session_state.tutorial_img_b64:
            img_bytes = b64_to_bytes(st.session_state.tutorial_img_b64)
            if img_bytes:
                st.image(img_bytes, width=360)
            else:
                st.info("이미지 표시 불가.")

            if st.button("수업 입장", key="tutorial_enter_class"):
                st.session_state.tutorial_done = True
                st.rerun()

    # 실전 수업
    else:
        steps = st.session_state.scenario.get("scenario", [])

        if not steps:
            st.warning("데이터 없음. 교사용 탭에서 생성 필요.")
            if st.button("새로고침"):
                st.rerun()

        else:
            idx = st.session_state.current_step
            total = len(steps)

            # 상단 제어(UX)
            top1, top2, top3 = st.columns([2, 1, 1])
            with top1:
                st.caption(f"주제: {st.session_state.topic or '미지정'}")
            with top2:
                show_img = st.toggle("이미지 보기", value=st.session_state.show_images_default)
            with top3:
                if st.button("처음으로(학생)", key="student_home"):
                    st.session_state.current_step = 0
                    st.session_state.tutorial_done = False
                    st.session_state.tutorial_step = 1
                    st.session_state.tutorial_img_b64 = None
                    st.session_state.chat_history = []
                    st.rerun()

            if idx >= total:
                st.success("수업 종료.")
                if st.session_state.logs:
                    st.download_button(
                        "학습 로그 다운로드(JSON)",
                        data=json.dumps(st.session_state.logs, ensure_ascii=False, indent=2),
                        file_name="ethics_class_log.json",
                        mime="application/json"
                    )
                if st.button("처음으로(다시)"):
                    st.session_state.current_step = 0
                    st.session_state.tutorial_done = False
                    st.session_state.tutorial_step = 1
                    st.session_state.tutorial_img_b64 = None
                    st.session_state.chat_history = []
                    st.rerun()
            else:
                data = steps[idx]
                st.progress((idx + 1) / total)
                st.subheader(f"단계 {idx+1}")

                # 이미지 로딩/캐시
                img_key = f"img_b64_{idx}"

                if show_img:
                    if img_key not in st.session_state:
                        with st.spinner("이미지 생성..."):
                            st.session_state[img_key] = generate_image_b64_cached(
                                data.get("story", "AI ethics"),
                                st.session_state.image_model
                            )
                    if st.session_state.get(img_key):
                        img_bytes = b64_to_bytes(st.session_state[img_key])
                        if img_bytes:
                            st.image(img_bytes)
                        else:
                            st.info("이미지 표시 불가.")
                    else:
                        st.caption("이미지 생성 실패(텍스트만 진행).")

                st.info(data.get("story", "내용 없음"))

                # 제출 폼
                with st.form(f"form_{idx}"):
                    sel = st.radio("선택", [data.get("choice_a", "A"), data.get("choice_b", "B")])
                    reason = st.text_area("이유")
                    submitted = st.form_submit_button("제출")

                    if submitted:
                        if reason:
                            with st.spinner("분석..."):
                                fb = ask_gpt_feedback_json(data.get("story", ""), sel, reason)

                            # 요약/피드백 카드
                            with st.container(border=True):
                                st.markdown("#### 🧾 제출 요약")
                                if fb.get("tags"):
                                    st.write("태그:", ", ".join(fb["tags"]))
                                if fb.get("summary"):
                                    st.write("요약:", fb["summary"])
                                st.write("피드백:", fb.get("feedback", ""))

                            # 채팅 히스토리(표시용)
                            st.session_state.chat_history.append({"role": "user", "content": f"[{sel}] {reason}"})
                            st.session_state.chat_history.append({"role": "assistant", "content": fb.get("feedback", "응답 불가.")})

                            # 로그 저장(다운로드/교사 리포트용)
                            st.session_state.logs.append({
                                "timestamp": now_kst_str(),
                                "student_name": st.session_state.student_name,
                                "topic": st.session_state.topic,
                                "step": idx + 1,
                                "story": data.get("story", ""),
                                "choice": sel,
                                "reason": reason,
                                "tags": fb.get("tags", []),
                                "summary": fb.get("summary", ""),
                                "feedback": fb.get("feedback", ""),
                            })
                        else:
                            st.warning("이유 입력 필요.")

                # 채팅 표시
                if st.session_state.chat_history:
                    st.divider()
                    for msg in st.session_state.chat_history:
                        role = "assistant" if msg["role"] == "assistant" else "user"
                        st.chat_message(role).write(msg["content"])

                    # 다음 단계 이동
                    if st.button("다음 단계 >", key=f"next_{idx}"):
                        st.session_state.current_step += 1
                        st.session_state.chat_history = []
                        st.rerun()

