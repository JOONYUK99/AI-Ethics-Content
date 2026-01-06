import streamlit as st
from openai import OpenAI
import json
import base64
import requests
from datetime import datetime

# =========================================================
# 1) Page config
# =========================================================
st.set_page_config(page_title="AI 윤리 교육", page_icon="🤖", layout="wide")

# =========================================================
# 2) OpenAI client
# =========================================================
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ API 키 오류: secrets.toml을 확인하세요.")
    st.stop()

# =========================================================
# 3) System persona (dry / bullet style)
# =========================================================
SYSTEM_PERSONA = """
당신은 AI 윤리 튜터입니다.
감정을 배제하고, 질문에 대해 핵심만 '단답형' 혹은 '개조식'으로 대답하세요.
인사말(안녕, 반가워)과 서술어(~입니다, ~해요)를 생략하세요.
예시: "선택 A의 윤리적 문제는 무엇인가?" -> "다수의 이익을 위해 소수를 희생하는 공리주의적 딜레마 발생."
"""

# =========================================================
# 4) Helpers / Functions
# =========================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_json_load(s: str):
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        # try extracting a JSON object portion
        try:
            a = s.find("{")
            b = s.rfind("}")
            if a != -1 and b != -1 and b > a:
                return json.loads(s[a:b+1])
        except Exception:
            return None
    return None

def ask_gpt_text(prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=st.session_state.text_model,
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        return (resp.choices[0].message.content or "").strip() or "응답 불가."
    except Exception:
        return "응답 불가."

def ask_gpt_json_object(prompt: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model=st.session_state.text_model,
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = safe_json_load(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def generate_scenario_3steps(topic: str) -> dict:
    prompt = f"""
주제 '{topic}'의 3단계 딜레마 시나리오를 생성.
반드시 JSON만 출력.
최상위 키: scenario (리스트, 길이=3)
각 원소 키: story, choice_a, choice_b
조건:
- 초등 고학년 수준
- 과도한 폭력/공포 배제
- 선택 A/B는 서로 다른 가치가 충돌하도록
"""
    data = ask_gpt_json_object(prompt)
    scenario = data.get("scenario", [])
    if not isinstance(scenario, list):
        return {"scenario": []}
    cleaned = []
    for s in scenario[:3]:
        if not isinstance(s, dict):
            continue
        cleaned.append({
            "story": str(s.get("story", "")).strip(),
            "choice_a": str(s.get("choice_a", "")).strip(),
            "choice_b": str(s.get("choice_b", "")).strip(),
        })
    # ensure exactly 3 if possible
    return {"scenario": cleaned if len(cleaned) == 3 else cleaned}

def regenerate_single_step(topic: str, step_index_1based: int) -> dict | None:
    prompt = f"""
주제 '{topic}'의 {step_index_1based}단계 딜레마를 다시 작성.
반드시 JSON만 출력.
키: story, choice_a, choice_b
조건:
- 초등 고학년 수준
- 과도한 폭력/공포 배제
- 선택 A/B 가치 충돌 명확
"""
    data = ask_gpt_json_object(prompt)
    if not all(k in data for k in ("story", "choice_a", "choice_b")):
        return None
    return {
        "story": str(data.get("story", "")).strip(),
        "choice_a": str(data.get("choice_a", "")).strip(),
        "choice_b": str(data.get("choice_b", "")).strip(),
    }

def feedback_with_tags(story: str, choice: str, reason: str) -> dict:
    prompt = f"""
상황: {story}
선택: {choice}
이유: {reason}

반드시 JSON만 출력.
키:
- tags: 문자열 리스트 (최대 3개)
- summary: 1줄 요약
- feedback: 단답형 피드백

tags 후보:
프라이버시, 공정성, 책임, 안전, 투명성, 존엄성, 데이터보호, 편향, 설명가능성
"""
    data = ask_gpt_json_object(prompt)
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip()][:3]
    summary = str(data.get("summary", "")).strip()
    fb = str(data.get("feedback", "")).strip() or "응답 불가."
    return {"tags": tags, "summary": summary, "feedback": fb}

@st.cache_data(show_spinner=False)
def generate_image_bytes_cached(prompt: str, image_model: str) -> bytes | None:
    """
    Returns image bytes reliably:
    1) try b64_json
    2) fallback to url and download
    """
    full_prompt = f"Minimalist, flat design illustration, educational context: {prompt}"

    # 1) b64_json
    try:
        r = client.images.generate(
            model=image_model,
            prompt=full_prompt,
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
        b64 = getattr(r.data[0], "b64_json", None)
        if b64:
            return base64.b64decode(b64)
    except Exception:
        pass

    # 2) url fallback
    try:
        r = client.images.generate(
            model=image_model,
            prompt=full_prompt,
            size="1024x1024",
            n=1,
        )
        url = getattr(r.data[0], "url", None)
        if not url:
            return None
        resp = requests.get(url, timeout=25)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None

def compute_report(logs: list[dict]):
    tag_counts = {}
    step_choice_counts = {}  # {step: {choice: count}}
    for row in logs:
        tags = row.get("tags", [])
        if isinstance(tags, list):
            for t in tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        step = row.get("step")
        choice = row.get("choice")
        if isinstance(step, int) and isinstance(choice, str) and choice.strip():
            step_choice_counts.setdefault(step, {})
            step_choice_counts[step][choice] = step_choice_counts[step].get(choice, 0) + 1
    return tag_counts, step_choice_counts

def reset_student_progress(keep_logs: bool = True):
    st.session_state.current_step = 0
    st.session_state.tutorial_done = False
    st.session_state.tutorial_step = 1
    st.session_state.chat_history = []
    st.session_state.confirm_student_reset = False

    st.session_state.tutorial_choice = ""
    st.session_state.tutorial_reason = ""
    st.session_state.tutorial_img_prompt = ""
    st.session_state.tutorial_img_bytes = None

    if not keep_logs:
        st.session_state.logs = []

# =========================================================
# 5) Session state init
# =========================================================
if "scenario" not in st.session_state or not isinstance(st.session_state.scenario, dict):
    st.session_state.scenario = {"scenario": []}

default_keys = {
    "analysis": "",
    "current_step": 0,
    "chat_history": [],
    "topic": "",
    "tutorial_done": False,
    "tutorial_step": 1,

    "text_model": "gpt-4o",
    "image_model": "dall-e-3",
    "show_images_default": True,

    "logs": [],
    "student_name": "",

    "confirm_student_reset": False,

    # ✅ tutorial for students (new)
    "tutorial_choice": "",
    "tutorial_reason": "",
    "tutorial_img_prompt": "",
    "tutorial_img_bytes": None,
}
for k, v in default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# 6) Sidebar
# =========================================================
st.sidebar.title("🤖 AI 윤리 학습")

with st.sidebar.expander("⚙️ 설정", expanded=False):
    st.session_state.text_model = st.selectbox(
        "텍스트 모델",
        options=["gpt-4o", "gpt-4o-mini"],
        index=0 if st.session_state.text_model == "gpt-4o" else 1,
        key="sb_text_model",
    )
    st.session_state.image_model = st.selectbox(
        "이미지 모델",
        options=["dall-e-3"],
        index=0,
        key="sb_image_model",
    )
    st.session_state.show_images_default = st.checkbox(
        "학생 모드: 이미지 기본 표시",
        value=st.session_state.show_images_default,
        key="sb_show_images_default",
    )

st.sidebar.divider()

if st.sidebar.button("⚠️ 앱 전체 초기화(완전 초기화)", key="sb_hard_reset"):
    st.session_state.clear()
    st.rerun()

mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"], key="sb_mode")

# Student tools in sidebar
if mode == "🙋‍♂️ 학생용":
    st.sidebar.subheader("🙋‍♂️ 학생 도구")

    st.session_state.student_name = st.sidebar.text_input(
        "이름(선택)",
        value=st.session_state.student_name,
        key="sb_student_name",
    )

    if st.sidebar.button("연습 다시하기(튜토리얼)", key="sb_restart_tutorial"):
        reset_student_progress(keep_logs=True)
        st.rerun()

    if not st.session_state.confirm_student_reset:
        if st.sidebar.button("진행 초기화(학생)", key="sb_student_reset_req"):
            st.session_state.confirm_student_reset = True
            st.rerun()
    else:
        st.sidebar.warning("정말 초기화?")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("초기화 확정", key="sb_student_reset_confirm"):
                reset_student_progress(keep_logs=True)
                st.rerun()
        with c2:
            if st.button("취소", key="sb_student_reset_cancel"):
                st.session_state.confirm_student_reset = False
                st.rerun()

    if st.session_state.logs:
        st.sidebar.download_button(
            "학습 로그 다운로드(JSON)",
            data=json.dumps(st.session_state.logs, ensure_ascii=False, indent=2),
            file_name="ethics_class_log.json",
            mime="application/json",
            key="sb_logs_download",
        )

# =========================================================
# 7) Main: Teacher mode
# =========================================================
if mode == "👨‍🏫 교사용":
    st.header("🛠️ 수업 생성")

    input_topic = st.text_input("주제 입력", value=st.session_state.topic, key="teacher_topic_input")

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("생성 시작", key="teacher_generate"):
            if not input_topic.strip():
                st.warning("주제 필요.")
            else:
                with st.spinner("데이터 생성 중..."):
                    st.session_state.topic = input_topic.strip()
                    st.session_state.scenario = generate_scenario_3steps(st.session_state.topic)
                    st.session_state.analysis = ask_gpt_text(
                        f"주제 '{st.session_state.topic}'의 핵심 가치, 교과, 목표를 개조식으로 요약."
                    )
                    st.session_state.current_step = 0
                    st.success("생성 완료.")

    with colB:
        if st.session_state.scenario.get("scenario"):
            pack = {
                "topic": st.session_state.topic,
                "analysis": st.session_state.analysis,
                "scenario": st.session_state.scenario.get("scenario", []),
            }
            st.download_button(
                "시나리오/분석 다운로드(JSON)",
                data=json.dumps(pack, ensure_ascii=False, indent=2),
                file_name="ethics_class_package.json",
                mime="application/json",
                key="teacher_pack_download",
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
                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"**🅰️ 선택:** {step.get('choice_a', '')}")
                with c2:
                    st.warning(f"**🅱️ 선택:** {step.get('choice_b', '')}")

        st.divider()
        st.subheader("✏️ 시나리오 편집 / 단계별 재생성")

        for i, step in enumerate(scenario_data):
            with st.expander(f"{i+1}단계 편집", expanded=False):
                story_val = st.text_area("상황(story)", value=step.get("story", ""), key=f"edit_story_{i}")
                a_val = st.text_input("선택 A(choice_a)", value=step.get("choice_a", ""), key=f"edit_a_{i}")
                b_val = st.text_input("선택 B(choice_b)", value=step.get("choice_b", ""), key=f"edit_b_{i}")

                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    if st.button("저장", key=f"teacher_save_{i}"):
                        st.session_state.scenario["scenario"][i] = {
                            "story": story_val,
                            "choice_a": a_val,
                            "choice_b": b_val,
                        }
                        st.success("저장 완료.")
                with c2:
                    if st.button("이 단계만 재생성", key=f"teacher_regen_{i}"):
                        with st.spinner("재생성 중..."):
                            new_step = regenerate_single_step(st.session_state.topic, i + 1)
                            if new_step:
                                st.session_state.scenario["scenario"][i] = new_step
                                st.success("재생성 완료.")
                                st.rerun()
                            else:
                                st.error("재생성 실패.")
                with c3:
                    if st.button("분석(가치/목표) 다시 생성", key=f"teacher_regen_analysis_{i}"):
                        with st.spinner("분석 생성 중..."):
                            st.session_state.analysis = ask_gpt_text(
                                f"주제 '{st.session_state.topic}'의 핵심 가치, 교과, 목표를 개조식으로 요약."
                            )
                            st.success("분석 갱신 완료.")
                            st.rerun()

        st.divider()
        st.subheader("📈 학습 로그 리포트(현재 세션)")

        if not st.session_state.logs:
            st.caption("아직 학생 제출 로그 없음.")
        else:
            tag_counts, step_choice_counts = compute_report(st.session_state.logs)

            with st.container(border=True):
                st.markdown("#### 태그(가치) 빈도")
                if tag_counts:
                    st.bar_chart(tag_counts)
                else:
                    st.caption("태그 데이터 없음.")

            with st.container(border=True):
                st.markdown("#### 단계별 선택 빈도")
                rows = []
                for step_no in sorted(step_choice_counts.keys()):
                    for choice_text, cnt in step_choice_counts[step_no].items():
                        rows.append({"step": step_no, "choice": choice_text, "count": cnt})
                if rows:
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.caption("선택 데이터 없음.")

            st.download_button(
                "학습 로그 다운로드(JSON)",
                data=json.dumps(st.session_state.logs, ensure_ascii=False, indent=2),
                file_name="ethics_class_log.json",
                mime="application/json",
                key="teacher_logs_download",
            )

# =========================================================
# 8) Main: Student mode
# =========================================================
else:
    # --------------------------
    # Tutorial (Guideline)
    # --------------------------
    if not st.session_state.tutorial_done:
        st.header("🎒 연습")
        st.progress(st.session_state.tutorial_step / 3)

        # 1) choice practice
        if st.session_state.tutorial_step == 1:
            st.subheader("1. 선택 연습")
            st.caption("목표: A/B 중 하나를 선택해보기(연습용)")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("A 선택", key="tut_choose_a"):
                    st.session_state.tutorial_choice = "A"
                    st.toast("선택: A")
                    st.session_state.tutorial_step = 2
                    st.rerun()
            with c2:
                if st.button("B 선택", key="tut_choose_b"):
                    st.session_state.tutorial_choice = "B"
                    st.toast("선택: B")
                    st.session_state.tutorial_step = 2
                    st.rerun()

        # 2) input practice
        elif st.session_state.tutorial_step == 2:
            st.subheader("2. 입력 연습")
            st.caption("목표: 간단한 문장 입력 후 전송")

            st.write(f"방금 선택: {st.session_state.tutorial_choice or '미선택'}")
            st.session_state.tutorial_reason = st.text_area(
                "이유(연습)",
                value=st.session_state.tutorial_reason,
                placeholder="예: A를 선택한 이유는 ...",
                key="tut_reason",
            )

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("전송", key="tut_send"):
                    if st.session_state.tutorial_reason.strip():
                        st.toast("입력 완료")
                        st.session_state.tutorial_step = 3
                        st.rerun()
                    else:
                        st.warning("이유 입력 필요.")
            with c2:
                if st.button("이전", key="tut_back_1"):
                    st.session_state.tutorial_step = 1
                    st.rerun()

        # 3) prompt -> image test
        elif st.session_state.tutorial_step == 3:
            st.subheader("3. 프롬프트 이미지 테스트")
            st.caption("목표: 간단한 프롬프트 입력 → 이미지 생성 확인")

            st.session_state.tutorial_img_prompt = st.text_input(
                "이미지 프롬프트(연습)",
                value=st.session_state.tutorial_img_prompt,
                placeholder="예: robot teacher in classroom",
                key="tut_img_prompt",
            )

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("이미지 생성", key="tut_gen_img"):
                    if st.session_state.tutorial_img_prompt.strip():
                        with st.spinner("생성..."):
                            st.session_state.tutorial_img_bytes = generate_image_bytes_cached(
                                st.session_state.tutorial_img_prompt.strip(),
                                st.session_state.image_model,
                            )
                        if not st.session_state.tutorial_img_bytes:
                            st.error("이미지 생성 실패(텍스트만 진행 가능).")
                    else:
                        st.warning("프롬프트 입력 필요.")
            with c2:
                if st.button("예시 넣기", key="tut_example"):
                    st.session_state.tutorial_img_prompt = "A student learning AI ethics with a robot tutor in a classroom"
                    st.rerun()
            with c3:
                if st.button("이전", key="tut_back_2"):
                    st.session_state.tutorial_step = 2
                    st.rerun()

            if st.session_state.tutorial_img_bytes:
                st.image(st.session_state.tutorial_img_bytes, width=360)

                if st.button("수업 입장", key="tut_enter"):
                    st.session_state.tutorial_done = True
                    st.rerun()

    # --------------------------
    # Real class
    # --------------------------
    else:
        steps = st.session_state.scenario.get("scenario", [])

        if not steps:
            st.warning("데이터 없음. 교사용 탭에서 생성 필요.")
            if st.button("새로고침", key="student_refresh"):
                st.rerun()
        else:
            idx = st.session_state.current_step
            total = len(steps)

            top1, top2, top3 = st.columns([2, 1, 1])
            with top1:
                st.caption(f"주제: {st.session_state.topic or '미지정'}")
            with top2:
                show_img = st.toggle("이미지 보기", value=st.session_state.show_images_default, key="student_show_img")
            with top3:
                if st.button("처음으로(학생)", key="student_to_tutorial"):
                    reset_student_progress(keep_logs=True)
                    st.rerun()

            if idx >= total:
                st.success("수업 종료.")
                if st.session_state.logs:
                    st.download_button(
                        "학습 로그 다운로드(JSON)",
                        data=json.dumps(st.session_state.logs, ensure_ascii=False, indent=2),
                        file_name="ethics_class_log.json",
                        mime="application/json",
                        key="student_logs_download_end",
                    )
                if st.button("처음으로(다시)", key="student_restart_all"):
                    reset_student_progress(keep_logs=True)
                    st.rerun()
            else:
                data = steps[idx]
                st.progress((idx + 1) / total)
                st.subheader(f"단계 {idx+1}")

                # Image per step (session key)
                img_key = f"img_bytes_{idx}"
                if show_img:
                    if img_key not in st.session_state:
                        with st.spinner("이미지 생성..."):
                            st.session_state[img_key] = generate_image_bytes_cached(
                                data.get("story", "AI ethics"),
                                st.session_state.image_model,
                            )
                    if st.session_state.get(img_key):
                        st.image(st.session_state[img_key])
                    else:
                        st.caption("이미지 생성 실패(텍스트만 진행).")

                st.info(data.get("story", "내용 없음"))

                with st.form(f"form_{idx}"):
                    sel = st.radio("선택", [data.get("choice_a", "A"), data.get("choice_b", "B")], key=f"radio_{idx}")
                    reason = st.text_area("이유", key=f"reason_{idx}")
                    submitted = st.form_submit_button("제출")

                if submitted:
                    if not reason.strip():
                        st.warning("이유 입력 필요.")
                    else:
                        with st.spinner("분석..."):
                            fb = feedback_with_tags(data.get("story", ""), sel, reason)

                        with st.container(border=True):
                            st.markdown("#### 🧾 제출 요약")
                            if fb.get("tags"):
                                st.write("태그:", ", ".join(fb["tags"]))
                            if fb.get("summary"):
                                st.write("요약:", fb["summary"])
                            st.write("피드백:", fb.get("feedback", "응답 불가."))

                        # chat history (display)
                        st.session_state.chat_history.append({"role": "user", "content": f"[{sel}] {reason}"})
                        st.session_state.chat_history.append({"role": "assistant", "content": fb.get("feedback", "응답 불가.")})

                        # logs
                        st.session_state.logs.append({
                            "timestamp": now_str(),
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

                if st.session_state.chat_history:
                    st.divider()
                    for msg in st.session_state.chat_history:
                        role = "assistant" if msg["role"] == "assistant" else "user"
                        st.chat_message(role).write(msg["content"])

                    if st.button("다음 단계 >", key=f"next_{idx}"):
                        st.session_state.current_step += 1
                        st.session_state.chat_history = []
                        st.rerun()
