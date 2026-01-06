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
# 2) Fixed model configuration (설정 UI 제거: 여기서 고정)
# =========================================================
TEXT_MODEL = "gpt-4o"
IMAGE_MODEL = "dall-e-3"

# 이미지에 글자(영어/한글 포함) 나오지 않게 강제
NO_TEXT_IMAGE_PREFIX = (
    "Minimalist, flat design illustration, educational context. "
    "ABSOLUTELY NO TEXT: no words, no letters, no numbers, no captions, no subtitles, "
    "no watermarks, no logos, no signs, no posters with writing. "
    "Only 그림/도형/사물. "
)

# =========================================================
# 3) OpenAI client
# =========================================================
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ API 키 오류: secrets.toml을 확인하세요.")
    st.stop()

# =========================================================
# 4) System persona (dry / bullet style)
# =========================================================
SYSTEM_PERSONA = """
당신은 AI 윤리 튜터입니다.
감정을 배제하고, 질문에 대해 핵심만 '단답형' 혹은 '개조식'으로 대답하세요.
인사말(안녕, 반가워)과 서술어(~입니다, ~해요)를 생략하세요.
예시: "선택 A의 윤리적 문제는 무엇인가?" -> "다수의 이익을 위해 소수를 희생하는 공리주의적 딜레마 발생."
"""

# =========================================================
# 5) Helpers / Functions
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
            model=TEXT_MODEL,
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
            model=TEXT_MODEL,
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
        cleaned.append(
            {
                "story": str(s.get("story", "")).strip(),
                "choice_a": str(s.get("choice_a", "")).strip(),
                "choice_b": str(s.get("choice_b", "")).strip(),
            }
        )
    return {"scenario": cleaned}

def regenerate_single_step(topic: str, step_index_1based: int):
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

def feedback_with_tags(story: str, choice: str, reason: str, extra_context: str = "") -> dict:
    prompt = f"""
상황: {story}
{extra_context}
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
def generate_image_bytes_cached(user_prompt: str, image_model: str):
    """
    이미지 bytes 반환.
    - 글자(영어/한글) 나오지 않도록 강제 프리픽스 추가
    - b64_json 우선, 실패 시 url 다운로드
    """
    full_prompt = f"{NO_TEXT_IMAGE_PREFIX}{user_prompt}"

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

def compute_report(logs):
    tag_counts = {}
    step_choice_counts = {}
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

def clear_generated_images_from_session():
    to_del = [
        k for k in st.session_state.keys()
        if str(k).startswith("img_bytes_")
        or str(k).startswith("user_img_bytes_")
        or str(k).startswith("tutorial_img_bytes")
    ]
    for k in to_del:
        del st.session_state[k]

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

def generate_copyright_example_lesson():
    """
    저작권 예시 수업을 '요구 흐름'대로 생성
    - 학생이 프롬프트로 이미지를 출력하게 되는 상황 부여
    - 그 이미지의 저작권/사용권/책임이 누구에게 있는지 토론 흐름
    - JSON으로 topic/analysis/teacher_guide/scenario 생성
    실패 시 하드코딩 예시로 fallback
    """
    prompt = """
초등 고학년 대상 '저작권 + 생성형 AI 이미지' 예시 수업 생성.

반드시 JSON만 출력.
키:
- topic: 문자열
- analysis: 개조식 문자열(핵심가치/교과연계/목표/핵심질문 포함)
- teacher_guide: 개조식 문자열(도입-활동-토론-정리 흐름, 교사용 질문 3개, 간단 평가 기준 포함)
- scenario: 리스트(길이=3)
  - 각 원소 키: story, choice_a, choice_b

필수 흐름(시나리오에 반영):
1) [상황 부여] 학생이 학교 과제/학급 포스터/발표자료를 위해 '프롬프트를 직접 입력해' AI 이미지 1장을 생성함.
   이어서 질문: "이 이미지의 저작권/사용 권한은 누구에게 있을까?"
2) 친구/팀원이 그 이미지를 쓰거나 수정해 쓰고 싶어 함(허락/출처표기/용도 제한 이슈).
3) 축제/스티커 판매/홍보물 등 '상업적 이용' 또는 '공유 범위 확장' 상황(약관/규정 확인, 대체자료 고려).

조건:
- 폭력/공포 배제
- 선택지는 가치 충돌이 명확(책임 vs 편의, 공정 vs 이익 등)
- 법 조항 단정 금지(“국가마다 다를 수 있음/약관 확인 필요” 관점으로 표현)
"""
    data = ask_gpt_json_object(prompt)
    ok = (
        isinstance(data, dict)
        and isinstance(data.get("topic"), str)
        and isinstance(data.get("analysis"), str)
        and isinstance(data.get("teacher_guide"), str)
        and isinstance(data.get("scenario"), list)
    )
    if ok:
        # scenario sanitize
        cleaned = []
        for s in data["scenario"][:3]:
            if isinstance(s, dict) and all(k in s for k in ("story", "choice_a", "choice_b")):
                cleaned.append({
                    "story": str(s.get("story", "")).strip(),
                    "choice_a": str(s.get("choice_a", "")).strip(),
                    "choice_b": str(s.get("choice_b", "")).strip(),
                })
        if len(cleaned) == 3:
            return data["topic"].strip(), data["analysis"].strip(), {"scenario": cleaned}, data["teacher_guide"].strip()

    # fallback (요구 흐름을 만족하는 고정 예시)
    topic = "저작권과 생성형 AI 이미지: 이 그림의 권리는 누구에게?"
    analysis = "\n".join([
        "- 핵심 가치: 책임, 공정성, 투명성",
        "- 교과 연계: 도덕(권리/책임), 실과(디지털 자료 활용)",
        "- 목표:",
        "  - 프롬프트로 만든 이미지의 권리/사용 이슈를 약관/규칙 관점으로 설명",
        "  - 허락·출처표기·용도 구분(과제/공유/판매) 판단",
        "- 핵심 질문:",
        "  - 프롬프트를 쓴 학생이 ‘저작권자’라고 말할 수 있을까?",
        "  - 플랫폼 약관/학교 규칙 확인이 왜 필요할까?",
        "  - 친구가 쓰거나 판매할 때 기준이 왜 달라질까?"
    ])
    teacher_guide = "\n".join([
        "수업 흐름(예시)",
        "1) 도입(5분): 'AI가 만든 그림의 권리는 누구에게?' 질문",
        "2) 활동(10분): 학생이 프롬프트 입력 → 이미지 1장 생성(글자 없는 그림만)",
        "3) 토론(20분): 아래 3단계 딜레마 순서대로 선택+이유 말하기",
        "4) 정리(5분): 다음 행동 1개(약관 확인/출처 표기/허락 받기 등)",
        "",
        "교사용 질문(예시)",
        "- 프롬프트 작성은 어떤 점에서 ‘창작 기여’일까?",
        "- 허락과 출처표기는 왜 분리해서 생각해야 할까?",
        "- 판매/홍보처럼 목적이 바뀌면 왜 더 신중해야 할까?",
        "",
        "평가(간단)",
        "- 근거 제시(규칙/약관/공정/책임 관점)",
        "- 타인 권리 고려(허락/표기/용도 제한)",
        "- 대안 제시(직접 제작/라이선스 명확 자료 사용/확인 후 사용)"
    ])
    scenario = [
        {
            "story": "학교 과제로 '학급 발표자료 표지'가 필요해 프롬프트를 직접 입력해 AI 이미지 1장을 만들었다. 친구가 묻는다: '이 이미지 저작권(사용 권한)은 누구에게 있어?'",
            "choice_a": "내가 프롬프트를 썼으니 내 것. 마음대로 써도 된다고 말한다.",
            "choice_b": "확실치 않음. 도구 약관/학교 규칙을 확인하고, 출처 표기와 사용 범위를 정한다."
        },
        {
            "story": "친구가 그 이미지를 자기 발표 자료에 쓰고 싶다고 한다. 일부 수정도 하겠다고 한다. 허락/출처표기/용도 제한을 어떻게 할까?",
            "choice_a": "조건부 허락: 출처(도구/프롬프트) 표기 + 용도(발표만) 제한 후 허락한다.",
            "choice_b": "허락하지 않는다: 내 이미지이니 다른 사람이 수정/사용하면 안 된다고 말한다."
        },
        {
            "story": "축제 때 그 이미지를 스티커로 만들어 판매하자는 의견이 나왔다. 상업적 이용이 가능한지(약관/규정) 확신이 없다.",
            "choice_a": "바로 판매한다: 어차피 우리가 만든 이미지라고 판단한다.",
            "choice_b": "판매 보류: 약관/규정 확인 후, 필요하면 직접 그린 그림이나 라이선스가 명확한 자료로 대체한다."
        },
    ]
    return topic, analysis, {"scenario": scenario}, teacher_guide


# =========================================================
# 6) Session state init
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

    "logs": [],
    "student_name": "",
    "confirm_student_reset": False,

    # lesson metadata
    "lesson_type": "general",      # general | copyright
    "teacher_guide": "",

    # tutorial for students
    "tutorial_choice": "",
    "tutorial_reason": "",
    "tutorial_img_prompt": "",
    "tutorial_img_bytes": None,
}
for k, v in default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# 7) Sidebar (설정 UI 없음)
# =========================================================
st.sidebar.title("🤖 AI 윤리 학습")

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
            if st.sidebar.button("초기화 확정", key="sb_student_reset_confirm"):
                reset_student_progress(keep_logs=True)
                st.rerun()
        with c2:
            if st.sidebar.button("취소", key="sb_student_reset_cancel"):
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
# 8) Main: Teacher mode
# =========================================================
if mode == "👨‍🏫 교사용":
    st.header("🛠️ 수업 생성")

    input_topic = st.text_input("주제 입력", value=st.session_state.topic, key="teacher_topic_input")
    colA, colB, colC = st.columns([1, 1, 1])

    with colA:
        if st.button("생성 시작", key="teacher_generate"):
            if not input_topic.strip():
                st.warning("주제 필요.")
            else:
                with st.spinner("데이터 생성 중..."):
                    st.session_state.topic = input_topic.strip()
                    st.session_state.lesson_type = "general"
                    st.session_state.teacher_guide = ""

                    st.session_state.scenario = generate_scenario_3steps(st.session_state.topic)
                    st.session_state.analysis = ask_gpt_text(
                        f"주제 '{st.session_state.topic}'의 핵심 가치, 교과, 목표를 개조식으로 요약."
                    )
                    st.session_state.current_step = 0
                    clear_generated_images_from_session()
                    st.success("생성 완료.")

    with colB:
        if st.button("예시 수업 생성(저작권)", key="teacher_example_copyright"):
            with st.spinner("예시 수업 생성 중..."):
                topic, analysis, scenario_obj, guide = generate_copyright_example_lesson()
                st.session_state.topic = topic
                st.session_state.analysis = analysis
                st.session_state.scenario = scenario_obj
                st.session_state.lesson_type = "copyright"
                st.session_state.teacher_guide = guide
                st.session_state.current_step = 0
                clear_generated_images_from_session()
                st.success("예시 수업 생성 완료.")

    with colC:
        if st.session_state.scenario.get("scenario"):
            pack = {
                "topic": st.session_state.topic,
                "lesson_type": st.session_state.lesson_type,
                "analysis": st.session_state.analysis,
                "teacher_guide": st.session_state.teacher_guide,
                "scenario": st.session_state.scenario.get("scenario", []),
            }
            st.download_button(
                "시나리오/분석 다운로드(JSON)",
                data=json.dumps(pack, ensure_ascii=False, indent=2),
                file_name="ethics_class_package.json",
                mime="application/json",
                key="teacher_pack_download",
            )

    if st.session_state.teacher_guide:
        st.divider()
        with st.expander("📌 교사용 수업 안내(저작권 예시)", expanded=True):
            st.text(st.session_state.teacher_guide)

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
                                st.session_state.lesson_type = "general"
                                st.session_state.teacher_guide = ""
                                clear_generated_images_from_session()
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
# 9) Main: Student mode
# =========================================================
else:
    # --------------------------
    # Tutorial (Guideline)
    # --------------------------
    if not st.session_state.tutorial_done:
        st.header("🎒 연습")
        st.progress(st.session_state.tutorial_step / 3)

        if st.session_state.tutorial_step == 1:
            st.subheader("1. 선택 연습")
            st.caption("목표: A/B 중 하나 선택")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("A 선택", key="tut_choose_a"):
                    st.session_state.tutorial_choice = "A"
                    st.session_state.tutorial_step = 2
                    st.rerun()
            with c2:
                if st.button("B 선택", key="tut_choose_b"):
                    st.session_state.tutorial_choice = "B"
                    st.session_state.tutorial_step = 2
                    st.rerun()

        elif st.session_state.tutorial_step == 2:
            st.subheader("2. 입력 연습")
            st.caption("목표: 이유 1문장 입력 후 전송")

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
                        st.session_state.tutorial_step = 3
                        st.rerun()
                    else:
                        st.warning("이유 입력 필요.")
            with c2:
                if st.button("이전", key="tut_back_1"):
                    st.session_state.tutorial_step = 1
                    st.rerun()

        elif st.session_state.tutorial_step == 3:
            st.subheader("3. 프롬프트 이미지 테스트")
            st.caption("목표: 프롬프트 입력 → 이미지 생성 확인 (글자 없이 그림만 나오게)")

            st.session_state.tutorial_img_prompt = st.text_input(
                "이미지 프롬프트(연습)",
                value=st.session_state.tutorial_img_prompt,
                placeholder="예: cute robot teacher and students in classroom",
                key="tut_img_prompt",
            )

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("이미지 생성", key="tut_gen_img"):
                    if st.session_state.tutorial_img_prompt.strip():
                        with st.spinner("생성..."):
                            st.session_state.tutorial_img_bytes = generate_image_bytes_cached(
                                st.session_state.tutorial_img_prompt.strip(),
                                IMAGE_MODEL,
                            )
                        if not st.session_state.tutorial_img_bytes:
                            st.error("이미지 생성 실패.")
                    else:
                        st.warning("프롬프트 입력 필요.")
            with c2:
                if st.button("예시 넣기", key="tut_example"):
                    st.session_state.tutorial_img_prompt = "a friendly robot and a child studying with books, no text"
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

            top1, top2 = st.columns([3, 1])
            with top1:
                st.caption(f"주제: {st.session_state.topic or '미지정'}")
            with top2:
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

                # ✅ 항상 이미지 표시 (토글 제거)
                img_key = f"img_bytes_{idx}"
                if img_key not in st.session_state:
                    with st.spinner("이미지 생성..."):
                        st.session_state[img_key] = generate_image_bytes_cached(
                            data.get("story", "AI ethics"),
                            IMAGE_MODEL,
                        )
                if st.session_state.get(img_key):
                    st.image(st.session_state[img_key])
                else:
                    st.caption("이미지 생성 실패(텍스트만 진행).")

                st.info(data.get("story", "내용 없음"))

                # ✅ 저작권 수업: '상황 부여 → 학생이 프롬프트로 이미지 출력 → 저작권 토론' 활동을 명시
                extra_context = ""
                if st.session_state.lesson_type == "copyright":
                    st.divider()
                    st.subheader("🎨 프롬프트로 이미지 제작")
                    st.caption("규칙: 글자/문장/로고 없이 그림만 나오게 프롬프트 작성")

                    user_prompt_key = f"user_img_prompt_{idx}"
                    user_img_key = f"user_img_bytes_{idx}"

                    user_prompt = st.text_input(
                        "내 프롬프트",
                        value=st.session_state.get(user_prompt_key, ""),
                        placeholder="예: colorful mascot character holding a paintbrush, no text",
                        key=user_prompt_key,
                    )

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        if st.button("내 이미지 생성", key=f"user_img_gen_{idx}"):
                            if user_prompt.strip():
                                with st.spinner("내 이미지 생성..."):
                                    st.session_state[user_img_key] = generate_image_bytes_cached(
                                        user_prompt.strip(),
                                        IMAGE_MODEL
                                    )
                            else:
                                st.warning("프롬프트 입력 필요.")
                    with c2:
                        if st.button("내 이미지 지우기", key=f"user_img_clear_{idx}"):
                            if user_img_key in st.session_state:
                                del st.session_state[user_img_key]
                            st.rerun()

                    if st.session_state.get(user_img_key):
                        st.image(st.session_state[user_img_key], caption="내가 만든 이미지(토론 기준 이미지)")
                        extra_context = f"학생이 생성에 사용한 프롬프트: {user_prompt.strip()}"
                    else:
                        extra_context = "학생이 이미지 생성(프롬프트 입력) 후 토론한다고 가정."

                with st.form(f"form_{idx}"):
                    sel = st.radio(
                        "선택",
                        [data.get("choice_a", "A"), data.get("choice_b", "B")],
                        key=f"radio_{idx}",
                    )
                    reason = st.text_area("이유", key=f"reason_{idx}")
                    submitted = st.form_submit_button("제출")

                if submitted:
                    if not reason.strip():
                        st.warning("이유 입력 필요.")
                    else:
                        with st.spinner("분석..."):
                            fb = feedback_with_tags(
                                data.get("story", ""),
                                sel,
                                reason,
                                extra_context=extra_context,
                            )

                        with st.container(border=True):
                            st.markdown("#### 🧾 제출 요약")
                            if fb.get("tags"):
                                st.write("태그:", ", ".join(fb["tags"]))
                            if fb.get("summary"):
                                st.write("요약:", fb["summary"])
                            st.write("피드백:", fb.get("feedback", "응답 불가."))

                        st.session_state.chat_history.append({"role": "user", "content": f"[{sel}] {reason}"})
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": fb.get("feedback", "응답 불가.")}
                        )

                        st.session_state.logs.append(
                            {
                                "timestamp": now_str(),
                                "student_name": st.session_state.student_name,
                                "topic": st.session_state.topic,
                                "lesson_type": st.session_state.lesson_type,
                                "step": idx + 1,
                                "story": data.get("story", ""),
                                "choice": sel,
                                "reason": reason,
                                "tags": fb.get("tags", []),
                                "summary": fb.get("summary", ""),
                                "feedback": fb.get("feedback", ""),
                                "student_image_prompt": st.session_state.get(f"user_img_prompt_{idx}", "") if st.session_state.lesson_type == "copyright" else "",
                            }
                        )

                if st.session_state.chat_history:
                    st.divider()
                    for msg in st.session_state.chat_history:
                        role = "assistant" if msg["role"] == "assistant" else "user"
                        st.chat_message(role).write(msg["content"])

                    if st.button("다음 단계 >", key=f"next_{idx}"):
                        st.session_state.current_step += 1
                        st.session_state.chat_history = []
                        st.rerun()
