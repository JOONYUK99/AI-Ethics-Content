import streamlit as st
from openai import OpenAI
import json
import base64
import requests
from datetime import datetime
import hashlib
import numpy as np
from pathlib import Path

# =========================================================
# 1) Page config
# =========================================================
st.set_page_config(page_title="AI 윤리 교육 (RAG)", page_icon="🤖", layout="wide")

# =========================================================
# 2) Model configuration
# =========================================================
TEXT_MODEL = "gpt-4o"
IMAGE_MODEL = "dall-e-3"
EMBED_MODEL = "text-embedding-3-small"

# ✅ 앱 내부(reference.txt)만 참고하는 RAG (항상 ON)
REFERENCE_PATH = "reference.txt"   # repo 루트에 reference.txt 두기
RAG_TOP_K = 4                      # 고정 (UI로 노출 안 함)

# ✅ 이미지에 글자(영어/한글 포함) 나오지 않게 강제
NO_TEXT_IMAGE_PREFIX = (
    "Minimalist, flat design illustration, educational context. "
    "ABSOLUTELY NO TEXT: no words, no letters, no numbers, no captions, no subtitles, "
    "no watermarks, no logos, no signs, no posters with writing. "
    "No text-like shapes. Only 그림/도형/사물. "
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
# 5) Helpers
# =========================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _clip(s: str, max_len: int = 1600) -> str:
    s = (s or "").strip()
    return s[:max_len] + ("…" if len(s) > max_len else "")

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
                return json.loads(s[a:b + 1])
        except Exception:
            return None
    return None

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

def normalize_steps(raw_steps):
    """
    Supports:
    - legacy dilemma steps: {"story","choice_a","choice_b"}
    - mixed steps:
      - image_task: {"type":"image_task","story","prompt_goal","question"}
      - dilemma: {"type":"dilemma","story","choice_a","choice_b"}
      - discussion: {"type":"discussion","story","question"}
    """
    if not isinstance(raw_steps, list):
        return []

    steps = []
    for s in raw_steps:
        if not isinstance(s, dict):
            continue

        # legacy dilemma -> convert
        if "type" not in s and all(k in s for k in ("story", "choice_a", "choice_b")):
            steps.append({
                "type": "dilemma",
                "story": str(s.get("story", "")).strip(),
                "choice_a": str(s.get("choice_a", "")).strip(),
                "choice_b": str(s.get("choice_b", "")).strip(),
            })
            continue

        t = str(s.get("type", "")).strip().lower()
        story = str(s.get("story", "")).strip()

        if t == "image_task":
            steps.append({
                "type": "image_task",
                "story": story,
                "prompt_goal": str(s.get("prompt_goal", "")).strip(),
                "prompt_hint": str(s.get("prompt_hint", "")).strip(),
                "question": str(s.get("question", "")).strip(),
            })
        elif t == "discussion":
            steps.append({
                "type": "discussion",
                "story": story,
                "question": str(s.get("question", "")).strip(),
            })
        else:
            steps.append({
                "type": "dilemma",
                "story": story,
                "choice_a": str(s.get("choice_a", "")).strip(),
                "choice_b": str(s.get("choice_b", "")).strip(),
            })

    return steps

def normalize_analysis(analysis_any):
    if isinstance(analysis_any, dict):
        return {
            "ethics_standards": analysis_any.get("ethics_standards", []),
            "curriculum_alignment": analysis_any.get("curriculum_alignment", []),
            "lesson_content": analysis_any.get("lesson_content", []),
        }
    if isinstance(analysis_any, str) and analysis_any.strip():
        return {"ethics_standards": [], "curriculum_alignment": [], "lesson_content": [analysis_any.strip()]}
    return {"ethics_standards": [], "curriculum_alignment": [], "lesson_content": []}

def render_bullets(items):
    if not items:
        st.caption("내용 없음.")
        return
    if isinstance(items, list):
        for x in items:
            st.write(f"- {str(x)}")
        return
    st.write(str(items))

def render_analysis_box(analysis_dict):
    a = normalize_analysis(analysis_dict)
    st.subheader("📊 분석 결과")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 인공지능 윤리기준")
        render_bullets(a.get("ethics_standards", []))
    with c2:
        st.markdown("### 연계 교육과정")
        render_bullets(a.get("curriculum_alignment", []))
    with c3:
        st.markdown("### 수업 내용")
        render_bullets(a.get("lesson_content", []))

# =========================================================
# 6) RAG (internal reference.txt)
# =========================================================
def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def chunk_text(text: str, max_chars: int = 900, overlap: int = 160):
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    # Split on blank lines first
    parts = []
    buf = []
    for line in text.split("\n"):
        if line.strip() == "":
            if buf:
                parts.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        parts.append("\n".join(buf).strip())

    # Re-pack into chunks
    chunks = []
    current = ""
    for p in parts:
        if len(current) + len(p) + 2 <= max_chars:
            current = (current + "\n\n" + p).strip() if current else p
        else:
            if current:
                chunks.append(current)
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = min(len(p), start + max_chars)
                    chunks.append(p[start:end])
                    start = max(0, end - overlap)
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)

    # Add overlap
    final = []
    for i, c in enumerate(chunks):
        if i == 0:
            final.append(c)
        else:
            tail = chunks[i - 1][-overlap:] if overlap > 0 else ""
            merged = (tail + "\n" + c).strip() if tail else c
            final.append(merged)
    return [x.strip() for x in final if x.strip()]

@st.cache_data(show_spinner=False)
def load_reference_text_cached(path_str: str, mtime: float) -> str:
    p = Path(path_str)
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if len(txt) > 1_200_000:
        txt = txt[:1_200_000]
    return txt

@st.cache_data(show_spinner=False)
def build_rag_index_cached(path_str: str, embed_model: str, mtime: float):
    txt = load_reference_text_cached(path_str, mtime)
    if not txt.strip():
        return {"chunks": [], "emb": None, "norms": None, "content_hash": ""}

    content_hash = sha256_text(txt)
    chunks = chunk_text(txt, max_chars=900, overlap=160)
    if not chunks:
        return {"chunks": [], "emb": None, "norms": None, "content_hash": content_hash}

    try:
        resp = client.embeddings.create(model=embed_model, input=chunks)
        vecs = [d.embedding for d in resp.data]
        emb = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1) + 1e-8
        return {"chunks": chunks, "emb": emb, "norms": norms, "content_hash": content_hash}
    except Exception:
        return {"chunks": chunks, "emb": None, "norms": None, "content_hash": content_hash}

def get_rag_index():
    p = Path(REFERENCE_PATH)
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    return build_rag_index_cached(REFERENCE_PATH, EMBED_MODEL, mtime)

def rag_retrieve(query: str, index: dict, top_k: int = RAG_TOP_K) -> str:
    query = (query or "").strip()
    if not query or not index or not index.get("chunks") or index.get("emb") is None:
        return ""
    try:
        q = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
        qv = np.array(q, dtype=np.float32)
        qn = np.linalg.norm(qv) + 1e-8

        emb = index["emb"]
        norms = index["norms"]
        sims = (emb @ qv) / (norms * qn)
        k = max(1, min(int(top_k), len(index["chunks"])))
        top_idx = np.argsort(-sims)[:k].tolist()

        ctx = "\n\n---\n\n".join(index["chunks"][i].strip() for i in top_idx)
        return _clip(ctx, 2200)
    except Exception:
        return ""

# =========================================================
# 7) Lesson generation (RAG-applied)
# =========================================================
def generate_scenario_3steps(topic: str, rag_ctx: str = "") -> dict:
    prompt = f"""
주제 '{topic}'의 3단계 딜레마 시나리오를 생성.
반드시 JSON만 출력.
최상위 키: scenario (리스트, 길이=3)
각 원소 키: story, choice_a, choice_b

[외부 참고자료(reference.txt 발췌)]
{rag_ctx if rag_ctx else "- 없음"}

조건:
- 초등 고학년 수준
- 과도한 폭력/공포 배제
- 선택 A/B는 서로 다른 가치가 충돌하도록
"""
    data = ask_gpt_json_object(prompt)
    return {"scenario": normalize_steps(data.get("scenario", []))}

def generate_mixed_lesson(topic: str, rag_ctx: str = "") -> tuple[str, dict, dict, str]:
    prompt = f"""
초등 고학년 대상 AI 윤리 수업(혼합형) 생성.
주제: '{topic}'

[외부 참고자료(reference.txt 발췌)]
{rag_ctx if rag_ctx else "- 없음"}

반드시 JSON만 출력.
키:
- topic: 문자열
- analysis: 객체
  - ethics_standards: 문자열 리스트
  - curriculum_alignment: 문자열 리스트
  - lesson_content: 문자열 리스트
- teacher_guide: 개조식 문자열(도입-활동-토론-정리, 교사용 질문 3개, 간단 평가 기준 포함)
- scenario: 리스트(길이 4~5)

scenario의 각 단계 type:
type="image_task" | "dilemma" | "discussion"

규칙:
- 최소 1개는 image_task 포함.
- image_task 키: type, story, prompt_goal, question (+prompt_hint 선택)
- dilemma 키: type, story, choice_a, choice_b
- discussion 키: type, story, question
- 폭력/공포 배제
- 법 조항 단정 금지(약관/학교 규칙 확인 필요 관점)
"""
    data = ask_gpt_json_object(prompt)

    t = str(data.get("topic", topic)).strip() or topic
    analysis = normalize_analysis(data.get("analysis", {}))
    guide = str(data.get("teacher_guide", "")).strip()
    steps = normalize_steps(data.get("scenario", []))

    if not any(s.get("type") == "image_task" for s in steps):
        steps = ([{
            "type": "image_task",
            "story": "프롬프트를 입력해 수업 주제와 관련된 이미지를 1장 생성한다.",
            "prompt_goal": "수업 주제를 상징하는 그림 만들기",
            "prompt_hint": "사람/사물/배경 3요소(글자 없음)",
            "question": "이 활동에서 중요한 점 1개를 말하라.",
        }] + steps)

    return t, analysis, {"scenario": steps}, guide

def generate_copyright_lesson(rag_ctx: str = "") -> tuple[str, dict, dict, str]:
    prompt = f"""
초등 고학년 대상 '저작권 + 생성형 AI 이미지' 수업 생성.

[외부 참고자료(reference.txt 발췌)]
{rag_ctx if rag_ctx else "- 없음"}

반드시 JSON만 출력.
키:
- topic
- analysis: 객체
  - ethics_standards: 문자열 리스트
  - curriculum_alignment: 문자열 리스트
  - lesson_content: 문자열 리스트
- teacher_guide
- scenario: 리스트(길이 4)

필수 구성(순서 중요):
1) image_task 1개 (첫 단계)
   - 학생이 프롬프트를 직접 입력해 이미지를 1장 생성하는 상황
   - 질문: "이 이미지의 저작권/사용권은 누구에게 있는가?"
2) dilemma 2개
   - 사용/수정 요청(허락/출처표기/용도 제한)
   - 공유 확장 또는 상업적 이용(약관/규정 확인, 대체자료 고려)
3) discussion 1개 (마지막)
   - '우리 반 규칙' 만들기 질문

규칙:
- 폭력/공포 배제
- 선택지는 가치 충돌 명확
- 법 조항 단정 금지(국가/플랫폼/약관/학교 규칙 확인 필요 관점)
"""
    data = ask_gpt_json_object(prompt)

    topic = str(data.get("topic", "저작권과 생성형 AI 이미지: 권리는 누구에게?")).strip()
    analysis = normalize_analysis(data.get("analysis", {}))
    guide = str(data.get("teacher_guide", "")).strip()
    steps = normalize_steps(data.get("scenario", []))

    if len(steps) < 4 or steps[0].get("type") != "image_task":
        steps = [
            {
                "type": "image_task",
                "story": "학교 과제로 발표자료 표지 그림이 필요하다. 너는 프롬프트를 직접 입력해 AI로 이미지를 1장 생성했다. 친구가 묻는다: '이 이미지의 저작권(사용 권한)은 누구에게 있어?'",
                "prompt_goal": "발표자료 표지에 쓸 ‘학습/학교’ 느낌 그림",
                "prompt_hint": "사람/사물/배경 3요소(글자 없음)",
                "question": "이 이미지의 권리·책임은 누구에게 있다고 생각하는가? 이유 1개",
            },
            {
                "type": "dilemma",
                "story": "친구가 그 이미지를 자기 발표 자료에 쓰고 싶다고 한다. 일부 수정도 하겠다고 한다. 허락/출처표기/용도 제한을 어떻게 할까?",
                "choice_a": "조건부 허락: 출처(도구/프롬프트) 표기 + 발표용으로만 허락",
                "choice_b": "허락하지 않음: 다른 사람이 사용/수정하면 안 된다고 말함",
            },
            {
                "type": "dilemma",
                "story": "축제 때 그 이미지를 스티커로 만들어 판매하자는 의견이 나왔다. 상업적 이용 가능 여부(약관/학교 규칙)가 확실치 않다.",
                "choice_a": "바로 판매한다: 우리가 만들었으니 문제 없다고 판단",
                "choice_b": "판매 보류: 약관/규칙 확인 후, 필요하면 직접 제작/라이선스 명확 자료로 대체",
            },
            {
                "type": "discussion",
                "story": "정리 토론: 앞으로 우리 반에서 AI로 만든 이미지를 사용할 때 지켜야 할 규칙을 정한다.",
                "question": "‘허락’, ‘출처표기’, ‘사용 목적(과제/공유/판매)’ 기준으로 우리 반 규칙 3가지를 적어라.",
            },
        ]

    if not guide:
        guide = "\n".join([
            "수업 흐름(예시)",
            "1) 도입: ‘AI가 만든 그림의 권리는 누구에게?’",
            "2) 활동: 프롬프트 입력 → 이미지 1장 생성(글자 없는 그림만)",
            "3) 토론: 선택형 딜레마 + 정리 토론(우리 반 규칙)",
            "4) 정리: 다음 행동 1개(약관 확인/출처표기/허락 받기)",
        ])

    return topic, analysis, {"scenario": steps}, guide

# =========================================================
# 8) Feedback (RAG + teacher rubric)
# =========================================================
def get_teacher_feedback_context() -> str:
    ctx = (st.session_state.get("teacher_feedback_context") or "").strip()
    return _clip(ctx, 900) if ctx else ""

def feedback_with_tags(step_story: str, answer_text: str, extra_context: str = "", mode: str = "generic", rag_ctx: str = "") -> dict:
    teacher_ctx = get_teacher_feedback_context()

    if mode == "copyright":
        tag_candidates = "저작권, 출처표기, 허락, 책임, 투명성, 공정성, 약관확인"
        caution = "주의: 법 조항 단정 금지. 플랫폼 약관/학교 규칙/사용 목적 확인 필요."
    else:
        tag_candidates = "프라이버시, 공정성, 책임, 안전, 투명성, 데이터보호, 편향, 설명가능성"
        caution = "주의: 단정적 사실 주장 금지. 상황 근거 중심."

    prompt = f"""
상황/활동: {step_story}

[외부 참고자료(reference.txt 발췌)]
{rag_ctx if rag_ctx else "- 없음"}

[교사 관점/평가기준(반영)]
{teacher_ctx if teacher_ctx else "- (교사 입력 없음)"}

[추가 맥락]
{_clip(extra_context, 700)}

[학생 답]
{answer_text}

{caution}

반드시 JSON만 출력.
키:
- tags: 문자열 리스트(최대 3개)
- summary: 1줄 요약
- feedback: 단답형 피드백(핵심만, 교사 기준 + reference.txt 관점을 반영)

tags 후보:
{tag_candidates}
"""
    data = ask_gpt_json_object(prompt)

    tags = data.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip()][:3]

    summary = str(data.get("summary", "")).strip()
    fb = str(data.get("feedback", "")).strip() or "응답 불가."
    return {"tags": tags, "summary": summary, "feedback": fb}

# =========================================================
# 9) Image generation
# =========================================================
@st.cache_data(show_spinner=False)
def generate_image_bytes_cached(user_prompt: str, image_model: str):
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

# =========================================================
# 10) Reports / Reset
# =========================================================
def compute_report(logs):
    tag_counts = {}
    step_type_counts = {}
    for row in logs:
        tags = row.get("tags", [])
        if isinstance(tags, list):
            for t in tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        stype = row.get("step_type", "")
        if stype:
            step_type_counts[stype] = step_type_counts.get(stype, 0) + 1
    return tag_counts, step_type_counts

def clear_generated_images_from_session():
    to_del = [k for k in st.session_state.keys()
              if str(k).startswith("img_bytes_") or str(k).startswith("user_img_bytes_")]
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

    st.session_state.last_student_image_prompt = ""
    st.session_state.last_student_image_done = False

    if not keep_logs:
        st.session_state.logs = []

# =========================================================
# 11) Session state init
# =========================================================
if "scenario" not in st.session_state or not isinstance(st.session_state.scenario, dict):
    st.session_state.scenario = {"scenario": []}

default_keys = {
    "analysis": {"ethics_standards": [], "curriculum_alignment": [], "lesson_content": []},
    "current_step": 0,
    "chat_history": [],
    "topic": "",
    "tutorial_done": False,
    "tutorial_step": 1,

    "logs": [],
    "student_name": "",
    "confirm_student_reset": False,

    "lesson_type": "general",      # general | copyright
    "teacher_guide": "",
    "teacher_feedback_context": "",

    # tutorial
    "tutorial_choice": "",
    "tutorial_reason": "",
    "tutorial_img_prompt": "",
    "tutorial_img_bytes": None,

    # last activity context
    "last_student_image_prompt": "",
    "last_student_image_done": False,
}
for k, v in default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.session_state.analysis = normalize_analysis(st.session_state.analysis)

# =========================================================
# 12) Sidebar
# =========================================================
st.sidebar.title("🤖 AI 윤리 학습 (RAG)")

if st.sidebar.button("⚠️ 앱 전체 초기화(완전 초기화)"):
    st.session_state.clear()
    st.rerun()

mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"], key="sb_mode")

# ✅ 요청사항: 사진에 있던 RAG UI(체크박스/URL/Top-K/새로고침/상태 버튼) 전부 제거
# ✅ 대신 “reference.txt를 내부에서 자동 참고(RAG 적용)”만 표시
rag_index = get_rag_index()
if rag_index and rag_index.get("chunks"):
    st.sidebar.caption(f"📚 RAG 적용: internal reference.txt (Top-K={RAG_TOP_K})")
else:
    st.sidebar.caption("📚 RAG 적용: internal reference.txt")
    if not Path(REFERENCE_PATH).exists():
        st.sidebar.warning("reference.txt 없음(레포에 포함 필요)")

if mode == "🙋‍♂️ 학생용":
    st.sidebar.subheader("🙋‍♂️ 학생 도구")
    st.session_state.student_name = st.sidebar.text_input("이름(선택)", value=st.session_state.student_name)
    if st.sidebar.button("연습 다시하기(튜토리얼)"):
        reset_student_progress(keep_logs=True)
        st.rerun()

    if not st.session_state.confirm_student_reset:
        if st.sidebar.button("진행 초기화(학생)"):
            st.session_state.confirm_student_reset = True
            st.rerun()
    else:
        st.sidebar.warning("정말 초기화?")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.sidebar.button("초기화 확정"):
                reset_student_progress(keep_logs=True)
                st.rerun()
        with c2:
            if st.sidebar.button("취소"):
                st.session_state.confirm_student_reset = False
                st.rerun()

    if st.session_state.logs:
        st.sidebar.download_button(
            "학습 로그 다운로드(JSON)",
            data=json.dumps(st.session_state.logs, ensure_ascii=False, indent=2),
            file_name="ethics_class_log.json",
            mime="application/json",
        )

# =========================================================
# 13) Teacher mode
# =========================================================
if mode == "👨‍🏫 교사용":
    st.header("🛠️ 수업 생성 (RAG: reference.txt 자동 적용)")

    with st.expander("📘 교사용 가이드라인(사용법)", expanded=True):
        st.markdown(
            """
**1) reference.txt 준비**
- GitHub 레포에 `reference.txt`를 앱 코드와 함께 포함
- 앱은 내부 파일을 자동으로 검색/근거로 활용(RAG)

**2) 수업 생성**
- 주제 입력 → 아래 버튼으로 생성
- 생성 시 reference.txt에서 관련 내용을 검색(Top-K 고정)해 수업/분석에 반영

**3) 교사 기준 반영**
- `교사 피드백 기준/관점`에 평가 기준 입력
- 학생 피드백에 교사 기준 + reference.txt 관점이 함께 반영됨

**4) 배포**
- `수업 패키지 다운로드(JSON)`로 배포
- 학생은 학생용 화면에서 JSON 업로드로 수업 불러오기
"""
        )

    with st.expander("🧑‍🏫 교사 피드백 기준/관점(학생 피드백에 반영)", expanded=False):
        st.session_state.teacher_feedback_context = st.text_area(
            "교사 기준 입력",
            value=st.session_state.teacher_feedback_context,
            height=140,
            placeholder="예) 1) 허락/출처표기/사용목적 구분  2) 약관/학교 규칙 확인 언급  3) 대체안 제시",
        )

    input_topic = st.text_input("주제 입력", value=st.session_state.topic)

    def teacher_rag_ctx(topic: str) -> str:
        if not rag_index:
            return ""
        q = f"{topic} 인공지능 윤리기준 교육과정 수업 설계"
        return rag_retrieve(q, rag_index, top_k=RAG_TOP_K)

    colA, colB, colC, colD = st.columns([1, 1, 1, 1])

    with colA:
        if st.button("딜레마 3단계 생성"):
            if not input_topic.strip():
                st.warning("주제 필요.")
            else:
                with st.spinner("생성 중..."):
                    st.session_state.topic = input_topic.strip()
                    st.session_state.lesson_type = "general"
                    st.session_state.teacher_guide = ""

                    rag_ctx = teacher_rag_ctx(st.session_state.topic)
                    st.session_state.scenario = generate_scenario_3steps(st.session_state.topic, rag_ctx=rag_ctx)

                    # analysis 생성(근거 포함)
                    steps = st.session_state.scenario.get("scenario", [])
                    short_steps = json.dumps(
                        [{"type": s.get("type", ""), "story": _clip(s.get("story", ""), 220)} for s in steps],
                        ensure_ascii=False
                    )
                    a_prompt = f"""
교사용 분석 결과 생성. 반드시 JSON.
주제: '{st.session_state.topic}'
[외부 참고자료(reference.txt 발췌)]
{rag_ctx if rag_ctx else "- 없음"}
[수업 단계 요약]
{short_steps}

키:
- ethics_standards: 문자열 리스트(5개 내외)
- curriculum_alignment: 문자열 리스트(4~6개, 초등 5~6 실과/도덕 중심)
- lesson_content: 문자열 리스트(4~6개, 도입-활동-토론-정리)
"""
                    st.session_state.analysis = normalize_analysis(ask_gpt_json_object(a_prompt))

                    st.session_state.current_step = 0
                    clear_generated_images_from_session()
                    st.success("생성 완료.")

    with colB:
        if st.button("혼합 수업 생성(활동+선택)"):
            if not input_topic.strip():
                st.warning("주제 필요.")
            else:
                with st.spinner("혼합 수업 구성 중..."):
                    rag_ctx = teacher_rag_ctx(input_topic.strip())
                    t, analysis, scenario_obj, guide = generate_mixed_lesson(input_topic.strip(), rag_ctx=rag_ctx)
                    st.session_state.topic = t
                    st.session_state.analysis = analysis
                    st.session_state.scenario = scenario_obj
                    st.session_state.lesson_type = "general"
                    st.session_state.teacher_guide = guide
                    st.session_state.current_step = 0
                    clear_generated_images_from_session()
                    st.success("생성 완료.")

    with colC:
        if st.button("예시 수업 생성(저작권)"):
            with st.spinner("저작권 수업 구성 중..."):
                rag_ctx = ""
                if rag_index:
                    rag_ctx = rag_retrieve("저작권 생성형 AI 이미지 출처표기 허락 사용목적 약관", rag_index, top_k=RAG_TOP_K)
                t, analysis, scenario_obj, guide = generate_copyright_lesson(rag_ctx=rag_ctx)
                st.session_state.topic = t
                st.session_state.analysis = analysis
                st.session_state.scenario = scenario_obj
                st.session_state.lesson_type = "copyright"
                st.session_state.teacher_guide = guide
                st.session_state.current_step = 0
                clear_generated_images_from_session()
                st.success("예시 수업 생성 완료.")

    with colD:
        if st.session_state.scenario.get("scenario"):
            pack = {
                "topic": st.session_state.topic,
                "lesson_type": st.session_state.lesson_type,
                "analysis": st.session_state.analysis,
                "teacher_guide": st.session_state.teacher_guide,
                "teacher_feedback_context": st.session_state.teacher_feedback_context,
                "rag": {"enabled": True, "source": "reference.txt", "top_k": RAG_TOP_K},
                "scenario": st.session_state.scenario.get("scenario", []),
            }
            st.download_button(
                "수업 패키지 다운로드(JSON)",
                data=json.dumps(pack, ensure_ascii=False, indent=2),
                file_name="ethics_class_package.json",
                mime="application/json",
            )

    if st.session_state.teacher_guide:
        st.divider()
        with st.expander("📌 교사용 수업 안내(자동 생성)", expanded=True):
            st.text(st.session_state.teacher_guide)

    if st.session_state.analysis:
        st.divider()
        render_analysis_box(st.session_state.analysis)

    steps = st.session_state.scenario.get("scenario", [])
    if steps:
        st.divider()
        st.subheader("📜 수업 단계 미리보기")
        for i, step in enumerate(steps):
            with st.container(border=True):
                st.markdown(f"### 🔹 단계 {i+1} ({step.get('type','')})")
                st.markdown(f"**📖 상황/활동:** {step.get('story','')}")
                if step.get("type") == "image_task":
                    if step.get("prompt_goal"):
                        st.write("🎯 프롬프트 목표:", step.get("prompt_goal"))
                    if step.get("prompt_hint"):
                        st.write("💡 힌트:", step.get("prompt_hint"))
                    if step.get("question"):
                        st.write("🗣️ 질문:", step.get("question"))
                elif step.get("type") == "discussion":
                    st.write("🗣️ 질문:", step.get("question", ""))
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.success(f"**🅰️ 선택:** {step.get('choice_a', '')}")
                    with c2:
                        st.warning(f"**🅱️ 선택:** {step.get('choice_b', '')}")

        st.divider()
        st.subheader("📈 학습 로그 리포트(현재 세션)")
        if not st.session_state.logs:
            st.caption("아직 학생 제출 로그 없음.")
        else:
            tag_counts, step_type_counts = compute_report(st.session_state.logs)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 태그 빈도")
                st.bar_chart(tag_counts) if tag_counts else st.caption("태그 없음.")
            with c2:
                st.markdown("#### 활동 유형 분포")
                st.bar_chart(step_type_counts) if step_type_counts else st.caption("데이터 없음.")

# =========================================================
# 14) Student mode
# =========================================================
else:
    with st.expander("📦 수업 불러오기(교사가 만든 JSON 업로드)", expanded=False):
        up = st.file_uploader("ethics_class_package.json", type=["json"])
        if up is not None:
            try:
                pack = json.load(up)
                st.session_state.topic = pack.get("topic", "")
                st.session_state.lesson_type = pack.get("lesson_type", "general")
                st.session_state.analysis = normalize_analysis(pack.get("analysis", {}))
                st.session_state.teacher_guide = pack.get("teacher_guide", "")
                st.session_state.teacher_feedback_context = pack.get("teacher_feedback_context", "")
                st.session_state.scenario = {"scenario": normalize_steps(pack.get("scenario", []))}
                st.session_state.current_step = 0
                st.session_state.chat_history = []
                clear_generated_images_from_session()
                st.success("수업 로드 완료")
                st.rerun()
            except Exception:
                st.error("JSON 로드 실패")

    # --------------------------
    # Tutorial
    # --------------------------
    if not st.session_state.tutorial_done:
        st.header("🎒 연습")
        st.progress(st.session_state.tutorial_step / 3)

        if st.session_state.tutorial_step == 1:
            st.subheader("1. 선택 연습")
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
            st.caption("글자 없이 그림만 출력")
            st.session_state.tutorial_img_prompt = st.text_input(
                "이미지 프롬프트(연습)",
                value=st.session_state.tutorial_img_prompt,
                placeholder="예: friendly robot and child studying with books",
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
                    st.session_state.tutorial_img_prompt = "cute robot teacher and students in classroom"
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
            st.warning("데이터 없음. 교사용에서 생성 후 JSON 업로드 필요.")
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
                step = steps[idx]
                st.progress((idx + 1) / total)
                st.subheader(f"단계 {idx + 1} ({step.get('type', '')})")

                # ✅ 항상 시나리오 이미지 표시
                img_key = f"img_bytes_{idx}"
                if img_key not in st.session_state:
                    with st.spinner("이미지 생성..."):
                        st.session_state[img_key] = generate_image_bytes_cached(
                            step.get("story", "AI ethics"),
                            IMAGE_MODEL,
                        )
                if st.session_state.get(img_key):
                    st.image(st.session_state[img_key])

                st.info(step.get("story", "내용 없음"))

                def step_rag_ctx():
                    if not rag_index:
                        return ""
                    q = f"{st.session_state.topic} {step.get('story','')} 윤리 기준 핵심"
                    return rag_retrieve(q, rag_index, top_k=RAG_TOP_K)

                # A) IMAGE TASK
                if step.get("type") == "image_task":
                    st.divider()
                    st.subheader("🎨 프롬프트로 이미지 만들기")
                    if step.get("prompt_goal"):
                        st.caption(f"목표: {step.get('prompt_goal')}")
                    if step.get("prompt_hint"):
                        st.caption(f"힌트: {step.get('prompt_hint')}")

                    user_prompt_key = f"user_img_prompt_{idx}"
                    user_img_key = f"user_img_bytes_{idx}"

                    user_prompt = st.text_input(
                        "내 프롬프트",
                        value=st.session_state.get(user_prompt_key, ""),
                        placeholder="예: a happy child and a robot painting together",
                        key=user_prompt_key,
                    )

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        if st.button("내 이미지 생성", key=f"user_img_gen_{idx}"):
                            if user_prompt.strip():
                                with st.spinner("내 이미지 생성..."):
                                    st.session_state[user_img_key] = generate_image_bytes_cached(
                                        user_prompt.strip(),
                                        IMAGE_MODEL,
                                    )
                                    st.session_state.last_student_image_prompt = user_prompt.strip()
                                    st.session_state.last_student_image_done = True
                            else:
                                st.warning("프롬프트 입력 필요.")
                    with c2:
                        if st.button("내 이미지 지우기", key=f"user_img_clear_{idx}"):
                            if user_img_key in st.session_state:
                                del st.session_state[user_img_key]
                            st.rerun()

                    if st.session_state.get(user_img_key):
                        st.image(st.session_state[user_img_key], caption="내가 만든 이미지")

                    q = step.get("question", "이 활동에서 중요한 점 1개")
                    st.markdown(f"**🗣️ 질문:** {q}")

                    with st.form(f"form_image_task_{idx}"):
                        opinion = st.text_area("내 생각(짧게)", key=f"img_opinion_{idx}")
                        submitted = st.form_submit_button("제출")

                    if submitted:
                        if not st.session_state.get(user_img_key):
                            st.warning("먼저 이미지를 생성해야 함.")
                        elif not opinion.strip():
                            st.warning("생각 입력 필요.")
                        else:
                            extra_context = f"학생 프롬프트: {user_prompt.strip()}" if user_prompt.strip() else ""
                            mode_hint = "copyright" if st.session_state.lesson_type == "copyright" else "generic"
                            rag_ctx = step_rag_ctx()

                            with st.spinner("피드백..."):
                                fb = feedback_with_tags(
                                    step.get("story", ""),
                                    opinion.strip(),
                                    extra_context=extra_context,
                                    mode=mode_hint,
                                    rag_ctx=rag_ctx,
                                )

                            with st.container(border=True):
                                if fb.get("tags"):
                                    st.write("태그:", ", ".join(fb["tags"]))
                                if fb.get("summary"):
                                    st.write("요약:", fb["summary"])
                                st.write("피드백:", fb["feedback"])

                            st.session_state.chat_history.append({"role": "user", "content": f"[활동] {opinion.strip()}"})
                            st.session_state.chat_history.append({"role": "assistant", "content": fb["feedback"]})

                            st.session_state.logs.append({
                                "timestamp": now_str(),
                                "student_name": st.session_state.student_name,
                                "topic": st.session_state.topic,
                                "lesson_type": st.session_state.lesson_type,
                                "step": idx + 1,
                                "step_type": "image_task",
                                "story": step.get("story", ""),
                                "prompt": user_prompt.strip(),
                                "opinion": opinion.strip(),
                                "tags": fb.get("tags", []),
                                "summary": fb.get("summary", ""),
                                "feedback": fb.get("feedback", ""),
                            })

                    if st.session_state.chat_history:
                        st.divider()
                        for msg in st.session_state.chat_history:
                            role = "assistant" if msg["role"] == "assistant" else "user"
                            st.chat_message(role).write(msg["content"])

                    if st.button("다음 단계 >", key=f"next_image_{idx}"):
                        st.session_state.current_step += 1
                        st.session_state.chat_history = []
                        st.rerun()

                # B) DISCUSSION
                elif step.get("type") == "discussion":
                    st.divider()
                    q = step.get("question", "토론 질문")
                    st.markdown(f"**🗣️ 토론 질문:** {q}")

                    with st.form(f"form_discussion_{idx}"):
                        opinion = st.text_area("내 의견", key=f"disc_opinion_{idx}")
                        submitted = st.form_submit_button("제출")

                    if submitted:
                        if not opinion.strip():
                            st.warning("의견 입력 필요.")
                        else:
                            extra_context = ""
                            if st.session_state.last_student_image_done and st.session_state.last_student_image_prompt:
                                extra_context = f"이전 활동 프롬프트: {st.session_state.last_student_image_prompt}"

                            mode_hint = "copyright" if st.session_state.lesson_type == "copyright" else "generic"
                            rag_ctx = step_rag_ctx()

                            with st.spinner("피드백..."):
                                fb = feedback_with_tags(
                                    step.get("story", ""),
                                    opinion.strip(),
                                    extra_context=extra_context,
                                    mode=mode_hint,
                                    rag_ctx=rag_ctx,
                                )

                            with st.container(border=True):
                                if fb.get("tags"):
                                    st.write("태그:", ", ".join(fb["tags"]))
                                if fb.get("summary"):
                                    st.write("요약:", fb["summary"])
                                st.write("피드백:", fb["feedback"])

                            st.session_state.chat_history.append({"role": "user", "content": f"[토론] {opinion.strip()}"})
                            st.session_state.chat_history.append({"role": "assistant", "content": fb["feedback"]})

                            st.session_state.logs.append({
                                "timestamp": now_str(),
                                "student_name": st.session_state.student_name,
                                "topic": st.session_state.topic,
                                "lesson_type": st.session_state.lesson_type,
                                "step": idx + 1,
                                "step_type": "discussion",
                                "story": step.get("story", ""),
                                "question": q,
                                "opinion": opinion.strip(),
                                "tags": fb.get("tags", []),
                                "summary": fb.get("summary", ""),
                                "feedback": fb.get("feedback", ""),
                            })

                    if st.session_state.chat_history:
                        st.divider()
                        for msg in st.session_state.chat_history:
                            role = "assistant" if msg["role"] == "assistant" else "user"
                            st.chat_message(role).write(msg["content"])

                    if st.button("다음 단계 >", key=f"next_disc_{idx}"):
                        st.session_state.current_step += 1
                        st.session_state.chat_history = []
                        st.rerun()

                # C) DILEMMA
                else:
                    with st.form(f"form_dilemma_{idx}"):
                        sel = st.radio(
                            "선택",
                            [step.get("choice_a", "A"), step.get("choice_b", "B")],
                            key=f"radio_{idx}",
                        )
                        reason = st.text_area("이유", key=f"reason_{idx}")
                        submitted = st.form_submit_button("제출")

                    if submitted:
                        if not reason.strip():
                            st.warning("이유 입력 필요.")
                        else:
                            extra_context = ""
                            if st.session_state.last_student_image_done and st.session_state.last_student_image_prompt:
                                extra_context = f"학생이 만든 이미지 프롬프트(참고): {st.session_state.last_student_image_prompt}"

                            mode_hint = "copyright" if st.session_state.lesson_type == "copyright" else "generic"
                            answer_text = f"선택: {sel}\n이유: {reason.strip()}"
                            rag_ctx = step_rag_ctx()

                            with st.spinner("분석..."):
                                fb = feedback_with_tags(
                                    step.get("story", ""),
                                    answer_text,
                                    extra_context=extra_context,
                                    mode=mode_hint,
                                    rag_ctx=rag_ctx,
                                )

                            with st.container(border=True):
                                st.markdown("#### 🧾 제출 요약")
                                if fb.get("tags"):
                                    st.write("태그:", ", ".join(fb["tags"]))
                                if fb.get("summary"):
                                    st.write("요약:", fb["summary"])
                                st.write("피드백:", fb["feedback"])

                            st.session_state.chat_history.append({"role": "user", "content": f"[{sel}] {reason.strip()}"})
                            st.session_state.chat_history.append({"role": "assistant", "content": fb["feedback"]})

                            st.session_state.logs.append({
                                "timestamp": now_str(),
                                "student_name": st.session_state.student_name,
                                "topic": st.session_state.topic,
                                "lesson_type": st.session_state.lesson_type,
                                "step": idx + 1,
                                "step_type": "dilemma",
                                "story": step.get("story", ""),
                                "choice": sel,
                                "reason": reason.strip(),
                                "tags": fb.get("tags", []),
                                "summary": fb.get("summary", ""),
                                "feedback": fb.get("feedback", ""),
                            })

                    if st.session_state.chat_history:
                        st.divider()
                        for msg in st.session_state.chat_history:
                            role = "assistant" if msg["role"] == "assistant" else "user"
                            st.chat_message(role).write(msg["content"])

                    if st.button("다음 단계 >", key=f"next_dilemma_{idx}"):
                        st.session_state.current_step += 1
                        st.session_state.chat_history = []
                        st.rerun()
