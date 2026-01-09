import streamlit as st
from openai import OpenAI
import json
import base64
import requests
from pathlib import Path
from datetime import datetime
import hashlib
import numpy as np

# =========================================================
# 1) Page config
# =========================================================
st.set_page_config(page_title="AI 윤리 교육 (수업유형 3종)", page_icon="🤖", layout="wide")

# =========================================================
# 1-1) Responsive image sizing (AUTO + MAX WIDTH CAP)
# =========================================================
IMAGE_MAX_WIDTH_PX = 520
st.markdown(
    f"""
<style>
div[data-testid="stImage"] img,
.stImage img {{
    width: 100% !important;
    max-width: {IMAGE_MAX_WIDTH_PX}px !important;
    height: auto !important;
    display: block;
    margin-left: auto;
    margin-right: auto;
}}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 2) Models
# =========================================================
TEXT_MODEL = "gpt-4o"
IMAGE_MODEL = "dall-e-3"
EMBED_MODEL = "text-embedding-3-small"

# =========================================================
# 3) Internal RAG (reference.txt only)
# =========================================================
REFERENCE_PATH = "reference.txt"
RAG_TOP_K = 4

# =========================================================
# 4) Image prompt policy: NO TEXT
# =========================================================
NO_TEXT_IMAGE_PREFIX = (
    "Minimalist, flat design illustration, educational context. "
    "ABSOLUTELY NO TEXT: no words, no letters, no numbers, no captions, no subtitles, "
    "no watermarks, no logos, no signs, no posters with writing. "
    "No text-like shapes. Only 그림/도형/사물. "
)

# =========================================================
# 4-1) UI-fixed text for Image Prompt lesson
# =========================================================
LOGO_CONTEST_GOAL = "학급 로고 제작 대회"
LOGO_FREE_TEXT_QUESTION = "어떤 내용의 로고를 제작했나요?"

# =========================================================
# 5) OpenAI client
# =========================================================
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ API 키 오류: secrets.toml을 확인하세요.")
    st.stop()

# =========================================================
# 6) System persona (UPDATED)
# =========================================================
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

# =========================================================
# 7) Utilities
# =========================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _clip(s: str, max_len: int = 1800) -> str:
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

def ask_gpt_json_object(prompt: str, system_prompt: str = SYSTEM_PERSONA) -> dict:
    try:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
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

def ask_gpt_text(prompt: str, system_prompt: str = SYSTEM_PERSONA) -> str:
    try:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return "응답 불가."

def normalize_analysis(x):
    if isinstance(x, dict):
        return {
            "ethics_standards": x.get("ethics_standards", []) if isinstance(x.get("ethics_standards", []), list) else [],
            "curriculum_alignment": x.get("curriculum_alignment", []) if isinstance(x.get("curriculum_alignment", []), list) else [],
            "lesson_content": x.get("lesson_content", []) if isinstance(x.get("lesson_content", []), list) else [],
        }
    return {"ethics_standards": [], "curriculum_alignment": [], "lesson_content": []}

def render_bullets(items):
    if not items:
        st.caption("내용 없음.")
        return
    if isinstance(items, list):
        for it in items:
            it = str(it).strip()
            if it:
                st.write(f"- {it}")
        return
    st.write(str(items))

def render_analysis_box(a):
    a = normalize_analysis(a)
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
# 8) Image generation (bytes) - cached
# =========================================================
@st.cache_data(show_spinner=False)
def generate_image_bytes_cached(user_prompt: str, model: str):
    full_prompt = f"{NO_TEXT_IMAGE_PREFIX}{user_prompt}"
    # 1) b64_json
    try:
        r = client.images.generate(
            model=model,
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
        r = client.images.generate(model=model, prompt=full_prompt, size="1024x1024", n=1)
        url = getattr(r.data[0], "url", None)
        if not url:
            return None
        resp = requests.get(url, timeout=25)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None

def clear_step_images_from_session():
    keys = [k for k in st.session_state.keys() if str(k).startswith("step_img_")]
    for k in keys:
        del st.session_state[k]

def clear_student_generated_images_from_session():
    keys = [k for k in st.session_state.keys() if str(k).startswith("stu_img_")]
    for k in keys:
        del st.session_state[k]

# =========================================================
# 9) RAG: reference.txt only
# =========================================================
def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def chunk_text(text: str, max_chars: int = 900, overlap: int = 160):
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    parts, buf = [], []
    for line in text.split("\n"):
        if line.strip() == "":
            if buf:
                parts.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        parts.append("\n".join(buf).strip())

    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = min(len(p), start + max_chars)
                    chunks.append(p[start:end])
                    start = max(0, end - overlap)
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)

    final = []
    for i, c in enumerate(chunks):
        if i == 0:
            final.append(c)
        else:
            tail = chunks[i - 1][-overlap:] if overlap > 0 else ""
            final.append((tail + "\n" + c).strip() if tail else c)

    return [x.strip() for x in final if x.strip()]

@st.cache_data(show_spinner=False)
def load_reference_text_cached(path_str: str, mtime: float) -> str:
    p = Path(path_str)
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8", errors="ignore")
    return txt[:1_200_000]

@st.cache_data(show_spinner=False)
def build_rag_index_cached(path_str: str, embed_model: str, mtime: float):
    txt = load_reference_text_cached(path_str, mtime)
    if not txt.strip():
        return {"chunks": [], "emb": None, "norms": None, "content_hash": ""}

    chunks = chunk_text(txt, max_chars=900, overlap=160)
    if not chunks:
        return {"chunks": [], "emb": None, "norms": None, "content_hash": sha256_text(txt)}

    try:
        resp = client.embeddings.create(model=embed_model, input=chunks)
        vecs = [d.embedding for d in resp.data]
        emb = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1) + 1e-8
        return {"chunks": chunks, "emb": emb, "norms": norms, "content_hash": sha256_text(txt)}
    except Exception:
        return {"chunks": chunks, "emb": None, "norms": None, "content_hash": sha256_text(txt)}

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
        emb, norms = index["emb"], index["norms"]
        sims = (emb @ qv) / (norms * qn)
        k = max(1, min(int(top_k), len(index["chunks"])))
        top_idx = np.argsort(-sims)[:k].tolist()
        ctx = "\n\n---\n\n".join(index["chunks"][i].strip() for i in top_idx)
        return _clip(ctx, 2400)
    except Exception:
        return ""

# =========================================================
# 10) Lesson types (3 buttons)
# =========================================================
LESSON_IMAGE_PROMPT = "이미지 프롬프트형"
LESSON_STORY_MODE = "스토리 모드형"
LESSON_DEEP_DEBATE = "심화 대화 토론형"

# 국가 인공지능 윤리기준(표현 고정)
NATIONAL_ETHICS_KEYS = ["프라이버시 보호", "연대성", "데이터 관리", "침해 금지", "안전성"]

def _force_image_revision_ui_text(steps: list):
    """첫 image_revision 단계에 목표/질문을 UI 고정 문구로 강제."""
    if not isinstance(steps, list):
        return steps
    for s in steps:
        if isinstance(s, dict) and s.get("type") == "image_revision":
            s["prompt_goal"] = LOGO_CONTEST_GOAL
            s["reflection_question"] = LOGO_FREE_TEXT_QUESTION
            return steps
    return steps

def generate_lesson_image_prompt(topic: str, rag_ctx: str) -> dict:
    prompt = f"""
교사용 설계 요청. (교사 관점으로 설계)

초등 고학년 대상 AI 윤리교육 수업 생성.
교사가 입력한 주제 1개만으로 수업 전체가 진행되게 구성.
주제: "{topic}"

[reference.txt 발췌]
{rag_ctx if rag_ctx else "- 없음"}

반드시 JSON만 출력.
키:
- topic: 문자열
- lesson_type: "{LESSON_IMAGE_PROMPT}"
- analysis: 객체
  - ethics_standards: 문자열 리스트
    * 반드시 아래 5개 국가 인공지능 윤리기준 명칭 중에서만 선택(표현 그대로), 3~5개:
      {", ".join(NATIONAL_ETHICS_KEYS)}
  - curriculum_alignment: 문자열 리스트(초등 5~6 실과/도덕 중심)
  - lesson_content: 문자열 리스트(도입-활동-토론-정리 요약)
- teacher_guide: 문자열(교사 관점 3~6개 항목 개조식)
- steps: 리스트(길이 3)

steps 규격:
1) type="image_revision"
   - story: 상황
   - prompt_goal: 목표(짧게)
   - reflection_question: 질문 1개
2) type="dilemma"
   - story, choice_a, choice_b
3) type="discussion"
   - story, question

규칙:
- 이미지 생성 단계는 "글자 없는 그림만" 전제
- 법 조항 단정 금지(약관/규정/상황 확인 필요 관점)
- 폭력/공포 배제
"""
    data = ask_gpt_json_object(prompt)
    steps = data.get("steps", [])
    if not isinstance(steps, list) or len(steps) < 3:
        steps = [
            {
                "type": "image_revision",
                "story": f"학급 로고 제작 대회에 낼 로고 이미지를 만든다. 주제는 '{topic}'. 글자 없이 그림만으로 의미를 담는다.",
                "prompt_goal": LOGO_CONTEST_GOAL,
                "reflection_question": LOGO_FREE_TEXT_QUESTION,
            },
            {
                "type": "dilemma",
                "story": "친구가 네 로고(또는 비슷한 로고)를 자기 팀에도 쓰고 싶다고 한다.",
                "choice_a": "조건부 허락(출처/목적/수정 범위 약속)",
                "choice_b": "허락하지 않음(내 팀만 사용)",
            },
            {
                "type": "discussion",
                "story": "정리: 우리 반에서 AI로 만든 로고를 사용할 때 규칙 만들기.",
                "question": "규칙 3가지(허락/출처/목적 기준)",
            },
        ]

    # UI 고정 문구 강제
    steps = _force_image_revision_ui_text(steps)

    analysis = normalize_analysis(data.get("analysis", {}))
    fixed = [x for x in analysis.get("ethics_standards", []) if x in NATIONAL_ETHICS_KEYS]
    if len(fixed) < 3:
        if "저작" in topic:
            fixed = ["데이터 관리", "침해 금지", "연대성", "안전성"]
        elif "개인" in topic or "프라이" in topic:
            fixed = ["프라이버시 보호", "데이터 관리", "침해 금지", "안전성"]
        else:
            fixed = ["안전성", "침해 금지", "데이터 관리", "연대성"]
    analysis["ethics_standards"] = fixed[:5]

    return {
        "topic": str(data.get("topic", topic)).strip() or topic,
        "lesson_type": LESSON_IMAGE_PROMPT,
        "analysis": analysis,
        "teacher_guide": str(data.get("teacher_guide", "")).strip(),
        "steps": steps,
    }

def generate_lesson_story_mode(topic: str, rag_ctx: str) -> dict:
    prompt = f"""
교사용 설계 요청. (교사 관점으로 설계)

초등 고학년 대상 AI 윤리교육 "스토리 모드" 수업 생성.
주제: "{topic}"

[reference.txt 발췌]
{rag_ctx if rag_ctx else "- 없음"}

반드시 JSON만 출력.
키:
- topic
- lesson_type: "{LESSON_STORY_MODE}"
- analysis(ethics_standards/curriculum_alignment/lesson_content)
  - ethics_standards는 반드시 아래 5개 국가 인공지능 윤리기준 명칭 중에서만 선택(표현 그대로), 3~5개:
    {", ".join(NATIONAL_ETHICS_KEYS)}
- teacher_guide(교사 관점 3~6개 항목 개조식)
- story_setup: setting, goal, characters(3~5), constraints(3~6)
- outline: 리스트(길이 5) 각 원소: chapter_title, learning_focus
- first_chapter: chapter_index=1, story(6~10문장), options(2개), question

규칙:
- 폭력/공포 배제
- 선택지는 문제 해결 전략 차이
- 법 조항 단정 금지(약관/규정/상황 확인 필요)
"""
    data = ask_gpt_json_object(prompt)

    setup = data.get("story_setup", {})
    if not isinstance(setup, dict):
        setup = {}

    outline = data.get("outline", [])
    if not isinstance(outline, list) or len(outline) < 5:
        outline = [
            {"chapter_title": "임무 시작", "learning_focus": "문제 파악/목표 설정"},
            {"chapter_title": "단서 수집", "learning_focus": "확인할 정보 찾기"},
            {"chapter_title": "대안 설계", "learning_focus": "조건/대체안 만들기"},
            {"chapter_title": "검증과 수정", "learning_focus": "리스크 점검/개선"},
            {"chapter_title": "규칙 만들기", "learning_focus": "원칙/규칙 정리"},
        ]

    first = data.get("first_chapter", {})
    if not isinstance(first, dict) or not first.get("story") or not isinstance(first.get("options", []), list):
        first = {
            "chapter_index": 1,
            "story": f"너는 학교 프로젝트 팀의 학생. 주제는 '{topic}'. 오늘 목표는 자료를 준비하는 것. "
                     f"그런데 자료를 만들다 보니 윤리적으로 확인할 점이 생김. "
                     f"팀원은 빨리 끝내자고 하고, 너는 안전하게 하자고 함. "
                     f"무엇부터 확인하고 어떻게 해결할지 선택 필요.",
            "options": ["먼저 확인 목록(허락/출처/개인정보/편향) 만들고 진행", "일단 만들고 나중에 문제 생기면 고치기"],
            "question": "왜 그 선택이 더 안전한가? 2문장",
        }

    analysis = normalize_analysis(data.get("analysis", {}))
    fixed = [x for x in analysis.get("ethics_standards", []) if x in NATIONAL_ETHICS_KEYS]
    if len(fixed) < 3:
        if "저작" in topic:
            fixed = ["데이터 관리", "침해 금지", "연대성", "안전성"]
        elif "개인" in topic or "프라이" in topic:
            fixed = ["프라이버시 보호", "데이터 관리", "침해 금지", "안전성"]
        else:
            fixed = ["안전성", "침해 금지", "데이터 관리", "연대성"]
    analysis["ethics_standards"] = fixed[:5]

    return {
        "topic": str(data.get("topic", topic)).strip() or topic,
        "lesson_type": LESSON_STORY_MODE,
        "analysis": analysis,
        "teacher_guide": str(data.get("teacher_guide", "")).strip(),
        "story_setup": {
            "setting": str(setup.get("setting", "학교 프로젝트")).strip(),
            "goal": str(setup.get("goal", f"주제 '{topic}'를 안전하고 공정하게 완성")).strip(),
            "characters": setup.get("characters", ["나", "팀원", "교사"]) if isinstance(setup.get("characters", []), list) else ["나", "팀원", "교사"],
            "constraints": setup.get("constraints", ["허락/출처 확인", "개인정보 보호", "편향/차별 표현 주의"]) if isinstance(setup.get("constraints", []), list) else ["허락/출처 확인", "개인정보 보호", "편향/차별 표현 주의"],
        },
        "outline": outline[:5],
        "first_chapter": {
            "chapter_index": 1,
            "story": str(first.get("story", "")).strip(),
            "options": first.get("options", [])[:2],
            "question": str(first.get("question", "선택 이유 2문장")).strip(),
        },
    }

def generate_story_next_chapter(topic: str, setup: dict, history: list, chapter_index: int, rag_ctx: str) -> dict:
    prompt = f"""
교사용 설계 요청. (교사 관점)
초등 고학년 스토리 모드: 다음 장면 생성.

주제: "{topic}"

[스토리 설정]
setting: {setup.get("setting","")}
goal: {setup.get("goal","")}
characters: {setup.get("characters",[])}
constraints(윤리 기준): {setup.get("constraints",[])}

[reference.txt 발췌]
{rag_ctx if rag_ctx else "- 없음"}

[이전 진행 기록]
{json.dumps(history, ensure_ascii=False) if history else "[]"}

현재 생성할 장(1~5): {chapter_index}

반드시 JSON만 출력.
키:
- chapter_index
- story(6~10문장)
- options(2개)  # 필요 없으면 빈 리스트 가능(마지막)
- question
- ending(boolean)
- debrief(ending=true일 때만, 3줄 개조식)
"""
    data = ask_gpt_json_object(prompt)
    out = {
        "chapter_index": int(data.get("chapter_index", chapter_index)),
        "story": str(data.get("story", "")).strip(),
        "options": data.get("options", []) if isinstance(data.get("options", []), list) else [],
        "question": str(data.get("question", "")).strip(),
        "ending": bool(data.get("ending", False)),
        "debrief": str(data.get("debrief", "")).strip(),
    }
    if not out["story"]:
        out["story"] = "다음 장면 생성 실패. 이전 선택을 바탕으로 다시 시도 필요."
    if out["chapter_index"] >= 5:
        out["ending"] = True
        out["options"] = out["options"][:2] if isinstance(out["options"], list) else []
    else:
        out["options"] = out["options"][:2]
        if not out["question"]:
            out["question"] = "왜 그 선택이 더 안전한가? 2문장"
    return out

def generate_lesson_deep_debate(topic: str, rag_ctx: str) -> dict:
    prompt = f"""
교사용 설계 요청. (교사 관점으로 설계)

초등 고학년 대상 AI 윤리교육 "심화 대화 토론형" 수업 생성.
주제: "{topic}"

[reference.txt 발췌]
{rag_ctx if rag_ctx else "- 없음"}

반드시 JSON만 출력.
키:
- topic
- lesson_type: "{LESSON_DEEP_DEBATE}"
- analysis
  - ethics_standards는 반드시 아래 5개 국가 인공지능 윤리기준 명칭 중에서만 선택(표현 그대로), 3~5개:
    {", ".join(NATIONAL_ETHICS_KEYS)}
- teacher_guide(교사 관점 3~6개 항목 개조식)
- debate_step: story(6~9문장), opening_question, constraints(4~6개), turns=3
- closing_step: story, question(2~3줄 결과)

주의:
- 학생 답에 맞춰 후속 질문을 던지는 형태(코드에서 구현)
- 폭력/공포 배제
- 법 조항 단정 금지(약관/규정/상황 확인 필요)
"""
    data = ask_gpt_json_object(prompt)
    debate = data.get("debate_step", {})
    closing = data.get("closing_step", {})

    if not isinstance(debate, dict) or not debate.get("story"):
        debate = {
            "story": f"학급에서 '{topic}' 주제로 활동을 했다. 결과물을 공유하자는 의견과, 확인 후 공유하자는 의견이 갈린다. "
                     f"너는 한 쪽 입장을 정하고 근거를 말해야 한다. 반대 입장도 생각하고, 타협안도 제시해야 한다.",
            "opening_question": "입장 1개 / 근거 1개",
            "constraints": ["근거 1개 이상", "반대 의견 1개", "대안 1개", "단정 금지", "약관/규칙 확인 언급 가능"],
            "turns": 3,
        }
    if not isinstance(closing, dict) or not closing.get("question"):
        closing = {
            "story": "정리: 토론을 바탕으로 실행 가능한 규칙을 만든다.",
            "question": "규칙 3줄(허락/출처/목적 또는 안전/공정/책임 기준)",
        }

    analysis = normalize_analysis(data.get("analysis", {}))
    fixed = [x for x in analysis.get("ethics_standards", []) if x in NATIONAL_ETHICS_KEYS]
    if len(fixed) < 3:
        if "저작" in topic:
            fixed = ["데이터 관리", "침해 금지", "연대성", "안전성"]
        elif "개인" in topic or "프라이" in topic:
            fixed = ["프라이버시 보호", "데이터 관리", "침해 금지", "안전성"]
        else:
            fixed = ["안전성", "침해 금지", "데이터 관리", "연대성"]
    analysis["ethics_standards"] = fixed[:5]

    return {
        "topic": str(data.get("topic", topic)).strip() or topic,
        "lesson_type": LESSON_DEEP_DEBATE,
        "analysis": analysis,
        "teacher_guide": str(data.get("teacher_guide", "")).strip(),
        "debate_step": {
            "story": str(debate.get("story", "")).strip(),
            "opening_question": str(debate.get("opening_question", "")).strip() or "입장 1개 / 근거 1개",
            "constraints": debate.get("constraints", []) if isinstance(debate.get("constraints", []), list) else [],
            "turns": int(debate.get("turns", 3)),
        },
        "closing_step": {
            "story": str(closing.get("story", "")).strip(),
            "question": str(closing.get("question", "")).strip(),
        },
    }

# =========================================================
# 11) Teacher feedback reflection (teacher rubric)
# =========================================================
def get_teacher_feedback_context() -> str:
    ctx = (st.session_state.get("teacher_feedback_context") or "").strip()
    return _clip(ctx, 900) if ctx else ""

def feedback_with_tags(step_story: str, answer_text: str, rag_ctx: str, extra_context: str = "") -> dict:
    teacher_ctx = get_teacher_feedback_context()
    prompt = f"""
[학생 피드백 생성]
상황/활동:
{step_story}

[reference.txt 발췌]
{rag_ctx if rag_ctx else "- 없음"}

[교사 기준/관점(반영)]
{teacher_ctx if teacher_ctx else "- (교사 입력 없음)"}

[추가 맥락]
{_clip(extra_context, 800) if extra_context else "- 없음"}

[학생 답]
{answer_text}

반드시 JSON만 출력.
키:
- tags: 문자열 리스트(최대 3개)
- summary: 1줄 요약
- feedback: 학생 피드백 템플릿 A 또는 B를 그대로 사용(4줄, 줄바꿈 포함)
"""
    data = ask_gpt_json_object(prompt)

    tags = data.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip()][:3]

    fb = str(data.get("feedback", "")).strip()
    if not fb:
        fb = "잘한 점: -\n위험 요소: -\n확인 질문: -\n다음 행동: -"

    return {
        "tags": tags,
        "summary": str(data.get("summary", "")).strip(),
        "feedback": fb,
    }

# =========================================================
# 12) Debate adaptive question generator
# =========================================================
DEBATE_Q_SYSTEM = """
너는 초등 5~6학년 토론 튜터.
인사말 금지.
출력: 질문 1문장만.
"""

def debate_next_question(topic: str, story: str, student_history: list, turn_index: int, rag_ctx: str) -> str:
    teacher_ctx = get_teacher_feedback_context()
    prompt = f"""
주제: "{topic}"

[토론 상황]
{story}

[reference.txt 발췌]
{rag_ctx if rag_ctx else "- 없음"}

[교사 기준(가능하면 반영)]
{teacher_ctx if teacher_ctx else "- 없음"}

[학생 발언 기록]
{json.dumps(student_history, ensure_ascii=False)}

이제 {turn_index}번째 후속 질문 1개.
원칙:
- 학생 답을 더 구체화(근거/반례/대안/조건)
- 단정 금지(약관/규칙/상황 확인 관점)
- 한 문장
"""
    q = ask_gpt_text(prompt, system_prompt=DEBATE_Q_SYSTEM).strip()
    if not q:
        q = "네 주장에 대한 가장 강한 반박 1개와 그에 대한 답 1개는?"
    return q

# =========================================================
# 13) Session state init
# =========================================================
default_state = {
    "mode": "👨‍🏫 교사용",
    "topic": "",
    "lesson_type": "",
    "analysis": {"ethics_standards": [], "curriculum_alignment": [], "lesson_content": []},
    "teacher_guide": "",
    "teacher_feedback_context": "",

    "steps": [],
    "current_step": 0,
    "chat_history": [],
    "logs": [],

    "story_setup": {},
    "story_outline": [],
    "story_history": [],
    "story_current": {},

    "debate": {},
    "closing": {},
    "debate_turn": 0,
    "debate_msgs": [],
}
for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# 14) Sidebar (minimal RAG indicator)
# =========================================================
st.sidebar.title("🤖 AI 윤리 교육")

rag_index = get_rag_index()
if rag_index and rag_index.get("chunks"):
    st.sidebar.caption(f"📚 RAG 적용: internal reference.txt (Top-K={RAG_TOP_K})")
else:
    st.sidebar.caption("📚 RAG 적용: internal reference.txt")
    if not Path(REFERENCE_PATH).exists():
        st.sidebar.warning("reference.txt 없음(레포에 포함 필요)")

if st.sidebar.button("⚠️ 전체 초기화"):
    st.session_state.clear()
    st.rerun()

mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"], key="mode_radio")
st.session_state.mode = mode

# =========================================================
# 15) Teacher UI
# =========================================================
if mode == "👨‍🏫 교사용":
    st.header("🛠️ 교사용 수업 생성 (주제 1개 + 수업유형 버튼 3개)")

    with st.expander("📘 교사용 가이드라인(사용법)", expanded=True):
        st.markdown(
            """
- 주제 1개 입력 → 아래 3개 버튼 중 1개로 수업 생성
- 수업 생성 시 reference.txt를 자동 참고(RAG)하여 ‘윤리기준/교육과정/수업 내용’을 구성
- 학생 피드백에 교사 관점 반영 가능(아래 입력칸)
- 생성 후 학생용 화면에서 동일 수업 진행
"""
        )

    topic = st.text_input("수업 주제 입력", value=st.session_state.topic, placeholder="예: 저작권, 개인정보, 추천 알고리즘, 편향, 딥페이크...")
    st.session_state.topic = topic

    st.session_state.teacher_feedback_context = st.text_area(
        "🧑‍🏫 교사 피드백 기준/관점(학생 피드백에 반영)",
        value=st.session_state.teacher_feedback_context,
        height=120,
        placeholder="예) 1) 출처/허락/목적 구분 강조  2) 약관/학교 규칙 확인 언급  3) 대안 제시 가점",
    )

    def get_rag_ctx_for_topic(tp: str) -> str:
        if not rag_index:
            return ""
        q = f"{tp} 국가 인공지능 윤리기준 프라이버시 보호 연대성 데이터 관리 침해 금지 안전성 교육과정 수업 설계"
        return rag_retrieve(q, rag_index, top_k=RAG_TOP_K)

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button(f"1) {LESSON_IMAGE_PROMPT}"):
            if not topic.strip():
                st.warning("주제 필요.")
            else:
                with st.spinner("수업 생성 중..."):
                    rag_ctx = get_rag_ctx_for_topic(topic.strip())
                    lesson = generate_lesson_image_prompt(topic.strip(), rag_ctx)
                    st.session_state.lesson_type = lesson["lesson_type"]
                    st.session_state.analysis = lesson["analysis"]
                    st.session_state.teacher_guide = lesson["teacher_guide"]
                    st.session_state.steps = lesson["steps"]
                    st.session_state.current_step = 0
                    st.session_state.chat_history = []
                    st.session_state.logs = []
                    st.session_state.story_setup = {}
                    st.session_state.story_outline = []
                    st.session_state.story_history = []
                    st.session_state.story_current = {}
                    st.session_state.debate = {}
                    st.session_state.closing = {}
                    st.session_state.debate_turn = 0
                    st.session_state.debate_msgs = []
                    clear_step_images_from_session()
                    clear_student_generated_images_from_session()
                    st.success("생성 완료.")

    with c2:
        if st.button(f"2) {LESSON_STORY_MODE}"):
            if not topic.strip():
                st.warning("주제 필요.")
            else:
                with st.spinner("스토리 모드 수업 생성 중..."):
                    rag_ctx = get_rag_ctx_for_topic(topic.strip())
                    lesson = generate_lesson_story_mode(topic.strip(), rag_ctx)
                    st.session_state.lesson_type = lesson["lesson_type"]
                    st.session_state.analysis = lesson["analysis"]
                    st.session_state.teacher_guide = lesson["teacher_guide"]
                    st.session_state.steps = []
                    st.session_state.current_step = 0
                    st.session_state.chat_history = []
                    st.session_state.logs = []
                    st.session_state.story_setup = lesson["story_setup"]
                    st.session_state.story_outline = lesson["outline"]
                    st.session_state.story_history = []
                    st.session_state.story_current = lesson["first_chapter"]
                    st.session_state.debate = {}
                    st.session_state.closing = {}
                    st.session_state.debate_turn = 0
                    st.session_state.debate_msgs = []
                    clear_step_images_from_session()
                    clear_student_generated_images_from_session()
                    st.success("생성 완료.")

    with c3:
        if st.button(f"3) {LESSON_DEEP_DEBATE}"):
            if not topic.strip():
                st.warning("주제 필요.")
            else:
                with st.spinner("심화 토론 수업 생성 중..."):
                    rag_ctx = get_rag_ctx_for_topic(topic.strip())
                    lesson = generate_lesson_deep_debate(topic.strip(), rag_ctx)
                    st.session_state.lesson_type = lesson["lesson_type"]
                    st.session_state.analysis = lesson["analysis"]
                    st.session_state.teacher_guide = lesson["teacher_guide"]
                    st.session_state.steps = []
                    st.session_state.current_step = 0
                    st.session_state.chat_history = []
                    st.session_state.logs = []
                    st.session_state.story_setup = {}
                    st.session_state.story_outline = []
                    st.session_state.story_history = []
                    st.session_state.story_current = {}
                    st.session_state.debate = lesson["debate_step"]
                    st.session_state.closing = lesson["closing_step"]
                    st.session_state.debate_turn = 0
                    st.session_state.debate_msgs = []
                    clear_step_images_from_session()
                    clear_student_generated_images_from_session()
                    st.success("생성 완료.")

    if st.session_state.lesson_type:
        st.divider()
        st.subheader("✅ 현재 선택된 수업 유형")
        st.write(f"- 주제: {st.session_state.topic}")
        st.write(f"- 수업 유형: {st.session_state.lesson_type}")

    if st.session_state.teacher_guide:
        with st.expander("📌 교사용 안내(자동 생성)", expanded=True):
            st.text(st.session_state.teacher_guide)

    if st.session_state.analysis:
        st.divider()
        render_analysis_box(st.session_state.analysis)

    # Preview for IMAGE_PROMPT lesson
    if st.session_state.lesson_type == LESSON_IMAGE_PROMPT and st.session_state.steps:
        st.divider()
        st.subheader("📜 단계 미리보기")
        for i, s in enumerate(st.session_state.steps):
            with st.container(border=True):
                st.markdown(f"### 단계 {i+1} ({s.get('type','')})")
                st.write(s.get("story", ""))
                if s.get("type") == "image_revision":
                    st.write("🎯 목표:", LOGO_CONTEST_GOAL)
                    st.write("🗣️ 질문:", LOGO_FREE_TEXT_QUESTION)
                elif s.get("type") == "dilemma":
                    cA, cB = st.columns(2)
                    with cA:
                        st.success("A: " + s.get("choice_a", ""))
                    with cB:
                        st.warning("B: " + s.get("choice_b", ""))
                elif s.get("type") == "discussion":
                    st.write("🗣️ 질문:", s.get("question", ""))

    # Preview for STORY_MODE
    if st.session_state.lesson_type == LESSON_STORY_MODE and st.session_state.story_current:
        st.divider()
        st.subheader("📖 스토리 모드 미리보기")
        with st.container(border=True):
            st.write("설정:", st.session_state.story_setup.get("setting", ""))
            st.write("목표:", st.session_state.story_setup.get("goal", ""))
            st.write("등장인물:", ", ".join(st.session_state.story_setup.get("characters", [])))
            st.write("제약/윤리 기준:", ", ".join(st.session_state.story_setup.get("constraints", [])))
        with st.container(border=True):
            st.markdown("### 5막 개요")
            for i, o in enumerate(st.session_state.story_outline[:5], start=1):
                st.write(f"- {i}막: {o.get('chapter_title','')} / {o.get('learning_focus','')}")
        with st.container(border=True):
            st.markdown("### 1막(첫 장면)")
            st.write(st.session_state.story_current.get("story", ""))
            opts = st.session_state.story_current.get("options", [])
            if isinstance(opts, list) and len(opts) >= 2:
                st.success("A: " + opts[0])
                st.warning("B: " + opts[1])
            st.write("질문:", st.session_state.story_current.get("question", ""))

    # Preview for DEEP_DEBATE
    if st.session_state.lesson_type == LESSON_DEEP_DEBATE and st.session_state.debate:
        st.divider()
        st.subheader("💬 심화 토론 미리보기")
        with st.container(border=True):
            st.write(st.session_state.debate.get("story", ""))
            st.write("오프닝 질문:", st.session_state.debate.get("opening_question", ""))
            cons = st.session_state.debate.get("constraints", [])
            if isinstance(cons, list) and cons:
                st.write("토론 규칙:")
                for it in cons:
                    st.write(f"- {it}")
            st.write("후속 질문 턴:", st.session_state.debate.get("turns", 3))
        with st.container(border=True):
            st.write("정리 질문:", st.session_state.closing.get("question", ""))

# =========================================================
# 16) Student UI
# =========================================================
else:
    st.header("🙋‍♂️ 학생용 학습")

    if not st.session_state.lesson_type:
        st.warning("교사용에서 주제 입력 후 수업 유형 버튼을 눌러 생성 필요.")
        st.stop()

    st.caption(f"주제: {st.session_state.topic}  |  수업 유형: {st.session_state.lesson_type}")

    def show_step_illustration(key: str, prompt_text: str):
        if key not in st.session_state:
            with st.spinner("이미지 생성..."):
                st.session_state[key] = generate_image_bytes_cached(prompt_text, IMAGE_MODEL)
        if st.session_state.get(key):
            st.image(st.session_state[key], use_container_width=True)

    def rag_ctx_for_step(text: str) -> str:
        if not rag_index:
            return ""
        q = f"{st.session_state.topic} {text} 국가 인공지능 윤리기준 프라이버시 보호 연대성 데이터 관리 침해 금지 안전성 근거"
        return rag_retrieve(q, rag_index, top_k=RAG_TOP_K)

    # =====================================================
    # A) IMAGE PROMPT LESSON
    # =====================================================
    if st.session_state.lesson_type == LESSON_IMAGE_PROMPT:
        steps = st.session_state.steps
        idx = st.session_state.current_step
        total = len(steps)

        if idx >= total:
            st.success("수업 종료.")
            if st.button("처음으로(학생)", key="img_restart"):
                st.session_state.current_step = 0
                st.session_state.chat_history = []
                st.session_state.logs = []
                clear_step_images_from_session()
                clear_student_generated_images_from_session()
                st.rerun()
            st.stop()

        step = steps[idx]
        st.progress((idx + 1) / total)
        st.subheader(f"단계 {idx+1} ({step.get('type','')})")

        show_step_illustration(f"step_img_{idx}", step.get("story", st.session_state.topic))
        st.info(step.get("story", ""))

        if step.get("type") == "image_revision":
            st.divider()
            st.subheader("🎨 프롬프트 → 이미지 → 수정")
            st.caption("글자 없는 그림만 생성(자동 적용)")
            st.write("목표:", LOGO_CONTEST_GOAL)

            p1_key = f"p1_{idx}"
            p2_key = f"p2_{idx}"
            img1_key = f"stu_img_{idx}_1"
            img2_key = f"stu_img_{idx}_2"

            p1 = st.text_input(
                "1차 프롬프트",
                value=st.session_state.get(p1_key, ""),
                key=p1_key,
                placeholder="예: simple class emblem logo symbol, flat illustration, no text",
            )
            cA, cB = st.columns([1, 1])
            with cA:
                if st.button("1차 이미지 생성", key=f"gen1_{idx}"):
                    if p1.strip():
                        with st.spinner("생성..."):
                            st.session_state[img1_key] = generate_image_bytes_cached(p1.strip(), IMAGE_MODEL)
                    else:
                        st.warning("프롬프트 입력 필요.")
            with cB:
                if st.button("1차 이미지 지우기", key=f"clr1_{idx}"):
                    if img1_key in st.session_state:
                        del st.session_state[img1_key]
                    st.rerun()

            if st.session_state.get(img1_key):
                st.image(st.session_state[img1_key], caption="1차 이미지", use_container_width=True)

            default_p2 = st.session_state.get(p2_key, "")
            if not default_p2 and p1:
                default_p2 = p1
            p2 = st.text_input(
                "2차 프롬프트(수정)",
                value=default_p2,
                key=p2_key,
                placeholder="예: make it more original, avoid famous brand style, no real logos, no text",
            )

            cC, cD = st.columns([1, 1])
            with cC:
                if st.button("2차 이미지 생성", key=f"gen2_{idx}"):
                    if p2.strip():
                        with st.spinner("생성..."):
                            st.session_state[img2_key] = generate_image_bytes_cached(p2.strip(), IMAGE_MODEL)
                    else:
                        st.warning("프롬프트 입력 필요.")
            with cD:
                if st.button("2차 이미지 지우기", key=f"clr2_{idx}"):
                    if img2_key in st.session_state:
                        del st.session_state[img2_key]
                    st.rerun()

            if st.session_state.get(img2_key):
                st.image(st.session_state[img2_key], caption="2차 이미지(수정본)", use_container_width=True)

            reflection = st.text_area(
                f"🗣️ {LOGO_FREE_TEXT_QUESTION}",
                key=f"ref_{idx}",
                placeholder="예: 친구와 협동을 나타내는 손 모양과 별을 넣은 로고...",
            )

            if st.button("제출(피드백 받기)", key=f"submit_rev_{idx}"):
                if not st.session_state.get(img1_key):
                    st.warning("1차 이미지를 먼저 생성해야 함.")
                elif not st.session_state.get(img2_key):
                    st.warning("2차 이미지를 생성(수정)해야 함.")
                elif not reflection.strip():
                    st.warning("답변 입력 필요.")
                else:
                    rag_ctx = rag_ctx_for_step(step.get("story", ""))
                    answer = f"""
[1차 프롬프트] {p1.strip()}
[2차 프롬프트] {p2.strip()}
[로고 설명] {reflection.strip()}
""".strip()
                    with st.spinner("피드백..."):
                        fb = feedback_with_tags(step.get("story", ""), answer, rag_ctx, extra_context="학급 로고 제작 대회 로고 제작/수정 활동")
                    with st.container(border=True):
                        if fb.get("tags"):
                            st.write("태그:", ", ".join(fb["tags"]))
                        if fb.get("summary"):
                            st.write("요약:", fb["summary"])
                        st.text(fb["feedback"])
                    st.session_state.logs.append({
                        "timestamp": now_str(),
                        "topic": st.session_state.topic,
                        "lesson_type": st.session_state.lesson_type,
                        "step": idx + 1,
                        "type": "image_revision",
                        "p1": p1.strip(),
                        "p2": p2.strip(),
                        "logo_desc": reflection.strip(),
                        "feedback": fb,
                    })

            if st.button("다음 단계 >", key=f"next_rev_{idx}"):
                st.session_state.current_step += 1
                st.rerun()

        elif step.get("type") == "dilemma":
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.success("A: " + step.get("choice_a", ""))
            with c2:
                st.warning("B: " + step.get("choice_b", ""))

            sel = st.radio("선택", ["A", "B"], horizontal=True, key=f"sel_{idx}")
            reason = st.text_area("이유", key=f"reason_{idx}", placeholder="2~4문장")

            if st.button("제출(피드백)", key=f"submit_dil_{idx}"):
                if not reason.strip():
                    st.warning("이유 입력 필요.")
                else:
                    rag_ctx = rag_ctx_for_step(step.get("story", ""))
                    choice_text = step.get("choice_a") if sel == "A" else step.get("choice_b")
                    answer = f"선택: {sel} / {choice_text}\n이유: {reason.strip()}"
                    with st.spinner("피드백..."):
                        fb = feedback_with_tags(step.get("story", ""), answer, rag_ctx)
                    with st.container(border=True):
                        if fb.get("tags"):
                            st.write("태그:", ", ".join(fb["tags"]))
                        if fb.get("summary"):
                            st.write("요약:", fb["summary"])
                        st.text(fb["feedback"])
                    st.session_state.logs.append({
                        "timestamp": now_str(),
                        "topic": st.session_state.topic,
                        "lesson_type": st.session_state.lesson_type,
                        "step": idx + 1,
                        "type": "dilemma",
                        "choice": sel,
                        "reason": reason.strip(),
                        "feedback": fb,
                    })

            if st.button("다음 단계 >", key=f"next_dil_{idx}"):
                st.session_state.current_step += 1
                st.rerun()

        elif step.get("type") == "discussion":
            st.divider()
            st.write("질문:", step.get("question", ""))
            opinion = st.text_area("내 답", key=f"disc_{idx}", placeholder="3~6줄")

            if st.button("제출(피드백)", key=f"submit_disc_{idx}"):
                if not opinion.strip():
                    st.warning("답 입력 필요.")
                else:
                    rag_ctx = rag_ctx_for_step(step.get("story", ""))
                    with st.spinner("피드백..."):
                        fb = feedback_with_tags(step.get("story", ""), opinion.strip(), rag_ctx)
                    with st.container(border=True):
                        if fb.get("tags"):
                            st.write("태그:", ", ".join(fb["tags"]))
                        if fb.get("summary"):
                            st.write("요약:", fb["summary"])
                        st.text(fb["feedback"])
                    st.session_state.logs.append({
                        "timestamp": now_str(),
                        "topic": st.session_state.topic,
                        "lesson_type": st.session_state.lesson_type,
                        "step": idx + 1,
                        "type": "discussion",
                        "answer": opinion.strip(),
                        "feedback": fb,
                    })

            if st.button("수업 종료 >", key=f"end_{idx}"):
                st.session_state.current_step = len(steps)
                st.rerun()

    # =====================================================
    # B) STORY MODE LESSON
    # =====================================================
    elif st.session_state.lesson_type == LESSON_STORY_MODE:
        chap = st.session_state.story_current
        if not chap:
            st.warning("스토리 데이터 없음. 교사용에서 다시 생성 필요.")
            st.stop()

        chap_idx = int(chap.get("chapter_index", 1))
        st.progress(chap_idx / 5)
        st.subheader(f"{chap_idx}막 / 5막")

        show_step_illustration(f"step_img_story_{chap_idx}", chap.get("story", st.session_state.topic))
        st.info(chap.get("story", ""))

        opts = chap.get("options", [])
        ending = bool(chap.get("ending", False))

        if ending:
            st.success("스토리 종료.")
            if chap.get("debrief"):
                st.write("배운 점:")
                st.text(chap.get("debrief"))
            if st.button("처음으로(학생)", key="story_restart"):
                st.session_state.story_history = []
                st.session_state.story_current = {}
                clear_step_images_from_session()
                st.rerun()
            st.stop()

        if not isinstance(opts, list) or len(opts) < 2:
            opts = ["A 선택", "B 선택"]

        st.success("A: " + opts[0])
        st.warning("B: " + opts[1])

        pick = st.radio("선택", ["A", "B"], horizontal=True, key=f"story_pick_{chap_idx}")
        q = chap.get("question", "왜 그 선택이 더 안전한가? 2문장")
        reason = st.text_area(f"🗣️ {q}", key=f"story_reason_{chap_idx}", placeholder="2~4문장")

        if st.button("다음 단계로", key=f"story_next_{chap_idx}"):
            if not reason.strip():
                st.warning("이유 입력 필요.")
            else:
                choice_text = opts[0] if pick == "A" else opts[1]
                st.session_state.story_history.append({
                    "chapter_index": chap_idx,
                    "story": chap.get("story", ""),
                    "choice": f"{pick}: {choice_text}",
                    "reason": reason.strip(),
                })

                rag_ctx = rag_ctx_for_step(chap.get("story", ""))
                next_idx = chap_idx + 1
                with st.spinner("다음 장면 생성..."):
                    nxt = generate_story_next_chapter(
                        st.session_state.topic,
                        st.session_state.story_setup,
                        st.session_state.story_history,
                        next_idx,
                        rag_ctx=rag_ctx
                    )
                st.session_state.story_current = nxt

                with st.spinner("피드백..."):
                    fb = feedback_with_tags(
                        chap.get("story", ""),
                        f"선택: {pick} / {choice_text}\n이유: {reason.strip()}",
                        rag_ctx=rag_ctx,
                        extra_context="스토리 모드 진행"
                    )
                with st.container(border=True):
                    if fb.get("tags"):
                        st.write("태그:", ", ".join(fb["tags"]))
                    if fb.get("summary"):
                        st.write("요약:", fb["summary"])
                    st.text(fb["feedback"])

                st.session_state.logs.append({
                    "timestamp": now_str(),
                    "topic": st.session_state.topic,
                    "lesson_type": st.session_state.lesson_type,
                    "chapter": chap_idx,
                    "choice": pick,
                    "reason": reason.strip(),
                    "feedback": fb,
                })

                st.rerun()

    # =====================================================
    # C) DEEP DEBATE LESSON
    # =====================================================
    elif st.session_state.lesson_type == LESSON_DEEP_DEBATE:
        debate = st.session_state.debate
        closing = st.session_state.closing
        if not debate:
            st.warning("토론 데이터 없음. 교사용에서 다시 생성 필요.")
            st.stop()

        st.subheader("토론 상황")
        show_step_illustration("step_img_debate", debate.get("story", st.session_state.topic))
        st.info(debate.get("story", ""))

        cons = debate.get("constraints", [])
        if isinstance(cons, list) and cons:
            with st.expander("토론 규칙", expanded=True):
                for it in cons:
                    st.write(f"- {it}")

        rag_ctx = rag_ctx_for_step(debate.get("story", ""))

        turns = int(debate.get("turns", 3))
        if turns != 3:
            turns = 3

        if st.session_state.debate_msgs:
            st.divider()
            for m in st.session_state.debate_msgs:
                role = m.get("role", "student")
                content = m.get("content", "")
                st.chat_message("assistant" if role == "assistant" else "user").write(content)

        st.divider()

        if st.session_state.debate_turn == 0:
            st.subheader("오프닝")
            opening_q = debate.get("opening_question", "입장 1개 / 근거 1개")
            opening = st.text_area(opening_q, key="deb_opening", placeholder="3~6줄")
            if st.button("제출(후속 질문 시작)", key="deb_start"):
                if not opening.strip():
                    st.warning("입력 필요.")
                else:
                    st.session_state.debate_msgs.append({"role": "student", "content": opening.strip()})
                    q1 = debate_next_question(st.session_state.topic, debate.get("story", ""), st.session_state.debate_msgs, 1, rag_ctx)
                    st.session_state.debate_msgs.append({"role": "assistant", "content": q1})
                    st.session_state.debate_turn = 1
                    st.rerun()

        elif 1 <= st.session_state.debate_turn <= turns:
            t = st.session_state.debate_turn
            st.subheader(f"후속 질문 {t}/{turns}")
            ans = st.text_area("답변", key=f"deb_ans_{t}", placeholder="2~6줄")
            if st.button("제출", key=f"deb_submit_{t}"):
                if not ans.strip():
                    st.warning("입력 필요.")
                else:
                    st.session_state.debate_msgs.append({"role": "student", "content": ans.strip()})
                    if t < turns:
                        qn = debate_next_question(st.session_state.topic, debate.get("story", ""), st.session_state.debate_msgs, t + 1, rag_ctx)
                        st.session_state.debate_msgs.append({"role": "assistant", "content": qn})
                        st.session_state.debate_turn = t + 1
                    else:
                        st.session_state.debate_turn = 4
                    st.rerun()

        else:
            st.subheader("정리")
            st.write(closing.get("story", ""))
            st.write("질문:", closing.get("question", ""))

            closing_ans = st.text_area("최종 정리 답", key="deb_close_ans", placeholder="2~6줄(원칙/규칙 형태)")
            if st.button("제출(최종 피드백)", key="deb_finish"):
                if not closing_ans.strip():
                    st.warning("입력 필요.")
                else:
                    transcript = "\n\n".join(
                        [("학생: " if m["role"] == "student" else "질문: ") + m["content"] for m in st.session_state.debate_msgs]
                    )
                    answer = f"[토론 기록]\n{transcript}\n\n[최종 정리]\n{closing_ans.strip()}"
                    with st.spinner("최종 피드백..."):
                        fb = feedback_with_tags(debate.get("story", ""), answer, rag_ctx, extra_context="심화 대화 토론(3턴)")
                    with st.container(border=True):
                        if fb.get("tags"):
                            st.write("태그:", ", ".join(fb["tags"]))
                        if fb.get("summary"):
                            st.write("요약:", fb["summary"])
                        st.text(fb["feedback"])

                    st.session_state.logs.append({
                        "timestamp": now_str(),
                        "topic": st.session_state.topic,
                        "lesson_type": st.session_state.lesson_type,
                        "debate_msgs": st.session_state.debate_msgs,
                        "closing": closing_ans.strip(),
                        "feedback": fb,
                    })

            if st.button("처음으로(학생)", key="deb_restart"):
                st.session_state.debate_turn = 0
                st.session_state.debate_msgs = []
                clear_step_images_from_session()
                st.rerun()

    if st.session_state.logs:
        st.divider()
        st.download_button(
            "학습 로그 다운로드(JSON)",
            data=json.dumps(st.session_state.logs, ensure_ascii=False, indent=2),
            file_name="ethics_learning_log.json",
            mime="application/json",
        )
