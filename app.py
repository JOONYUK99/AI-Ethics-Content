import streamlit as st
from openai import OpenAI
import json
import base64
import requests
from pathlib import Path
from datetime import datetime
import hashlib
import numpy as np
import re

# =========================================================
# 1) Page config
# =========================================================
st.set_page_config(page_title="AI 윤리 교육 (수업유형 3종)", page_icon="🤖", layout="wide")

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
# 5) OpenAI client
# =========================================================
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ API 키 오류: secrets.toml을 확인하세요.")
    st.stop()

# =========================================================
# 6) System prompts
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

SYSTEM_JSON_DESIGNER = """
너는 초등 5~6학년 대상 AI 윤리교육 수업 설계자.
출력은 반드시 JSON 객체만.
코드블록/설명/여분 문장 금지.
모든 필드는 한국어로.
"""

SYSTEM_FEEDBACK_JSON = """
너는 초등 5~6학년 AI 윤리교육 보조교사.
출력은 반드시 JSON 객체만.
반드시 "칭찬(구체적)"을 포함.
교사가 준 기준/관점이 있으면 '다음 행동' 또는 '확인 질문'에 반드시 반영.
문장은 짧고 쉬운 말 사용.
"""

DEBATE_Q_SYSTEM = """
너는 다정한 초등 5~6학년 토론 선생님.
출력은 정확히 2줄.
1줄: 공감/칭찬 1문장(짧게)
2줄: 질문 1문장(왜/근거/반대/대안/조건 중 1개 포함)
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
            temperature=0.6,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""

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

# ---- Story line rendering (한줄에 하나씩) ----
def split_to_lines(text: str, max_lines: int = 50) -> list:
    t = (text or "").strip()
    if not t:
        return []
    t = re.sub(r"\s+", " ", t)
    parts = re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=다\?)\s+|(?<=요\?)\s+", t)
    lines = [p.strip() for p in parts if p and p.strip()]
    return lines[:max_lines]

def render_story_box(text: str):
    lines = split_to_lines(text, max_lines=60)
    if not lines:
        return
    with st.container(border=True):
        st.markdown("<br>".join(lines), unsafe_allow_html=True)

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

def clear_story_prompt_assets():
    for k in ["story_act1_prompt", "story_act1_prompt_final", "story_act1_img"]:
        if k in st.session_state:
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

    # split on blank lines
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

    # pack
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

    # overlap merge
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
    return txt[:1_200_000]  # safety cap

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
# 10) Lesson types / standards
# =========================================================
LESSON_IMAGE_PROMPT = "이미지 프롬프트형"
LESSON_STORY_MODE = "스토리 모드형"
LESSON_DEEP_DEBATE = "심화 대화 토론형"

NATIONAL_ETHICS_KEYS = ["프라이버시 보호", "연대성", "데이터 관리", "침해 금지", "안전성"]

def ensure_analysis_defaults(topic: str, analysis_obj) -> dict:
    a = normalize_analysis(analysis_obj if isinstance(analysis_obj, dict) else {})
    fixed = [x for x in a.get("ethics_standards", []) if x in NATIONAL_ETHICS_KEYS]

    if len(fixed) < 3:
        if "저작" in topic:
            fixed = ["데이터 관리", "침해 금지", "연대성", "안전성"]
        elif "개인" in topic or "프라이" in topic:
            fixed = ["프라이버시 보호", "데이터 관리", "침해 금지", "안전성"]
        else:
            fixed = ["안전성", "침해 금지", "데이터 관리", "연대성"]

    a["ethics_standards"] = fixed[:5]
    if not a["curriculum_alignment"]:
        a["curriculum_alignment"] = ["초등 5~6 실과", "초등 5~6 도덕"]
    if not a["lesson_content"]:
        a["lesson_content"] = ["도입: 사례 확인", "활동: 선택/수정", "토론: 근거/대안", "정리: 규칙 만들기"]
    return a

# =========================================================
# 11) Fixed Story Mode (요청하신 브러시 스토리 고정)
# =========================================================
FIXED_STORY_TITLE = "인공지능 화가 '브러시'와 비밀의 그림"

FIXED_STORY_CHAPTERS = [
    {
        "chapter_index": 1,
        "chapter_title": "1단계: 인공지능의 편리함을 발견하다",
        "story": (
            "하늘이는 학교 미술 숙제를 하다가, 무엇이든 그려주는 인공지능 화가 '브러시'를 알게 되었어요. "
            "하늘이가 “멋진 숲속 마을을 그려줘!”라고 말하자, 브러시는 순식간에 화려한 그림을 만들어 줬어요. "
            "하늘이는 그 그림을 자신이 그린 것처럼 제출했고, 칭찬도 받았답니다."
        ),
        "question": "생각해보기: 내가 직접 그리지 않은 그림을 내 이름으로 내도 괜찮을까요?",
        "act1_prompt_activity": True,
        "prompt_activity_desc": "브러시에게 그려달라고 할 ‘숲속 마을’ 이미지를 글자 없이 그리도록 프롬프트를 만들어 보세요.",
    },
    {
        "chapter_index": 2,
        "chapter_title": "2단계: 작가의 노력을 알게 되다",
        "story": (
            "어느 날, 하늘이는 학교 게시판에서 유명 동화 작가의 원본 그림을 보고 깜짝 놀랐어요. "
            "브러시가 그려준 그림과 아주 비슷했거든요. "
            "알고 보니 브러시는 작가가 오래 노력해 만든 그림을 허락 없이 학습해 흉내 내고 있었어요."
        ),
        "question": "생각해보기: 인공지능은 누구의 도움으로 그림을 그릴 수 있는 걸까요?",
    },
    {
        "chapter_index": 3,
        "chapter_title": "3단계: 딜레마 - 허락받지 않은 학습",
        "story": (
            "하늘이는 작가가 ‘내 그림이 허락 없이 쓰였다’는 말을 듣고 속상해한다는 소식을 들었어요. "
            "하지만 짝꿍은 “인공지능이 공부하는 건데 뭐가 문제야? 편하면 그만이지!”라고 말해요. "
            "하늘이는 편리함과 공정함 사이에서 고민이 커졌어요."
        ),
        "question": "생각해보기: 작가의 허락 없이 그림을 학습시키는 것은 정당한 일일까요?",
    },
    {
        "chapter_index": 4,
        "chapter_title": "4단계: 저작권의 규칙을 배우다",
        "story": (
            "선생님은 하늘이에게 ‘저작권’에 대해 알려주셨어요. "
            "“남이 만든 소중한 작품을 쓸 때는 만든 이의 노력을 존중해야 해. 허락을 받거나 출처를 밝혀야 해.” "
            "하늘이는 브러시가 작가의 권리를 지키지 못했음을 깨달았어요."
        ),
        "question": "생각해보기: 인공지능을 사용하면서 저작권을 지킬 수 있는 방법은 무엇일까요?",
    },
    {
        "chapter_index": 5,
        "chapter_title": "5단계: 올바른 인공지능 사용자가 되다",
        "story": (
            "하늘이는 작가에게 사과 편지를 쓰고, 앞으로 인공지능을 쓸 때는 ‘내가 만든 부분’과 ‘도움 받은 부분’을 솔직하게 밝히기로 했어요. "
            "이제 하늘이는 편리함을 누리면서도 다른 사람의 노력을 소중히 여기는 멋진 어린이가 되었답니다."
        ),
        "question": "생각해보기: 인공지능과 사람이 함께 행복해지려면 어떤 약속이 필요할까요?",
        "ending": True,
        "debrief": "배운 점: 1) 남의 작품은 허락/출처가 필요해요.\n배운 점: 2) AI도 누군가의 자료로 배워요.\n배운 점: 3) 사용 목적과 공개범위를 먼저 확인해요.",
    },
]

# =========================================================
# 12) Lesson generators
# =========================================================
def generate_lesson_image_prompt(topic: str, rag_ctx: str) -> dict:
    prompt = f"""
교사용 설계 요청. (교사 관점)

주제: "{topic}"

[반드시 포함할 국가 인공지능 윤리기준(명칭 고정)]
{", ".join(NATIONAL_ETHICS_KEYS)}

[reference.txt 발췌]
{rag_ctx if rag_ctx else "- 없음"}

반드시 JSON만 출력.
키:
- topic
- lesson_type: "{LESSON_IMAGE_PROMPT}"
- analysis(ethics_standards/curriculum_alignment/lesson_content)
- teacher_guide
- steps: 리스트(길이 3)

steps[0] image_revision:
- story: 학급 로고 제작 대회 상황(2~3문장)
- prompt_goal: "학급 로고 제작 대회에 낼 우리 반 로고(글자 없음)" 관련 1문장
- checklist_items: 6~9개(보기용)
- reflection_question: 반드시 "어떤 내용의 로고를 제작했나요?"

steps[1] dilemma: story, choice_a, choice_b
steps[2] discussion: story, question

규칙:
- 글자 없는 그림만 전제
- 법 단정 금지(약관/규정/상황 확인 필요)
- 폭력/공포 배제
"""
    data = ask_gpt_json_object(prompt, system_prompt=SYSTEM_JSON_DESIGNER)

    steps = data.get("steps", [])
    if not isinstance(steps, list) or len(steps) < 3:
        steps = [
            {
                "type": "image_revision",
                "story": "학급 로고 제작 대회가 열린다. 우리 반을 나타내는 로고를 AI로 만든다.",
                "prompt_goal": "학급 로고 제작 대회에 낼 우리 반 로고(글자 없음) 만들기",
                "checklist_items": [
                    "유명 캐릭터/로고와 비슷함?",
                    "다른 사람 그림을 그대로 따라함?",
                    "출처/허락 확인이 필요한 요소가 있음?",
                    "특정 사람/집단을 놀리거나 차별함?",
                    "공유 범위(반/학교/온라인)와 맞음?",
                    "너무 복잡해서 의미가 흐려짐?",
                ],
                "reflection_question": "어떤 내용의 로고를 제작했나요?",
            },
            {
                "type": "dilemma",
                "story": "친구가 네 로고를 자기 발표에도 쓰고 싶다고 한다.",
                "choice_a": "조건부 허락(출처/사용 목적/수정 범위 약속)",
                "choice_b": "허락하지 않음(대신 새 아이디어를 함께 찾기)",
            },
            {
                "type": "discussion",
                "story": "정리: 우리 반에서 AI 로고/이미지 사용할 때 규칙을 만든다.",
                "question": "규칙 3가지(허락/출처/목적/공개범위 기준)",
            },
        ]

    analysis = ensure_analysis_defaults(topic, data.get("analysis", {}))
    return {
        "topic": str(data.get("topic", topic)).strip() or topic,
        "lesson_type": LESSON_IMAGE_PROMPT,
        "analysis": analysis,
        "teacher_guide": str(data.get("teacher_guide", "")).strip(),
        "steps": steps[:3],
    }

def generate_lesson_story_mode_fixed(topic: str) -> dict:
    # 고정 스토리(브러시)로 진행
    analysis = ensure_analysis_defaults("저작권", {})
    teacher_guide = "\n".join([
        "- 1막은 ‘편리함’에 끌린 선택을 다루고, 출처/허락 개념을 가볍게 던진다.",
        "- 2~3막에서 ‘누구의 노력으로 학습했는가’와 ‘허락’ 딜레마를 중심으로 질문한다.",
        "- 4막에서 저작권(남의 작품 존중/허락/출처)을 학생 언어로 정리한다.",
        "- 5막에서 ‘내가 한 것/AI가 도운 것’ 구분과 공개범위 약속으로 마무리한다.",
    ])

    outline = [
        {"chapter_title": FIXED_STORY_CHAPTERS[0]["chapter_title"], "learning_focus": "편리함 vs 정직"},
        {"chapter_title": FIXED_STORY_CHAPTERS[1]["chapter_title"], "learning_focus": "노력/출처"},
        {"chapter_title": FIXED_STORY_CHAPTERS[2]["chapter_title"], "learning_focus": "허락 딜레마"},
        {"chapter_title": FIXED_STORY_CHAPTERS[3]["chapter_title"], "learning_focus": "저작권 규칙"},
        {"chapter_title": FIXED_STORY_CHAPTERS[4]["chapter_title"], "learning_focus": "약속/실천"},
    ]

    return {
        "topic": str(topic).strip() or "저작권",
        "lesson_type": LESSON_STORY_MODE,
        "analysis": analysis,
        "teacher_guide": teacher_guide,
        "story_title": FIXED_STORY_TITLE,
        "outline": outline,
        "chapters": FIXED_STORY_CHAPTERS,
        "first_chapter": FIXED_STORY_CHAPTERS[0],
    }

def generate_lesson_deep_debate(topic: str, rag_ctx: str) -> dict:
    prompt = f"""
교사용 설계 요청. (교사 관점)

초등 고학년 대상 AI 윤리교육 "심화 대화 토론형(딜레마 기반)" 수업 생성.
주제: "{topic}"

[반드시 포함할 국가 인공지능 윤리기준(명칭 고정)]
{", ".join(NATIONAL_ETHICS_KEYS)}

[reference.txt 발췌]
{rag_ctx if rag_ctx else "- 없음"}

[중요: 딜레마 토론 구성]
- 발췌 안에 "사례01~사례05"가 있으면, 그 중 1개를 골라 토론을 구성.
- debate_step.case_title / case_summary에 반영.
- A/B 선택지: debate_step.choice_a / choice_b.
- opening_question은 "A/B 중 무엇을 선택하고, 왜 그렇게 생각하나요?" 포함.
- turns는 3 고정.

반드시 JSON만 출력.
키:
- topic
- lesson_type: "{LESSON_DEEP_DEBATE}"
- analysis(ethics_standards/curriculum_alignment/lesson_content)
- teacher_guide
- debate_step: case_title, case_summary, story, choice_a, choice_b, opening_question, constraints, turns=3
- closing_step: story, question

규칙:
- 폭력/공포 배제
- 법 단정 금지(약관/규정/상황 확인 필요)
"""
    data = ask_gpt_json_object(prompt, system_prompt=SYSTEM_JSON_DESIGNER)

    debate = data.get("debate_step", {})
    closing = data.get("closing_step", {})

    if not isinstance(debate, dict) or not debate.get("story"):
        debate = {
            "case_title": f"{topic} 관련 사례",
            "case_summary": f"'{topic}' 활동에서 공유/사용 과정에서 확인할 점이 생겼다.",
            "story": f"학급에서 '{topic}' 주제로 활동을 했다. 공유하려고 하니 확인이 필요하다는 의견이 나온다. "
                     f"너는 한 가지를 선택하고 이유를 말해야 한다.",
            "choice_a": "조건부 진행(허락/출처/목적/공개범위 확인 후 진행)",
            "choice_b": "보류(확인 전까지 멈추고 대안을 찾기)",
            "opening_question": "좋아, 네 생각이 궁금해.\nA/B 중 무엇을 선택하고, 왜 그렇게 생각하나요?",
            "constraints": ["근거 1개 이상", "반대 의견 1개", "대안 1개", "단정 금지", "약관/학교 규칙 확인 언급"],
            "turns": 3,
        }

    oq = str(debate.get("opening_question", "")).strip()
    if "A/B" not in oq or "왜" not in oq:
        oq = "좋아, 네 생각이 궁금해.\nA/B 중 무엇을 선택하고, 왜 그렇게 생각하나요?"
    if "\n" not in oq:
        oq = "좋아, 네 생각이 궁금해.\n" + oq

    ca = str(debate.get("choice_a", "")).strip() or "A 선택(조건부 진행: 허락/출처/목적 확인)"
    cb = str(debate.get("choice_b", "")).strip() or "B 선택(보류/대안 찾기)"

    if not isinstance(closing, dict) or not closing.get("question"):
        closing = {
            "story": "정리: 토론을 바탕으로 실행 가능한 규칙을 만든다.",
            "question": "우리 반 규칙 3줄(허락/출처/목적/공개범위 기준)",
        }

    analysis = ensure_analysis_defaults(topic, data.get("analysis", {}))

    return {
        "topic": str(data.get("topic", topic)).strip() or topic,
        "lesson_type": LESSON_DEEP_DEBATE,
        "analysis": analysis,
        "teacher_guide": str(data.get("teacher_guide", "")).strip(),
        "debate_step": {
            "case_title": str(debate.get("case_title", "")).strip(),
            "case_summary": str(debate.get("case_summary", "")).strip(),
            "story": str(debate.get("story", "")).strip(),
            "choice_a": ca,
            "choice_b": cb,
            "opening_question": oq,
            "constraints": debate.get("constraints", []) if isinstance(debate.get("constraints", []), list) else [],
            "turns": 3,
        },
        "closing_step": {
            "story": str(closing.get("story", "")).strip(),
            "question": str(closing.get("question", "")).strip(),
        },
    }

# =========================================================
# 13) Teacher feedback (칭찬 + 교사기준 반영 강제)
# =========================================================
def get_teacher_feedback_context() -> str:
    ctx = (st.session_state.get("teacher_feedback_context") or "").strip()
    return _clip(ctx, 900) if ctx else ""

def _format_feedback(template: str, praise: str, risk: str, q: str, next_action: str) -> str:
    praise = praise.strip() or "-"
    risk = risk.strip() or "-"
    q = q.strip() or "-"
    next_action = next_action.strip() or "-"

    if template == "B":
        return f"핵심 판단: {praise}\n근거: {risk}\n확인 질문: {q}\n다음 행동: {next_action}"
    return f"잘한 점: {praise}\n위험 요소: {risk}\n확인 질문: {q}\n다음 행동: {next_action}"

def feedback_with_tags(step_story: str, answer_text: str, rag_ctx: str, extra_context: str = "") -> dict:
    teacher_ctx = get_teacher_feedback_context()
    prompt = f"""
[학생 피드백 생성: 교사 기준 강반영 + 칭찬 포함]

상황/활동:
{step_story}

[reference.txt 발췌]
{rag_ctx if rag_ctx else "- 없음"}

[교사 기준/관점(반영 필수)]
{teacher_ctx if teacher_ctx else "- (교사 입력 없음)"}

[추가 맥락]
{_clip(extra_context, 800) if extra_context else "- 없음"}

[학생 답]
{answer_text}

반드시 JSON만 출력.
키:
- tags: 문자열 리스트(최대 3개)
- summary: 1줄 요약
- template: "A" 또는 "B"
- praise: 칭찬(구체적 1문장)
- risk: 위험/주의점(1문장)
- check_question: 확인 질문(1문장)
- next_action: 다음 행동(1문장, 교사 기준/관점이 있으면 반드시 반영)
"""
    data = ask_gpt_json_object(prompt, system_prompt=SYSTEM_FEEDBACK_JSON)

    tags = data.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip()][:3]

    template = str(data.get("template", "A")).strip().upper()
    if template not in ["A", "B"]:
        template = "A"

    fb = _format_feedback(
        template,
        str(data.get("praise", "")).strip(),
        str(data.get("risk", "")).strip(),
        str(data.get("check_question", "")).strip(),
        str(data.get("next_action", "")).strip(),
    )

    return {
        "tags": tags,
        "summary": str(data.get("summary", "")).strip(),
        "feedback": fb,
    }

# =========================================================
# 14) Debate adaptive question generator (2 lines)
# =========================================================
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

이제 {turn_index}번째 후속 질문을 만든다.

조건:
- 출력은 2줄
- 1줄: 공감/칭찬 1문장
- 2줄: 질문 1문장(왜/근거/반대/대안/조건 중 1개 포함)
- 단정 금지(약관/규칙/상황 확인 관점)
"""
    q = ask_gpt_text(prompt, system_prompt=DEBATE_Q_SYSTEM).strip()
    if not q:
        q = "좋아, 네 생각이 또렷해.\n그 생각의 근거를 한 가지로 말해볼래?"
    lines = [ln.strip() for ln in q.split("\n") if ln.strip()]
    if len(lines) == 1:
        q = f"좋아, 잘 설명했어.\n{lines[0]}"
    else:
        q = "\n".join(lines[:2])
    return q

# =========================================================
# 15) Session state init
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
    "logs": [],

    # story mode (fixed)
    "story_title": "",
    "story_outline": [],
    "story_chapters": [],
    "story_chapter_index": 1,

    # debate mode
    "debate": {},
    "closing": {},
    "debate_turn": 0,
    "debate_msgs": [],
}
for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# 16) Sidebar
# =========================================================
st.sidebar.title("🤖 AI 윤리 교육")

rag_index = get_rag_index()
if rag_index and rag_index.get("chunks"):
    st.sidebar.caption(f"📚 RAG 적용: reference.txt (Top-K={RAG_TOP_K})")
else:
    st.sidebar.caption("📚 RAG 적용: reference.txt")
    if not Path(REFERENCE_PATH).exists():
        st.sidebar.warning("reference.txt 없음(레포에 포함 필요)")

if st.sidebar.button("⚠️ 전체 초기화"):
    st.session_state.clear()
    st.rerun()

mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"], key="mode_radio")
st.session_state.mode = mode

# =========================================================
# 17) Small image renderer (스토리 모드 이미지 대폭 축소)
# =========================================================
def show_step_illustration_small(key: str, prompt_text: str, width_px: int = 300):
    if key not in st.session_state:
        with st.spinner("이미지 생성..."):
            st.session_state[key] = generate_image_bytes_cached(prompt_text, IMAGE_MODEL)

    img = st.session_state.get(key)
    if img:
        cL, cM, cR = st.columns([6, 2, 6])
        with cM:
            st.image(img, width=width_px)

def show_step_illustration_medium(key: str, prompt_text: str, width_px: int = 420):
    if key not in st.session_state:
        with st.spinner("이미지 생성..."):
            st.session_state[key] = generate_image_bytes_cached(prompt_text, IMAGE_MODEL)

    img = st.session_state.get(key)
    if img:
        cL, cM, cR = st.columns([4, 4, 4])
        with cM:
            st.image(img, width=width_px)

# =========================================================
# 18) Teacher UI
# =========================================================
if mode == "👨‍🏫 교사용":
    st.header("🛠️ 교사용 수업 생성")

    with st.expander("📘 교사용 가이드라인(사용법)", expanded=True):
        st.markdown(
            """
- 주제 1개 입력 → 아래 3개 버튼 중 1개로 수업 생성  
- 생성 시 reference.txt를 자동 참고(RAG)  
- 학생 피드백은 교사 기준/관점을 반영  
- 스토리 모드는 ‘브러시’ 이야기로 5막 고정 진행
"""
        )

    topic = st.text_input(
        "수업 주제 입력",
        value=st.session_state.topic,
        placeholder="예: 저작권, 개인정보, 추천 알고리즘, 편향, 딥페이크..."
    )
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
        q = f"{tp} 사례01 사례02 사례03 사례04 사례05 딜레마 토론 국가 인공지능 윤리기준 프라이버시 보호 연대성 데이터 관리 침해 금지 안전성"
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

                    # clear others
                    st.session_state.story_title = ""
                    st.session_state.story_outline = []
                    st.session_state.story_chapters = []
                    st.session_state.story_chapter_index = 1

                    st.session_state.debate = {}
                    st.session_state.closing = {}
                    st.session_state.debate_turn = 0
                    st.session_state.debate_msgs = []

                    st.session_state.logs = []
                    clear_step_images_from_session()
                    clear_student_generated_images_from_session()
                    clear_story_prompt_assets()
                    st.success("생성 완료.")

    with c2:
        if st.button(f"2) {LESSON_STORY_MODE}"):
            if not topic.strip():
                st.warning("주제 필요.")
            else:
                with st.spinner("스토리 모드 수업 생성 중..."):
                    lesson = generate_lesson_story_mode_fixed(topic.strip())

                    st.session_state.lesson_type = lesson["lesson_type"]
                    st.session_state.analysis = lesson["analysis"]
                    st.session_state.teacher_guide = lesson["teacher_guide"]

                    st.session_state.story_title = lesson["story_title"]
                    st.session_state.story_outline = lesson["outline"]
                    st.session_state.story_chapters = lesson["chapters"]
                    st.session_state.story_chapter_index = 1

                    # clear others
                    st.session_state.steps = []
                    st.session_state.current_step = 0

                    st.session_state.debate = {}
                    st.session_state.closing = {}
                    st.session_state.debate_turn = 0
                    st.session_state.debate_msgs = []

                    st.session_state.logs = []
                    clear_step_images_from_session()
                    clear_student_generated_images_from_session()
                    clear_story_prompt_assets()
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

                    # clear others
                    st.session_state.steps = []
                    st.session_state.current_step = 0

                    st.session_state.story_title = ""
                    st.session_state.story_outline = []
                    st.session_state.story_chapters = []
                    st.session_state.story_chapter_index = 1

                    st.session_state.debate = lesson["debate_step"]
                    st.session_state.closing = lesson["closing_step"]
                    st.session_state.debate_turn = 0
                    st.session_state.debate_msgs = []

                    st.session_state.logs = []
                    clear_step_images_from_session()
                    clear_student_generated_images_from_session()
                    clear_story_prompt_assets()
                    st.success("생성 완료.")

    if st.session_state.lesson_type:
        st.divider()
        st.subheader("✅ 현재 선택된 수업")
        st.write(f"- 주제: {st.session_state.topic}")
        st.write(f"- 유형: {st.session_state.lesson_type}")

    if st.session_state.teacher_guide:
        with st.expander("📌 교사용 안내(자동 생성)", expanded=True):
            st.text(st.session_state.teacher_guide)

    if st.session_state.analysis:
        st.divider()
        render_analysis_box(st.session_state.analysis)

    # teacher preview for story
    if st.session_state.lesson_type == LESSON_STORY_MODE and st.session_state.story_chapters:
        st.divider()
        st.subheader("📖 스토리 모드 미리보기(고정)")
        st.write(f"🎨 제목: {st.session_state.story_title}")
        with st.container(border=True):
            st.markdown("### 5막 개요")
            for i, o in enumerate(st.session_state.story_outline[:5], start=1):
                st.write(f"- {i}막: {o.get('chapter_title','')} / {o.get('learning_focus','')}")
        with st.container(border=True):
            st.markdown("### 1막")
            ch1 = st.session_state.story_chapters[0]
            st.write(ch1.get("chapter_title", ""))
            render_story_box(ch1.get("story", ""))
            st.write(ch1.get("question", ""))

# =========================================================
# 19) Student UI
# =========================================================
else:
    st.header("🙋‍♂️ 학생용 학습")

    if not st.session_state.lesson_type:
        st.warning("교사용에서 주제 입력 후 수업 유형 버튼을 눌러 생성 필요.")
        st.stop()

    st.caption(f"주제: {st.session_state.topic}  |  수업 유형: {st.session_state.lesson_type}")

    def rag_ctx_for_step(text: str) -> str:
        if not rag_index:
            return ""
        q = f"{st.session_state.topic} {text} 저작권 출처 허락 사례01 사례02 사례03 사례04 사례05 국가 인공지능 윤리기준 프라이버시 보호 연대성 데이터 관리 침해 금지 안전성"
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
                st.session_state.logs = []
                clear_step_images_from_session()
                clear_student_generated_images_from_session()
                st.rerun()
            st.stop()

        step = steps[idx]
        st.progress((idx + 1) / total)
        st.subheader(f"단계 {idx+1} ({step.get('type','')})")

        show_step_illustration_medium(f"step_img_{idx}", step.get("story", st.session_state.topic), width_px=420)
        render_story_box(step.get("story", ""))

        if step.get("type") == "image_revision":
            st.divider()
            st.subheader("🎨 프롬프트 → 이미지 → 수정")
            st.caption("글자 없는 그림만 생성(자동 적용)")
            st.write("목표:", "학급 로고 제작 대회에 낼 우리 반 로고(글자 없음) 만들기")

            # 체크 선택 기능 제거(보기만)
            items = step.get("checklist_items", [])
            if isinstance(items, list) and items:
                with st.expander("점검 포인트(보기)", expanded=False):
                    for it in items:
                        it = str(it).strip()
                        if it:
                            st.write(f"- {it}")

            p1_key = f"p1_{idx}"
            p2_key = f"p2_{idx}"
            img1_key = f"stu_img_{idx}_1"
            img2_key = f"stu_img_{idx}_2"

            p1 = st.text_input(
                "1차 프롬프트",
                value=st.session_state.get(p1_key, ""),
                key=p1_key,
                placeholder="예: simple class logo concept, flat illustration, mascot style, no text"
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
                show_step_illustration_small(img1_key + "_view", "preview", width_px=1)  # no-op safety
                cL, cM, cR = st.columns([6, 2, 6])
                with cM:
                    st.image(st.session_state[img1_key], width=360, caption="1차 이미지")

            default_p2 = st.session_state.get(p2_key, "")
            if not default_p2 and p1:
                default_p2 = p1
            p2 = st.text_input(
                "2차 프롬프트(수정)",
                value=default_p2,
                key=p2_key,
                placeholder="예: make it more original, avoid famous characters, simple shapes, no logos, no text"
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
                cL, cM, cR = st.columns([6, 2, 6])
                with cM:
                    st.image(st.session_state[img2_key], width=360, caption="2차 이미지(수정본)")

            reflection = st.text_area(
                "🗣️ 어떤 내용의 로고를 제작했나요?",
                key=f"ref_{idx}",
                placeholder="예: 우리 반을 상징하는 ○○(동물/색/모양)을 넣고, 글자 없이 단순한 도형으로 만들었어요."
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
                        fb = feedback_with_tags(step.get("story", ""), answer, rag_ctx, extra_context="학급 로고 제작 대회: 로고 만들기/수정 활동")
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
                        "reflection": reflection.strip(),
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
            reason = st.text_area("왜 그렇게 생각하나요?", key=f"reason_{idx}", placeholder="2~4문장")

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
    # B) STORY MODE LESSON (고정 5막 + 이미지 작게 + 문장 한줄씩)
    # =====================================================
    elif st.session_state.lesson_type == LESSON_STORY_MODE:
        if not st.session_state.story_chapters:
            st.warning("스토리 데이터 없음. 교사용에서 다시 생성 필요.")
            st.stop()

        chap_idx = int(st.session_state.story_chapter_index)
        chap_idx = max(1, min(5, chap_idx))
        chapters = st.session_state.story_chapters
        chap = next((c for c in chapters if int(c.get("chapter_index", 0)) == chap_idx), None)
        if not chap:
            st.warning("현재 막 데이터 없음.")
            st.stop()

        st.progress(chap_idx / 5)
        st.subheader(f"{chap_idx}막 / 5막")
        st.write(f"🎨 제목: {st.session_state.story_title}")

        # ✅ 스토리 모드 이미지: 매우 작게
        show_step_illustration_small(f"step_img_story_{chap_idx}", chap.get("story", st.session_state.topic), width_px=280)

        # ✅ 한 줄씩 출력
        st.write(chap.get("chapter_title", ""))
        render_story_box(chap.get("story", ""))

        # ✅ 1막: 학생이 직접 프롬프트 작성/출력 + (선택) 이미지 생성
        if chap_idx == 1 and chap.get("act1_prompt_activity"):
            st.divider()
            st.subheader("🧩 1막 활동: 내가 만드는 이미지 프롬프트")
            st.caption(chap.get("prompt_activity_desc", ""))

            st.session_state["story_act1_prompt"] = st.text_area(
                "프롬프트 작성(글자 없는 그림)",
                value=st.session_state.get("story_act1_prompt", ""),
                placeholder="예: colorful forest village, cozy houses, winding paths, soft sunlight, flat illustration, no text"
            )

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("프롬프트 출력(저장)", key="story_prompt_save"):
                    p = (st.session_state.get("story_act1_prompt") or "").strip()
                    if not p:
                        st.warning("프롬프트를 먼저 작성해 주세요.")
                    else:
                        st.session_state["story_act1_prompt_final"] = p
                        st.success("프롬프트를 저장했어요.")
            with c2:
                if st.button("프롬프트 예시 넣기", key="story_prompt_example"):
                    st.session_state["story_act1_prompt"] = (
                        "warm forest village, small cozy cottages, river and bridge, friendly animals, "
                        "soft pastel colors, flat illustration, simple shapes, no text"
                    )
                    st.rerun()

            if st.session_state.get("story_act1_prompt_final"):
                with st.container(border=True):
                    st.write("내 프롬프트:")
                    st.code(st.session_state["story_act1_prompt_final"], language="text")

                # (선택) 실제 이미지 생성
                if st.button("이 프롬프트로 이미지 만들기(선택)", key="story_prompt_make_img"):
                    with st.spinner("이미지 생성..."):
                        st.session_state["story_act1_img"] = generate_image_bytes_cached(
                            st.session_state["story_act1_prompt_final"], IMAGE_MODEL
                        )
                    st.rerun()

                if st.session_state.get("story_act1_img"):
                    cL, cM, cR = st.columns([6, 2, 6])
                    with cM:
                        st.image(st.session_state["story_act1_img"], width=280)

        st.divider()
        st.write(chap.get("question", ""))

        answer_key = f"story_answer_{chap_idx}"
        ans = st.text_area("내 생각", key=answer_key, placeholder="2~6줄")

        if st.button("제출(피드백)", key=f"story_submit_{chap_idx}"):
            if not ans.strip():
                st.warning("답을 입력해 주세요.")
            else:
                rag_ctx = rag_ctx_for_step(chap.get("story", ""))
                extra = "스토리 모드(고정 5막)"
                if chap_idx == 1 and st.session_state.get("story_act1_prompt_final"):
                    extra += f" / 1막 프롬프트: {st.session_state.get('story_act1_prompt_final')}"
                with st.spinner("피드백..."):
                    fb = feedback_with_tags(
                        chap.get("story", ""),
                        f"[질문] {chap.get('question','')}\n[답] {ans.strip()}",
                        rag_ctx=rag_ctx,
                        extra_context=extra
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
                    "question": chap.get("question", ""),
                    "answer": ans.strip(),
                    "act1_prompt": st.session_state.get("story_act1_prompt_final", "") if chap_idx == 1 else "",
                    "feedback": fb,
                })

        # 다음 단계 이동
        if chap.get("ending"):
            st.divider()
            st.success("스토리 종료.")
            if chap.get("debrief"):
                st.write("정리")
                render_story_box(chap.get("debrief", ""))
            if st.button("처음으로(학생)", key="story_restart"):
                st.session_state.story_chapter_index = 1
                st.session_state.logs = []
                clear_step_images_from_session()
                clear_story_prompt_assets()
                st.rerun()
        else:
            if st.button("다음 단계로", key=f"story_next_{chap_idx}"):
                st.session_state.story_chapter_index = chap_idx + 1
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

        st.subheader("딜레마 토론 상황")
        show_step_illustration_small("step_img_debate", debate.get("story", st.session_state.topic), width_px=300)

        if debate.get("case_title"):
            st.write("사례:", debate.get("case_title", ""))
        if debate.get("case_summary"):
            st.write("요약:", debate.get("case_summary", ""))

        render_story_box(debate.get("story", ""))

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
            st.subheader("선택")
            c1, c2 = st.columns(2)
            with c1:
                st.success("A: " + debate.get("choice_a", ""))
            with c2:
                st.warning("B: " + debate.get("choice_b", ""))

            pick = st.radio("A/B 선택", ["A", "B"], horizontal=True, key="deb_pick")
            opening_reason = st.text_area("왜 그렇게 생각하나요?", key="deb_opening_reason", placeholder="2~6줄")

            if st.button("제출(후속 질문 시작)", key="deb_start"):
                if not opening_reason.strip():
                    st.warning("이유 입력 필요.")
                else:
                    choice_text = debate.get("choice_a") if pick == "A" else debate.get("choice_b")
                    msg = f"선택: {pick} / {choice_text}\n이유: {opening_reason.strip()}"
                    st.session_state.debate_msgs.append({"role": "student", "content": msg})

                    q1 = debate_next_question(
                        st.session_state.topic,
                        debate.get("story", ""),
                        st.session_state.debate_msgs,
                        1,
                        rag_ctx
                    )
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
                        qn = debate_next_question(
                            st.session_state.topic,
                            debate.get("story", ""),
                            st.session_state.debate_msgs,
                            t + 1,
                            rag_ctx
                        )
                        st.session_state.debate_msgs.append({"role": "assistant", "content": qn})
                        st.session_state.debate_turn = t + 1
                    else:
                        st.session_state.debate_turn = 4
                    st.rerun()

        else:
            st.subheader("정리")
            st.write(closing.get("story", ""))
            st.write("질문:", closing.get("question", ""))

            closing_ans = st.text_area("최종 정리 답", key="deb_close_ans", placeholder="2~6줄(규칙/원칙 형태)")
            if st.button("제출(최종 피드백)", key="deb_finish"):
                if not closing_ans.strip():
                    st.warning("입력 필요.")
                else:
                    transcript = "\n\n".join(
                        [("학생: " if m["role"] == "student" else "선생님: ") + m["content"] for m in st.session_state.debate_msgs]
                    )
                    answer = f"[토론 기록]\n{transcript}\n\n[최종 정리]\n{closing_ans.strip()}"

                    with st.spinner("최종 피드백..."):
                        fb = feedback_with_tags(
                            debate.get("story", ""),
                            answer,
                            rag_ctx,
                            extra_context="딜레마 토론(3턴) 최종 정리"
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
                        "debate_msgs": st.session_state.debate_msgs,
                        "closing": closing_ans.strip(),
                        "feedback": fb,
                    })

            if st.button("처음으로(학생)", key="deb_restart"):
                st.session_state.debate_turn = 0
                st.session_state.debate_msgs = []
                clear_step_images_from_session()
                st.rerun()

    # =====================================================
    # Logs download
    # =====================================================
    if st.session_state.logs:
        st.divider()
        st.download_button(
            "학습 로그 다운로드(JSON)",
            data=json.dumps(st.session_state.logs, ensure_ascii=False, indent=2),
            file_name="ethics_learning_log.json",
            mime="application/json",
        )
