import streamlit as st
from openai import OpenAI
import json
import math
import hashlib
import requests
from typing import List, Dict, Any, Tuple

# =========================
# 0) 기본 설정
# =========================
st.set_page_config(page_title="AI 윤리 학습", page_icon="🤖", layout="wide")

# =========================
# 1) OpenAI 클라이언트
# =========================
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ API 키 오류: secrets.toml을 확인하세요.")
    st.stop()

TEXT_MODEL = "gpt-4o"
IMAGE_MODEL = "dall-e-3"
EMBED_MODEL = "text-embedding-3-small"

# =========================
# 2) RAG: reference.txt (GitHub RAW URL) - UI 없이 내부에서만 사용
# =========================
# TODO: 본인 GitHub raw 링크로 교체
REFERENCE_URL = "https://raw.githubusercontent.com/USERNAME/REPO/main/reference.txt"

def _safe_strip(s: str) -> str:
    return (s or "").strip()

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_reference_text(url: str) -> str:
    """GitHub raw의 reference.txt를 로드 (캐시)"""
    if not url or "raw.githubusercontent.com" not in url:
        return ""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.text
        return ""
    except Exception:
        return ""

def split_chunks(text: str) -> List[str]:
    """빈 줄 기준 chunk 분리"""
    text = _safe_strip(text)
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    # 너무 짧은 chunk 제거
    return [p for p in parts if len(p) >= 30]

def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def embed_one(text: str) -> List[float]:
    res = client.embeddings.create(model=EMBED_MODEL, input=text)
    return res.data[0].embedding

@st.cache_resource(show_spinner=False)
def build_rag_index(url: str) -> Dict[str, Any]:
    """reference.txt -> chunks + embeddings (캐시 리소스)"""
    raw = load_reference_text(url)
    chunks = split_chunks(raw)
    if not chunks:
        return {"chunks": [], "embeddings": []}

    # 임베딩은 batch로 요청
    try:
        emb_res = client.embeddings.create(model=EMBED_MODEL, input=chunks)
        embs = [d.embedding for d in emb_res.data]
        return {"chunks": chunks, "embeddings": embs}
    except Exception:
        return {"chunks": chunks, "embeddings": []}

def retrieve_rag_context(query: str, top_k: int = 6) -> str:
    """query 기반 top-k chunk 반환"""
    idx = build_rag_index(REFERENCE_URL)
    chunks = idx.get("chunks", [])
    embs = idx.get("embeddings", [])

    if not chunks:
        return ""
    if not embs or len(embs) != len(chunks):
        # 임베딩 실패 시 키워드 기반 간단 fallback
        q = query.lower()
        hits = []
        for c in chunks:
            score = 0
            for token in q.split():
                if token and token in c.lower():
                    score += 1
            if score > 0:
                hits.append((score, c))
        hits.sort(key=lambda x: x[0], reverse=True)
        return "\n\n".join([h[1] for h in hits[:top_k]])

    q_emb = embed_one(query)
    scored = [(cosine_sim(q_emb, e), c) for e, c in zip(embs, chunks)]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for s, c in scored[:top_k] if s > 0]
    return "\n\n".join(top)

# =========================
# 3) 시스템 페르소나
# =========================
SYSTEM_PERSONA = """
당신은 AI 윤리 교육 튜터입니다.
- 핵심만 간단히, 개조식 중심
- 학생 수준(초등 고학년) 고려
- 법적 결론을 단정하지 말고, '확인해야 할 것'을 제시
- 가능하면 '근거 1개 + 대안 1개' 형태로 제시
"""

# =========================
# 4) 공통 호출 함수
# =========================
def ask_gpt_json_object(prompt: str) -> Dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        data = json.loads(response.choices[0].message.content.strip())
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}

def ask_gpt_text(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "응답 불가."

def generate_image_url(user_prompt: str) -> str:
    """항상 '그림만' 나오도록 강한 제약 프롬프트를 추가"""
    base = _safe_strip(user_prompt)
    if not base:
        return ""

    safety = (
        "Create a minimalist flat illustration. "
        "NO TEXT, NO LETTERS, NO NUMBERS, NO WORDS, NO LOGOS, NO WATERMARKS. "
        "No captions, no signs, no posters, no book covers, no UI text. "
        "Simple shapes, child-friendly, clean background."
    )
    final_prompt = f"{safety}\nScene instruction (Korean allowed): {base}"

    try:
        res = client.images.generate(
            model=IMAGE_MODEL,
            prompt=final_prompt,
            size="1024x1024",
            n=1,
        )
        return res.data[0].url
    except Exception:
        return ""

def _hash_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

# =========================
# 5) 세션 상태 초기화
# =========================
def init_state():
    defaults = {
        "mode": "👨‍🏫 교사용",
        "topic": "",
        "lesson_type": "",
        "lesson": {},
        "analysis_pack": {"ethics": [], "curriculum": [], "lesson_content": []},
        "teacher_guide": [],
        "teacher_feedback_context": "",
        "tutorial_done": False,
        "tutorial_step": 1,
        "current_step": 0,
        "chat_history": [],
        # story mode
        "story_setup": {},
        "story_outline": [],
        "story_current": {},
        "story_history": [],
        # deep debate
        "debate_turn": 0,
        "debate_msgs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================
# 6) 캐시/세션 정리 유틸
# =========================
def clear_generated_images():
    # 이미지 관련 세션 키 제거
    for k in list(st.session_state.keys()):
        if k.startswith(("img_", "g_img_", "tut_img_")):
            del st.session_state[k]

def clear_lesson_runtime_state():
    st.session_state.current_step = 0
    st.session_state.chat_history = []
    st.session_state.story_current = {}
    st.session_state.story_history = []
    st.session_state.debate_turn = 0
    st.session_state.debate_msgs = []
    clear_generated_images()

def hard_refresh_all():
    # 캐시 제거(요청하신 "캐시값 삭제"가 여기)
    st.cache_data.clear()
    st.cache_resource.clear()

    # 수업 관련 세션 제거
    keys_to_reset = [
        "lesson_type", "lesson", "analysis_pack", "teacher_guide",
        "current_step", "chat_history",
        "story_setup", "story_outline", "story_current", "story_history",
        "debate_turn", "debate_msgs",
    ]
    for k in keys_to_reset:
        if k in st.session_state:
            del st.session_state[k]
    init_state()
    clear_generated_images()

# =========================
# 7) 수업 생성(3유형 + 예시)
# =========================
LESSON_IMAGE_PROMPT = "이미지 프롬프트형"
LESSON_STORY_MODE = "스토리 모드형"
LESSON_DEEP_DEBATE = "심화 대화 토론형"

def normalize_analysis_pack(obj: Any) -> Dict[str, List[str]]:
    """analysis_pack을 항상 {ethics:[], curriculum:[], lesson_content:[]} 형태로 강제"""
    base = {"ethics": [], "curriculum": [], "lesson_content": []}
    if not isinstance(obj, dict):
        return base
    for k in base.keys():
        v = obj.get(k, [])
        if isinstance(v, str):
            v = [s.strip() for s in v.split("\n") if s.strip()]
        if isinstance(v, list):
            base[k] = [str(x).strip() for x in v if str(x).strip()]
    return base

def generate_lesson_image_prompt(topic: str, rag_ctx: str) -> Dict[str, Any]:
    prompt = f"""
주제: {topic}

아래 참고자료(RAG)를 근거로, '이미지 프롬프트형' 수업을 JSON으로 생성.
- 초등 고학년
- 학생이 프롬프트를 입력해 이미지를 만들고(글자 없는 그림), 점검 후 수정(2차 생성)하는 단계 포함
- 이후 선택+이유(간단 딜레마) 1회 포함
- 마지막에 토론 정리(규칙 만들기) 포함

[필수 JSON 구조]
{{
  "analysis_pack": {{
    "ethics": ["...","..."],
    "curriculum": ["...","..."],
    "lesson_content": ["...","..."]
  }},
  "teacher_guide": ["사용법 1", "사용법 2", "..."],
  "steps": [
    {{
      "type": "image_activity",
      "title": "...",
      "scenario": "...",
      "prompt_instruction": "학생에게 안내할 프롬프트 작성 지시(한국어)",
      "checklist": ["...","..."],
      "reflection_questions": ["...","..."]
    }},
    {{
      "type": "choice_activity",
      "story": "...",
      "choice_a": "...",
      "choice_b": "...",
      "question": "A/B 중 무엇을 선택? 이유는?"
    }},
    {{
      "type": "wrapup",
      "task": "토론 정리 과제(규칙 3줄 등)",
      "starter_questions": ["...","..."]
    }}
  ]
}}

[RAG 참고자료]
{rag_ctx}
"""
    data = ask_gpt_json_object(prompt)
    steps = data.get("steps", [])
    if not isinstance(steps, list) or len(steps) < 3:
        steps = [
            {
                "type": "image_activity",
                "title": "이미지 만들기 및 수정",
                "scenario": f"'{topic}' 관련 상황에서 학생이 직접 이미지를 만들어 본다.",
                "prompt_instruction": "글자 없는 그림이 나오도록, 대상/배경/행동을 구체적으로 써서 프롬프트 작성.",
                "checklist": ["글자/숫자 없음", "로고/상표 없음", "실존 인물 유사 없음", "개인정보 암시 없음"],
                "reflection_questions": ["무엇을 왜 수정했나?", "더 안전한 대안은?"],
            },
            {
                "type": "choice_activity",
                "story": f"'{topic}' 상황에서 친구가 네가 만든 결과물을 그대로 쓰자고 한다.",
                "choice_a": "허락/출처 확인 후 사용하자",
                "choice_b": "교육 목적이니 그냥 쓰자",
                "question": "어느 쪽? 이유 2문장.",
            },
            {
                "type": "wrapup",
                "task": "우리 반 규칙 3줄 만들기(허락/출처/목적).",
                "starter_questions": ["누구의 권리가 관련?", "확인해야 할 것 2가지?"],
            },
        ]

    return {
        "lesson_type": LESSON_IMAGE_PROMPT,
        "analysis_pack": normalize_analysis_pack(data.get("analysis_pack", {})),
        "teacher_guide": data.get("teacher_guide", []),
        "steps": steps,
        "rag_used": True,
    }

def generate_lesson_story_mode(topic: str, rag_ctx: str) -> Dict[str, Any]:
    prompt = f"""
주제: {topic}

아래 RAG 자료를 근거로, '스토리 모드형(문제해결 5막)' 수업 JSON 생성.
- 초등 고학년
- 5막 개요(outline)와 1막(first_chapter)을 포함
- first_chapter는 선택지 2개(A/B) + 질문 포함
- 분석 결과는 analysis_pack(윤리기준/연계교육과정/수업내용)로 채우기

[필수 JSON 구조]
{{
  "analysis_pack": {{ "ethics": [...], "curriculum": [...], "lesson_content": [...] }},
  "teacher_guide": ["..."],
  "story_setup": {{
    "setting": "...",
    "goal": "...",
    "characters": ["..."],
    "constraints": ["..."]
  }},
  "outline": ["1막 ...", "2막 ...", "3막 ...", "4막 ...", "5막 ..."],
  "first_chapter": {{
    "chapter_index": 1,
    "story": "...",
    "options": {{
      "A": "...",
      "B": "..."
    }},
    "question": "A/B 선택 + 이유"
  }}
}}

[RAG 참고자료]
{rag_ctx}
"""
    data = ask_gpt_json_object(prompt)

    story_setup = data.get("story_setup", {})
    outline = data.get("outline", [])
    first = data.get("first_chapter", {})

    if not isinstance(outline, list) or len(outline) < 5:
        outline = [
            "1막: 갈등 제시", "2막: 단서 선택", "3막: 대안 설계", "4막: 검증/수정", "5막: 규칙 만들기"
        ]
    if not isinstance(first, dict) or not first.get("story"):
        first = {
            "chapter_index": 1,
            "story": f"{topic} 관련 갈등이 생겼다. 무엇을 먼저 확인해야 할까?",
            "options": {"A": "출처/허락/목적부터 확인", "B": "일단 빠르게 진행"},
            "question": "A/B 선택하고 이유를 말해보기",
        }

    return {
        "lesson_type": LESSON_STORY_MODE,
        "analysis_pack": normalize_analysis_pack(data.get("analysis_pack", {})),
        "teacher_guide": data.get("teacher_guide", []),
        "story_setup": story_setup if isinstance(story_setup, dict) else {},
        "outline": outline,
        "first_chapter": first,
        "rag_used": True,
    }

def generate_story_next_chapter(topic: str, setup: Dict[str, Any], history: List[Dict[str, Any]], next_idx: int, rag_ctx: str) -> Dict[str, Any]:
    prompt = f"""
주제: {topic}
스토리 설정: {json.dumps(setup, ensure_ascii=False)}
이전 기록: {json.dumps(history, ensure_ascii=False)}

다음 {next_idx}막(장면)을 JSON으로 생성.
- 초등 고학년
- story: 3~5문장
- options: A/B 각 1문장
- question: 1문장
- ending: true/false (5막이면 true)
- debrief: ending=true일 때 배운점 3줄

[JSON]
{{
  "chapter_index": {next_idx},
  "story": "...",
  "options": {{ "A": "...", "B": "..." }},
  "question": "...",
  "ending": false,
  "debrief": ["...","...","..."]
}}

[RAG 참고자료]
{rag_ctx}
"""
    data = ask_gpt_json_object(prompt)
    if not isinstance(data, dict) or not data.get("story"):
        data = {
            "chapter_index": next_idx,
            "story": f"{topic} 문제를 해결하기 위해 더 확인할 점이 있다.",
            "options": {"A": "안전한 대안을 선택", "B": "편한 길을 선택"},
            "question": "A/B 선택 + 이유",
            "ending": next_idx >= 5,
            "debrief": ["근거 확인", "대안 제시", "규칙 만들기"] if next_idx >= 5 else [],
        }
    return data

def generate_lesson_deep_debate(topic: str, rag_ctx: str) -> Dict[str, Any]:
    prompt = f"""
주제: {topic}

아래 RAG 자료를 근거로, '심화 대화 토론형(후속질문 3턴)' 수업 JSON 생성.
- 초등 고학년
- debate_step(상황/오프닝질문/규칙) + closing_step(정리 질문)
- 분석 결과는 analysis_pack로 채우기

[필수 JSON 구조]
{{
  "analysis_pack": {{ "ethics": [...], "curriculum": [...], "lesson_content": [...] }},
  "teacher_guide": ["..."],
  "debate_step": {{
    "story": "...",
    "opening_question": "...",
    "rules": ["근거 제시", "반대 의견 존중", "대안 제시"]
  }},
  "closing_step": {{
    "question": "최종 규칙 3줄로 정리"
  }}
}}

[RAG 참고자료]
{rag_ctx}
"""
    data = ask_gpt_json_object(prompt)
    debate = data.get("debate_step", {})
    closing = data.get("closing_step", {})
    if not isinstance(debate, dict) or not debate.get("story"):
        debate = {
            "story": f"{topic} 상황에서 서로 다른 입장이 생겼다.",
            "opening_question": "입장 1개 선택 + 근거 1개",
            "rules": ["근거를 말하기", "상대 존중", "대안 1개 제시"],
        }
    if not isinstance(closing, dict) or not closing.get("question"):
        closing = {"question": "우리 반 규칙 3줄로 정리(허락/출처/목적 등 포함)"}

    return {
        "lesson_type": LESSON_DEEP_DEBATE,
        "analysis_pack": normalize_analysis_pack(data.get("analysis_pack", {})),
        "teacher_guide": data.get("teacher_guide", []),
        "debate_step": debate,
        "closing_step": closing,
        "rag_used": True,
    }

def generate_example_copyright_lesson() -> Dict[str, Any]:
    """교사용 예시: 저작권 + 이미지 생성 + 토론 흐름(LLM 호출 없이 고정 예시)"""
    return {
        "lesson_type": LESSON_IMAGE_PROMPT,
        "analysis_pack": {
            "ethics": ["프라이버시 보호(개인정보/식별 위험 점검)", "데이터 관리(동의/출처/사용 범위 확인)", "침해 금지(무단 사용/오용 예방)", "안전성(검증/통제)"],
            "curriculum": ["도덕: 규칙/책임/배려 기반 토론", "실과: 디지털 도구 활용과 창작/공유 책임"],
            "lesson_content": ["학생이 프롬프트로 이미지를 생성(글자 없는 그림)", "생성물의 권리/허락/출처/목적을 근거로 토론", "우리 반 창작물 사용 규칙 3줄 작성"],
        },
        "teacher_guide": [
            "주제 입력 후 수업 유형 버튼 클릭",
            "분석 결과(윤리기준/교육과정/수업내용)를 확인하고 수업 흐름 안내",
            "학생 활동 중: 허락/출처/목적/범위 확인 질문을 반복",
            "피드백 기준(교사 의견)을 입력하면 학생 피드백에 반영됨",
        ],
        "steps": [
            {
                "type": "image_activity",
                "title": "AI로 이미지 만들기(저작권 토론용)",
                "scenario": "학교 행사 안내 포스터에 넣을 '상징 그림'이 필요하다. 학생이 프롬프트로 이미지를 만든다.",
                "prompt_instruction": "글자 없는 그림으로, 대상/배경/분위기를 구체적으로 적어 프롬프트를 작성하라.",
                "checklist": ["글자/숫자 없음", "로고/상표 없음", "유명 캐릭터 유사 없음", "실존 인물 유사 없음"],
                "reflection_questions": ["이 이미지의 저작권/권리는 누구에게 있다고 볼 수 있나?", "무엇을 확인하면 공정해지나(약관/허락/출처/목적)?"],
            },
            {
                "type": "choice_activity",
                "story": "친구가 네가 만든 이미지를 다른 반 포스터에도 그대로 쓰자고 한다.",
                "choice_a": "사용 목적과 범위를 정하고, 출처/허락을 확인한 뒤 사용",
                "choice_b": "학교 일이니까 자유롭게 가져다 써도 된다",
                "question": "A/B 중 선택하고 이유 2문장",
            },
            {
                "type": "wrapup",
                "task": "우리 반 규칙 3줄(허락/출처/목적/범위 포함) 작성",
                "starter_questions": ["누가 어떤 기여를 했나?", "문제가 생기면 어떻게 수정/중단할까?"],
            },
        ],
        "rag_used": True,
    }

# =========================
# 8) 피드백(교사 의견 반영 + RAG 반영)
# =========================
def feedback_with_teacher_context(activity_context: str, student_input: str, rag_ctx: str, teacher_ctx: str) -> str:
    teacher_ctx = _safe_strip(teacher_ctx)
    prompt = f"""
[활동 맥락]
{activity_context}

[학생 입력]
{student_input}

[교사 피드백 기준/강조점]
{teacher_ctx if teacher_ctx else "(없음)"}

[RAG 참고자료]
{rag_ctx}

요구:
- 초등 고학년 수준
- 개조식으로 3~6줄
- (근거 1개) + (대안 1개) 포함
- 법적 결론 단정 금지(대신 확인해야 할 것 제시)
"""
    return ask_gpt_text(prompt)

def debate_next_question(topic: str, story: str, debate_msgs: List[Dict[str, str]], turn_index: int, rag_ctx: str) -> str:
    prompt = f"""
주제: {topic}
상황: {story}
지금까지 학생 발언/질문 기록: {json.dumps(debate_msgs, ensure_ascii=False)}

이제 {turn_index}번째 후속 질문 1개만 생성.
- 초등 고학년
- 근거/반례/대안/규칙 중 하나를 더 깊게 묻기
- 1문장

[RAG 참고자료]
{rag_ctx}
"""
    return ask_gpt_text(prompt).strip()

# =========================
# 9) 사이드바
# =========================
st.sidebar.title("🤖 AI 윤리 학습")

mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"], index=0)
st.session_state.mode = mode

st.sidebar.caption("RAG: reference.txt 내부 적용(설정 UI 없음)")

if st.sidebar.button("🧹 콘텐츠 새로고침(캐시/세션 초기화)"):
    hard_refresh_all()
    st.rerun()

# =========================
# 10) 교사용 화면
# =========================
if mode == "👨‍🏫 교사용":
    st.header("🛠️ 수업 생성")

    topic = st.text_input("수업 주제 입력", value=st.session_state.topic, placeholder="예: 저작권, 개인정보, 딥페이크, 편향, 추천 알고리즘...")
    st.session_state.topic = topic

    st.subheader("수업 유형 선택")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])

    with c4:
        st.text_area(
            "교사 피드백 기준/강조점(학생 피드백에 반영)",
            key="teacher_feedback_context",
            height=130,
            placeholder="예: 출처/허락/목적/범위 확인을 꼭 언급하게 하기. 단정 금지. 대안 제시 유도."
        )

    def _run_generate(which: str):
        if not _safe_strip(st.session_state.topic) and which != "EXAMPLE":
            st.warning("주제 입력 필요.")
            return

        with st.spinner("생성 중..."):
            if which == "EXAMPLE":
                lesson = generate_example_copyright_lesson()
            else:
                q = f"{st.session_state.topic} / {which}"
                rag_ctx = retrieve_rag_context(q, top_k=6)
                if which == LESSON_IMAGE_PROMPT:
                    lesson = generate_lesson_image_prompt(st.session_state.topic, rag_ctx)
                elif which == LESSON_STORY_MODE:
                    lesson = generate_lesson_story_mode(st.session_state.topic, rag_ctx)
                else:
                    lesson = generate_lesson_deep_debate(st.session_state.topic, rag_ctx)

            # ---- 핵심: 분석 결과 3칸이 항상 뜨도록 키/형식 통일 저장 ----
            st.session_state.lesson = lesson
            st.session_state.lesson_type = lesson.get("lesson_type", "")
            st.session_state.analysis_pack = normalize_analysis_pack(lesson.get("analysis_pack", {}))
            st.session_state.teacher_guide = lesson.get("teacher_guide", [])
            clear_lesson_runtime_state()

        st.success("생성 완료.")
        st.rerun()  # 화면에 바로 반영

    with c1:
        if st.button("이미지 프롬프트형", use_container_width=True):
            _run_generate(LESSON_IMAGE_PROMPT)

    with c2:
        if st.button("스토리 모드형", use_container_width=True):
            _run_generate(LESSON_STORY_MODE)

    with c3:
        if st.button("심화 대화 토론형", use_container_width=True):
            _run_generate(LESSON_DEEP_DEBATE)

    st.divider()

    # 교사용 사용법 가이드(요청: "사용법" 중심)
    with st.expander("⚙️ 교사용 사용법(가이드)", expanded=True):
        guide = st.session_state.get("teacher_guide", [])
        if isinstance(guide, list) and guide:
            for g in guide:
                st.markdown(f"- {g}")
        else:
            st.markdown(
                "- 주제 입력 후 수업 유형 버튼 클릭\n"
                "- 상단 분석 결과(윤리기준/연계교육과정/수업내용) 확인\n"
                "- 아래 미리보기로 수업 흐름 파악\n"
                "- 교사 피드백 기준을 입력하면 학생 피드백에 반영"
            )

    # ---- 분석 결과(3칸) 출력: 무조건 analysis_pack에서만 읽음 ----
    st.subheader("📊 분석 결과")
    ap = st.session_state.get("analysis_pack", {"ethics": [], "curriculum": [], "lesson_content": []})
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("### 인공지능 윤리기준")
        ethics = ap.get("ethics", [])
        if ethics:
            st.markdown("\n".join([f"- {x}" for x in ethics]))
        else:
            st.caption("내용 없음.")

    with colB:
        st.markdown("### 연계 교육과정")
        cur = ap.get("curriculum", [])
        if cur:
            st.markdown("\n".join([f"- {x}" for x in cur]))
        else:
            st.caption("내용 없음.")

    with colC:
        st.markdown("### 수업 내용")
        content = ap.get("lesson_content", [])
        if content:
            st.markdown("\n".join([f"- {x}" for x in content]))
        else:
            st.caption("내용 없음.")

    st.divider()

    # ---- 미리보기 ----
    lesson_type = st.session_state.get("lesson_type", "")
    lesson = st.session_state.get("lesson", {})

    if not lesson_type:
        st.info("수업 유형을 선택하면 미리보기가 표시됩니다.")
    else:
        if lesson_type == LESSON_IMAGE_PROMPT:
            st.subheader("🧩 이미지 프롬프트형 미리보기")
            for i, step in enumerate(lesson.get("steps", []), start=1):
                with st.container(border=True):
                    st.markdown(f"#### 단계 {i} - {step.get('type','')}")
                    if step.get("type") == "image_activity":
                        st.markdown(f"**제목:** {step.get('title','')}")
                        st.markdown(f"**상황:** {step.get('scenario','')}")
                        st.markdown(f"**프롬프트 안내:** {step.get('prompt_instruction','')}")
                    elif step.get("type") == "choice_activity":
                        st.markdown(f"**상황:** {step.get('story','')}")
                        st.success(f"🅰️ {step.get('choice_a','')}")
                        st.warning(f"🅱️ {step.get('choice_b','')}")
                    elif step.get("type") == "wrapup":
                        st.markdown(f"**정리 과제:** {step.get('task','')}")
        elif lesson_type == LESSON_STORY_MODE:
            st.subheader("📘 스토리 모드 미리보기")
            setup = lesson.get("story_setup", {})
            with st.container(border=True):
                st.markdown(f"**설정:** {setup.get('setting','')}")
                st.markdown(f"**목표:** {setup.get('goal','')}")
                st.markdown(f"**등장인물:** {', '.join(setup.get('characters', [])) if isinstance(setup.get('characters', []), list) else ''}")
                cons = setup.get("constraints", [])
                if isinstance(cons, list) and cons:
                    st.markdown("**제약/윤리 기준:** " + ", ".join(cons))
            with st.container(border=True):
                st.markdown("### 5막 개요")
                outline = lesson.get("outline", [])
                for x in outline:
                    st.markdown(f"- {x}")
        else:
            st.subheader("💬 심화 토론 미리보기")
            debate = lesson.get("debate_step", {})
            with st.container(border=True):
                st.markdown(debate.get("story", ""))
                st.markdown(f"**오프닝 질문:** {debate.get('opening_question','')}")
                rules = debate.get("rules", [])
                if isinstance(rules, list) and rules:
                    st.markdown("**토론 규칙:**")
                    for r in rules:
                        st.markdown(f"- {r}")

# =========================
# 11) 학생용 화면
# =========================
else:
    st.header("🙋‍♂️ 학생용")

    # --- 튜토리얼(요청 반영: 탕수육 제거 + 선택지 연습 + 프롬프트 입력해 이미지 출력 확인) ---
    if not st.session_state.tutorial_done:
        st.subheader("🎒 연습(선택 + 프롬프트 이미지 테스트)")
        st.progress(st.session_state.tutorial_step / 3)

        if st.session_state.tutorial_step == 1:
            st.markdown("### 1) 선택지 연습")
            a, b = st.columns(2)
            with a:
                if st.button("선택지 A", use_container_width=True):
                    st.toast("선택: A")
                    st.session_state.tutorial_step = 2
                    st.rerun()
            with b:
                if st.button("선택지 B", use_container_width=True):
                    st.toast("선택: B")
                    st.session_state.tutorial_step = 2
                    st.rerun()

        elif st.session_state.tutorial_step == 2:
            st.markdown("### 2) 프롬프트 입력 연습")
            st.caption("짧게 입력해도 됨. 예: '교실에서 책 읽는 로봇, 글자 없는 그림'")
            tut_prompt = st.text_input("이미지 프롬프트", key="tut_prompt")
            if st.button("이미지 생성 테스트"):
                with st.spinner("생성 중..."):
                    url = generate_image_url(tut_prompt)
                    if url:
                        st.session_state["tut_img_1"] = url
                        st.session_state.tutorial_step = 3
                        st.rerun()
                    else:
                        st.warning("이미지 생성 실패. 프롬프트를 바꿔보세요.")

        elif st.session_state.tutorial_step == 3:
            st.markdown("### 3) 확인")
            url = st.session_state.get("tut_img_1", "")
            if url:
                st.image(url, caption="테스트 이미지(항상 표시)", use_container_width=False)
            if st.button("수업 입장"):
                st.session_state.tutorial_done = True
                st.rerun()

    # --- 실전 수업 ---
    else:
        lesson = st.session_state.get("lesson", {})
        lesson_type = st.session_state.get("lesson_type", "")

        if not lesson_type or not lesson:
            st.warning("수업 데이터 없음. 교사용에서 수업 생성 필요.")
            if st.button("연습으로 돌아가기"):
                st.session_state.tutorial_done = False
                st.session_state.tutorial_step = 1
                st.rerun()
        else:
            topic = st.session_state.get("topic", "")
            rag_ctx = retrieve_rag_context(f"{topic} / {lesson_type}", top_k=6)
            teacher_ctx = st.session_state.get("teacher_feedback_context", "")

            # ========== 1) 이미지 프롬프트형 ==========
            if lesson_type == LESSON_IMAGE_PROMPT:
                steps = lesson.get("steps", [])
                idx = st.session_state.current_step

                if idx >= len(steps):
                    st.success("수업 종료.")
                    if st.button("처음으로"):
                        st.session_state.current_step = 0
                        st.session_state.tutorial_done = False
                        st.session_state.tutorial_step = 1
                        clear_lesson_runtime_state()
                        st.rerun()
                else:
                    step = steps[idx]
                    st.progress((idx + 1) / len(steps))
                    st.subheader(f"단계 {idx+1}")

                    if step.get("type") == "image_activity":
                        st.info(step.get("scenario", ""))
                        st.markdown(f"**과제:** {step.get('title','')}")
                        st.caption("이미지는 항상 표시됩니다(설정 UI 없음). 글자 없는 그림만 나오도록 작성하세요.")

                        p1 = st.text_input("1차 프롬프트", key=f"p1_{idx}")
                        if st.button("1차 이미지 생성", key=f"gen1_{idx}"):
                            with st.spinner("생성 중..."):
                                key = "g_img_" + _hash_key(p1)
                                if key not in st.session_state:
                                    st.session_state[key] = generate_image_url(p1)
                                st.session_state[f"img_1_{idx}"] = st.session_state[key]
                            st.rerun()

                        if st.session_state.get(f"img_1_{idx}", ""):
                            st.image(st.session_state[f"img_1_{idx}"], caption="1차 이미지", use_container_width=True)

                        checklist = step.get("checklist", [])
                        picked = st.multiselect("점검 체크리스트", options=checklist, key=f"chk_{idx}")

                        p2 = st.text_input("2차 프롬프트(수정)", key=f"p2_{idx}")
                        if st.button("2차 이미지 생성", key=f"gen2_{idx}"):
                            with st.spinner("생성 중..."):
                                key = "g_img_" + _hash_key(p2)
                                if key not in st.session_state:
                                    st.session_state[key] = generate_image_url(p2)
                                st.session_state[f"img_2_{idx}"] = st.session_state[key]
                            st.rerun()

                        if st.session_state.get(f"img_2_{idx}", ""):
                            st.image(st.session_state[f"img_2_{idx}"], caption="2차 이미지", use_container_width=True)

                        refl = st.text_area("수정 이유/느낀 점(2~3문장)", key=f"refl_{idx}")

                        if st.button("제출(피드백)", key=f"submit_{idx}"):
                            student_input = f"1차프롬프트:{p1}\n점검:{picked}\n2차프롬프트:{p2}\n수정이유:{refl}"
                            fb = feedback_with_teacher_context(
                                activity_context=f"[이미지 활동] {step.get('scenario','')}\n프롬프트 안내:{step.get('prompt_instruction','')}",
                                student_input=student_input,
                                rag_ctx=rag_ctx,
                                teacher_ctx=teacher_ctx,
                            )
                            st.session_state.chat_history = [
                                {"role": "user", "content": student_input},
                                {"role": "assistant", "content": fb},
                            ]
                            st.rerun()

                    elif step.get("type") == "choice_activity":
                        st.info(step.get("story", ""))
                        st.success(f"🅰️ {step.get('choice_a','A')}")
                        st.warning(f"🅱️ {step.get('choice_b','B')}")

                        with st.form(f"choice_form_{idx}"):
                            sel = st.radio("선택", ["A", "B"])
                            reason = st.text_area("이유(2~3문장)")
                            ok = st.form_submit_button("제출(피드백)")
                        if ok:
                            a_text = step.get("choice_a", "")
                            b_text = step.get("choice_b", "")
                            chosen = a_text if sel == "A" else b_text
                            student_input = f"선택:{sel}({chosen})\n이유:{reason}"
                            fb = feedback_with_teacher_context(
                                activity_context=f"[선택 활동] {step.get('story','')}",
                                student_input=student_input,
                                rag_ctx=rag_ctx,
                                teacher_ctx=teacher_ctx,
                            )
                            st.session_state.chat_history = [
                                {"role": "user", "content": student_input},
                                {"role": "assistant", "content": fb},
                            ]
                            st.rerun()

                    elif step.get("type") == "wrapup":
                        st.info(step.get("task", ""))
                        qs = step.get("starter_questions", [])
                        if isinstance(qs, list) and qs:
                            st.markdown("**생각해 볼 질문**")
                            for q in qs:
                                st.markdown(f"- {q}")
                        out = st.text_area("최종 정리(규칙 3줄)", key=f"wrap_{idx}")
                        if st.button("제출(피드백)", key=f"wrap_submit_{idx}"):
                            fb = feedback_with_teacher_context(
                                activity_context="[정리 활동] 규칙 만들기",
                                student_input=out,
                                rag_ctx=rag_ctx,
                                teacher_ctx=teacher_ctx,
                            )
                            st.session_state.chat_history = [
                                {"role": "user", "content": out},
                                {"role": "assistant", "content": fb},
                            ]
                            st.rerun()

                    # 채팅/피드백 표시
                    if st.session_state.chat_history:
                        st.divider()
                        for msg in st.session_state.chat_history:
                            st.chat_message("assistant" if msg["role"] == "assistant" else "user").write(msg["content"])

                    # 다음 단계
                    st.divider()
                    if st.button("다음 단계 >", use_container_width=True):
                        st.session_state.current_step += 1
                        st.session_state.chat_history = []
                        st.rerun()

            # ========== 2) 스토리 모드형 ==========
            elif lesson_type == LESSON_STORY_MODE:
                # 최초 세팅
                if not st.session_state.story_current:
                    st.session_state.story_setup = lesson.get("story_setup", {})
                    st.session_state.story_outline = lesson.get("outline", [])
                    st.session_state.story_current = lesson.get("first_chapter", {})
                    st.session_state.story_history = []

                chap = st.session_state.story_current
                chap_idx = int(chap.get("chapter_index", 1))
                ending = bool(chap.get("ending", False))

                st.subheader(f"스토리 {chap_idx}막")
                st.info(chap.get("story", ""))

                if ending:
                    st.success("스토리 종료")
                    debrief = chap.get("debrief", [])
                    if isinstance(debrief, list) and debrief:
                        st.markdown("**정리(배운 점)**")
                        for d in debrief:
                            st.markdown(f"- {d}")
                    if st.button("처음으로"):
                        clear_lesson_runtime_state()
                        st.session_state.tutorial_done = False
                        st.session_state.tutorial_step = 1
                        st.rerun()
                else:
                    opts = chap.get("options", {})
                    a_txt = opts.get("A", "A")
                    b_txt = opts.get("B", "B")
                    st.success(f"🅰️ {a_txt}")
                    st.warning(f"🅱️ {b_txt}")

                    with st.form("story_form"):
                        sel = st.radio("선택", ["A", "B"])
                        reason = st.text_area("이유(2~3문장)")
                        ok = st.form_submit_button("제출하고 다음 막으로")
                    if ok:
                        chosen = a_txt if sel == "A" else b_txt
                        st.session_state.story_history.append({
                            "chapter_index": chap_idx,
                            "choice": sel,
                            "choice_text": chosen,
                            "reason": reason,
                            "story": chap.get("story", "")
                        })

                        nxt = generate_story_next_chapter(
                            topic=topic,
                            setup=st.session_state.story_setup,
                            history=st.session_state.story_history,
                            next_idx=chap_idx + 1,
                            rag_ctx=rag_ctx
                        )
                        st.session_state.story_current = nxt

                        # 피드백 1회(선택 제출 시)
                        fb = feedback_with_teacher_context(
                            activity_context=f"[스토리 모드] {chap_idx}막 선택",
                            student_input=f"선택:{sel}({chosen})\n이유:{reason}",
                            rag_ctx=rag_ctx,
                            teacher_ctx=teacher_ctx,
                        )
                        st.session_state.chat_history = [
                            {"role": "user", "content": f"{sel} / {reason}"},
                            {"role": "assistant", "content": fb},
                        ]
                        st.rerun()

                    if st.session_state.chat_history:
                        st.divider()
                        for msg in st.session_state.chat_history:
                            st.chat_message("assistant" if msg["role"] == "assistant" else "user").write(msg["content"])

            # ========== 3) 심화 대화 토론형 ==========
            else:
                debate = lesson.get("debate_step", {})
                closing = lesson.get("closing_step", {})

                st.subheader("심화 대화 토론")
                st.info(debate.get("story", ""))

                rules = debate.get("rules", [])
                if isinstance(rules, list) and rules:
                    with st.expander("토론 규칙", expanded=True):
                        for r in rules:
                            st.markdown(f"- {r}")

                # 초기 assistant 질문 세팅
                if st.session_state.debate_turn == 0 and not st.session_state.debate_msgs:
                    opening_q = debate.get("opening_question", "입장 1개 + 근거 1개")
                    st.session_state.debate_msgs.append({"role": "assistant", "content": opening_q})

                # 메시지 표시
                for m in st.session_state.debate_msgs:
                    st.chat_message("assistant" if m["role"] == "assistant" else "user").write(m["content"])

                # turn 진행
                if st.session_state.debate_turn < 4:
                    user_in = st.text_area("답변 입력", key=f"deb_in_{st.session_state.debate_turn}")
                    if st.button("제출", use_container_width=True):
                        if not _safe_strip(user_in):
                            st.warning("답변 입력 필요.")
                            st.stop()

                        st.session_state.debate_msgs.append({"role": "user", "content": user_in.strip()})

                        # 후속 질문 1~3 생성
                        next_turn = st.session_state.debate_turn + 1
                        if next_turn <= 3:
                            qn = debate_next_question(
                                topic=topic,
                                story=debate.get("story", ""),
                                debate_msgs=st.session_state.debate_msgs,
                                turn_index=next_turn,
                                rag_ctx=rag_ctx
                            )
                            st.session_state.debate_msgs.append({"role": "assistant", "content": qn})
                            st.session_state.debate_turn = next_turn
                        else:
                            # closing 단계로
                            st.session_state.debate_turn = 4
                        st.rerun()
                else:
                    st.divider()
                    st.markdown("### 최종 정리")
                    st.markdown(closing.get("question", "규칙 3줄로 정리"))
                    final = st.text_area("최종 정리 답", key="deb_final")
                    if st.button("최종 피드백 받기", use_container_width=True):
                        transcript = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.debate_msgs])
                        fb = feedback_with_teacher_context(
                            activity_context="[심화 대화 토론] 전체 대화 기반 정리",
                            student_input=f"대화기록:\n{transcript}\n\n최종정리:\n{final}",
                            rag_ctx=rag_ctx,
                            teacher_ctx=teacher_ctx,
                        )
                        st.session_state.chat_history = [
                            {"role": "user", "content": final},
                            {"role": "assistant", "content": fb},
                        ]
                        st.rerun()

                    if st.session_state.chat_history:
                        st.divider()
                        for msg in st.session_state.chat_history:
                            st.chat_message("assistant" if msg["role"] == "assistant" else "user").write(msg["content"])

                    if st.button("처음으로"):
                        clear_lesson_runtime_state()
                        st.session_state.tutorial_done = False
                        st.session_state.tutorial_step = 1
                        st.rerun()
