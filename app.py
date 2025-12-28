import streamlit as st
from openai import OpenAI
import re
import os
import json
import datetime

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 윤리 및 교육 학습 시스템", page_icon="🤖", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    # Streamlit Cloud의 Secrets에서 API 키를 가져옵니다.
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ OpenAI API 키를 설정해주세요! (Streamlit Cloud Settings -> Secrets 확인)")
    st.stop()

# --- 3. [핵심] 시스템 페르소나 ---
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)을 위한 교육 튜터 '테스트 봇'입니다.
교사나 학생이 입력한 주제가 무엇이든, 그 상황 속에서 고민해볼 만한 '선택의 순간'을 포함한 교육 시나리오를 만들어 학습을 돕습니다.

[핵심 행동 수칙]
1. [자유로운 생성]: 주제가 무엇이든 거부하지 말고 재미있는 교육용 딜레마 시나리오를 만드세요.
2. [교육과정 연계]: 설명할 때 도덕, 사회, 실과 등 초등 교과 과정과 자연스럽게 연결하세요.
3. [개인정보 철벽 방어]: 학생이 개인정보를 말하려 하면 즉시 교육적으로 제지하세요.
4. [말투]: "안녕! 나는 테스트 봇이야", "~했니?" 처럼 다정하고 친근한 초등 교사 말투를 사용하세요.
"""

# --- 4. 함수 정의 ---

def ask_gpt_json(prompt, max_tokens=2048):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI 응답 오류: {e}")
        return None

def ask_gpt_text(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI 응답 오류: {e}")
        return None

def pii_filter(text):
    original_text = text
    text = re.sub(r'01\d{1}[-\s]?\d{3,4}[-\s]?\d{4}', '[전화번호]', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[이메일 주소]', text)
    text = re.sub(r'\d{6}[-\s]?[1-4]\d{6}', '[주민번호]', text)
    if original_text != text:
        st.warning("⚠️ 개인정보 보호를 위해 일부 내용이 마스킹되었습니다.")
    return text

def create_scenario(topic):
    prompt = (
        f"# 주제: '{topic}'\n\n"
        "이 주제를 바탕으로 초등학생이 고민해볼 만한 선택지가 포함된 교육용 시나리오를 생성하세요.\n"
        "규칙: 3~5단계로 구성하고, 각 단계는 짧은 2~3문장으로 작성하세요. 반드시 JSON 형식으로만 응답하세요.\n"
        "출력형식: {\"scenario\": [{\"story\": \"내용\", \"choice_a\": \"선택A\", \"choice_b\": \"선택B\"}]}"
    )
    raw_json = ask_gpt_json(prompt)
    return json.loads(raw_json) if raw_json else None

def analyze_scenario(topic, parsed_scenario):
    story_context = "\n".join([f"[{i+1}단계] {item.get('story')}" for i, item in enumerate(parsed_scenario)])
    prompt = (
        f"주제 '{topic}'에 대한 시나리오입니다:\n{story_context}\n\n"
        "이 시나리오를 분석하여 [핵심 가치], [연계 교과], [학습 목표]를 3줄로 요약해줘."
    )
    analysis = ask_gpt_text(prompt)
    return analysis

# --- 5. 메인 앱 로직 ---

# 세션 초기화
for key in ['scenario', 'topic', 'current_step', 'feedback_stage', 'analysis']:
    if key not in st.session_state:
        st.session_state[key] = 0 if key in ['current_step', 'feedback_stage'] else ""

st.sidebar.title("🏫 AI 학습 지원 시스템")
mode = st.sidebar.radio("모드 선택", ["학생용 (학습 참여)", "교사용 (수업 개설)"])

if mode == "교사용 (수업 개설)":
    st.header("👨‍🏫 교사용: 자유 주제 수업 설계")
    topic = st.text_input("수업 주제를 입력하세요", value=st.session_state.topic)
    
    if st.button("🚀 시나리오 생성"):
        with st.spinner("AI가 수업을 구성 중입니다..."):
            data = create_scenario(topic)
            if data:
                st.session_state.scenario = data['scenario']
                st.session_state.topic = topic
                st.session_state.analysis = analyze_scenario(topic, data['scenario'])
                st.session_state.current_step = 0
                st.success("생성 완료!")

    if st.session_state.analysis:
        st.info(st.session_state.analysis)

elif mode == "학생용 (학습 참여)":
    if not st.session_state.scenario:
        st.warning("선생님이 먼저 수업을 개설해야 합니다.")
    else:
        idx = st.session_state.current_step
        step = st.session_state.scenario[idx]
        st.subheader(f"📖 {st.session_state.topic} 이야기 ({idx+1}/{len(st.session_state.scenario)})")
        st.write(step['story'])
        
        c1, c2 = st.columns(2)
        if c1.button(f"🅰️ {step['choice_a']}"):
            st.session_state.feedback_stage = 1; st.rerun()
        if c2.button(f"🅱️ {step['choice_b']}"):
            st.session_state.feedback_stage = 1; st.rerun()
            
        if st.session_state.feedback_stage == 1:
            st.success("잘 선택했어! 다음 단계로 넘어가볼까?")
            if st.button("다음 이야기로 ➡️"):
                if idx < len(st.session_state.scenario) - 1:
                    st.session_state.current_step += 1
                    st.session_state.feedback_stage = 0
                else:
                    st.balloons(); st.success("오늘의 학습 완료!")
                st.rerun()
