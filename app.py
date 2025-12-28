import streamlit as st
from openai import OpenAI
import re
import os
import json
import datetime

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="테스트 봇과 함께하는 AI 교육 학습", page_icon="🤖", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    # 환경 변수에서 API 키를 가져옵니다.
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ OpenAI API 키를 설정해주세요! (Streamlit Cloud Settings -> Secrets 확인)")
    st.stop()

# --- 3. [핵심] 시스템 페르소나 ---
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)을 위한 교육 튜터 '테스트 봇'입니다.
학생이나 교사가 입력한 주제가 무엇이든, 그 상황 속에서 고민해볼 만한 '선택의 순간(딜레마)'을 포함한 시나리오를 만들어 학습을 돕습니다.

[핵심 행동 수칙]
1. [자유로운 생성]: 입력 주제가 무엇이든 거부하지 말고 재미있는 교육 시나리오로 만드세요.
2. [교육과정 연계]: 설명할 때 도덕, 사회, 실과 등 초등 교과 과정과 자연스럽게 연결해주세요.
3. [개인정보 철벽 방어]: 학생이 개인정보를 말하려 하면 즉시 교육적으로 제지하세요.
4. [말투]: "안녕! 나는 테스트 봇이야", "~했니?" 처럼 다정하고 친근한 초등 교사 말투를 사용하세요.
"""

# --- 4. 함수 정의 ---

def ask_gpt_json(prompt, max_tokens=2048):
    """GPT-4o에게 JSON 형식의 응답을 요청하는 함수"""
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
        st.error(f"GPT-4o JSON 요청 오류: {e}")
        return None

def ask_gpt_text(prompt):
    """GPT-4o에게 일반 텍스트 응답을 요청하는 함수"""
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
        st.error(f"GPT-4o 텍스트 요청 오류: {e}")
        return None

def generate_image(prompt):
    """DALL-E 3 이미지 생성 (교육용 삽화)"""
    try:
        dalle_prompt = f"A friendly, educational cartoon-style illustration for elementary school textbook, depicting: {prompt}"
        response = client.images.generate(
            model="dall-e-3", prompt=dalle_prompt, size="1024x1024", quality="standard", n=1
        )
        return response.data[0].url
    except:
        return None

def pii_filter(text):
    """정규 표현식을 사용하여 사용자 입력에서 개인정보를 탐지하고 마스킹합니다."""
    original_text = text
    text = re.sub(r'01\d{1}[-\s]?\d{3,4}[-\s]?\d{4}', '[전화번호]', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[이메일 주소]', text)
    text = re.sub(r'\d{6}[-\s]?[1-4]\d{6}', '[주민번호]', text)
    
    if original_text != text:
        st.warning("⚠️ 개인정보가 감지되어 메시지의 일부가 필터링되었습니다.")
    return text

def create_scenario(topic): 
    """주제에 따른 딜레마 시나리오 생성 요청 (주제 제한 없음)"""
    prompt = (
        f"# 주제: '{topic}'\n\n"
        "이 주제를 바탕으로 초등학생이 고민해볼 만한 선택지가 포함된 교육 시나리오를 생성하세요.\n"
        "규칙 1: 3~6단계 사이로 단계 수를 결정해.\n"
        "규칙 2: 각 단계는 2~3문장 이내로 짧게 작성해야 해. 어려운 단어는 쓰지 마.\n"
        "규칙 3: 반드시 아래 JSON 형식으로만 응답해.\n\n"
        "# 출력 형식 (JSON): \n"
        "{\"scenario\": [\n"
        "  {\"story\": \"스토리 내용\", \"choice_a\": \"선택지 A\", \"choice_b\": \"선택지 B\"}\n"
        "]}"
    )
    raw_json = ask_gpt_json(prompt)
    
    if raw_json:
        try:
            return json.loads(raw_json)
        except:
            return None
    return None

def analyze_scenario(topic, parsed_scenario):
    """생성된 시나리오를 분석하여 학습 목표 추출"""
    story_context = "\n".join([f"[{i+1}단계] {item.get('story')}" for i, item in enumerate(parsed_scenario)])

    prompt = (
        f"교사가 '{topic}' 주제로 아래 시나리오를 만들었습니다:\n{story_context}\n\n"
        "이 시나리오를 분석하여 다음 3가지 항목을 추출해 주세요.\n"
        "[핵심 가치] [이 시나리오에 근거가 되는 가치나 원칙]\n"
        "[연계 교과] [이 시나리오와 관련된 교육과정 내용]\n"
        "[학습 목표] [이 시나리오를 통해 배우게 될 핵심 내용]\n"
    )
    analysis = ask_gpt_text(prompt)
    
    result = {}
    try:
        def safe_extract(pattern, text):
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else '분석 중...'
            
        result['ethical_standard'] = safe_extract(r"\[핵심 가치\](.*?)\[연계 교과\]", analysis)
        result['achievement_std'] = safe_extract(r"\[연계 교과\](.*?)\[학습 목표\]", analysis)
        result['learning_content'] = safe_extract(r"\[학습 목표\](.*)", analysis)
    except:
        result = {'ethical_standard': '분석 실패', 'achievement_std': '분석 실패', 'learning_content': '분석 실패'}
    return result

def get_feedback(choice, reason, story_context):
    """학생에게 줄 피드백 생성"""
    prompt = (
        f"상황: {story_context}\n학생의 선택: {choice}, 이유: {reason}\n\n"
        "초등학생에게 따뜻한 말투로 공감과 칭찬을 해주고, 사고를 넓힐 수 있는 질문을 하나 던져줘. 3문장 이내로 작성해."
    )
    return ask_gpt_text(prompt)

# --- 5. 메인 앱 로직 ---

# 세션 상태 초기화
for key in ['scenario', 'scenario_images', 'current_step', 'chat_log', 'topic', 'tutorial_complete', 'feedback_stage', 'learning_records', 'lesson_complete']:
    if key not in st.session_state:
        if key in ['scenario_images', 'chat_log', 'learning_records']: st.session_state[key] = []
        elif key in ['current_step', 'feedback_stage']: st.session_state[key] = 0
        elif key in ['tutorial_complete', 'lesson_complete']: st.session_state[key] = False
        else: st.session_state[key] = ""

st.sidebar.title("🏫 AI 교육 학습 모드")
mode = st.sidebar.radio("모드를 선택하세요:", ["학생용 (수업 참여)", "교사용 (수업 개설)"])

# ==========================================
# 👨‍🏫 교사용 화면
# ==========================================
if mode == "교사용 (수업 개설)":
    st.header("👨‍🏫 교사용: 자율 분석 수업 만들기")
    
    input_topic = st.text_area("오늘의 수업 주제", value=st.session_state.topic, height=100)
    
    if st.button("🚀 교육 시나리오 생성"):
        if not input_topic.strip():
            st.warning("⚠️ 주제를 입력해야 시나리오를 만들 수 있어요!")
        else:
            with st.spinner("AI가 수업을 설계 중입니다..."):
                parsed = create_scenario(input_topic)
                if parsed and 'scenario' in parsed:
                    st.session_state.scenario = parsed['scenario']
                    st.session_state.topic = input_topic
                    st.session_state.scenario_analysis = analyze_scenario(input_topic, st.session_state.scenario)
                    st.session_state.current_step = 0
                    st.session_state.scenario_images = [None] * len(st.session_state.scenario)
                    st.success("수업 생성 완료!")
                else:
                    st.error("생성에 실패했습니다. 다시 시도해 주세요.")

    if st.session_state.scenario and 'scenario_analysis' in st.session_state:
        st.write("---")
        st.subheader("📊 AI의 수업 분석")
        analysis = st.session_state.scenario_analysis
        st.markdown(f"**1. 핵심 가치:** {analysis['ethical_standard']}")
        st.markdown(f"**2. 연계 교과:** {analysis['achievement_std']}")
        st.markdown(f"**3. 학습 목표:** {analysis['learning_content']}")

# ==========================================
# 🙋‍♂️ 학생용 화면
# ==========================================
elif mode == "학생용 (수업 참여)":
    if not st.session_state.scenario:
        st.info("선생님이 수업을 개설할 때까지 기다려주세요!")
    elif not st.session_state.lesson_complete:
        idx = st.session_state.current_step
        data = st.session_state.scenario[idx]
        
        st.header(f"🙋‍♂️ 학습하기: {st.session_state.topic}")
        st.subheader(f"📖 이야기 {idx + 1}")
        st.info(data['story'])
        
        if st.session_state.feedback_stage == 0:
            c1, c2 = st.columns(2)
            if c1.button(f"🅰️ {data['choice_a']}", use_container_width=True):
                st.session_state.selected_choice = data['choice_a']; st.session_state.feedback_stage = 1; st.rerun()
            if c2.button(f"🅱️ {data['choice_b']}", use_container_width=True):
                st.session_state.selected_choice = data['choice_b']; st.session_state.feedback_stage = 1; st.rerun()
        
        elif st.session_state.feedback_stage == 1:
            st.success(f"선택: {st.session_state.selected_choice}")
            reason = st.text_area("그렇게 선택한 이유는?")
            if st.button("제출하기"):
                safe_reason = pii_filter(reason)
                feedback = get_feedback(st.session_state.selected_choice, safe_reason, data['story'])
                st.session_state.chat_log = feedback
                st.session_state.feedback_stage = 2; st.rerun()
        
        elif st.session_state.feedback_stage == 2:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(st.session_state.chat_log)
            if st.button("다음 단계로 ➡️"):
                if idx < len(st.session_state.scenario) - 1:
                    st.session_state.current_step += 1; st.session_state.feedback_stage = 0; st.rerun()
                else:
                    st.session_state.lesson_complete = True; st.rerun()
    else:
        st.balloons(); st.header("🎉 학습을 모두 마쳤어! 정말 훌륭해!")
        if st.button("처음으로 돌아가기"):
            st.session_state.lesson_complete = False; st.session_state.scenario = None; st.rerun()
