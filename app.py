import streamlit as st
from openai import OpenAI
import re
import json 
import datetime

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="테스트 봇과 함께하는 AI 윤리 학습", page_icon="🤖", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ OpenAI API 키를 설정해주세요! (Streamlit Cloud Secrets 확인)")
    st.stop()

# --- 3. 시스템 페르소나 (RAG 없이 자체 지식 활용) ---
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)을 위한 AI 윤리 교육 튜터 '테스트 봇'입니다.
당신은 '국가 인공지능 윤리기준'과 학교 교육과정에 대한 깊은 지식을 가지고 있습니다.

[핵심 행동 수칙]
1. [자체 지식 활용]: 외부 데이터 없이도 AI 윤리 원칙(인권 보장, 프라이버시 보호 등)을 바탕으로 시나리오를 생성하고 분석하세요.
2. [교육과정 연계]: 설명할 때 초등학교 도덕이나 실과 시간에 배우는 내용과 연결하여 설명하세요.
3. [개인정보 철벽 방어]: 학생이 개인정보를 말하려 하면 즉시 교육적으로 제지하세요.
4. [말투]: "안녕! 나는 테스트 봇이야", "~했니?" 처럼 다정하고 친근한 초등 교사 말투를 사용하세요.
"""

# --- 4. 함수 정의 ---

def ask_gpt_json(prompt, max_tokens=2048):
    """GPT-4o에게 JSON 형식의 응답을 요청"""
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
        st.error(f"오류 발생: {e}")
        return None

def ask_gpt_text(prompt):
    """GPT-4o에게 일반 텍스트 응답을 요청"""
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
        st.error(f"오류 발생: {e}")
        return None

def generate_image(prompt):
    """DALL-E 3 이미지 생성"""
    try:
        dalle_prompt = f"A friendly, educational cartoon-style illustration for elementary school textbook, depicting: {prompt}"
        response = client.images.generate(
            model="dall-e-3", prompt=dalle_prompt, size="1024x1024", quality="standard", n=1
        )
        return response.data[0].url
    except:
        return None

def pii_filter(text):
    """개인정보 필터링"""
    original_text = text
    text = re.sub(r'01\d{1}[-\s]?\d{3,4}[-\s]?\d{4}', '[전화번호]', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[이메일 주소]', text)
    text = re.sub(r'\d{6}[-\s]?[1-4]\d{6}', '[주민번호]', text)
    
    if original_text != text:
        st.warning("⚠️ 개인정보가 감지되어 마스킹되었습니다.")
    return text

def create_scenario(topic): 
    """주제에 따른 딜레마 시나리오 생성"""
    prompt = (
        f"# 주제: '{topic}'\n\n"
        "아래 규칙을 지켜서 초등학생용 AI 윤리 딜레마 시나리오를 생성하세요.\n"
        "1. 주제가 윤리 교육과 무관하면 {'error': '윤리교육과 상관없는 내용입니다'}를 반환하세요.\n"
        "2. 시나리오는 3~6단계로 구성하세요.\n"
        "3. 각 단계는 짧고 쉬운 문장으로 작성하세요.\n\n"
        "# 출력 형식 (JSON):\n"
        "{\"scenario\": [{\"story\": \"...\", \"choice_a\": \"...\", \"choice_b\": \"...\"}]}"
    )
    raw_json = ask_gpt_json(prompt)
    
    if raw_json:
        try:
            return json.loads(raw_json)
        except:
            return None
    return None

def analyze_scenario(topic, parsed_scenario):
    """생성된 시나리오 분석"""
    story_summary = "\n".join([f"[{i+1}단계] {item['story']}" for i, item in enumerate(parsed_scenario)])
    prompt = (
        f"주제 '{topic}'에 대한 다음 시나리오를 분석하세요:\n{story_summary}\n\n"
        "다음 형식으로 분석결과를 출력하세요:\n"
        "[윤리 기준] [관련 윤리 원칙]\n"
        "[성취기준] [초등 교육과정 연계 내용]\n"
        "[학습 내용] [핵심 학습 목표]"
    )
    analysis = ask_gpt_text(prompt)
    
    result = {}
    try:
        result['ethical_standard'] = re.search(r"\[윤리 기준\](.*?)\[성취기준\]", analysis, re.S).group(1).strip()
        result['achievement_std'] = re.search(r"\[성취기준\](.*?)\[학습 내용\]", analysis, re.S).group(1).strip()
        result['learning_content'] = re.search(r"\[학습 내용\](.*)", analysis, re.S).group(1).strip()
    except:
        result = {'ethical_standard': '분석 실패', 'achievement_std': '분석 실패', 'learning_content': '분석 실패'}
    return result

def get_feedback(choice, reason, story_context):
    """학생 선택에 대한 피드백 및 질문 생성"""
    prompt_1 = (
        f"상황: {story_context}\n학생 선택: {choice}, 이유: {reason}\n"
        "따뜻한 공감과 함께 이 선택이 윤리적으로 어떤 의미가 있는지 2문장 이내로 설명하세요."
    )
    prompt_2 = f"학생에게 사고를 넓힐 수 있는 질문을 하나만 던지세요."
    
    f1 = ask_gpt_text(prompt_1)
    f2 = ask_gpt_text(prompt_2)
    return [{"role": "assistant", "content": f1}, {"role": "assistant", "content": f2}]

# --- 5. 메인 앱 로직 ---

# 세션 상태 초기화
for key in ['scenario', 'scenario_images', 'current_step', 'chat_log', 'topic', 'scenario_analysis', 'feedback_stage', 'lesson_complete']:
    if key not in st.session_state:
        if key == 'scenario_images': st.session_state[key] = []
        elif key == 'chat_log': st.session_state[key] = []
        elif key == 'current_step': st.session_state[key] = 0
        elif key == 'feedback_stage': st.session_state[key] = 0
        else: st.session_state[key] = None

st.sidebar.title("🏫 AI 윤리 학습")
mode = st.sidebar.radio("모드 선택:", ["학생용", "교사용"])

if mode == "교사용":
    st.header("👨‍🏫 수업 개설 (AI 자율 모드)")
    input_topic = st.text_input("수업 주제 입력", placeholder="예: 생성형 AI를 이용한 숙제")
    
    if st.button("🚀 시나리오 생성"):
        with st.spinner("AI가 시나리오를 구성 중입니다..."):
            data = create_scenario(input_topic)
            if data and "scenario" in data:
                st.session_state.scenario = data["scenario"]
                st.session_state.topic = input_topic
                st.session_state.scenario_analysis = analyze_scenario(input_topic, data["scenario"])
                st.session_state.scenario_images = [None] * len(data["scenario"])
                st.session_state.current_step = 0
                st.success("생성 완료!")
            else:
                st.error("윤리 교육에 적합한 주제를 입력해주세요.")

    if st.session_state.scenario_analysis:
        st.subheader("📊 분석 결과")
        st.write(f"**윤리 기준:** {st.session_state.scenario_analysis['ethical_standard']}")
        st.write(f"**성취 기준:** {st.session_state.scenario_analysis['achievement_std']}")
        st.write(f"**학습 내용:** {st.session_state.scenario_analysis['learning_content']}")

elif mode == "학생용":
    if not st.session_state.scenario:
        st.info("교사용 모드에서 먼저 시나리오를 생성해주세요.")
    elif not st.session_state.lesson_complete:
        step_idx = st.session_state.current_step
        step_data = st.session_state.scenario[step_idx]
        
        st.subheader(f"📖 이야기 {step_idx + 1}")
        st.info(step_data['story'])
        
        if st.session_state.feedback_stage == 0:
            c1, c2 = st.columns(2)
            if c1.button(f"A: {step_data['choice_a']}"):
                st.session_state.selected_choice = step_data['choice_a']
                st.session_state.feedback_stage = 1
                st.rerun()
            if c2.button(f"B: {step_data['choice_b']}"):
                st.session_state.selected_choice = step_data['choice_b']
                st.session_state.feedback_stage = 1
                st.rerun()
        
        elif st.session_state.feedback_stage == 1:
            reason = st.text_input("그렇게 생각한 이유는?")
            if st.button("제출"):
                st.session_state.chat_log = get_feedback(st.session_state.selected_choice, reason, step_data['story'])
                st.session_state.feedback_stage = 2
                st.rerun()
        
        elif st.session_state.feedback_stage == 2:
            for msg in st.session_state.chat_log:
                with st.chat_message("assistant"): st.write(msg["content"])
            
            if st.button("다음 단계로"):
                if st.session_state.current_step < len(st.session_state.scenario) - 1:
                    st.session_state.current_step += 1
                    st.session_state.feedback_stage = 0
                else:
                    st.session_state.lesson_complete = True
                st.rerun()
    else:
        st.balloons()
        st.header("🎉 모든 학습을 마쳤습니다!")
        if st.button("다시 시작하기"):
            st.session_state.lesson_complete = False
            st.session_state.scenario = None
            st.rerun()
