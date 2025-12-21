import streamlit as st
from openai import OpenAI
import re
import json 
import datetime

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="테스트 봇과 함께하는 자유 시나리오 학습", page_icon="🤖", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ OpenAI API 키를 설정해주세요!")
    st.stop()

# --- 3. 시스템 페르소나 ---
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)을 위한 교육 튜터 '테스트 봇'입니다.
학생이 입력한 주제가 무엇이든, 그 상황 속에서 생각할 거리가 있는 '선택의 순간(딜레마)'을 포함한 시나리오를 만들어 학습을 돕습니다.

[핵심 행동 수칙]
1. [유연한 생성]: 주제가 무엇이든 거부하지 말고 재미있는 교육 시나리오로 만드세요.
2. [사례 중심]: 학교 생활이나 초등학생이 이해하기 쉬운 사례로 구성하세요.
3. [말투]: "안녕! 나는 테스트 봇이야", "~했니?" 처럼 다정하고 친근한 초등 교사 말투를 사용하세요.
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
        st.error(f"JSON 요청 오류: {e}")
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
        st.error(f"텍스트 요청 오류: {e}")
        return None

def pii_filter(text):
    """개인정보 필터링"""
    original_text = text
    text = re.sub(r'01\d{1}[-\s]?\d{3,4}[-\s]?\d{4}', '[전화번호]', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[이메일 주소]', text)
    text = re.sub(r'\d{6}[-\s]?[1-4]\d{6}', '[주민번호]', text)
    if original_text != text:
        st.warning("⚠️ 개인정보 보호를 위해 일부 내용이 마스킹되었습니다.")
    return text

def create_scenario(topic): 
    """주제 제한 없이 시나리오 생성 요청"""
    prompt = (
        f"# 주제: '{topic}'\n\n"
        "이 주제를 바탕으로 초등학생이 고민해볼 만한 선택지가 포함된 시나리오를 생성하세요.\n"
        "규칙 1: 3~5단계로 구성해줘.\n"
        "규칙 2: 각 단계는 2~3문장 이내로 짧게 작성해줘.\n"
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
    """시나리오의 교육적 의미 분석"""
    story_context = "\n".join([f"[{i+1}단계] {item.get('story')}" for i, item in enumerate(parsed_scenario)])
    prompt = (
        f"주제: '{topic}'\n"
        f"내용:\n{story_context}\n\n"
        "이 내용을 분석하여 다음 3가지를 알려줘.\n"
        "[핵심 가치] [이 시나리오에서 중요하게 다루는 가치나 원칙]\n"
        "[연계 교과] [초등학교 교과목과 연계할 수 있는 부분]\n"
        "[학습 목표] [학생이 이 시나리오를 통해 배우게 될 점]"
    )
    analysis = ask_gpt_text(prompt)
    result = {}
    try:
        def safe_extract(pattern, text):
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else '분석 중'
        
        result['ethical_standard'] = safe_extract(r"\[핵심 가치\](.*?)\[연계 교과\]", analysis)
        result['achievement_std'] = safe_extract(r"\[연계 교과\](.*?)\[학습 목표\]", analysis)
        result['learning_content'] = safe_extract(r"\[학습 목표\](.*)", analysis)
    except:
        result = {'ethical_standard': '자율 분석', 'achievement_std': '자율 연계', 'learning_content': '자율 목표'}
    return result

# --- 5. 메인 앱 로직 ---

# 세션 초기화
for key in ['scenario', 'current_step', 'chat_log', 'topic', 'scenario_analysis', 'feedback_stage', 'lesson_complete']:
    if key not in st.session_state:
        if key == 'chat_log': st.session_state[key] = []
        elif key in ['current_step', 'feedback_stage']: st.session_state[key] = 0
        elif key == 'lesson_complete': st.session_state[key] = False
        else: st.session_state[key] = None

st.sidebar.title("🏫 AI 교육 튜터")
mode = st.sidebar.radio("모드 선택:", ["학생용", "교사용"])

if mode == "교사용":
    st.header("👨‍🏫 수업 개설")
    input_topic = st.text_input("수업 주제 (어떤 주제든 입력 가능)", value=st.session_state.topic if st.session_state.topic else "")
    
    if st.button("🚀 시나리오 및 분석 생성"):
        if not input_topic:
            st.warning("주제를 입력하세요.")
        else:
            with st.spinner("AI가 수업을 설계 중입니다..."):
                data = create_scenario(input_topic)
                if data and 'scenario' in data:
                    st.session_state.scenario = data['scenario']
                    st.session_state.topic = input_topic
                    st.session_state.scenario_analysis = analyze_scenario(input_topic, data['scenario'])
                    st.session_state.current_step = 0
                    st.session_state.feedback_stage = 0
                    st.session_state.lesson_complete = False
                    st.success("수업 생성 완료!")
                else:
                    st.error("생성에 실패했습니다. 다시 시도해 주세요.")

    if st.session_state.scenario_analysis:
        st.write("---")
        st.subheader("📊 AI의 수업 분석")
        st.markdown(f"**1. 핵심 가치:** {st.session_state.scenario_analysis['ethical_standard']}")
        st.markdown(f"**2. 연계 교과:** {st.session_state.scenario_analysis['achievement_std']}")
        st.markdown(f"**3. 학습 목표:** {st.session_state.scenario_analysis['learning_content']}")

elif mode == "학생용":
    if not st.session_state.scenario:
        st.info("교사용 모드에서 먼저 주제를 입력하고 시나리오를 만들어주세요!")
    elif not st.session_state.lesson_complete:
        idx = st.session_state.current_step
        data = st.session_state.scenario[idx]
        
        st.header(f"🙋‍♂️ {st.session_state.topic} 공부하기")
        st.progress((idx + 1) / len(st.session_state.scenario))
        
        st.subheader(f"Step {idx + 1}")
        st.info(data['story'])
        
        if st.session_state.feedback_stage == 0:
            c1, c2 = st.columns(2)
            if c1.button(f"🅰️ {data['choice_a']}", use_container_width=True):
                st.session_state.selected_choice = data['choice_a']
                st.session_state.feedback_stage = 1
                st.rerun()
            if c2.button(f"🅱️ {data['choice_b']}", use_container_width=True):
                st.session_state.selected_choice = data['choice_b']
                st.session_state.feedback_stage = 1
                st.rerun()
        
        elif st.session_state.feedback_stage == 1:
            st.success(f"나의 선택: {st.session_state.selected_choice}")
            reason = st.text_area("그렇게 선택한 이유를 말해줘!")
            if st.button("제출하기"):
                if not reason.strip():
                    st.warning("이유를 입력해줘!")
                else:
                    with st.spinner("테스트 봇이 생각 중..."):
                        # 간단 피드백 로직
                        prompt = f"상황: {data['story']}\n학생 선택: {st.session_state.selected_choice}\n이유: {reason}\n학생에게 줄 따뜻한 격려와 생각할 거리를 2문장으로 말해줘."
                        feedback = ask_gpt_text(prompt)
                        st.session_state.chat_log = [{"role": "assistant", "content": feedback}]
                        st.session_state.feedback_stage = 2
                        st.rerun()
        
        elif st.session_state.feedback_stage == 2:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(st.session_state.chat_log[0]["content"])
            
            if st.button("다음으로 넘어가기 ➡️"):
                if idx < len(st.session_state.scenario) - 1:
                    st.session_state.current_step += 1
                    st.session_state.feedback_stage = 0
                else:
                    st.session_state.lesson_complete = True
                st.rerun()
    else:
        st.balloons()
        st.header("🎉 학습을 모두 마쳤어!")
        if st.button("처음으로 돌아가기"):
            st.session_state.lesson_complete = False
            st.session_state.current_step = 0
            st.rerun()
