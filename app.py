import streamlit as st
from openai import OpenAI
import re
import json

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 토론 학습 지원 시스템", page_icon="🗣️", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ API 키를 설정해주세요!")
    st.stop()

# --- 3. [핵심] 토론 중심 시스템 페르소나 ---
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)의 비판적 사고를 돕는 'AI 토론 튜터'입니다.
모든 시나리오는 정답이 없는 '딜레마 상황'으로 구성하며, 학생이 스스로 근거를 들어 주장할 수 있도록 유도합니다.

[행동 수칙]
1. [토론 유도]: 단순히 지식을 전달하지 말고 "왜 그렇게 생각하니?", "다른 입장에서는 어떨까?" 같은 질문을 던지세요.
2. [다양한 관점]: 특정 선택이 무조건 옳다고 하기보다, 각 선택이 가질 수 있는 장단점과 가치를 비교하게 하세요.
3. [눈높이 교육]: 초등학생이 이해하기 쉬운 비유를 사용하고 따뜻한 격려를 잊지 마세요.
"""

# --- 4. 함수 정의 ---

def create_debate_scenario(topic):
    """토론용 딜레마 시나리오 생성"""
    prompt = (
        f"# 주제: '{topic}'\n\n"
        "이 주제로 초등학생용 토론 수업 시나리오를 3~4단계로 만드세요.\n"
        "각 단계는 대립하는 두 가지 가치가 부딪히는 상황이어야 합니다.\n"
        "반드시 아래 JSON 형식으로만 응답하세요.\n"
        "{\"scenario\": [{\"story\": \"상황 설명\", \"choice_a\": \"찬성/입장1\", \"choice_b\": \"반대/입장2\", \"debate_point\": \"교사가 참고할 토론의 핵심\"}]}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PERSONA}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content.strip())
    except:
        return None

def get_debate_feedback(choice, reason, story):
    """학생의 주장에 대한 토론형 피드백"""
    prompt = (
        f"상황: {story}\n학생의 주장: {choice}\n이유: {reason}\n\n"
        "1. 학생의 의견을 존중하며 요약해줘.\n"
        "2. 반대 입장에서는 어떤 걱정을 할 수 있을지 '반론'을 부드럽게 제기해줘.\n"
        "3. 다시 한번 생각해보게 하는 질문으로 마무리해줘. (3문장 이내)"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PERSONA}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except:
        return "오류가 발생했습니다. 다시 시도해주세요."

# --- 5. 메인 앱 로직 ---

if 'scenario' not in st.session_state: st.session_state.scenario = None
if 'current_step' not in st.session_state: st.session_state.current_step = 0
if 'feedback' not in st.session_state: st.session_state.feedback = ""

st.sidebar.title("🏫 토론 수업 플랫폼")
mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용 (수업 설계)", "🙋‍♂️ 학생용 (토론 참여)"])

# ==========================================
# 👨‍🏫 교사용 화면: 설계 및 미리보기
# ==========================================
if mode == "👨‍🏫 교사용 (수업 설계)":
    st.header("🛠️ 토론 수업 설계 및 미리보기")
    topic = st.text_input("토론 주제 입력", placeholder="예: 무인 상점의 AI 감시 카메라 설치")

    if st.button("🚀 토론 시나리오 구성"):
        with st.spinner("AI가 토론 흐름을 짜는 중..."):
            data = create_debate_scenario(topic)
            if data:
                st.session_state.scenario = data['scenario']
                st.session_state.topic = topic
                st.success("토론 수업이 구성되었습니다!")

    if st.session_state.scenario:
        st.write("---")
        st.subheader(f"📊 '{st.session_state.topic}' 수업 흐름 미리보기")
        
        # 교사를 위한 미리보기 테이블
        preview_data = []
        for i, s in enumerate(st.session_state.scenario):
            preview_data.append({
                "단계": f"{i+1}단계",
                "상황": s['story'],
                "논쟁 지점": s['debate_point']
            })
        st.table(preview_data)
        
        st.info("💡 위 내용을 확인하신 후, 왼쪽 사이드바에서 '학생용' 모드로 변경하여 수업을 진행하세요.")

# ==========================================
# 🙋‍♂️ 학생용 화면: 실제 토론 진행
# ==========================================
elif mode == "🙋‍♂️ 학생용 (토론 참여)":
    if not st.session_state.scenario:
        st.warning("선생님이 아직 토론 주제를 정하지 않았어요!")
    else:
        idx = st.session_state.current_step
        step = st.session_state.scenario[idx]
        
        st.header(f"🗣️ 토론: {st.session_state.topic}")
        st.progress((idx + 1) / len(st.session_state.scenario))
        
        st.subheader(f"Step {idx + 1}")
        st.chat_message("assistant", avatar="🤖").write(step['story'])

        # 선택 및 이유 입력
        col1, col2 = st.columns(2)
        if col1.button(f"🅰️ {step['choice_a']}", use_container_width=True):
            st.session_state.temp_choice = step['choice_a']
        if col2.button(f"🅱️ {step['choice_b']}", use_container_width=True):
            st.session_state.temp_choice = step['choice_b']

        if 'temp_choice' in st.session_state:
            st.write(f"**나의 입장:** {st.session_state.temp_choice}")
            reason = st.text_area("그렇게 생각하는 근거는 무엇인가요?", key=f"reason_{idx}")
            
            if st.button("내 주장 전달하기 ✉️"):
                with st.spinner("테스트 봇이 답변을 읽고 있어요..."):
                    feedback = get_debate_feedback(st.session_state.temp_choice, reason, step['story'])
                    st.session_state.feedback = feedback
            
            if st.session_state.feedback:
                st.chat_message("assistant", avatar="🤖").write(st.session_state.feedback)
                
                if st.button("다음 논제로 넘어가기 ➡️"):
                    if idx < len(st.session_state.scenario) - 1:
                        st.session_state.current_step += 1
                        st.session_state.feedback = ""
                        del st.session_state.temp_choice
                        st.rerun()
                    else:
                        st.balloons()
                        st.success("오늘의 모든 토론을 마쳤습니다! 훌륭한 비판적 사고였어요!")
