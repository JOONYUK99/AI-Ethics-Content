import streamlit as st
from openai import OpenAI
import re
import json

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 토론 및 창의 학습 시스템", page_icon="🎨", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ OpenAI API 키를 설정해주세요! (Streamlit Cloud Settings -> Secrets 확인)")
    st.stop()

# --- 3. [핵심] 시스템 페르소나 ---
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)의 비판적 사고와 창의성을 돕는 'AI 토론&아트 튜터'입니다.
정답을 제시하기보다 학생이 스스로 이유를 생각하고 표현하도록 유도하며, 필요한 경우 시각적 자료(이미지)를 통해 이해를 돕습니다.
말투는 항상 다정하고 친근한 초등 교사의 말투(~했니?, ~단다)를 사용하세요.
"""

# --- 4. 주요 기능 함수 ---

def ask_gpt_json(prompt):
    """JSON 형식의 수업 설계 데이터를 생성"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PERSONA}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content.strip())
    except:
        return None

def ask_gpt_text(prompt):
    """텍스트 기반의 분석 및 피드백 생성"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PERSONA}, {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except:
        return "답변을 가져오지 못했어요. 다시 시도해볼까요?"

def generate_image(prompt):
    """수업용 삽화 생성 (DALL-E 3)"""
    try:
        dalle_prompt = f"A friendly, educational cartoon-style illustration for elementary school textbook, depicting: {prompt}"
        response = client.images.generate(
            model="dall-e-3", prompt=dalle_prompt, size="1024x1024", quality="standard", n=1
        )
        return response.data[0].url
    except:
        return None

# --- 5. 메인 앱 로직 ---

# 세션 상태 초기화
if 'scenario' not in st.session_state: st.session_state.scenario = None
if 'analysis' not in st.session_state: st.session_state.analysis = None
if 'current_step' not in st.session_state: st.session_state.current_step = 0
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

st.sidebar.title("🏫 AI 지능형 학습 시스템")
mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용 (수업 설계)", "🙋‍♂️ 학생용 (토론 및 창작)"])

# ==========================================
# 👨‍🏫 교사용 화면: 상세 분석 및 미리보기
# ==========================================
if mode == "👨‍🏫 교사용 (수업 설계)":
    st.header("🛠️ 맞춤형 토론 수업 설계")
    topic = st.text_input("오늘의 토론 주제를 입력하세요", placeholder="예: 우리 학교에 AI 로봇 선생님이 온다면?")

    if st.button("🚀 수업 시나리오 및 분석 생성"):
        with st.spinner("AI가 수업을 설계 중입니다..."):
            # 시나리오 생성
            scenario_prompt = f"주제 '{topic}'에 대해 초등학생용 3단계 토론 시나리오를 만들어줘. 각 단계는 'story', 'choice_a', 'choice_b', 'debate_point'를 포함한 JSON 형식이어야 해."
            st.session_state.scenario = ask_gpt_json(scenario_prompt)
            
            # 상세 분석 생성 (분리된 데이터 요청)
            analysis_prompt = f"주제 '{topic}'의 수업 내용을 분석해서 [핵심 가치], [연계 교과], [학습 목표]를 각각 짧은 문장으로 따로따로 알려줘."
            st.session_state.analysis = ask_gpt_text(analysis_prompt)
            st.session_state.topic = topic
            st.session_state.current_step = 0
            st.success("수업 설계가 완료되었습니다!")

    if st.session_state.analysis:
        st.write("---")
        st.subheader("📊 AI의 수업 상세 분석")
        
        # 분석 내용을 줄 단위로 분리하여 시각화 (한 줄 출력을 개별 칸으로 분리)
        lines = st.session_state.analysis.split('\n')
        for line in lines:
            if line.strip():
                st.info(line)
        
        with st.expander("📜 전체 시나리오 미리보기"):
            st.table(st.session_state.scenario['scenario'])

# ==========================================
# 🙋‍♂️ 학생용 화면: 토론 및 그림 생성 기능
# ==========================================
elif mode == "🙋‍♂️ 학생용 (토론 및 창작)":
    if not st.session_state.scenario:
        st.warning("선생님이 수업을 설계할 때까지 잠시만 기다려주세요! 😊")
    else:
        idx = st.session_state.current_step
        steps = st.session_state.scenario['scenario']
        current_data = steps[idx]

        st.header(f"🗣️ 토론 학습: {st.session_state.topic}")
        st.progress((idx + 1) / len(steps))

        # 1. 상황 제시
        with st.chat_message("assistant", avatar="🤖"):
            st.write(f"**{idx+1}단계:** {current_data['story']}")
            st.write(f"💡 **생각해볼 점:** {current_data['debate_point']}")

        # 2. 입장 선택 및 이유 입력 (텍스트 기반 토론)
        st.write("---")
        st.subheader("📝 나의 생각 적기")
        choice = st.radio("당신의 입장은?", [current_data['choice_a'], current_data['choice_b']], index=0)
        reason = st.text_area("그렇게 생각한 이유를 구체적으로 적어줄래?", placeholder="내 생각에는...")

        col_debate, col_draw = st.columns(2)
        
        with col_debate:
            if st.button("내 의견 전달하기 📩"):
                debate_prompt = f"상황: {current_data['story']}\n학생 선택: {choice}\n이유: {reason}\n이 주장에 대해 따뜻하게 공감해주고, 반대 입장에서는 어떤 걱정을 할 수 있을지 질문을 하나만 던져줘."
                feedback = ask_gpt_text(debate_prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": feedback})
        
        with col_draw:
            # 그림 그리기 기능 추가
            if st.button("🎨 이 상황을 그림으로 보기"):
                with st.spinner("AI 화가가 그림을 그리고 있어요..."):
                    img_url = generate_image(current_data['story'])
                    if img_url:
                        st.session_state.chat_history.append({"role": "image", "content": img_url})

        # 3. 채팅 기록 출력 (피드백 및 생성된 이미지)
        for chat in st.session_state.chat_history:
            if chat["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(chat["content"])
            elif chat["role"] == "image":
                st.image(chat["content"], caption="AI가 그린 수업 장면")

        # 4. 다음 단계 이동
        st.write("---")
        if st.button("다음 논제로 이동하기 ➡️"):
            if idx < len(steps) - 1:
                st.session_state.current_step += 1
                st.session_state.chat_history = []
                st.rerun()
            else:
                st.balloons()
                st.success("와! 모든 토론과 학습을 성공적으로 마쳤어! 정말 대단해! 🎉")
