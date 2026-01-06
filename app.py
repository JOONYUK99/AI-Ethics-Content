import streamlit as st
from openai import OpenAI
import re
import json

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 토론 학습 시스템", page_icon="🎨", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ API 키를 설정해주세요! (Streamlit Cloud -> Secrets 확인)")
    st.stop()

# --- 3. 시스템 페르소나 ---
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)의 비판적 사고를 돕는 'AI 토론&아트 튜터'입니다.
학생이 스스로 생각하게 유도하고, 다정한 초등 교사 말투(~했니?, ~단다)를 사용하세요.
주제가 무엇이든 거부하지 말고 교육적인 토론 시나리오로 만듭니다.
"""

# --- 4. 주요 함수 ---

def ask_gpt_json(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PERSONA}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content.strip())
    except:
        return {"scenario": []}

def ask_gpt_text(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PERSONA}, {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except:
        return "데이터를 가져오지 못했습니다."

def generate_image(prompt):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"A friendly cartoon-style illustration for elementary school: {prompt}",
            size="1024x1024", n=1
        )
        return response.data[0].url
    except:
        return None

# --- 5. 세션 상태 초기화 ---

if 'scenario' not in st.session_state: st.session_state.scenario = {"scenario": []}
if 'analysis' not in st.session_state: st.session_state.analysis = ""
if 'current_step' not in st.session_state: st.session_state.current_step = 0
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'topic' not in st.session_state: st.session_state.topic = ""
if 'tutorial_done' not in st.session_state: st.session_state.tutorial_done = False
if 'tutorial_step' not in st.session_state: st.session_state.tutorial_step = 1

# --- 6. 메인 로직 ---

st.sidebar.title("🏫 AI 지능형 학습")
mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"])

if mode == "👨‍🏫 교사용":
    st.header("🛠️ 토론 수업 설계 및 분석")
    input_topic = st.text_input("토론 주제 입력", value=st.session_state.topic)
    
    if st.button("🚀 수업 설계하기"):
        with st.spinner("AI가 수업을 구성 중입니다..."):
            s_prompt = f"주제 '{input_topic}'로 초등용 3단계 토론 시나리오를 JSON으로 만들어줘. 키는 'scenario'이고 내부 키는 'story', 'choice_a', 'choice_b', 'debate_point'야."
            st.session_state.scenario = ask_gpt_json(s_prompt)
            
            a_prompt = f"주제 '{input_topic}'의 [핵심 가치], [연계 교과], [학습 목표]를 각각 짧게 한 문장씩 따로 작성해줘."
            st.session_state.analysis = ask_gpt_text(a_prompt)
            st.session_state.topic = input_topic
            st.success("수업 생성이 완료되었습니다!")

    if st.session_state.analysis:
        st.subheader("📊 수업 분석 리포트")
        # 정규표현식으로 태그 내용 추출하여 개별 상자에 표시
        parts = re.split(r'\[|\]', st.session_state.analysis)
        for i in range(1, len(parts), 2):
            if i+1 < len(parts):
                st.info(f"**{parts[i]}**: {parts[i+1].strip()}")

    if st.session_state.scenario.get('scenario'):
        with st.expander("📜 전체 시나리오 미리보기"):
            st.table(st.session_state.scenario['scenario'])

elif mode == "🙋‍♂️ 학생용":
    if not st.session_state.tutorial_done:
        st.header("🎒 수업 전 가이드 연습")
        t_step = st.session_state.tutorial_step
        
        if t_step == 1:
            st.subheader("1. 입장 선택 연습")
            st.info("토론 중 자신의 입장을 고르는 방법이야! 버튼을 눌러볼래?")
            if st.button("😊 토론이 기대돼요!"): st.session_state.tutorial_step = 2; st.rerun()
            
        elif t_step == 2:
            st.subheader("2. 생각 적기 연습")
            st.info("너의 주장을 글로 입력하는 연습이야. '안녕'이라고 써볼까?")
            t_input = st.text_input("여기에 입력")
            if st.button("연습 제출"):
                if t_input: st.session_state.tutorial_step = 3; st.rerun()
                else: st.warning("내용을 입력해줘!")
                
        elif t_step == 3:
            st.subheader("3. 그림 요청 연습")
            st.info("수업 장면을 그림으로 그려달라고 할 수 있어!")
            if st.button("🎨 연습용 그림 그리기"):
                with st.spinner("AI 화가가 그리는 중..."):
                    img = generate_image("A friendly robot helping kids in class")
                    if img:
                        st.image(img, caption="연습 그림 완성!")
                        if st.button("진짜 수업 시작하기 🚀"):
                            st.session_state.tutorial_done = True; st.rerun()
    else:
        if not st.session_state.scenario.get('scenario'):
            st.warning("선생님이 아직 수업을 준비 중이야! 잠시만 기다려줘.")
        else:
            idx = st.session_state.current_step
            data = st.session_state.scenario['scenario'][idx]
            st.header(f"🗣️ {st.session_state.topic}")
            st.subheader(f"{idx+1}단계: 토론 상황")
            st.info(data['story'])
            
            choice = st.radio("나의 입장은?", [data['choice_a'], data['choice_b']], key=f"r_{idx}")
            reason = st.text_area("이유를 말해줄래?", key=f"a_{idx}")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("주장 제출 📩", key=f"s_{idx}"):
                    f_prompt = f"상황: {data['story']}\n선택: {choice}\n이유: {reason}\n부드럽게 공감하며 다른 생각도 질문해줘."
                    st.session_state.chat_history.append({"role": "bot", "content": ask_gpt_text(f_prompt)})
            with c2:
                if st.button("🎨 장면 그림으로 보기", key=f"i_{idx}"):
                    with st.spinner("그리는 중..."):
                        url = generate_image(data['story'])
                        if url: st.session_state.chat_history.append({"role": "img", "content": url})
            
            for msg in st.session_state.chat_history:
                if msg["role"] == "bot": st.chat_message("assistant").write(msg["content"])
                else: st.image(msg["content"])

            if st.button("다음 논제로 ➡️", key=f"n_{idx}"):
                st.session_state.current_step += 1
                st.session_state.chat_history = []
                st.rerun()
