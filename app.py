import streamlit as st
from openai import OpenAI
import re
import json

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 토론 및 창작 시스템", page_icon="🎨", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ API 키를 설정해주세요! (Streamlit Cloud Secrets 확인)")
    st.stop()

# --- 3. 시스템 페르소나 ---
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)의 비판적 사고와 창의성을 돕는 'AI 토론&아트 튜터'입니다.
학생이 스스로 생각하게 유도하고, 다정한 초등 교사 말투(~했니?, ~단다)를 사용하세요.
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
        return None

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

# --- 5. 메인 로직 ---

# 세션 초기화
if 'scenario' not in st.session_state: st.session_state.scenario = None
if 'analysis' not in st.session_state: st.session_state.analysis = None
if 'current_step' not in st.session_state: st.session_state.current_step = 0
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'topic' not in st.session_state: st.session_state.topic = ""
# 튜토리얼 관련 세션
if 'tutorial_done' not in st.session_state: st.session_state.tutorial_done = False
if 'tutorial_step' not in st.session_state: st.session_state.tutorial_step = 1

st.sidebar.title("🏫 AI 지능형 학습")
mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"])

if mode == "👨‍🏫 교사용":
    st.header("🛠️ 토론 수업 설계")
    input_topic = st.text_input("토론 주제", value=st.session_state.topic)
    
    if st.button("🚀 수업 생성"):
        with st.spinner("AI가 분석 중..."):
            s_prompt = f"주제 '{input_topic}'로 초등용 3단계 토론 시나리오를 JSON으로 만들어줘. 키는 'scenario'이고 내부 키는 'story', 'choice_a', 'choice_b', 'debate_point'야."
            st.session_state.scenario = ask_gpt_json(s_prompt)
            
            a_prompt = f"주제 '{input_topic}'의 [핵심 가치], [연계 교과], [학습 목표]를 각각 한 문장씩 작성해줘."
            st.session_state.analysis = ask_gpt_text(a_prompt)
            st.session_state.topic = input_topic
            st.success("생성 완료!")

    if st.session_state.analysis:
        st.subheader("📊 수업 분석 결과")
        content = st.session_state.analysis
        parts = re.split(r'\[|\]', content)
        for i in range(1, len(parts), 2):
            if i+1 < len(parts):
                st.info(f"**{parts[i]}**: {parts[i+1].strip()}")

    if st.session_state.scenario and 'scenario' in st.session_state.scenario:
        with st.expander("📜 시나리오 미리보기"):
            st.table(st.session_state.scenario['scenario'])

elif mode == "🙋‍♂️ 학생용":
    # 튜토리얼 단계 (가이드라인 테스트)
    if not st.session_state.tutorial_done:
        st.header("🎒 수업 전 연습하기 (가이드라인)")
        st.write("본격적인 수업 전에 테스트 봇과 함께 기능을 연습해보자!")
        
        t_step = st.session_state.tutorial_step
        
        if t_step == 1:
            st.subheader("1단계: 선택지 고르기 연습")
            st.info("아래의 버튼 중 하나를 눌러서 너의 기분을 알려줘!")
            c1, c2 = st.columns(2)
            if c1.button("😄 기분 좋아!"): st.session_state.tutorial_step = 2; st.rerun()
            if c2.button("😎 기대돼!"): st.session_state.tutorial_step = 2; st.rerun()
            
        elif t_step == 2:
            st.subheader("2단계: 생각 적기 연습")
            st.info("토론할 때는 너의 생각을 글로 적는 게 중요해. 아래 칸에 '안녕'이라고 적어볼래?")
            t_input = st.text_input("여기에 입력해줘")
            if st.button("제출"):
                if t_input: st.session_state.tutorial_step = 3; st.rerun()
                else: st.warning("글자를 입력해줘!")
                
        elif t_step == 3:
            st.subheader("3단계: 그림 그리기 연습")
            st.info("공부하다가 그림이 필요할 땐 AI 화가에게 부탁할 수 있어! 버튼을 눌러볼까?")
            if st.button("🎨 연습용 그림 그리기"):
                with st.spinner("AI 화가가 그림을 그리고 있어..."):
                    test_img = generate_image("A happy robot teaching children in a futuristic classroom")
                    if test_img:
                        st.image(test_img, caption="AI 화가가 그린 연습 그림이야!")
                        st.success("대단해! 모든 기능을 다 익혔어.")
                        if st.button("진짜 수업 시작하기 🚀"):
                            st.session_state.tutorial_done = True
                            st.rerun()

    # 실제 수업 단계
    else:
        if not st.session_state.scenario:
            st.warning("선생님이 아직 수업을 만들어주지 않았어! 조금만 기다려줘. 😊")
            if st.button("다시 연습하기"): st.session_state.tutorial_done = False; st.rerun()
        else:
            idx = st.session_state.current_step
            steps = st.session_state.scenario.get('scenario', [])
            if idx < len(steps):
                data = steps[idx]
                st.header(f"🗣️ 토론 학습: {st.session_state.topic}")
                st.subheader(f"{idx+1}단계 토론")
                st.info(data['story'])
                st.write(f"💡 **토론 거리**: {data['debate_point']}")
                
                choice = st.radio("나의 선택은?", [data['choice_a'], data['choice_b']], key=f"radio_{idx}")
                reason = st.text_area("그렇게 생각한 이유는?", key=f"area_{idx}")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("주장 제출 📩"):
                        f_prompt = f"상황: {data['story']}\n선택: {choice}\n이유: {reason}\n따뜻하게 공감하고 반대 의견을 질문해줘."
                        st.session_state.chat_history.append({"role": "bot", "content": ask_gpt_text(f_prompt)})
                with c2:
                    if st.button("🎨 그림으로 보기"):
                        with st.spinner("그리는 중..."):
                            url = generate_image(data['story'])
                            if url: st.session_state.chat_history.append({"role": "img", "content": url})
                
                for msg in st.session_state.chat_history:
                    if msg["role"] == "bot": st.chat_message("assistant").write(msg["content"])
                    else: st.image(msg["content"])

                if st.button("다음 단계로 ➡️"):
                    st.session_state.current_step += 1
                    st.session_state.chat_history = []
                    st.rerun()
            else:
                st.balloons()
                st.success("와! 모든 토론과 학습을 성공적으로 마쳤어! 정말 대단해! 🎉")
                if st.button("처음으로 돌아가기"):
                    st.session_state.current_step = 0
                    st.session_state.tutorial_done = False
                    st.rerun()
