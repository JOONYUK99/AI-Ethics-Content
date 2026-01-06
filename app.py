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
당신은 초등학생(5~6학년)의 비판적 사고와 창의성을 돕는 'AI 토론&아트 튜터'입니다.
정답을 내리기보다 학생이 스스로 생각하게 유도하고, 다정한 말투(~했니?, ~단다)를 사용하세요.
"""

# --- 4. 주요 함수 (KeyError 방어 로직 포함) ---

def ask_gpt_json(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PERSONA}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        data = json.loads(response.choices[0].message.content.strip())
        # 데이터 구조 보장 (KeyError 방지)
        if 'scenario' not in data:
            data = {'scenario': []}
        return data
    except:
        return {'scenario': []}

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
keys = {
    'scenario': {'scenario': []},
    'analysis': '',
    'current_step': 0,
    'chat_history': [],
    'topic': '',
    'tutorial_done': False,
    'tutorial_step': 1
}
for key, value in keys.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- 6. 메인 로직 ---

st.sidebar.title("🏫 AI 지능형 학습")
mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"])

if mode == "👨‍🏫 교사용":
    st.header("🛠️ 토론 수업 설계 및 분석")
    input_topic = st.text_input("토론 주제", value=st.session_state.topic)
    
    if st.button("🚀 수업 설계하기"):
        with st.spinner("AI가 분석 중..."):
            s_prompt = f"주제 '{input_topic}'로 초등용 3단계 토론 시나리오를 JSON으로 만들어줘. 키는 'scenario'이고 내부 키는 'story', 'choice_a', 'choice_b', 'debate_point'야."
            st.session_state.scenario = ask_gpt_json(s_prompt)
            
            a_prompt = f"주제 '{input_topic}'의 [핵심 가치], [연계 교과], [학습 목표]를 각각 짧게 한 문장씩 작성해줘."
            st.session_state.analysis = ask_gpt_text(a_prompt)
            st.session_state.topic = input_topic
            st.success("수업 생성이 완료되었습니다!")

    if st.session_state.analysis:
        st.subheader("📊 수업 분석 리포트")
        # 정규표현식으로 각 항목을 분리하여 개별 상자에 표시
        parts = re.split(r'\[|\]', st.session_state.analysis)
        for i in range(1, len(parts), 2):
            if i+1 < len(parts):
                st.info(f"**{parts[i]}**: {parts[i+1].strip()}")

    # KeyError 방어하며 시나리오 미리보기 출력
    scenario_data = st.session_state.scenario.get('scenario', [])
    if scenario_data:
        with st.expander("📜 전체 시나리오 미리보기"):
            st.table(scenario_data)

elif mode == "🙋‍♂️ 학생용":
    if not st.session_state.tutorial_done:
        st.header("🎒 수업 전 가이드 연습")
        t_step = st.session_state.tutorial_step
        
        if t_step == 1:
            st.subheader("1. 입장 선택 연습")
            st.info("버튼을 눌러보세요!")
            if st.button("😊 토론 시작!"):
                st.session_state.tutorial_step = 2
                st.rerun()
            
        elif t_step == 2:
            st.subheader("2. 생각 적기 연습")
            st.info("주장을 적어보세요.")
            t_input = st.text_input("연습 입력창")
            if st.button("연습 제출"):
                if t_input:
                    st.session_state.tutorial_step = 3
                    st.rerun()
                
        elif t_step == 3:
            st.subheader("3. 그림 연습")
            st.info("그림 생성 버튼을 확인해볼까요?")
            if st.button("🎨 연습용 그림 그리기"):
                with st.spinner("그리는 중..."):
                    img = generate_image("Happy children learning AI")
                    if img:
                        st.image(img, width=400)
                        if st.button("진짜 수업 시작하기 🚀"):
                            st.session_state.tutorial_done = True
                            st.rerun()
    else:
        # 실제 수업 진행 (KeyError 방지 적용)
        steps = st.session_state.scenario.get('scenario', [])
        if not steps:
            st.warning("선생님이 아직 수업을 준비 중이야!")
        else:
            idx = st.session_state.current_step
            if idx < len(steps):
                data = steps[idx]
                st.header(f"🗣️ {st.session_state.topic}")
                st.info(data.get('story', '상황을 불러오는 중...'))
                
                choice = st.radio("나의 선택은?", [data.get('choice_a', 'A'), data.get('choice_b', 'B')], key=f"r_{idx}")
                reason = st.text_area("이유를 말해줄래?", key=f"a_{idx}")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("주장 제출 📩", key=f"s_{idx}"):
                        st.session_state.chat_history.append({"role": "bot", "content": ask_gpt_text(f"공감과 피드백 해줘: {reason}")})
                with c2:
                    if st.button("🎨 장면 그림으로 보기", key=f"i_{idx}"):
                        with st.spinner("그리는 중..."):
                            url = generate_image(data.get('story', ''))
                            if url: st.session_state.chat_history.append({"role": "img", "content": url})
                
                for msg in st.session_state.chat_history:
                    if msg["role"] == "bot": st.chat_message("assistant").write(msg["content"])
                    else: st.image(msg["content"], width=400)

                if st.button("다음 논제로 ➡️", key=f"n_{idx}"):
                    st.session_state.current_step += 1
                    st.session_state.chat_history = []
                    st.rerun()
            else:
                st.balloons()
                st.success("오늘의 학습을 마쳤어!")
                if st.button("처음으로"):
                    st.session_state.current_step = 0
                    st.session_state.tutorial_done = False
                    st.rerun()
