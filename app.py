import streamlit as st
from openai import OpenAI
import json

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 윤리 교육", page_icon="🤖", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ API 키 오류: secrets.toml을 확인하세요.")
    st.stop()

# --- 3. 시스템 페르소나 (단답형/건조한 말투) ---
SYSTEM_PERSONA = """
당신은 AI 윤리 튜터입니다.
감정을 배제하고, 질문에 대해 핵심만 '단답형' 혹은 '개조식'으로 대답하세요.
인사말(안녕, 반가워)과 서술어(~입니다, ~해요)를 생략하세요.
예시: "선택 A의 윤리적 문제는 무엇인가?" -> "다수의 이익을 위해 소수를 희생하는 공리주의적 딜레마 발생."
"""

# --- 4. 주요 함수 ---

def ask_gpt_json(prompt):
    """JSON 응답 요청 (오류 발생 시 빈 구조 반환)"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        data = json.loads(response.choices[0].message.content.strip())
        if "scenario" not in data: return {"scenario": []}
        return data
    except Exception:
        return {"scenario": []}

def ask_gpt_text(prompt):
    """텍스트 응답 요청"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "응답 불가."

def generate_image(prompt):
    """이미지 생성"""
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"Minimalist, flat design illustration, educational context: {prompt}",
            size="1024x1024",
            n=1
        )
        return response.data[0].url
    except Exception:
        return None

# --- 5. 세션 상태 안전한 초기화 ---
if 'scenario' not in st.session_state or not isinstance(st.session_state.scenario, dict):
    st.session_state.scenario = {"scenario": []}

default_keys = {
    'analysis': "", 'current_step': 0, 'chat_history': [],
    'topic': "", 'tutorial_done': False, 'tutorial_step': 1
}
for k, v in default_keys.items():
    if k not in st.session_state: st.session_state[k] = v

# --- 6. 사이드바 ---
st.sidebar.title("🤖 AI 윤리 학습")

# [비상 버튼] 에러가 날 때 누르는 버튼
if st.sidebar.button("⚠️ 에러 해결 / 초기화"):
    st.session_state.clear()
    st.rerun()

mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"])

# --- 7. 메인 로직 ---

# [교사용 모드]
if mode == "👨‍🏫 교사용":
    st.header("🛠️ 수업 생성")
    input_topic = st.text_input("주제 입력", value=st.session_state.topic)
    
    if st.button("생성 시작"):
        if not input_topic:
            st.warning("주제 필요.")
        else:
            with st.spinner("데이터 생성 중..."):
                s_prompt = f"주제 '{input_topic}'의 3단계 딜레마 시나리오 JSON 생성. 키: scenario, 내부 키: story, choice_a, choice_b."
                result = ask_gpt_json(s_prompt)
                st.session_state.scenario = result
                
                a_prompt = f"주제 '{input_topic}'의 핵심 가치, 교과, 목표를 개조식으로 요약."
                st.session_state.analysis = ask_gpt_text(a_prompt)
                
                st.session_state.topic = input_topic
                st.session_state.current_step = 0
                
                keys_to_del = [k for k in st.session_state.keys() if k.startswith("img_url_")]
                for k in keys_to_del: del st.session_state[k]
                    
                st.success("생성 완료.")

    # [수정된 부분] 복잡한 표(Table) 대신 깔끔한 카드 디자인 적용
    scenario_data = st.session_state.scenario.get('scenario', [])
    
    if st.session_state.analysis:
        st.divider()
        st.subheader("📊 분석 결과")
        st.info(st.session_state.analysis) # 박스 안에 넣어 깔끔하게

    if scenario_data:
        st.divider()
        st.subheader("📜 시나리오 미리보기")
        
        # 반복문을 돌면서 카드 형태로 출력
        for i, step in enumerate(scenario_data):
            with st.container(border=True): # 테두리가 있는 박스 생성
                st.markdown(f"### 🔹 {i+1}단계")
                st.markdown(f"**📖 상황:** {step.get('story', '')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**🅰️ 선택:** {step.get('choice_a', '')}")
                with col2:
                    st.warning(f"**🅱️ 선택:** {step.get('choice_b', '')}")

# [학생용 모드]
elif mode == "🙋‍♂️ 학생용":
    
    # 튜토리얼
    if not st.session_state.tutorial_done:
        st.header("🎒 연습")
        st.progress(st.session_state.tutorial_step / 3)

        if st.session_state.tutorial_step == 1:
            st.subheader("1. 선택")
            c1, c2 = st.columns(2)
            with c1: 
                if st.button("A: 탕수육 찍먹"):
                    st.toast("선택: 찍먹")
                    st.session_state.tutorial_step = 2
                    st.rerun()
            with c2:
                if st.button("B: 탕수육 부먹"):
                    st.toast("선택: 부먹")
                    st.session_state.tutorial_step = 2
                    st.rerun()

        elif st.session_state.tutorial_step == 2:
            st.subheader("2. 입력")
            t_input = st.text_input("입력창")
            if st.button("전송"):
                if t_input:
                    st.toast("완료")
                    st.session_state.tutorial_step = 3
                    st.rerun()

        elif st.session_state.tutorial_step == 3:
            st.subheader("3. 생성")
            if st.button("테스트 이미지 생성"):
                with st.spinner("생성..."):
                    img = generate_image("Robot teacher")
                    if img:
                        st.image(img, width=300)
                        if st.button("수업 입장"):
                            st.session_state.tutorial_done = True
                            st.rerun()

    # 실전 수업
    else:
        steps = st.session_state.scenario.get('scenario', [])
        
        if not steps:
            st.warning("데이터 없음. 교사용 탭에서 생성 필요.")
            if st.button("새로고침"):
                st.rerun()
        
        else:
            idx = st.session_state.current_step
            total = len(steps)
            
            if idx >= total:
                st.success("수업 종료.")
                if st.button("처음으로"):
                    st.session_state.current_step = 0
                    st.session_state.tutorial_done = False
                    st.session_state.chat_history = []
                    st.rerun()
            else:
                data = steps[idx]
                st.progress((idx + 1) / total)
                st.subheader(f"단계 {idx+1}")

                img_key = f"img_url_{idx}"
                if img_key not in st.session_state:
                    with st.spinner("이미지 로딩..."):
                        st.session_state[img_key] = generate_image(data.get('story', ''))
                
                if st.session_state.get(img_key):
                    st.image(st.session_state[img_key])

                st.info(data.get('story', '내용 없음'))

                with st.form(f"form_{idx}"):
                    sel = st.radio("선택", [data.get('choice_a', 'A'), data.get('choice_b', 'B')])
                    reason = st.text_area("이유")
                    if st.form_submit_button("제출"):
                        if reason:
                            prompt = f"상황:{data.get('story')}, 선택:{sel}, 이유:{reason}. 핵심만 단답형 피드백."
                            with st.spinner("분석..."):
                                res = ask_gpt_text(prompt)
                                st.session_state.chat_history.append({"role": "user", "content": f"[{sel}] {reason}"})
                                st.session_state.chat_history.append({"role": "assistant", "content": res})
                        else:
                            st.warning("이유 입력 필요.")

                if st.session_state.chat_history:
                    st.divider()
                    for msg in st.session_state.chat_history:
                        role = "assistant" if msg["role"] == "assistant" else "user"
                        st.chat_message(role).write(msg["content"])
                    
                    if st.button("다음 단계 >"):
                        st.session_state.current_step += 1
                        st.session_state.chat_history = []
                        st.rerun()
