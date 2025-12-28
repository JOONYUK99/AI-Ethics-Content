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
    st.error("⚠️ API 키를 설정해주세요! (.streamlit/secrets.toml 파일 확인)")
    st.stop()

# --- 3. 시스템 페르소나 (말투 수정: 단답형) ---
SYSTEM_PERSONA = """
당신은 초등학생의 비판적 사고를 돕는 AI 튜터입니다.
질문에 대해 핵심만 간결하게 '단답형'으로 대답하세요.
불필요한 미사여구(안녕, 반가워 등)는 생략하고 사실과 질문 위주로 짧게 말하세요.
"""

# --- 4. 주요 함수 ---

def ask_gpt_json(prompt):
    """JSON 형식으로 응답을 요청하는 함수 (시나리오 생성용)"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        # 에러 발생 시 빈 시나리오 구조 반환 (KeyError 방지)
        return {"scenario": []}

def ask_gpt_text(prompt):
    """일반 텍스트 응답을 요청하는 함수 (피드백용)"""
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
        return "응답 생성 실패."

def generate_image(prompt):
    """DALL-E 3 이미지 생성 함수"""
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"Simple, clear cartoon style illustration: {prompt}",
            size="1024x1024",
            n=1
        )
        return response.data[0].url
    except Exception:
        return None

# --- 5. 세션 상태 초기화 ---
# 데이터가 없어도 에러가 나지 않도록 초기값을 확실하게 설정합니다.
default_values = {
    'scenario': {"scenario": []}, # 기본 구조 보장
    'analysis': "",
    'current_step': 0,
    'chat_history': [],
    'topic': "",
    'tutorial_done': False,
    'tutorial_step': 1
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- 6. 사이드바 메뉴 ---
st.sidebar.title("🏫 AI 지능형 학습")
mode = st.sidebar.radio("모드 선택", ["👨‍🏫 교사용 (수업 만들기)", "🙋‍♂️ 학생용 (수업 참여)"])

# --- 7. 메인 로직 ---

# [모드 1] 교사용: 수업 설계
if mode == "👨‍🏫 교사용 (수업 만들기)":
    st.header("🛠️ 토론 수업 설계")
    
    input_topic = st.text_input("토론 주제 입력", value=st.session_state.topic, placeholder="주제를 입력하세요")
    
    if st.button("🚀 수업 생성"):
        if not input_topic:
            st.warning("주제를 입력하세요.")
        else:
            with st.spinner("수업 데이터 생성 중..."):
                # 1. 시나리오 생성
                s_prompt = f"""
                주제 '{input_topic}'로 초등학생용 3단계 딜레마 시나리오 JSON 생성.
                형식: {{ "scenario": [ {{ "story": "상황설명", "choice_a": "선택A", "choice_b": "선택B" }} ] }}
                """
                st.session_state.scenario = ask_gpt_json(s_prompt)
                
                # 2. 수업 분석 생성
                a_prompt = f"주제 '{input_topic}'의 [핵심가치], [교과], [목표]를 단답형으로 요약."
                st.session_state.analysis = ask_gpt_text(a_prompt)
                
                # 3. 상태 업데이트
                st.session_state.topic = input_topic
                st.session_state.current_step = 0
                
                # 기존 이미지 캐시 삭제
                for key in list(st.session_state.keys()):
                    if key.startswith("img_url_"):
                        del st.session_state[key]
                        
                st.success("생성 완료.")

    # 생성된 수업 내용 미리보기 (KeyError 방지 로직 적용)
    if st.session_state.analysis:
        st.divider()
        st.subheader("📊 수업 분석")
        st.write(st.session_state.analysis)

    # [수정 포인트 1] 데이터가 있고, 키가 확실히 존재할 때만 테이블 표시
    scenario_data = st.session_state.scenario.get('scenario', [])
    if scenario_data:
        with st.expander("📜 시나리오 확인"):
            st.table(scenario_data)

# [모드 2] 학생용: 튜토리얼 -> 실전 수업
elif mode == "🙋‍♂️ 학생용 (수업 참여)":
    
    # PART A. 튜토리얼
    if not st.session_state.tutorial_done:
        st.header("🎒 튜토리얼 (연습)")
        st.progress(st.session_state.tutorial_step / 3)

        if st.session_state.tutorial_step == 1:
            st.subheader("1. 선택 연습")
            snack = st.radio("좋아하는 간식은?", ["초콜릿", "과자", "아이스크림"])
            if st.button("확인"):
                st.toast(f"선택: {snack}")
                st.session_state.tutorial_step = 2
                st.rerun()

        elif st.session_state.tutorial_step == 2:
            st.subheader("2. 입력 연습")
            t_input = st.text_area("오늘 기분 입력")
            if st.button("제출"):
                if len(t_input) > 0:
                    st.toast("입력 완료")
                    st.session_state.tutorial_step = 3
                    st.rerun()
                else:
                    st.warning("내용을 입력하세요.")

        elif st.session_state.tutorial_step == 3:
            st.subheader("3. 그림 생성 연습")
            prompt_input = st.text_input("그릴 내용 입력 (예: 고양이)")
            if st.button("생성"):
                if prompt_input:
                    with st.spinner("생성 중..."):
                        img_url = generate_image(prompt_input)
                        if img_url:
                            st.image(img_url)
                            if st.button("수업 시작하기"):
                                st.session_state.tutorial_done = True
                                st.rerun()
                else:
                    st.warning("내용을 입력하세요.")

    # PART B. 실제 수업
    else:
        # [수정 포인트 2] KeyError 완벽 차단: .get() 사용 및 리스트 확인
        steps = st.session_state.scenario.get('scenario', [])
        
        if not steps:
            st.warning("수업 내용이 없습니다. 선생님이 수업을 생성할 때까지 기다리세요.")
            if st.button("새로고침"):
                st.rerun()
        
        else:
            idx = st.session_state.current_step
            total_steps = len(steps)

            st.progress((idx + 1) / total_steps)

            if idx < total_steps:
                data = steps[idx]
                
                st.subheader(f"단계 {idx+1}/{total_steps}")

                # 이미지 자동 생성
                img_key = f"img_url_{idx}"
                if img_key not in st.session_state:
                    with st.spinner("이미지 생성 중..."):
                        st.session_state[img_key] = generate_image(data['story'])
                
                if st.session_state.get(img_key):
                    st.image(st.session_state[img_key], use_container_width=True)

                st.info(data['story'])
                
                with st.form(key=f"form_{idx}"):
                    choice = st.radio("선택", [data['choice_a'], data['choice_b']])
                    reason = st.text_area("이유 입력")
                    submit_btn = st.form_submit_button("제출")

                if submit_btn:
                    if not reason.strip():
                        st.warning("이유를 입력하세요.")
                    else:
                        # 말투 단답형 요청
                        f_prompt = f"상황: {data['story']}\n선택: {choice}\n이유: {reason}\n이에 대해 단답형으로 핵심만 피드백하고, 짧은 질문 하나 던져줘."
                        with st.spinner("분석 중..."):
                            feedback = ask_gpt_text(f_prompt)
                            st.session_state.chat_history.append({"role": "user", "content": f"선택: {choice}\n이유: {reason}"})
                            st.session_state.chat_history.append({"role": "assistant", "content": feedback})

                if st.session_state.chat_history:
                    st.write("---")
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "assistant":
                            st.chat_message("assistant").write(msg["content"])
                        else:
                            st.chat_message("user").write(msg["content"])

                if st.session_state.chat_history:
                    if st.button("다음 단계"):
                        st.session_state.current_step += 1
                        st.session_state.chat_history = []
                        st.rerun()

            else:
                st.success("수업 종료.")
                if st.button("처음으로"):
                    st.session_state.current_step = 0
                    st.session_state.tutorial_done = False
                    st.session_state.chat_history = []
                    st.rerun()
