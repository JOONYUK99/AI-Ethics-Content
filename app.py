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

# --- 3. 시스템 페르소나 ---
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)의 비판적 사고와 창의성을 돕는 'AI 토론&아트 튜터'입니다.
학생이 스스로 생각하게 유도하고, 다정한 초등 교사 말투(~했니?, ~단다, ~해요)를 사용하세요.
어려운 단어는 피하고 쉽게 설명해주세요.
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
        st.error(f"데이터 생성 중 오류 발생: {e}")
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
        return "죄송해요, 지금은 대답하기가 조금 힘들어요. 잠시 후 다시 시도해주세요."

def generate_image(prompt):
    """DALL-E 3 이미지 생성 함수"""
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"A safe, friendly, cartoon-style illustration suitable for elementary school education: {prompt}",
            size="1024x1024",
            n=1
        )
        return response.data[0].url
    except Exception:
        return None

# --- 5. 세션 상태 초기화 ---
# 프로그램이 다시 실행돼도 데이터가 유지되도록 변수들을 초기화합니다.
default_values = {
    'scenario': {"scenario": []},
    'analysis': "",
    'current_step': 0,
    'chat_history': [],
    'topic': "",
    'tutorial_done': False,  # 튜토리얼 완료 여부
    'tutorial_step': 1       # 튜토리얼 진행 단계
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
    st.info("학생들이 학습할 주제를 입력하면 AI가 3단계 시나리오를 자동으로 만들어줍니다.")
    
    input_topic = st.text_input("토론 주제 입력", value=st.session_state.topic, placeholder="예: 동물원 폐지, 노키즈존, AI 숙제 허용")
    
    if st.button("🚀 수업 생성하기"):
        if not input_topic:
            st.warning("주제를 입력해주세요!")
        else:
            with st.spinner("AI 선생님이 수업 자료를 만들고 있습니다... (약 10~20초 소요)"):
                # 1. 시나리오 생성
                s_prompt = f"""
                주제 '{input_topic}'에 대해 초등학생이 토론할 수 있는 3단계 딜레마 시나리오를 JSON으로 만들어줘.
                형식: {{ "scenario": [ {{ "story": "상황설명", "choice_a": "선택A", "choice_b": "선택B" }}, ... ] }}
                이야기는 이어지도록 구성해줘.
                """
                st.session_state.scenario = ask_gpt_json(s_prompt)
                
                # 2. 수업 분석 생성
                a_prompt = f"주제 '{input_topic}'의 [핵심 가치], [관련 교과], [학습 목표]를 각각 한 문장으로 정리해줘."
                st.session_state.analysis = ask_gpt_text(a_prompt)
                
                # 3. 상태 업데이트 및 기존 이미지 캐시 삭제(새 수업 시작 시)
                st.session_state.topic = input_topic
                st.session_state.current_step = 0
                for key in list(st.session_state.keys()):
                    if key.startswith("img_url_"):
                        del st.session_state[key]
                        
                st.success("수업 생성이 완료되었습니다! 이제 학생용 모드로 전환하세요.")

    # 생성된 수업 내용 미리보기
    if st.session_state.analysis:
        st.divider()
        st.subheader("📊 수업 분석 내용")
        content = st.session_state.analysis
        parts = re.split(r'\[|\]', content)
        for i in range(1, len(parts), 2):
            if i+1 < len(parts):
                st.caption(f"**{parts[i]}**")
                st.write(parts[i+1].strip())

    if st.session_state.scenario.get('scenario'):
        with st.expander("📜 생성된 시나리오 확인하기"):
            for idx, item in enumerate(st.session_state.scenario['scenario']):
                st.markdown(f"**{idx+1}단계:** {item['story']}")
                st.text(f"A: {item['choice_a']} / B: {item['choice_b']}")

# [모드 2] 학생용: 튜토리얼 -> 실전 수업
elif mode == "🙋‍♂️ 학생용 (수업 참여)":
    
    # ---------------------------------------------------------
    # PART A. 튜토리얼 (수업 전 연습)
    # ---------------------------------------------------------
    if not st.session_state.tutorial_done:
        st.header("🎒 수업 준비 운동 (튜토리얼)")
        st.markdown("#### AI 선생님과 함께하는 즐거운 토론 수업!")
        st.info("본격적인 수업 전에 사용 방법을 먼저 익혀볼까요?")
        
        # 튜토리얼 진행률
        st.progress(st.session_state.tutorial_step / 3)

        # Mission 1: 선택하기
        if st.session_state.tutorial_step == 1:
            st.subheader("Mission 1. 선택 연습하기 👈")
            st.write("선생님이 질문을 하면, 네 생각을 골라야 해. 아래에서 가장 좋아하는 간식을 골라볼까?")
            snack = st.radio("가장 좋아하는 간식은?", ["달콤한 초콜릿 🍫", "바삭한 과자 🍪", "시원한 아이스크림 🍦"])
            
            if st.button("선택 완료! (다음으로)"):
                st.toast(f"와! {snack}을(를) 좋아하는구나! 아주 잘 골랐어.")
                st.session_state.tutorial_step = 2
                st.rerun()

        # Mission 2: 글쓰기
        elif st.session_state.tutorial_step == 2:
            st.subheader("Mission 2. 글쓰기 연습 ✍️")
            st.write("선택을 했으면 이유를 적어야겠지? 키보드로 네 생각을 적는 연습을 해보자.")
            t_input = st.text_area("오늘 기분이 어떤지 적어주세요! (예: 날씨가 좋아서 신나!)")
            
            if st.button("다 썼어요! (다음으로)"):
                if len(t_input) > 2:
                    st.toast("멋진 문장이야! 글쓰기 실력이 대단한걸?")
                    st.session_state.tutorial_step = 3
                    st.rerun()
                else:
                    st.warning("너무 짧아요! 조금만 더 길게 써볼까?")

        # Mission 3: 이미지 생성
        elif st.session_state.tutorial_step == 3:
            st.subheader("Mission 3. AI 화가와 그림 그리기 🎨")
            st.write("내가 상상한 장면을 글로 설명하면, AI가 그림을 그려줘. 한번 해볼까?")
            prompt_input = st.text_input("그리고 싶은 것을 설명해줘 (예: 우주복을 입은 귀여운 고양이)")
            
            if st.button("그림 생성하기 ✨"):
                if prompt_input:
                    with st.spinner("AI 화가가 붓을 들고 그림을 그리고 있어요..."):
                        img_url = generate_image(prompt_input)
                        if img_url:
                            st.image(img_url, caption="네가 주문한 그림이야! 정말 멋진데?")
                            st.success("모든 준비 운동 끝! 이제 진짜 수업으로 가보자.")
                            if st.button("수업 입장하기 🚀"):
                                st.session_state.tutorial_done = True
                                st.rerun()
                        else:
                            st.error("앗, 그림을 그리는 도중에 문제가 생겼어. 다시 한번 눌러볼래?")
                else:
                    st.warning("어떤 그림을 그릴지 먼저 적어줘야 해!")

    # ---------------------------------------------------------
    # PART B. 실제 수업 (시나리오 진행)
    # ---------------------------------------------------------
    else:
        # 안전장치: 시나리오 데이터 확인 (KeyError 방지)
        steps = st.session_state.scenario.get('scenario', [])
        
        if not steps:
            st.header("🏫 교실 대기 중...")
            st.image("https://media.giphy.com/media/l0HlBO7eyxdzTZtSS/giphy.gif", width=300)
            st.warning("선생님이 아직 수업 내용을 만들고 계셔! 잠시만 기다려줘.")
            if st.button("새로고침 🔄"):
                st.rerun()
        
        else:
            # 변수 설정
            idx = st.session_state.current_step
            total_steps = len(steps)

            # 진행률 표시 바
            st.progress((idx + 1) / total_steps)
            st.caption(f"현재 진행률: {idx + 1} / {total_steps} 단계")

            if idx < total_steps:
                data = steps[idx]
                
                st.title(f"🗣️ 토론 주제: {st.session_state.topic}")
                st.subheader(f"제 {idx+1}장. 어떻게 해야 할까?")

                # --- 이미지 자동 생성 및 캐싱 ---
                img_key = f"img_url_{idx}" # 단계별 고유 키 생성
                
                # 이미지가 없으면 생성 시도
                if img_key not in st.session_state:
                    with st.spinner("AI가 현재 상황을 그림으로 그리고 있어요..."):
                        scene_prompt = f"Scene describing: {data['story']}. Cartoon style for kids."
                        st.session_state[img_key] = generate_image(scene_prompt)
                
                # 이미지가 있으면 표시
                if st.session_state.get(img_key):
                    st.image(st.session_state[img_key], use_container_width=True, caption=f"{idx+1}단계 상황")

                # --- 스토리 및 선택 ---
                st.info(data['story']) # 상황 설명 박스
                
                with st.form(key=f"form_{idx}"):
                    choice = st.radio("너의 선택은?", [data['choice_a'], data['choice_b']])
                    reason = st.text_area("그렇게 선택한 이유는 뭐야?", placeholder="친구들에게 말하듯이 편하게 적어봐.")
                    submit_btn = st.form_submit_button("나의 주장 제출하기 📩")

                # 제출 시 피드백 로직
                if submit_btn:
                    if not reason.strip():
                        st.warning("이유를 적어야 AI 선생님과 이야기할 수 있어!")
                    else:
                        f_prompt = f"상황: {data['story']}\n학생선택: {choice}\n학생이유: {reason}\n따뜻하게 공감해주고, 반대 측면에서 생각할 거리를 질문 하나 해줘."
                        with st.spinner("AI 선생님이 답변을 생각 중..."):
                            feedback = ask_gpt_text(f_prompt)
                            # 챗 히스토리에 추가
                            st.session_state.chat_history.append({"role": "user", "content": f"선택: {choice}\n이유: {reason}"})
                            st.session_state.chat_history.append({"role": "assistant", "content": feedback})

                # 대화 기록 표시 (채팅 UI)
                if st.session_state.chat_history:
                    st.write("---")
                    st.subheader("💬 토론 내용")
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "assistant":
                            st.chat_message("assistant", avatar="🤖").write(msg["content"])
                        else:
                            st.chat_message("user", avatar="🙋‍♂️").write(msg["content"])

                # 다음 단계 이동 버튼 (대화가 1회 이상 오갔을 때 활성화 추천하지만, 편의상 항상 노출)
                if st.session_state.chat_history:
                    if st.button("다음 이야기로 넘어가기 ➡️"):
                        st.session_state.current_step += 1
                        st.session_state.chat_history = [] # 대화 기록 초기화
                        st.rerun()

            else:
                # 모든 단계 종료 시
                st.balloons()
                st.success("모든 토론을 훌륭하게 마쳤어! 정말 멋진 생각이란다. 🎉")
                
                if st.button("처음으로 돌아가기"):
                    st.session_state.current_step = 0
                    st.session_state.tutorial_done = False
                    st.session_state.chat_history = []
                    # 캐시된 이미지 삭제
                    for key in list(st.session_state.keys()):
                        if key.startswith("img_url_"):
                            del st.session_state[key]
                    st.rerun()
