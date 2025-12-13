import streamlit as st
import google.generativeai as genai
import re

# --- 1. 예시 주제 데이터 (지식 베이스 내용 제거, 프롬프트만 유지) ---
EXAMPLE_TOPICS = {
    "AI 예술과 저작권": "최근 열린 미술 대회에서 AI로 그린 그림이 1등을 차지해 큰 논란이 되었습니다. 그림을 그린 학생은 AI에게 수백 번의 지시어를 입력하며 원하는 그림을 얻었다고 주장합니다. 이 그림의 저작권은 누구에게 있어야 할까요?",
    "자율주행차의 딜레마": "자율주행차가 갑자기 나타난 아이들을 피하려고 핸들을 꺾으면, 차에 타고 있던 내가 다칠 수 있는 위험한 상황에 처했습니다. 이때 자율주행차는 어떤 선택을 해야 할까요?",
    "AI 튜터와 개인정보": "나의 모든 것을 알고 나에게 딱 맞는 공부법을 알려주는 AI 학습 로봇이 생겼습니다. 그런데 로봇이 나의 모든 학습 기록을 데이터 센터로 전송하고 있다는 사실을 알게 되었습니다.",
    "딥페이크와 가짜 뉴스": "친한 친구의 얼굴이 담긴 이상한 동영상을 인터넷에서 보게 되었습니다. 친구는 그런 영상을 찍은 적이 없다고 말하는데, 영상은 너무나 진짜 같아서 반 친구들 사이에 소문이 퍼지기 시작했습니다.",
    "AI 추천의 편향성": "새로 나온 동영상 앱을 사용하는데, 나에게는 항상 아이돌 춤 영상만 추천되고, 내 남동생에게는 게임 영상만 추천되는 것을 발견했습니다. 나는 게임도 좋아하는데 왜 앱은 나에게 게임 영상을 보여주지 않는 걸까요?"
}

# --- 2. 기본 설정 및 함수 ---
# API 키 설정
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ 구글 API 키를 설정해주세요! (Streamlit secrets)")
    st.stop()

def get_model():
    """Gemini 모델을 가져오는 함수"""
    return genai.GenerativeModel('gemini-pro')

# --- 3. AI 로직 함수들 (RAG 제거됨) ---

def transform_scenario(teacher_input):
    """
    선생님의 입력(teacher_input)만을 바탕으로 시나리오를 생성합니다.
    참고 자료(Context) 주입 부분이 제거되었습니다.
    """
    model = get_model()
    prompt = (
        "당신은 초등학생 고학년 눈높이에 맞춰 AI 윤리 교육용 인터랙티브 시나리오를 작성하는 전문 작가입니다.\n"
        "아래 '입력 내용'을 바탕으로, 학생들이 몰입할 수 있는 완결된 이야기를 만들어 주세요.\n"
        "이야기는 총 4개의 파트로 구성되며, 각 파트 끝에는 주인공의 고민이 드러나는 두 가지 선택지를 제시해야 합니다.\n\n"
        
        "# 필수 출력 형식:\n"
        "[STORY 1] ... [CHOICE 1A] ... [CHOICE 1B] ...\n---\n"
        "[STORY 2] ... [CHOICE 2A] ... [CHOICE 2B] ...\n---\n"
        "[STORY 3] ... [CHOICE 3A] ... [CHOICE 3B] ...\n---\n"
        "[STORY 4] ... [CHOICE 4A] ... [CHOICE 4B] ...\n\n"
        
        f"--- 입력 내용 ---\n{teacher_input}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"시나리오 생성 중 오류: {e}")
        return None

def analyze_student_response(debate_history):
    """학생의 답변을 분석하여 토론 방향을 결정하는 감독 AI"""
    model = get_model()
    prompt = (
        "당신은 학생의 토론 능력을 분석하는 교육 심리 전문가입니다.\n"
        "아래 '토론 내용'에서 가장 마지막에 학생이 한 답변을 보고, 학생의 이해도와 논리력을 판단해주세요.\n"
        "반드시 다음 네 가지 중 하나로만 평가하고, 그 이유를 간략하게 설명해주세요.\n\n"
        "# 평가 옵션:\n"
        "- 깊게 파고들기: 학생이 주제를 잘 이해함. 심화 질문 필요.\n"
        "- 토론 이어가기: 무난하게 토론 진행 가능.\n"
        "- 쉽게 질문하기: 학생이 어려워함. 더 쉬운 질문 필요.\n"
        "- 토론 마무리하기: 주제 이탈 혹은 부담감. 마무리 필요.\n\n"
        f"--- 토론 내용 ---\n{debate_history}\n\n"
        "# 평가 결과 (옵션 중 하나만 선택):"
    )
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if "깊게 파고들기" in text: return "deepen"
        elif "쉽게 질문하기" in text: return "simplify"
        elif "토론 마무리하기" in text: return "end_early"
        else: return "continue"
    except Exception:
        return "continue"

def generate_simpler_question(debate_history):
    """쉬운 질문 생성"""
    model = get_model()
    prompt = (
        "당신은 매우 다정하고 친절한 AI 윤리 선생님입니다.\n"
        "학생이 답변을 어려워합니다. 학생을 격려하며 '예/아니오'나 '선택지' 형태의 쉬운 질문을 해주세요.\n\n"
        f"--- 토론 내용 ---\n{debate_history}\n\n"
        "AI 선생님의 쉬운 질문:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "조금 어려운가요? 괜찮아요. 그럼 간단하게 생각해보죠."

def continue_debate(debate_history, level="normal"):
    """일반/심화 질문 생성"""
    model = get_model()
    instruction = "반론을 제기하거나 관점을 바꾸는 심화 질문을 해주세요." if level == "deepen" else "생각의 폭을 넓히는 다음 질문을 해주세요."
    prompt = (
        f"당신은 다정한 AI 윤리 선생님입니다. {instruction}\n"
        "학생의 의견을 깊이 공감하고 존중해주세요.\n\n"
        f"--- 토론 내용 ---\n{debate_history}\n\n"
        "AI 선생님의 다음 질문:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "좋은 생각이네요! 또 다른 생각은 없나요?"

def parse_and_store_scenario(generated_text):
    """생성된 텍스트를 파싱하여 세션에 저장"""
    st.session_state.full_scenario = []
    parts = generated_text.split('---')
    if len(parts) < 4: return False
    for part in parts:
        try:
            story = re.search(r"\[STORY\s?\d\](.*?)(?=\[CHOICE\s?\dA\])", part, re.DOTALL).group(1).strip()
            choice_a = re.search(r"\[CHOICE\s?\dA\](.*?)(?=\[CHOICE\s?\dB\])", part, re.DOTALL).group(1).strip()
            choice_b = re.search(r"\[CHOICE\s?\dB\](.*)", part, re.DOTALL).group(1).strip()
            st.session_state.full_scenario.append({"story": story, "choice_a": choice_a, "choice_b": choice_b})
        except Exception:
            continue
    return len(st.session_state.full_scenario) >= 4

def start_debate(history, choice):
    """토론 시작 질문 생성"""
    model = get_model()
    prompt = (
        "당신은 다정한 AI 윤리 선생님입니다. 학생의 선택을 지지하며 자연스럽게 토론을 시작하는 첫 질문을 해주세요.\n"
        f"--- 이야기와 선택 ---\n{history}\n학생의 선택: {choice}\n\nAI 선생님:"
    )
    try:
        response = model.generate_content(prompt); return response.text.strip()
    except Exception: return "선택했군요! 왜 그런 선택을 했는지 궁금해요."

def generate_conclusion(final_history):
    """최종 피드백 생성"""
    model = get_model()
    prompt = (
        "당신은 다정한 AI 윤리 선생님입니다. 학생의 전체 토론 기록을 보고, 정답보다는 고민하는 과정이 중요했음을 칭찬하는 따뜻한 마무리 멘트를 해주세요.\n"
        f"--- 전체 기록 ---\n{final_history}"
    )
    try:
        response = model.generate_content(prompt); return response.text.strip()
    except Exception: return "정말 수고했어요! 훌륭한 토론이었습니다."

# --- 4. 메인 앱 로직 ---

def run_main_app():
    st.header("✨ 초등학생을 위한 AI 윤리 교육 (Basic ver.)")
    st.caption("AI의 기본 지식만을 활용하여 시나리오를 생성합니다.")

    # 세션 초기화
    if 'stage' not in st.session_state:
        st.session_state.stage = 'start'
        st.session_state.full_scenario = []
        st.session_state.full_log = ""
        st.session_state.current_part = -1
        st.session_state.debate_turns = 0
        st.session_state.debate_finished = False

    MAX_DEBATE_REPLIES = 3

    def restart():
        for key in ['stage', 'full_scenario', 'full_log', 'current_part', 'debate_turns', 'debate_finished']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

    # [1단계] 시작 화면
    if st.session_state.stage == 'start':
        st.info("AI 윤리 주제를 입력하거나 선택하면, AI가 즉시 시나리오를 만들어줍니다.")
        
        # 주제 선택 (RAG 내용 없이 프롬프트만 가져옴)
        options_list = ["직접 입력..."] + list(EXAMPLE_TOPICS.keys())
        selected_topic = st.selectbox("주제 선택:", options_list)

        default_text = ""
        if selected_topic != "직접 입력...":
            default_text = EXAMPLE_TOPICS[selected_topic]
        
        teacher_input = st.text_area("시나리오 소재 입력:", value=default_text, height=150)

        if st.button("교육 콘텐츠 생성하기"):
            if not teacher_input.strip():
                st.warning("내용을 입력해주세요.")
            else:
                with st.spinner("AI가 상상력을 발휘해 이야기를 만들고 있어요..."):
                    # RAG 과정 없이 바로 생성 함수 호출
                    scenario_text = transform_scenario(teacher_input)
                    
                    if scenario_text and parse_and_store_scenario(scenario_text):
                        st.session_state.full_log = f"**주제:** {teacher_input[:50]}..."
                        st.session_state.current_part = 0
                        st.session_state.stage = 'story'
                        st.rerun()
                    else:
                        st.error("시나리오 생성 실패. 다시 시도해주세요.")

    # [2단계] 이야기 진행 및 선택
    elif st.session_state.stage == 'story':
        part = st.session_state.full_scenario[st.session_state.current_part]
        current_story = f"\n\n---\n\n### 📖 이야기 #{st.session_state.current_part + 1}\n{part['story']}"
        
        # 로그에 없으면 추가
        if current_story.strip() not in st.session_state.full_log:
            st.session_state.full_log += current_story
            
        st.markdown(current_story)
        st.info("어떤 선택을 하시겠습니까?")
        
        c1, c2 = st.columns(2)
        if c1.button(f"A: {part['choice_a']}", use_container_width=True):
            st.session_state.full_log += f"\n\n**>> 나의 선택(A):** {part['choice_a']}"
            st.session_state.stage = 'debate'
            st.rerun()
        if c2.button(f"B: {part['choice_b']}", use_container_width=True):
            st.session_state.full_log += f"\n\n**>> 나의 선택(B):** {part['choice_b']}"
            st.session_state.stage = 'debate'
            st.rerun()

    # [3단계] 토론 진행
    elif st.session_state.stage == 'debate':
        # 채팅 UI 표시
        for msg in st.session_state.full_log.split('\n\n'):
            msg = msg.strip()
            if not msg: continue
            if msg.startswith(">> 나의 선택"): st.chat_message("user", avatar="🙋").write(msg)
            elif msg.startswith("AI 선생님:"): st.chat_message("assistant", avatar="🤖").write(msg.replace("AI 선생님:", ""))
            elif msg.startswith("나 (의견"): st.chat_message("user", avatar="🙋").write(msg)
            elif "### 📖 이야기" in msg: st.markdown(msg) # 이야기 부분은 텍스트로

        # 토론 종료 처리
        if st.session_state.debate_finished:
            st.success("토론이 끝났습니다.")
            is_last = st.session_state.current_part >= len(st.session_state.full_scenario) - 1
            if st.button("다음 단계로" if not is_last else "최종 결과 보기"):
                st.session_state.debate_turns = 0
                st.session_state.debate_finished = False
                st.session_state.current_part += 1
                st.session_state.stage = 'conclusion' if is_last else 'story'
                st.rerun()
        
        else:
            # 1. AI 첫 질문
            if st.session_state.debate_turns == 0:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("생각 중..."):
                        last_choice = st.session_state.full_log.split('>> 나의 선택')[-1]
                        q = start_debate(st.session_state.full_log, last_choice)
                        st.write(q)
                        st.session_state.full_log += f"\n\nAI 선생님: {q}"
                        st.session_state.debate_turns = 1

            # 2. 학생 답변 입력
            elif st.session_state.debate_turns % 2 != 0:
                if user_input := st.chat_input(f"답변하기 ({ (st.session_state.debate_turns+1)//2 }/{MAX_DEBATE_REPLIES})"):
                    st.session_state.full_log += f"\n\n나 (의견): {user_input}"
                    st.session_state.debate_turns += 1
                    st.rerun()

            # 3. AI 반응 및 다음 질문
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("분석 중..."):
                        decision = analyze_student_response(st.session_state.full_log)
                        
                        # 종료 조건 확인
                        if decision == "end_early" or (st.session_state.debate_turns / 2) >= MAX_DEBATE_REPLIES:
                            msg = "좋은 의견 감사합니다. 이 주제에 대해 충분히 이야기를 나눈 것 같네요!"
                            st.write(msg)
                            st.session_state.full_log += f"\n\nAI 선생님: {msg}"
                            st.session_state.debate_finished = True
                            st.rerun()
                        else:
                            # 질문 생성
                            if decision == "simplify": q = generate_simpler_question(st.session_state.full_log)
                            elif decision == "deepen": q = continue_debate(st.session_state.full_log, "deepen")
                            else: q = continue_debate(st.session_state.full_log, "normal")
                            
                            st.write(q)
                            st.session_state.full_log += f"\n\nAI 선생님: {q}"
                            st.session_state.debate_turns += 1
                            st.rerun()

    # [4단계] 최종 결과
    elif st.session_state.stage == 'conclusion':
        st.balloons()
        st.header("🎉 수업 종료")
        st.subheader("전체 기록")
        st.text_area("활동 로그", st.session_state.full_log, height=300)
        
        st.subheader("👩‍🏫 선생님의 피드백")
        with st.spinner("피드백 작성 중..."):
            final_comment = generate_conclusion(st.session_state.full_log)
            st.write(final_comment)
            
        if st.button("처음으로 돌아가기"):
            restart()

# 앱 실행
if __name__ == "__main__":
    run_main_app()
