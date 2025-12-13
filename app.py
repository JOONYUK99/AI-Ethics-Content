import streamlit as st
from openai import OpenAI
import re

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="쭈니봇과 함께 토론하기", page_icon="🤖")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    # secrets.toml 파일에 OPENAI_API_KEY가 있어야 합니다.
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ OpenAI API 키를 설정해주세요! (.streamlit/secrets.toml 파일 확인)")
    st.stop()

# --- 3. 예시 주제 데이터 ---
EXAMPLE_TOPICS = {
    "AI 예술과 저작권": "최근 열린 미술 대회에서 AI로 그린 그림이 1등을 차지해 큰 논란이 되었습니다. 그림을 그린 학생은 AI에게 수백 번의 지시어를 입력하며 원하는 그림을 얻었다고 주장합니다. 이 그림의 저작권은 누구에게 있어야 할까요?",
    "자율주행차의 딜레마": "자율주행차가 갑자기 나타난 아이들을 피하려고 핸들을 꺾으면, 차에 타고 있던 내가 다칠 수 있는 위험한 상황에 처했습니다. 이때 자율주행차는 어떤 선택을 해야 할까요?",
    "AI 튜터와 개인정보": "나의 모든 것을 알고 나에게 딱 맞는 공부법을 알려주는 AI 학습 로봇이 생겼습니다. 그런데 로봇이 나의 모든 학습 기록을 데이터 센터로 전송하고 있다는 사실을 알게 되었습니다.",
    "딥페이크와 가짜 뉴스": "친한 친구의 얼굴이 담긴 이상한 동영상을 인터넷에서 보게 되었습니다. 친구는 그런 영상을 찍은 적이 없다고 말하는데, 영상은 너무나 진짜 같아서 반 친구들 사이에 소문이 퍼지기 시작했습니다.",
    "AI 추천의 편향성": "새로 나온 동영상 앱을 사용하는데, 나에게는 항상 아이돌 춤 영상만 추천되고, 내 남동생에게는 게임 영상만 추천되는 것을 발견했습니다. 나는 게임도 좋아하는데 왜 앱은 나에게 게임 영상을 보여주지 않는 걸까요?"
}

# --- 4. AI 로직 함수들 (GPT-4o 적용) ---

def ask_gpt(prompt):
    """OpenAI GPT-4o 모델에게 질문하고 답을 받는 공통 함수"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 최신 GPT-4o 모델 사용
            messages=[
                {"role": "system", "content": "당신은 초등학생을 위한 다정한 AI 윤리 선생님 '쭈니봇'입니다. 답변은 항상 친절하고 이해하기 쉽게 해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7  # 창의성과 일관성의 균형
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT 응답 오류가 발생했어요: {e}")
        return None

def transform_scenario(teacher_input):
    """시나리오 생성"""
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
    return ask_gpt(prompt)

def analyze_student_response(debate_history):
    """학생 답변 분석 (감독 AI)"""
    prompt = (
        "아래 토론 내용 중 학생의 마지막 답변을 보고 다음 4가지 중 하나로 평가해주세요.\n"
        "답변은 오직 키워드만 출력하세요.\n\n"
        "# 평가 옵션:\n"
        "- 깊게 파고들기\n"
        "- 토론 이어가기\n"
        "- 쉽게 질문하기\n"
        "- 토론 마무리하기\n\n"
        f"--- 토론 내용 ---\n{debate_history}"
    )
    response = ask_gpt(prompt)
    if response:
        if "깊게" in response: return "deepen"
        if "쉽게" in response: return "simplify"
        if "마무리" in response: return "end_early"
    return "continue"

def generate_simpler_question(debate_history):
    """쉬운 질문 생성"""
    prompt = (
        "학생이 답변을 어려워합니다. 격려하며 '예/아니오'나 '선택지' 형태의 쉬운 질문을 해주세요.\n"
        f"--- 토론 내용 ---\n{debate_history}\n\n"
        "쭈니봇의 쉬운 질문:"
    )
    return ask_gpt(prompt)

def continue_debate(debate_history, level="normal"):
    """일반/심화 질문 생성"""
    instruction = "반론을 제기하거나 관점을 바꾸는 심화 질문을 해주세요." if level == "deepen" else "생각의 폭을 넓히는 다음 질문을 해주세요."
    prompt = (
        f"{instruction}\n"
        "학생의 의견을 깊이 공감하고 존중해주세요. 말투는 친근하게 해요체(~해요)를 써주세요.\n"
        f"--- 토론 내용 ---\n{debate_history}\n\n"
        "쭈니봇의 다음 질문:"
    )
    return ask_gpt(prompt)

def parse_and_store_scenario(generated_text):
    """생성된 텍스트 파싱"""
    if not generated_text: return False
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
    """토론 시작 질문"""
    prompt = (
        "학생의 선택을 지지하며 자연스럽게 토론을 시작하는 첫 질문을 해주세요.\n"
        f"--- 이야기와 선택 ---\n{history}\n학생의 선택: {choice}\n\n쭈니봇:"
    )
    return ask_gpt(prompt)

def generate_conclusion(final_history):
    """최종 피드백"""
    prompt = (
        "학생의 전체 토론 기록을 보고, 정답보다는 고민하는 과정이 중요했음을 칭찬하는 따뜻한 마무리 멘트를 해주세요.\n"
        f"--- 전체 기록 ---\n{final_history}"
    )
    return ask_gpt(prompt)

# --- 5. 메인 앱 로직 (UI) ---

def run_main_app():
    st.header("🤖 쭈니봇과 함께 토론하기 (GPT-4o ver.)")
    st.caption("OpenAI GPT-4o 모델이 이야기를 만들어줍니다.")

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
        st.info("토론하고 싶은 주제를 고르거나 직접 입력하면, 쭈니봇이 이야기를 만들어줄 거야!")
        
        options_list = ["직접 입력..."] + list(EXAMPLE_TOPICS.keys())
        selected_topic = st.selectbox("주제 선택:", options_list)

        default_text = ""
        if selected_topic != "직접 입력...":
            default_text = EXAMPLE_TOPICS[selected_topic]
        
        teacher_input = st.text_area("시나리오 소재 입력:", value=default_text, height=150)

        if st.button("토론 시작하기 ✨"):
            if not teacher_input.strip():
                st.warning("내용을 입력해주세요.")
            else:
                with st.spinner("쭈니봇(GPT)이 이야기를 만들고 있어요..."):
                    scenario_text = transform_scenario(teacher_input)
                    
                    if scenario_text and parse_and_store_scenario(scenario_text):
                        st.session_state.full_log = f"**주제:** {teacher_input[:50]}..."
                        st.session_state.current_part = 0
                        st.session_state.stage = 'story'
                        st.rerun()
                    else:
                        st.error("이야기를 만드는 데 실패했어요. 다시 시도해주세요.")

    # [2단계] 이야기 진행 및 선택
    elif st.session_state.stage == 'story':
        part = st.session_state.full_scenario[st.session_state.current_part]
        current_story = f"\n\n---\n\n### 📖 이야기 #{st.session_state.current_part + 1}\n{part['story']}"
        
        if current_story.strip() not in st.session_state.full_log:
            st.session_state.full_log += current_story
            
        st.markdown(current_story)
        st.info("어떤 선택을 할래?")
        
        c1, c2 = st.columns(2)
        if c1.button(f"🅰️ {part['choice_a']}", use_container_width=True):
            st.session_state.full_log += f"\n\n**>> 나의 선택(A):** {part['choice_a']}"
            st.session_state.stage = 'debate'
            st.rerun()
        if c2.button(f"🅱️ {part['choice_b']}", use_container_width=True):
            st.session_state.full_log += f"\n\n**>> 나의 선택(B):** {part['choice_b']}"
            st.session_state.stage = 'debate'
            st.rerun()

    # [3단계] 토론 진행
    elif st.session_state.stage == 'debate':
        for msg in st.session_state.full_log.split('\n\n'):
            msg = msg.strip()
            if not msg: continue
            if msg.startswith(">> 나의 선택"): st.chat_message("user", avatar="🙋").write(msg)
            elif msg.startswith("쭈니봇:"): st.chat_message("assistant", avatar="🤖").write(msg.replace("쭈니봇:", "**쭈니봇:**"))
            elif msg.startswith("나 (의견"): st.chat_message("user", avatar="🙋").write(msg)
            elif "### 📖 이야기" in msg: st.markdown(msg)

        if st.session_state.debate_finished:
            st.success("이번 토론이 끝났어!")
            is_last = st.session_state.current_part >= len(st.session_state.full_scenario) - 1
            if st.button("다음 이야기로 가기 ➡️" if not is_last else "최종 결과 보기 🏆"):
                st.session_state.debate_turns = 0
                st.session_state.debate_finished = False
                st.session_state.current_part += 1
                st.session_state.stage = 'conclusion' if is_last else 'story'
                st.rerun()
        
        else:
            if st.session_state.debate_turns == 0:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("쭈니봇이 생각 중..."):
                        last_choice = st.session_state.full_log.split('>> 나의 선택')[-1]
                        q = start_debate(st.session_state.full_log, last_choice)
                        st.write(f"**쭈니봇:** {q}")
                        st.session_state.full_log += f"\n\n쭈니봇: {q}"
                        st.session_state.debate_turns = 1

            elif st.session_state.debate_turns % 2 != 0:
                if user_input := st.chat_input(f"내 생각 말하기 ({ (st.session_state.debate_turns+1)//2 }/{MAX_DEBATE_REPLIES})"):
                    st.session_state.full_log += f"\n\n나 (의견): {user_input}"
                    st.session_state.debate_turns += 1
                    st.rerun()

            else:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("쭈니봇이 답변을 읽고 있어요..."):
                        decision = analyze_student_response(st.session_state.full_log)
                        
                        if decision == "end_early" or (st.session_state.debate_turns / 2) >= MAX_DEBATE_REPLIES:
                            msg = "좋은 의견이야! 네 생각을 들려줘서 고마워. 우리 이제 다음으로 넘어가 볼까?"
                            st.write(f"**쭈니봇:** {msg}")
                            st.session_state.full_log += f"\n\n쭈니봇: {msg}"
                            st.session_state.debate_finished = True
                            st.rerun()
                        else:
                            if decision == "simplify": q = generate_simpler_question(st.session_state.full_log)
                            elif decision == "deepen": q = continue_debate(st.session_state.full_log, "deepen")
                            else: q = continue_debate(st.session_state.full_log, "normal")
                            
                            st.write(f"**쭈니봇:** {q}")
                            st.session_state.full_log += f"\n\n쭈니봇: {q}"
                            st.session_state.debate_turns += 1
                            st.rerun()

    # [4단계] 최종 결과
    elif st.session_state.stage == 'conclusion':
        st.balloons()
        st.header("🎉 토론 완료!")
        st.subheader("우리가 나눈 이야기들")
        st.text_area("활동 기록", st.session_state.full_log, height=300)
        
        st.subheader("💌 쭈니봇의 편지")
        with st.spinner("편지 쓰는 중..."):
            final_comment = generate_conclusion(st.session_state.full_log)
            st.write(final_comment)
            
        if st.button("처음으로 돌아가기"):
            restart()

if __name__ == "__main__":
    run_main_app()
