import streamlit as st
import google.generativeai as genai
import re

# --- 1. AI 핵심 기능 함수 정의 ---
def get_model():
    return genai.GenerativeModel('gemini-pro-latest')

def generate_story_part(topic, history_summary=""):
    model = get_model()
    # <--- 수정: 이야기를 '반드시 2문장'으로 매우 짧게 만들도록 프롬프트 강화
    if not history_summary:
        prompt = f"'{topic}'라는 주제로, 초등학생 저학년도 이해할 수 있는 AI 윤리 동화의 '첫 부분'을 만들어줘. 이야기는 반드시 간결한 두 문장으로만 구성하고, 주인공이 중요한 결정을 내려야 하는 순간에서 끝나야 해. 절대 길게 쓰지 마."
    else:
        prompt = f"다음은 지금까지 진행된 이야기의 요약이야: '{history_summary}'. 이 이야기에 이어서, 학생의 선택으로 인해 벌어지는 '다음 사건'을 반드시 간결한 두 문장으로 만들어줘. 그리고 이야기가 또 다른 중요한 결정을 내려야 하는 순간에서 끝나도록 해줘."
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e: return f"이야기 생성 중 오류: {e}"

def generate_choices_for_story(story_part):
    model = get_model()
    prompt = f"아래 이야기의 마지막 상황에서 주인공이 할 수 있는, 윤리적으로 상반된 두 가지 선택지를 초등학생 눈높이에 맞춰서 간결하게 만들어줘.\n[출력 형식]\nA: [A 선택지 내용]\nB: [B 선택지 내용]\n\n--- 이야기 ---\n{story_part}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e: return f"선택지 생성 중 오류: {e}"

def start_debate(history, choice):
    model = get_model()
    prompt = (
        "당신은 학생들을 아주 아끼는 다정한 AI 윤리 선생님입니다. 학생의 선택을 격려하며 토론을 시작해주세요.\n"
        f"--- 지금까지의 이야기와 학생의 선택 ---\n{history}\n학생의 선택: {choice}\n\nAI 선생님의 따뜻한 첫 질문:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"토론 시작 중 오류: {e}"

def continue_debate(debate_history):
    model = get_model()
    prompt = (
        "당신은 다정한 AI 윤리 선생님입니다. 학생의 의견에 공감하며 토론을 이어가주세요.\n"
        f"--- 지금까지의 토론 내용 ---\n{debate_history}\n\nAI 선생님의 다음 질문:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"토론 중 오류: {e}"

def generate_conclusion(final_history):
    model = get_model()
    prompt = (
        "당신은 학생의 성장을 지켜본 다정한 AI 윤리 선생님입니다.\n"
        "다음은 한 학생이 AI 윤리 문제에 대해 거친 전체 기록입니다. 이 기록을 바탕으로, 학생의 고민 과정을 칭찬하고, 정답 찾기보다 과정 자체가 중요했다는 점을 강조하는 따뜻하고 격려가 되는 마무리 메시지를 작성해주세요.\n\n"
        f"--- 전체 기록 ---\n{final_history}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"결론 생성 중 오류: {e}"

# --- 2. Streamlit 앱 UI 및 로직 ---
st.set_page_config(page_title="AI 윤리 교육 챗봇", page_icon="✨", layout="centered")
st.title("✨ 초등학생을 위한 AI 윤리 교육")

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("API 키를 설정해주세요!")
    st.stop()

if 'stage' not in st.session_state:
    st.session_state.stage = 'start'
    st.session_state.full_scenario = []
    st.session_state.full_log = ""
    st.session_state.current_part = -1
    st.session_state.debate_turns = 0

def parse_and_store_scenario(generated_text):
    st.session_state.full_scenario = []
    parts = generated_text.split('---')
    for i, part in enumerate(parts):
        try:
            story = re.search(rf"\[STORY {i+1}\](.*?)(?=\[CHOICE {i+1}A\])", part, re.DOTALL).group(1).strip()
            choice_a = re.search(rf"\[CHOICE {i+1}A\](.*?)(?=\[CHOICE {i+1}B\])", part, re.DOTALL).group(1).strip()
            choice_b = re.search(rf"\[CHOICE {i+1}B\](.*)", part, re.DOTALL).group(1).strip()
            st.session_state.full_scenario.append({"story": story, "choice_a": choice_a, "choice_b": choice_b})
        except Exception:
            continue
    return len(st.session_state.full_scenario) >= 4

def restart_lesson():
    st.session_state.stage = 'start'
    st.session_state.full_scenario = []
    st.session_state.full_log = ""
    st.session_state.current_part = -1
    st.session_state.debate_turns = 0

if st.session_state.stage == 'start':
    st.info("안녕하세요, 친구들! AI 윤리 문제에 대해 함께 고민해보는 수업에 오신 것을 환영해요.")
    topics = ["자율주행 자동차의 윤리적 딜레마", "인공지능 판사의 공정성 문제", "AI 창작물의 저작권", "개인정보를 학습한 AI 챗봇"]
    selected_topic = st.selectbox("오늘 탐구해볼 주제를 선택해볼까요?", topics)

    if st.button("수업 시작하기"):
        st.session_state.topic = selected_topic
        st.session_state.stage = 'story'
        st.rerun()

elif st.session_state.stage == 'story':
    if st.session_state.current_part == -1:
        st.session_state.current_part = 0
        st.session_state.full_log = f"**주제:** {st.session_state.topic}"
    
    st.markdown(st.session_state.full_log, unsafe_allow_html=True)
    
    with st.spinner(f"AI가 이야기 #{st.session_state.current_part + 1}을(를) 생성 중입니다..."):
        history_summary = st.session_state.full_log[-500:] if st.session_state.current_part > 0 else ""
        story_part = generate_story_part(st.session_state.topic, history_summary)
        choices_text = generate_choices_for_story(story_part)

    st.markdown(f"### 이야기 #{st.session_state.current_part + 1}")
    st.write(story_part)
    
    try:
        match_a = re.search(r"A:\s*(.*)", choices_text, re.DOTALL)
        match_b = re.search(r"B:\s*(.*)", choices_text, re.DOTALL)
        if not (match_a and match_b): raise ValueError("선택지 형식 오류")
        choice_a_text = match_a.group(1).strip()
        choice_b_text = match_b.group(1).strip()
        
        st.info("자, 이제 어떤 선택을 해볼까요?")
        col1, col2 = st.columns(2)
        if col1.button(f"A: {choice_a_text}", use_container_width=True, key=f"A_{st.session_state.current_part}"):
            st.session_state.full_log += f"\n\n---\n\n### 이야기 #{st.session_state.current_part + 1}\n{story_part}\n\n**>> 나의 선택:** {choice_a_text}"
            st.session_state.stage = 'debate'
            st.rerun()
        if col2.button(f"B: {choice_b_text}", use_container_width=True, key=f"B_{st.session_state.current_part}"):
            st.session_state.full_log += f"\n\n---\n\n### 이야기 #{st.session_state.current_part + 1}\n{story_part}\n\n**>> 나의 선택:** {choice_b_text}"
            st.session_state.stage = 'debate'
            st.rerun()
    except Exception as e:
        st.error(f"선택지를 만드는 데 실패했어요. AI의 답변 형식이 달랐을 수 있어요.")
        if st.button("이야기 다시 만들기"):
            st.rerun()

elif st.session_state.stage == 'debate':
    log_parts = st.session_state.full_log.split('\n\n')
    for p in log_parts:
        if p.startswith("**>> 나의 선택"): st.chat_message("user").write(p)
        elif p.startswith("**AI 선생님:**"): st.chat_message("assistant").write(p)
        elif p.startswith("**나 (의견"): st.chat_message("user").write(p)
        else: st.markdown(p, unsafe_allow_html=True)
    
    if st.session_state.debate_turns == 0:
        with st.chat_message("assistant"):
            with st.spinner("AI 선생님이 질문을 준비하고 있어요..."):
                choice = st.session_state.full_log.split('>> 나의 선택')[-1]
                question = start_debate(st.session_state.full_log, choice)
                st.session_state.full_log += f"\n\n**AI 선생님:** {question}"; st.session_state.debate_turns = 1; st.rerun()
    elif st.session_state.debate_turns == 1:
        if reply := st.chat_input("첫 번째 의견을 이야기해주세요:"):
            st.session_state.full_log += f"\n\n**나 (의견 1):** {reply}"; st.session_state.debate_turns = 2; st.rerun()
    elif st.session_state.debate_turns == 2:
        with st.chat_message("assistant"):
            with st.spinner("AI 선생님이 다음 질문을 생각 중이에요..."):
                question = continue_debate(st.session_state.full_log)
                st.session_state.full_log += f"\n\n**AI 선생님:** {question}"; st.session_state.debate_turns = 3; st.rerun()
    elif st.session_state.debate_turns == 3:
        if reply := st.chat_input("두 번째 의견을 이야기해주세요:"):
            st.session_state.full_log += f"\n\n**나 (의견 2):** {reply}"; st.session_state.debate_turns = 4; st.rerun()
    elif st.session_state.debate_turns == 4:
        st.info("토론이 완료되었어요. 아래 버튼을 눌러 다음으로 넘어가요!")
        st.session_state.current_part += 1
        if st.button("다음 이야기로" if st.session_state.current_part < st.session_state.MAX_CHOICES else "최종 정리 보기"):
            st.session_state.debate_turns = 0
            if st.session_state.current_part >= st.session_state.MAX_CHOICES:
                st.session_state.stage = 'conclusion'
            else:
                st.session_state.stage = 'story'
            st.rerun()

elif st.session_state.stage == 'conclusion':
    st.markdown(st.session_state.full_log, unsafe_allow_html=True)
    with st.spinner("AI 선생님이 우리의 멋진 여정을 정리하고 있어요..."):
        conclusion = generate_conclusion(st.session_state.full_log)
        st.balloons(); st.success("모든 이야기가 끝났어요! 정말 수고 많았어요!")
        st.markdown("---"); st.markdown("### 최종 정리"); st.write(conclusion)
    if st.button("새로운 주제로 다시 시작하기"):
        restart_lesson(); st.rerun()
