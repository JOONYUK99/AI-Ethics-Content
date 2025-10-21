import streamlit as st
import google.generativeai as genai
import re

# --- 1. AI 핵심 기능 함수 정의 ---
def get_model():
    # 사용 가능한 안정적인 모델로 지정
    return genai.GenerativeModel('gemini-pro-latest')

def generate_full_scenario(topic):
    model = get_model()
    # 문법 오류를 피하기 위해 프롬프트 정의 방식을 수정
    prompt = (
        "당신은 초등학생들에게 이야기를 들려주는 다정하고 친절한 동화 작가입니다.\n"
        f"'{topic}'라는 주제로, 학생들이 몰입할 수 있고 따뜻한 감성이 담긴, 총 4개의 파트로 구성된 완결된 이야기를 만들어주세요.\n"
        "각 파트의 끝에는 주인공의 고민이 잘 드러나는 두 가지 선택지를 포함해주세요.\n\n"
        "# 필수 출력 형식:\n"
        "[STORY 1] (이야기 내용) [CHOICE 1A] (A 선택지) [CHOICE 1B] (B 선택지)\n---\n"
        "[STORY 2] (이야기 내용) [CHOICE 2A] (A 선택지) [CHOICE 2B] (B 선택지)\n---\n"
        "[STORY 3] (이야기 내용) [CHOICE 3A] (A 선택지) [CHOICE 3B] (B 선택지)\n---\n"
        "[STORY 4] (이야기 내용) [CHOICE 4A] (A 선택지) [CHOICE 4B] (B 선택지)"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"시나리오 생성 중 오류: {e}"

def start_debate(history, choice):
    model = get_model()
    prompt = (
        "당신은 학생들을 아주 아끼는 다정한 AI 윤리 선생님입니다. 학생의 선택을 격려하며 토론을 시작해주세요.\n\n"
        "# 역할:\n"
        "1. 학생의 선택을 칭찬해주세요. (예: \"멋진 선택이에요.\", \"그렇게 생각할 수 있군요!\")\n"
        "2. 학생이 왜 그런 선택을 했는지, 그 생각의 깊이를 더 탐구할 수 있는 부드럽고 친절한 '첫 질문'을 던져주세요.\n\n"
        "--- 지금까지의 이야기와 학생의 선택 ---\n"
        f"{history}\n"
        f"학생의 선택: {choice}\n\n"
        "AI 선생님의 따뜻한 첫 질문:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"토론 시작 중 오류: {e}"

def continue_debate(debate_history):
    model = get_model()
    prompt = (
        "당신은 다정한 AI 윤리 선생님입니다. 학생의 의견에 공감하며 토론을 이어가주세요.\n\n"
        "# 역할:\n"
        "1. 학생의 의견을 먼저 긍정적으로 인정해주세요. (예: \"아, 그런 깊은 뜻이 있었군요.\", \"좋은 생각이에요.\")\n"
        "2. 그 다음, \"혹시 이런 점은 어떨까요?\" 와 같이 부드러운 말투로 반대 관점이나 새로운 생각해볼 거리를 질문으로 제시해주세요.\n\n"
        "--- 지금까지의 토론 내용 ---\n"
        f"{debate_history}\n\n"
        "AI 선생님의 다음 질문:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"토론 중 오류: {e}"

def generate_conclusion(final_history):
    model = get_model()
    prompt = (
        "당신은 학생의 성장을 지켜본 다정한 AI 윤리 선생님입니다.\n"
        "다음은 한 학생이 AI 윤리 문제에 대해 총 4번의 선택과 토론을 거친 전체 기록입니다. 이 기록을 바탕으로, 학생의 고민과 성장의 과정을 칭찬하고, 정답 찾기보다 고민하는 과정 자체가 중요했다는 점을 강조하는, 아주 따뜻하고 격려가 되는 마무리 메시지를 작성해주세요.\n\n"
        "--- 전체 기록 ---\n"
        f"{final_history}"
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
        with st.spinner("AI 선생님이 여러분을 위한 특별한 이야기를 만들고 있어요. 잠시만 기다려주세요..."):
            scenario_text = generate_full_scenario(st.session_state.topic)
            if parse_and_store_scenario(scenario_text):
                st.session_state.full_log = f"**주제:** {st.session_state.topic}"
                st.session_state.current_part = 0
                st.session_state.stage = 'story'
                st.rerun()
            else:
                st.error("AI가 이야기를 만들다 조금 힘들어하네요. 잠시 후 다시 시도해주세요.")
                st.code(scenario_text)

elif st.session_state.stage == 'story':
    part = st.session_state.full_scenario[st.session_state.current_part]
    st.session_state.full_log += f"\n\n---\n\n### 이야기 #{st.session_state.current_part + 1}\n{part['story']}"
    st.markdown(st.session_state.full_log, unsafe_allow_html=True)
    st.info("자, 이제 어떤 선택을 해볼까요?")
    col1, col2 = st.columns(2)
    if col1.button(f"A: {part['choice_a']}", use_container_width=True, key=f"A_{st.session_state.current_part}"):
        st.session_state.full_log += f"\n\n**>> 나의 선택 #{st.session_state.current_part + 1}:** {part['choice_a']}"
        st.session_state.stage = 'debate'
        st.rerun()
    if col2.button(f"B: {part['choice_b']}", use_container_width=True, key=f"B_{st.session_state.current_part}"):
        st.session_state.full_log += f"\n\n**>> 나의 선택 #{st.session_state.current_part + 1}:** {part['choice_b']}"
        st.session_state.stage = 'debate'
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
        if st.button("다음 이야기로" if st.session_state.current_part < 4 else "최종 정리 보기"):
            st.session_state.debate_turns = 0
            if st.session_state.current_part >= 4:
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
