import streamlit as st
import google.generativeai as genai
import re
import urllib.parse

# --- 1. AI 핵심 기능 함수 정의 ---
def get_model():
    # 사용 가능한 안정적인 모델로 지정
    return genai.GenerativeModel('gemini-pro-latest')

def generate_full_scenario(teacher_input):
    model = get_model()
    prompt = (
        "당신은 초등학생 고학년을 위한 AI 윤리 교육용 인터랙티브 시나리오 작가입니다.\n"
        f"아래의 '입력 내용'을 바탕으로, 학생들이 몰입할 수 있고 총 4번의 선택을 하게 되는 완결된 이야기를 만들어주세요.\n"
        "각 파트의 끝에는 주인공의 고민이 잘 드러나는 두 가지 선택지를 포함해주세요.\n\n"
        "# 필수 출력 형식:\n"
        "[STORY 1] (이야기 내용) [CHOICE 1A] (A 선택지) [CHOICE 1B] (B 선택지)\n---\n"
        "[STORY 2] (이야기 내용) [CHOICE 2A] (A 선택지) [CHOICE 2B] (B 선택지)\n---\n"
        "[STORY 3] (이야기 내용) [CHOICE 3A] (A 선택지) [CHOICE 3B] (B 선택지)\n---\n"
        "[STORY 4] (이야기 내용) [CHOICE 4A] (A 선택지) [CHOICE 4B] (B 선택지)\n\n"
        f"--- 입력 내용 ---\n{teacher_input}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"시나리오 생성 중 오류: {e}"

# <--- 이미지 키워드 추출 기능 추가 --->
def generate_image_keywords(story_part):
    model = get_model()
    prompt = f"다음 한국어 문장의 핵심 내용을 대표하는 영어 단어 2개를 쉼표로 구분하여 짧게 요약해줘. 예: '슬픈 아이가 로봇과 함께 있다' -> 'sad child, robot'\n\n문장: {story_part}"
    try:
        response = model.generate_content(prompt)
        # AI 답변에서 특수문자 제거 후 키워드 정리
        keywords = [keyword.strip() for keyword in response.text.strip().replace('*','').split(',')]
        return ",".join(keywords)
    except Exception:
        # 실패 시 기본 키워드 반환
        return "AI,thinking"

def start_debate(history, choice):
    model = get_model()
    prompt = (
        "당신은 학생들을 아주 아끼는 다정한 AI 윤리 선생님입니다. 학생의 선택을 격려하며 토론을 시작해주세요.\n\n"
        "1. 학생의 선택을 칭찬해주세요. (예: \"멋진 선택이에요.\")\n"
        "2. 학생이 왜 그런 선택을 했는지, 그 생각의 깊이를 더 탐구할 수 있는 부드럽고 친절한 '첫 질문'을 던져주세요.\n\n"
        f"--- 지금까지의 이야기와 학생의 선택 ---\n{history}\n학생의 선택: {choice}\n\nAI 선생님의 따뜻한 첫 질문:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"토론 시작 중 오류: {e}"

def continue_debate(debate_history):
    model = get_model()
    prompt = (
        "당신은 다정한 AI 윤리 선생님입니다. 학생의 의견에 공감하며 토론을 이어가주세요.\n\n"
        "1. 학생의 의견을 먼저 긍정적으로 인정해주세요. (예: \"아, 그런 깊은 뜻이 있었군요.\")\n"
        "2. 그 다음, \"혹시 이런 점은 어떨까요?\" 와 같이 부드러운 말투로 반대 관점을 제시하는 질문을 던져주세요.\n\n"
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
        "다음은 한 학생이 AI 윤리 문제에 대해 총 4번의 선택과 토론을 거친 전체 기록입니다. 이 기록을 바탕으로, 학생의 고민 과정을 칭찬하고, 정답 찾기보다 고민하는 과정 자체가 중요했다는 점을 강조하는, 아주 따뜻하고 격려가 되는 마무리 메시지를 작성해주세요.\n\n"
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
    st.info("안녕하세요! AI 윤리 교육 콘텐츠로 만들고 싶은 실제 사례, 뉴스 기사 등을 아래에 입력해주세요.")
    teacher_text = st.text_area("시나리오 입력:", height=150, placeholder="예시: AI 그림 대회에서 인공지능으로 그린 그림이 1등을 차지해서 논란이 되었습니다...")

    if st.button("이 내용으로 교육 콘텐츠 생성하기"):
        if not teacher_text:
            st.warning("시나리오를 입력해주세요.")
        else:
            st.session_state.teacher_input = teacher_text
            with st.spinner("AI가 입력하신 내용을 바탕으로 멋진 시나리오를 만들고 있어요. 잠시만 기다려주세요..."):
                scenario_text = generate_full_scenario(st.session_state.teacher_input)
                if parse_and_store_scenario(scenario_text):
                    st.session_state.full_log = f"**입력 내용:** {st.session_state.teacher_input[:70]}..."
                    st.session_state.current_part = 0
                    st.session_state.stage = 'story'
                    st.rerun()
                else:
                    st.error("AI가 이야기를 만들다 조금 힘들어하네요. 입력 내용을 조금 더 구체적으로 작성한 후 다시 시도해주세요.")
                    st.code(scenario_text)

elif st.session_state.stage == 'story':
    part = st.session_state.full_scenario[st.session_state.current_part]
    
    st.markdown(f"### 이야기 #{st.session_state.current_part + 1}")
    
    # <--- 수정: 이미지 생성 및 표시 로직 추가
    with st.spinner("AI가 장면에 맞는 이미지를 생성 중입니다..."):
        keywords = generate_image_keywords(part['story'])
        encoded_keywords = urllib.parse.quote(keywords)
        st.image(f"https://placehold.co/600x300/E8E8E8/313131?text={encoded_keywords}", caption=f"AI가 생각한 이미지 키워드: {keywords}")

    st.write(part['story']) # 이미지 아래에 이야기 표시
    
    st.session_state.full_log += f"\n\n---\n\n### 이야기 #{st.session_state.current_part + 1}\n{part['story']}"
    
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
    # ... (이하 토론 및 결론 로직은 이전과 동일)
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
