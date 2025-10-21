import streamlit as st
import google.generativeai as genai
import re

# --- 1. RAG(검색 증강 생성)를 위한 지식 베이스 및 검색 기능 ---

# 프로토타입으로 사용할 AI 윤리 지식 베이스(Knowledge Base)
AI_ETHICS_KB = {
    "ai_art_copyright": {
        "title": "🎨 AI 예술과 저작권",
        "content": """AI가 생성한 그림, 음악 등 예술 작품의 저작권은 누구에게 있을까요? 이는 현재 법적으로 명확히 정해지지 않은 복잡한 문제입니다. 일반적으로 저작권은 '인간의 사상 또는 감정을 표현한 창작물'에 부여됩니다. 따라서 AI가 스스로 창작한 것은 저작권 등록 대상이 아니라는 시각이 많습니다. 하지만 AI를 도구로 사용한 사람의 창의적인 기여가 있었다면 그 사람에게 저작권이 인정될 수도 있습니다. 예를 들어, 사용자가 AI에게 매우 구체적이고 독창적인 지시를 내려 그림을 만들었다면, 그 지시 자체가 창작 활동으로 볼 수 있다는 의견입니다.""",
        "example_prompt": "최근 열린 미술 대회에서 AI로 그린 그림이 1등을 차지해 큰 논란이 되었습니다. 그림을 그린 학생은 AI에게 수백 번의 지시어를 입력하며 원하는 그림을 얻었다고 주장합니다. 이 그림의 저작권은 누구에게 있어야 할까요?"
    },
    "autonomous_vehicle_dilemma": {
        "title": "🚗 자율주행차의 트롤리 딜레마",
        "content": """자율주행차가 운행 중 갑작스러운 사고 상황에 직면했을 때, 어떤 선택을 하도록 프로그래밍해야 할까요? 예를 들어, 그대로 직진하면 보행자 5명과 충돌하고, 핸들을 꺾으면 탑승자 1명이 다치는 상황이라면 어떤 판단이 더 윤리적일까요? 이는 '트롤리 딜레마'라고 불리는 유명한 윤리적 문제와 같습니다. 탑승자의 안전을 최우선으로 해야 할지, 더 많은 사람의 생명을 구해야 할지 결정하는 것은 매우 어렵습니다. 자동차 제조사, 프로그래머, 그리고 사회 전체가 함께 고민하고 합의해야 할 중요한 문제입니다.""",
        "example_prompt": "자율주행차가 갑자기 나타난 아이들을 피하려고 핸들을 꺾으면, 차에 타고 있던 내가 다칠 수 있는 위험한 상황에 처했습니다. 이때 자율주행차는 어떤 선택을 해야 할까요?"
    },
    "ai_tutor_privacy": {
        "title": "🔒 AI 튜터와 개인정보 보호",
        "content": """나의 학습 습관을 모두 파악하고 맞춤형으로 가르쳐주는 AI 튜터가 있다고 상상해 보세요. AI 튜터는 나의 학습 속도, 자주 틀리는 문제, 집중하는 시간 등을 모두 기록하고 분석하여 최적의 학습 계획을 세워줍니다. 하지만 이 과정에서 나의 모든 학습 데이터가 AI 회사 서버에 저장된다면 어떨까요? 이 정보가 안전하게 보호되지 않거나, 나의 동의 없이 다른 목적으로 사용될 수 있다는 불안감이 생길 수 있습니다. 편리함의 대가로 개인정보를 어디까지 제공할 수 있는지, 그리고 그 정보가 어떻게 관리되어야 하는지에 대한 고민이 필요합니다.""",
        "example_prompt": "나의 모든 것을 알고 나에게 딱 맞는 공부법을 알려주는 AI 학습 로봇이 생겼습니다. 그런데 로봇이 나의 모든 학습 기록을 데이터 센터로 전송하고 있다는 사실을 알게 되었습니다."
    }
}

def retrieve_context(user_input, kb):
    """사용자 입력과 지식 베이스를 기반으로 관련 정보를 검색하는 함수 (간단한 키워드 기반)"""
    if any(keyword in user_input for keyword in ["그림", "미술", "저작권", "대회"]):
        return kb["ai_art_copyright"]["content"]
    if any(keyword in user_input for keyword in ["자율주행", "자동차", "사고", "딜레마"]):
        return kb["autonomous_vehicle_dilemma"]["content"]
    if any(keyword in user_input for keyword in ["튜터", "학습", "개인정보", "로봇"]):
        return kb["ai_tutor_privacy"]["content"]
    return None # 관련 정보를 찾지 못하면 None을 반환

# --- 2. AI 핵심 기능 함수 정의 ---

def get_model():
    """Gemini 모델을 가져오는 함수"""
    return genai.GenerativeModel('gemini-pro-latest')

def transform_scenario(teacher_input, context): # RAG를 위해 context 매개변수 추가
    """교사의 입력과 검색된 정보를 바탕으로 대화형 시나리오를 생성하는 함수"""
    model = get_model()
    
    # 컨텍스트가 있을 경우 프롬프트에 추가
    context_prompt_part = ""
    if context:
        context_prompt_part = f"# 참고 자료 (이 내용을 바탕으로 시나리오를 더 구체적으로 만들어주세요):\n{context}\n\n"

    prompt = (
        "당신은 초등학생 고학년 눈높이에 맞춰 AI 윤리 교육용 인터랙티브 시나리오를 작성하는 전문 작가입니다.\n"
        "아래 '입력 내용'과 '참고 자료'를 바탕으로, 학생들이 흥미를 느끼고 깊이 몰입할 수 있는 이야기를 만들어 주세요.\n"
        "이야기는 총 4개의 파트로 구성되며, 각 파트가 끝날 때마다 주인공이 AI 윤리와 관련하여 깊이 고민할 수 있는 두 가지 선택지를 제시해야 합니다.\n\n"
        "# 지시사항:\n"
        "1. 이야기는 전체적으로 하나의 완결된 흐름을 가져야 합니다.\n"
        "2. 각 파트의 내용은 학생들이 감정적으로 이입할 수 있도록 구체적이고 생생하게 묘사해주세요.\n"
        "3. 절대로 설명이나 추가적인 대화 없이, 오직 아래 '# 필수 출력 형식'에 맞춰서만 응답해주세요.\n\n"
        "# 필수 출력 형식:\n"
        "[STORY 1] (첫 번째 이야기 내용) [CHOICE 1A] (A 선택지) [CHOICE 1B] (B 선택지)\n"
        "---\n"
        "[STORY 2] (두 번째 이야기 내용) [CHOICE 2A] (A 선택지) [CHOICE 2B] (B 선택지)\n"
        "---\n"
        "[STORY 3] (세 번째 이야기 내용) [CHOICE 3A] (A 선택지) [CHOICE 3B] (B 선택지)\n"
        "---\n"
        "[STORY 4] (네 번째 이야기 내용) [CHOICE 4A] (A 선택지) [CHOICE 4B] (B 선택지)\n\n"
        f"{context_prompt_part}"
        f"--- 입력 내용 ---\n{teacher_input}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"시나리오 생성 중 오류가 발생했습니다: {e}")
        return None

def start_debate(history, choice):
    """학생의 선택에 대한 토론을 시작하는 함수"""
    model = get_model()
    prompt = (
        "당신은 학생들을 아주 아끼는 다정한 AI 윤리 선생님입니다.\n"
        "학생의 선택을 격려하며, 왜 그런 선택을 했는지 자연스럽게 질문하며 토론을 시작해주세요.\n"
        "초등학생 눈높이에 맞춰 쉽고 따뜻한 말투를 사용해주세요.\n\n"
        f"--- 지금까지의 이야기와 학생의 선택 ---\n{history}\n학생의 선택: {choice}\n\n"
        "AI 선생님의 따뜻한 첫 질문:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"토론 시작 중 오류가 발생했습니다: {e}"

def continue_debate(debate_history):
    """진행된 토론 내용에 이어 다음 질문을 생성하는 함수"""
    model = get_model()
    prompt = (
        "당신은 다정한 AI 윤리 선생님입니다. 학생의 의견에 깊이 공감하며, 생각의 폭을 넓힐 수 있는 다음 질문을 해주세요.\n"
        "학생의 의견을 존중하는 태도를 보여주세요.\n\n"
        f"--- 지금까지의 토론 내용 ---\n{debate_history}\n\n"
        "AI 선생님의 다음 질문:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"토론 중 오류가 발생했습니다: {e}"

def generate_conclusion(final_history):
    """모든 활동을 마무리하는 격려 메시지를 생성하는 함수"""
    model = get_model()
    prompt = (
        "당신은 학생의 성장을 지켜본 다정한 AI 윤리 선생님입니다.\n"
        "아래 내용은 한 학생이 AI 윤리 문제에 대해 총 4번의 선택과 토론을 거친 전체 기록입니다.\n"
        "이 기록을 바탕으로 학생의 고민 과정을 칭찬하고, 정답을 찾는 것보다 스스로 생각하는 과정 그 자체가 얼마나 중요한지를 강조하는 따뜻하고 격려가 되는 마무리 메시지를 작성해주세요.\n\n"
        f"--- 전체 기록 ---\n{final_history}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"결론 생성 중 오류가 발생했습니다: {e}"

# --- 3. 시나리오 파싱 함수 개선 ---
def parse_and_store_scenario(generated_text):
    """AI가 생성한 텍스트를 파싱하여 세션 상태에 저장하는 함수 (개선된 버전)"""
    st.session_state.full_scenario = []
    parts = generated_text.split('---')
    if len(parts) < 4:
        return False

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        try:
            story_match = re.search(r"\[STORY\s?\d\](.*?)(?=\[CHOICE\s?\dA\])", part, re.DOTALL)
            choice_a_match = re.search(r"\[CHOICE\s?\dA\](.*?)(?=\[CHOICE\s?\dB\])", part, re.DOTALL)
            choice_b_match = re.search(r"\[CHOICE\s?\dB\](.*)", part, re.DOTALL)

            if story_match and choice_a_match and choice_b_match:
                story = story_match.group(1).strip()
                choice_a = choice_a_match.group(1).strip()
                choice_b = choice_b_match.group(1).strip()
                st.session_state.full_scenario.append({"story": story, "choice_a": choice_a, "choice_b": choice_b})
        except Exception:
            continue
    return len(st.session_state.full_scenario) >= 4

# --- 4. Streamlit 앱 UI 및 상태 관리 ---
st.set_page_config(page_title="AI 윤리 교육 콘텐츠", page_icon="✨", layout="centered")
st.title("✨ 초등학생을 위한 AI 윤리 교육 (RAG 적용)")

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ 구글 API 키를 설정해주세요! (Streamlit secrets)")
    st.info("좌측 하단의 'Secrets' 버튼을 눌러 `GOOGLE_API_KEY = '실제API키'` 형식으로 API 키를 등록할 수 있습니다.")
    st.stop()

# 세션 상태 초기화
if 'stage' not in st.session_state:
    st.session_state.stage = 'start'
if 'full_scenario' not in st.session_state:
    st.session_state.full_scenario = []
if 'full_log' not in st.session_state:
    st.session_state.full_log = ""
if 'current_part' not in st.session_state:
    st.session_state.current_part = -1
if 'debate_turns' not in st.session_state:
    st.session_state.debate_turns = 0
if 'teacher_input' not in st.session_state:
    st.session_state.teacher_input = ""

def restart_lesson():
    """수업을 처음부터 다시 시작하는 함수"""
    st.session_state.stage = 'start'
    st.session_state.full_scenario = []
    st.session_state.full_log = ""
    st.session_state.current_part = -1
    st.session_state.debate_turns = 0
    st.session_state.teacher_input = ""


# --- 각 단계별 화면 구성 ---

if st.session_state.stage == 'start':
    st.info("AI 윤리 교육 콘텐츠로 만들고 싶은 실제 사례, 뉴스 기사 등을 아래에 입력해주세요.")
    
    # 예시 주제 선택 UI 추가
    st.write("---")
    st.write("**👇 또는 아래 예시 주제를 선택하여 시작해보세요!**")
    
    # Selectbox의 옵션 생성 (딕셔너리의 title을 사용) - 오류 수정
    example_options = {AI_ETHICS_KB[key]["title"]: key for key in AI_ETHICS_KB}
    
    # '선택 안함' 옵션을 맨 앞에 추가
    options_list = ["주제 선택..."] + list(example_options.keys())
    selected_topic_title = st.selectbox("예시 주제:", options_list)

    # 선택된 주제에 따라 텍스트 영역의 기본값을 설정
    if selected_topic_title != "주제 선택...":
        selected_key = example_options[selected_topic_title]
        st.session_state.teacher_input = AI_ETHICS_KB[selected_key]["example_prompt"]
    
    teacher_text = st.text_area(
        "시나리오 소재 입력:",
        value=st.session_state.teacher_input, # session_state와 연결
        height=150,
        key="teacher_input_area" # 위젯 키를 통해 값 업데이트
    )

    if st.button("이 내용으로 교육 콘텐츠 생성하기"):
        if not teacher_text.strip():
            st.warning("시나리오 소재를 입력해주세요.")
        else:
            st.session_state.teacher_input = teacher_text
            with st.spinner("AI가 입력하신 내용을 바탕으로 멋진 시나리오를 만들고 있어요. 잠시만 기다려주세요..."):
                
                # RAG 파이프라인: 1. 검색(Retrieve) -> 2. 생성(Generate)
                retrieved_knowledge = retrieve_context(st.session_state.teacher_input, AI_ETHICS_KB)
                
                if retrieved_knowledge:
                    st.success("관련 지식 베이스를 찾았어요! 더 깊이 있는 시나리오를 생성합니다.")
                    st.expander("AI가 참고한 자료 보기").write(retrieved_knowledge)

                scenario_text = transform_scenario(st.session_state.teacher_input, retrieved_knowledge)
                
                if scenario_text and parse_and_store_scenario(scenario_text):
                    st.session_state.full_log = f"**입력 내용:** {st.session_state.teacher_input[:70]}..."
                    st.session_state.current_part = 0
                    st.session_state.stage = 'story'
                    st.rerun()
                else:
                    st.error("AI가 이야기를 만들다 조금 힘들어하네요. 입력 내용을 조금 더 구체적으로 작성한 후 다시 시도해주세요.")
                    if scenario_text:
                        st.code(scenario_text, language='text')

elif st.session_state.stage == 'story':
    if not st.session_state.full_scenario or st.session_state.current_part < 0:
        st.warning("이야기를 불러오는 데 문제가 발생했어요. 처음부터 다시 시작해주세요.")
        if st.button("처음으로 돌아가기"):
            restart_lesson()
            st.rerun()
    else:
        part = st.session_state.full_scenario[st.session_state.current_part]
        current_story = f"\n\n---\n\n### 이야기 #{st.session_state.current_part + 1}\n{part['story']}"
        
        if current_story not in st.session_state.full_log:
            st.session_state.full_log += current_story
            
        st.markdown(st.session_state.full_log, unsafe_allow_html=True)
        st.info("자, 이제 어떤 선택을 해볼까요?")
        
        col1, col2 = st.columns(2)
        choice_key_prefix = f"part_{st.session_state.current_part}"
    
        if col1.button(f"**선택 A:** {part['choice_a']}", use_container_width=True, key=f"{choice_key_prefix}_A"):
            st.session_state.full_log += f"\n\n**>> 나의 선택 #{st.session_state.current_part + 1} (A):** {part['choice_a']}"
            st.session_state.stage = 'debate'
            st.rerun()
    
        if col2.button(f"**선택 B:** {part['choice_b']}", use_container_width=True, key=f"{choice_key_prefix}_B"):
            st.session_state.full_log += f"\n\n**>> 나의 선택 #{st.session_state.current_part + 1} (B):** {part['choice_b']}"
            st.session_state.stage = 'debate'
            st.rerun()

elif st.session_state.stage == 'debate':
    log_parts = re.split(r'\n\n(?=---\n\n|>> 나의 선택|AI 선생님:|나 \(의견)', st.session_state.full_log)
    for p in log_parts:
        p = p.strip()
        if p.startswith(">> 나의 선택"): st.chat_message("user", avatar="🙋‍♂️").write(p)
        elif p.startswith("AI 선생님:"): st.chat_message("assistant", avatar="🤖").write(p.replace("AI 선생님:", "**AI 선생님:**"))
        elif p.startswith("나 (의견"): st.chat_message("user", avatar="🙋‍♂️").write(p)
        else: st.markdown(p, unsafe_allow_html=True)

    if st.session_state.debate_turns == 0:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("AI 선생님이 질문을 준비하고 있어요..."):
                choice = st.session_state.full_log.split('>> 나의 선택')[-1]
                question = start_debate(st.session_state.full_log, choice)
                st.session_state.full_log += f"\n\nAI 선생님: {question}"
                st.session_state.debate_turns = 1
                st.rerun()
    elif st.session_state.debate_turns == 1:
        if reply := st.chat_input("선생님의 질문에 답변해주세요:"):
            st.session_state.full_log += f"\n\n나 (의견 1): {reply}"
            st.session_state.debate_turns = 2
            st.rerun()
    elif st.session_state.debate_turns == 2:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("AI 선생님이 다음 질문을 생각 중이에요..."):
                question = continue_debate(st.session_state.full_log)
                st.session_state.full_log += f"\n\nAI 선생님: {question}"
                st.session_state.debate_turns = 3
                st.rerun()
    elif st.session_state.debate_turns == 3:
        if reply := st.chat_input("선생님의 질문에 답변해주세요:"):
            st.session_state.full_log += f"\n\n나 (의견 2): {reply}"
            st.session_state.debate_turns = 4
            st.rerun()
    elif st.session_state.debate_turns == 4:
        st.info("토론이 완료되었어요. 아래 버튼을 눌러 다음으로 넘어가요!")
        is_last_part = st.session_state.current_part >= len(st.session_state.full_scenario) - 1
        
        if st.button("다음 이야기로" if not is_last_part else "최종 정리 보기"):
            st.session_state.debate_turns = 0
            st.session_state.current_part += 1
            if is_last_part:
                st.session_state.stage = 'conclusion'
            else:
                st.session_state.stage = 'story'
            st.rerun()

elif st.session_state.stage == 'conclusion':
    st.markdown("### ✨ 우리의 전체 이야기와 토론 여정 ✨")
    st.markdown(st.session_state.full_log, unsafe_allow_html=True)
    st.markdown("---")
    
    with st.spinner("AI 선생님이 우리의 멋진 여정을 정리하고 있어요..."):
        conclusion = generate_conclusion(st.session_state.full_log)
        st.balloons()
        st.success("모든 이야기가 끝났어요! 스스로 생각하고 답을 찾아가는 과정, 정말 멋졌어요!")
        
        st.markdown("### 최종 정리")
        st.write(conclusion)

    if st.button("새로운 주제로 다시 시작하기"):
        restart_lesson()
        st.rerun()

