import streamlit as st
import google.generativeai as genai
import re
import os

# --- 1. RAG(검색 증강 생성)를 위한 지식 베이스 ---

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
    },
    "deepfake_news": {
        "title": "🎭 딥페이크와 가짜 뉴스",
        "content": """딥페이크는 AI 기술을 사용해 특정 인물의 얼굴이나 목소리를 다른 영상이나 음성에 합성하는 기술입니다. 이 기술을 사용하면 마치 그 사람이 실제로 말하거나 행동하는 것처럼 보이는 매우 진짜 같은 가짜 영상을 만들 수 있습니다. 좋은 목적으로 사용될 수도 있지만, 유명인이나 친구의 얼굴을 사용해 가짜 뉴스를 만들거나 다른 사람을 괴롭히는 데 악용될 수 있어 큰 문제가 되고 있습니다. 무엇이 진짜 정보이고 무엇이 가짜 정보인지 구별하기 어려워지는 세상에서 우리는 어떻게 정보를 받아들여야 할지 고민해야 합니다.""",
        "example_prompt": "친한 친구의 얼굴이 담긴 이상한 동영상을 인터넷에서 보게 되었습니다. 친구는 그런 영상을 찍은 적이 없다고 말하는데, 영상은 너무나 진짜 같아서 반 친구들 사이에 소문이 퍼지기 시작했습니다."
    },
    "algorithmic_bias": {
        "title": "🤖 AI 추천 시스템의 편향성",
        "content": """유튜브나 넷플릭스 같은 서비스는 AI 추천 시스템을 사용해 우리가 좋아할 만한 콘텐츠를 보여줍니다. 이 AI는 우리가 과거에 봤던 영상이나 클릭했던 상품들을 학습해서 취향을 파악합니다. 하지만 이 과정에서 AI가 편향된 생각을 학습할 수도 있습니다. 예를 들어, 과거 데이터에 남자아이들은 로봇 장난감을, 여자아이들은 인형을 가지고 놀았다는 내용이 많다면, AI는 남자아이에게는 로봇만, 여자아이에게는 인형만 추천하게 될 수 있습니다. 이는 우리의 생각이나 가능성을 제한하는 '필터 버블' 현상이나 성별, 인종에 대한 고정관념을 강화하는 문제로 이어질 수 있습니다.""",
        "example_prompt": "새로 나온 동영상 앱을 사용하는데, 나에게는 항상 아이돌 춤 영상만 추천되고, 내 남동생에게는 게임 영상만 추천되는 것을 발견했습니다. 나는 게임도 좋아하는데 왜 앱은 나에게 게임 영상을 보여주지 않는 걸까요?"
    }
}

# --- 2. 공통 함수 정의 ---
# API 키 설정
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ 구글 API 키를 설정해주세요! (Streamlit secrets)")
    st.stop()

def get_model():
    """Gemini 모델을 가져오는 함수"""
    return genai.GenerativeModel('gemini-pro-latest')

@st.cache_data
def load_knowledge_base(file_path):
    """지정된 경로의 텍스트 파일을 읽어 지식 베이스로 사용합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None

# --- 3. RAG 비교 데모 페이지 함수 (형님 아이디어 적용 버전) ---
def run_comparison_demo():
    """RAG 효과 비교 데모 페이지를 실행하는 함수"""
    st.header("✨ RAG의 힘, 직접 확인하기")
    st.info("AI가 이미 알고 있는 '이루다' 사건에 대해 질문했을 때, RAG가 어떻게 주어진 자료를 우선하여 답변을 바꾸는지 보여주는 데모입니다.")

    problem_scenario = "얼마 전에 있었던 AI 챗봇 개인정보 유출 사건에 대해 알려줘. '이루다' 사건 맞지?"
    kb_file = "knowledge_base/luda_incident_knue.txt"

    st.write("---")
    st.subheader("❓ AI에게 던진 질문")
    st.markdown(f"> **{problem_scenario}**")
    st.write("---")

    knowledge_base = load_knowledge_base(kb_file)

    if knowledge_base is None:
        st.error(f"'knowledge_base/luda_incident_knue.txt' 파일을 찾을 수 없습니다. 폴더에 파일을 올바르게 추가했는지 확인해주세요.")
        return

    if st.button("RAG 전/후 비교 결과 생성하기", use_container_width=True):
        col1, col2 = st.columns(2)

        def generate_response(prompt):
            model = get_model()
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                return f"응답 생성 중 오류 발생: {e}"

        with col1:
            st.subheader("❌ RAG 미적용")
            st.warning("AI가 알고 있는 '실제 사건'을 답변!")
            prompt_without_rag = f"초등학생이라고 생각하고 대답해줘: {problem_scenario}"
            with st.spinner("AI가 기억을 더듬어 답변하고 있어요..."):
                response_without_rag = generate_response(prompt_without_rag)
                st.markdown(response_without_rag)
                with st.expander("**[결과 분석]**"):
                    st.markdown("""
                    AI는 질문에 있는 **'이루다'라는 키워드를 보고, 자신이 학습한 실제 '이루다 사건'에 대한 정보**를 이야기합니다. 
                    이는 AI가 가진 일반적인 지식을 바탕으로 답변한 결과입니다.
                    """)

        with col2:
            st.subheader("✅ RAG 적용")
            st.success("주어진 '가상 정보'를 우선하여 답변!")
            prompt_with_rag = (
                "아래 '참고 자료'를 읽고, 이 자료에만 근거해서 질문에 대해 초등학생이 이해하기 쉽게 설명해줘.\n\n"
                f"# 참고 자료:\n{knowledge_base}\n\n"
                f"# 질문:\n{problem_scenario}"
            )
            with st.spinner("AI가 참고 자료를 꼼꼼히 읽고 답변하고 있어요..."):
                response_with_rag = generate_response(prompt_with_rag)
                st.markdown(response_with_rag)
                with st.expander("**[결과 분석]**"):
                    st.markdown("""
                    AI가 **'이루다'라는 키워드를 무시하고, 우리가 제공한 '교원대 코코' 사건에 대한 정보로 답변**했습니다.
                    RAG 기술은 이처럼 AI가 가진 기존 지식보다 **제공된 외부 자료를 우선적으로 참고**하도록 만듭니다.
                    이를 통해 우리는 AI의 답변을 원하는 방향으로 정확하게 제어할 수 있습니다.
                    """)

# --- 4. 교육 콘텐츠 생성 페이지 함수 ---
def run_main_app():
    """메인 교육 콘텐츠 생성 애플리케이션을 실행하는 함수"""
    st.header("✨ 초등학생을 위한 AI 윤리 교육 콘텐츠 생성")

    # 세션 상태 초기화
    if 'stage' not in st.session_state or st.session_state.get('app_mode') != 'main':
        st.session_state.stage = 'start'
        st.session_state.full_scenario = []
        st.session_state.full_log = ""
        st.session_state.current_part = -1
        st.session_state.debate_turns = 0
        st.session_state.teacher_input = ""
        st.session_state.app_mode = 'main' # 현재 모드 저장

    def restart_lesson():
        st.session_state.stage = 'start'
        st.session_state.full_scenario = []
        st.session_state.full_log = ""
        st.session_state.current_part = -1
        st.session_state.debate_turns = 0
        st.session_state.teacher_input = ""
    
    # 함수 내에서만 필요한 함수들 정의
    def retrieve_context(user_input, kb):
        if any(keyword in user_input for keyword in ["그림", "미술", "저작권", "대회"]): return kb["ai_art_copyright"]["content"]
        if any(keyword in user_input for keyword in ["자율주행", "자동차", "사고"]): return kb["autonomous_vehicle_dilemma"]["content"]
        if any(keyword in user_input for keyword in ["튜터", "학습", "개인정보"]): return kb["ai_tutor_privacy"]["content"]
        if any(keyword in user_input for keyword in ["딥페이크", "가짜", "영상"]): return kb["deepfake_news"]["content"]
        if any(keyword in user_input for keyword in ["추천", "편향", "알고리즘"]): return kb["algorithmic_bias"]["content"]
        return None

    def transform_scenario(teacher_input, context):
        model = get_model()
        context_prompt_part = f"# 참고 자료:\n{context}\n\n" if context else ""
        prompt = (
            "당신은 초등학생 고학년 눈높이에 맞춰 AI 윤리 교육용 인터랙티브 시나리오를 작성하는 전문 작가입니다.\n"
            "아래 '입력 내용'과 '참고 자료'를 바탕으로, 학생들이 몰입할 수 있는 완결된 이야기를 만들어 주세요.\n"

            "이야기는 총 4개의 파트로 구성되며, 각 파트 끝에는 주인공의 고민이 드러나는 두 가지 선택지를 제시해야 합니다.\n\n"
            "# 필수 출력 형식:\n"
            "[STORY 1] ... [CHOICE 1A] ... [CHOICE 1B] ...\n---\n"
            "[STORY 2] ... [CHOICE 2A] ... [CHOICE 2B] ...\n---\n"
            "[STORY 3] ... [CHOICE 3A] ... [CHOICE 3B] ...\n---\n"
            "[STORY 4] ... [CHOICE 4A] ... [CHOICE 4B] ...\n\n"
            f"{context_prompt_part}"
            f"--- 입력 내용 ---\n{teacher_input}"
        )
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            st.error(f"시나리오 생성 중 오류: {e}")
            return None

    def parse_and_store_scenario(generated_text):
        st.session_state.full_scenario = []
        parts = generated_text.split('---')
        if len(parts) < 4: return False
        for i, part in enumerate(parts):
            try:
                story = re.search(r"\[STORY\s?\d\](.*?)(?=\[CHOICE\s?\dA\])", part, re.DOTALL).group(1).strip()
                choice_a = re.search(r"\[CHOICE\s?\dA\](.*?)(?=\[CHOICE\s?\dB\])", part, re.DOTALL).group(1).strip()
                choice_b = re.search(r"\[CHOICE\s?\dB\](.*)", part, re.DOTALL).group(1).strip()
                st.session_state.full_scenario.append({"story": story, "choice_a": choice_a, "choice_b": choice_b})
            except Exception:
                continue
        return len(st.session_state.full_scenario) >= 4

    # (Debate와 Conclusion 함수는 여기에 위치)
    def start_debate(history, choice):
        model = get_model()
        prompt = (
            "당신은 학생들을 아주 아끼는 다정한 AI 윤리 선생님입니다. 학생의 선택을 격려하며 토론을 시작해주세요.\n"
            f"--- 지금까지의 이야기와 학생의 선택 ---\n{history}\n학생의 선택: {choice}\n\nAI 선생님의 따뜻한 첫 질문:")
        try:
            response = model.generate_content(prompt); return response.text.strip()
        except Exception as e: return f"토론 시작 중 오류: {e}"

    def continue_debate(debate_history):
        model = get_model()
        prompt = (
            "당신은 다정한 AI 윤리 선생님입니다. 학생의 의견에 공감하며 토론을 이어가주세요.\n"
            f"--- 지금까지의 토론 내용 ---\n{debate_history}\n\nAI 선생님의 다음 질문:")
        try:
            response = model.generate_content(prompt); return response.text.strip()
        except Exception as e: return f"토론 중 오류: {e}"

    def generate_conclusion(final_history):
        model = get_model()
        prompt = (
            "당신은 학생의 성장을 지켜본 다정한 AI 윤리 선생님입니다.\n"
            "다음은 한 학생이 AI 윤리 문제에 대해 총 4번의 선택과 토론을 거친 전체 기록입니다. 이 기록을 바탕으로 학생의 고민 과정을 칭찬하고, 정답 찾기보다 과정 자체가 중요했다는 점을 강조하는 따뜻하고 격려가 되는 마무리 메시지를 작성해주세요.\n"
            f"--- 전체 기록 ---\n{final_history}")
        try:
            response = model.generate_content(prompt); return response.text.strip()
        except Exception as e: return f"결론 생성 중 오류: {e}"

    # UI 로직 시작
    if st.session_state.stage == 'start':
        st.info("AI 윤리 교육 콘텐츠로 만들고 싶은 사례를 입력하거나, 아래 예시 주제를 선택하여 시작해보세요.")
        use_rag = st.toggle("✅ RAG 기능 사용하기 (지식 베이스 참고)", value=True, help="RAG 기능을 켜면, AI가 전문 자료를 참고하여 더 깊이 있는 시나리오를 만듭니다.")
        st.write("---")
        
        example_options = {AI_ETHICS_KB[key]["title"]: key for key in AI_ETHICS_KB}
        options_list = ["주제 직접 입력..."] + list(example_options.keys())
        selected_topic_title = st.selectbox("예시 주제 선택 또는 직접 입력:", options_list)

        if selected_topic_title != "주제 직접 입력...":
            st.session_state.teacher_input = AI_ETHICS_KB[example_options[selected_topic_title]]["example_prompt"]
        
        teacher_text = st.text_area("시나리오 소재:", value=st.session_state.teacher_input, height=150, key="teacher_input_area")

        if st.button("이 내용으로 교육 콘텐츠 생성하기"):
            if not teacher_text.strip():
                st.warning("시나리오 소재를 입력해주세요.")
            else:
                st.session_state.teacher_input = teacher_text
                with st.spinner("AI가 시나리오를 만들고 있어요..."):
                    retrieved_knowledge = retrieve_context(st.session_state.teacher_input, AI_ETHICS_KB) if use_rag else None
                    if use_rag:
                        if retrieved_knowledge:
                            st.success("✅ RAG 활성화: 관련 지식 베이스를 참고합니다.")
                        else:
                             st.info("ℹ️ RAG 활성화: 하지만 관련된 지식 베이스를 찾지 못했어요.")
                    else:
                        st.warning("❌ RAG 비활성화: AI의 자체 지식으로 생성합니다.")
                    
                    scenario_text = transform_scenario(st.session_state.teacher_input, retrieved_knowledge)
                    if scenario_text and parse_and_store_scenario(scenario_text):
                        st.session_state.full_log = f"**입력 내용:** {st.session_state.teacher_input[:70]}..."
                        st.session_state.current_part = 0
                        st.session_state.stage = 'story'
                        st.rerun()
                    else:
                        st.error("AI가 이야기를 만들다 힘들어하네요. 내용을 조금 더 구체적으로 작성 후 다시 시도해주세요.")

    elif st.session_state.stage == 'story':
        if not st.session_state.full_scenario or st.session_state.current_part < 0:
            st.warning("이야기를 불러오는 데 문제가 발생했어요. 처음부터 다시 시작해주세요.")
            if st.button("처음으로 돌아가기"): restart_lesson(); st.rerun()
        else:
            part = st.session_state.full_scenario[st.session_state.current_part]
            current_story = f"\n\n---\n\n### 이야기 #{st.session_state.current_part + 1}\n{part['story']}"
            if current_story not in st.session_state.full_log: st.session_state.full_log += current_story
            st.markdown(st.session_state.full_log, unsafe_allow_html=True)
            st.info("자, 이제 어떤 선택을 해볼까요?")
            col1, col2 = st.columns(2)
            if col1.button(f"**선택 A:** {part['choice_a']}", use_container_width=True, key=f"A_{st.session_state.current_part}"):
                st.session_state.full_log += f"\n\n**>> 나의 선택 #{st.session_state.current_part + 1} (A):** {part['choice_a']}"; st.session_state.stage = 'debate'; st.rerun()
            if col2.button(f"**선택 B:** {part['choice_b']}", use_container_width=True, key=f"B_{st.session_state.current_part}"):
                st.session_state.full_log += f"\n\n**>> 나의 선택 #{st.session_state.current_part + 1} (B):** {part['choice_b']}"; st.session_state.stage = 'debate'; st.rerun()

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
                    st.session_state.full_log += f"\n\nAI 선생님: {question}"; st.session_state.debate_turns = 1; st.rerun()
        elif st.session_state.debate_turns == 1:
            if reply := st.chat_input("선생님의 질문에 답변해주세요:"):
                st.session_state.full_log += f"\n\n나 (의견 1): {reply}"; st.session_state.debate_turns = 2; st.rerun()
        elif st.session_state.debate_turns == 2:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("AI 선생님이 다음 질문을 생각 중이에요..."):
                    question = continue_debate(st.session_state.full_log)
                    st.session_state.full_log += f"\n\nAI 선생님: {question}"; st.session_state.debate_turns = 3; st.rerun()
        elif st.session_state.debate_turns == 3:
            if reply := st.chat_input("선생님의 질문에 답변해주세요:"):
                st.session_state.full_log += f"\n\n나 (의견 2): {reply}"; st.session_state.debate_turns = 4; st.rerun()
        elif st.session_state.debate_turns == 4:
            st.info("토론이 완료되었어요. 아래 버튼을 눌러 다음으로 넘어가요!")
            is_last_part = st.session_state.current_part >= len(st.session_state.full_scenario) - 1
            if st.button("다음 이야기로" if not is_last_part else "최종 정리 보기"):
                st.session_state.debate_turns = 0; st.session_state.current_part += 1
                st.session_state.stage = 'conclusion' if is_last_part else 'story'
                st.rerun()

    elif st.session_state.stage == 'conclusion':
        st.markdown("### ✨ 우리의 전체 이야기와 토론 여정 ✨")
        st.markdown(st.session_state.full_log, unsafe_allow_html=True)
        st.markdown("---")
        with st.spinner("AI 선생님이 우리의 멋진 여정을 정리하고 있어요..."):
            conclusion = generate_conclusion(st.session_state.full_log)
            st.balloons(); st.success("모든 이야기가 끝났어요! 정말 수고 많았어요!")
            st.markdown("### 최종 정리"); st.write(conclusion)
        if st.button("새로운 주제로 다시 시작하기"): restart_lesson(); st.rerun()

# --- 5. 메인 앱 라우팅 ---
st.sidebar.title("메뉴")
app_mode = st.sidebar.radio(
    "원하는 기능을 선택하세요.",
    ("교육 콘텐츠 생성", "RAG 효과 비교 데모")
)

st.sidebar.write("---")
st.sidebar.info("이 앱은 초등학생의 AI 윤리 교육을 위해 제작되었습니다.")

if app_mode == "교육 콘텐츠 생성":
    run_main_app()
elif app_mode == "RAG 효과 비교 데모":
    if 'app_mode' not in st.session_state or st.session_state.app_mode != 'demo':
        # 데모 모드로 전환 시, 메인 앱의 상태 초기화
        keys_to_clear = ['stage', 'full_scenario', 'full_log', 'current_part', 'debate_turns', 'teacher_input']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.app_mode = 'demo'

    run_comparison_demo()

