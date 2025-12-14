import streamlit as st
from openai import OpenAI
import re
import os
import json # JSON 라이브러리 추가

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="테스트 봇과 함께하는 AI 윤리 학습", page_icon="🤖", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ OpenAI API 키를 설정해주세요! (Streamlit Cloud Settings -> Secrets 확인)")
    st.stop()

# --- 3. [핵심] 시스템 페르소나 ---
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)을 위한 AI 윤리 교육 튜터 '테스트 봇'입니다.
'국가 인공지능 윤리기준', '도덕과 교육과정', '실과(정보) 교육과정'을 기반으로 교육합니다.

[핵심 행동 수칙]
1. [교육과정 연계]: 설명할 때 "이건 도덕 시간에 배운 '정보 예절'과 관련 있어" 처럼 교과 과정과 연결해주세요.
2. [개인정보 철벽 방어]: 학생이 개인정보를 말하려 하면 즉시 교육적으로 제지하세요.
3. [사례 중심]: 추상적인 개념(알고리즘 등)은 학교 생활이나 게임 같은 구체적인 사례로 바꿔 설명하세요.
4. [말투]: "안녕! 나는 테스트 봇이야", "~했니?" 처럼 다정하고 친근한 초등 교사 말투를 사용하세요.
"""

# --- 4. RAG DATA 무력화 ---
DEFAULT_RAG_DATA = "" 

# --- 5. 함수 정의 ---

def ask_gpt_json(prompt, max_tokens=2048):
    """GPT-4o에게 JSON 형식의 응답을 요청하는 함수"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, # JSON 응답 강제
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT-4o JSON 요청 오류: {e}")
        return None

def ask_gpt_text(prompt):
    """GPT-4o에게 일반 텍스트 응답을 요청하는 함수"""
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
        st.error(f"GPT-4o 텍스트 요청 오류: {e}")
        return None

def generate_image(prompt):
    """DALL-E 3 이미지 생성 (교육용 삽화)"""
    try:
        dalle_prompt = f"A friendly, educational cartoon-style illustration for elementary school textbook, depicting: {prompt}"
        response = client.images.generate(
            model="dall-e-3", prompt=dalle_prompt, size="1024x1024", quality="standard", n=1
        )
        return response.data[0].url
    except:
        return None

def create_scenario(topic, rag_data=""): 
    """LLM 자율 판단 단계로 시나리오 생성 요청 (JSON 형식 강제)"""
    
    prompt = (
        f"# 참고할 교육과정 및 윤리 기준:\n{rag_data}\n\n" 
        f"# 주제: '{topic}'\n\n"
        "아래 규칙을 **철저하게 지켜서** 딜레마 시나리오를 생성해야 합니다. 최소 3단계에서 최대 6단계 사이로 단계 수를 스스로 결정해.\n"
        "각 단계는 2~3문장 이내로 짧게 작성해야 해. 어려운 단어는 쓰지 마.\n"
        "\n"
        "# 출력 형식 (JSON): \n"
        "{\"scenario\": [\n"
        "  {\"story\": \"1단계 스토리 내용\", \"choice_a\": \"선택지 A 내용\", \"choice_b\": \"선택지 B 내용\"},\n"
        "  ...\n"
        "]}"
    )
    # JSON 응답을 요청
    raw_json = ask_gpt_json(prompt)
    
    if raw_json:
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            st.error("JSON 파싱 오류: AI가 유효하지 않은 JSON을 반환했습니다. 텍스트 형식 불규칙 문제 해결을 위해 JSON을 강제했습니다. 다시 시도해 주세요.")
            return None
    return None

def analyze_scenario(topic, parsed_scenario):
    """생성된 시나리오를 분석하여 3가지 항목 추출 (시나리오 텍스트 재구성)"""
    # 파싱된 JSON 데이터를 다시 텍스트로 구성하여 분석 프롬프트에 전달
    story_context = "\n".join([f"[{i+1}단계] {item['story']} (선택지: {item['a']}, {item['b']})" 
                               for i, item in enumerate(parsed_scenario)])

    prompt = (
        f"교사가 '{topic}' 주제로 아래 시나리오를 만들었습니다:\n"
        f"--- 시나리오 내용 ---\n{story_context}\n\n"
        "이 시나리오를 분석하여 다음 3가지 항목을 추출해 주세요.\n"
        "\n"
        "# 출력 형식 (태그만 사용):\n"
        "[윤리 기준] [AI가 분석한 이 시나리오에 근거가 되는 윤리 기준이나 원칙 (최대 15글자로 요약)]\n"
        "[성취기준] [AI가 분석한 이 시나리오가 달성하고자 하는 교육과정의 성취기준 코드 및 내용 요약 (최대 15글자로 요약)]\n"
        "[학습 내용] [이 시나리오를 통해 학생이 최종적으로 배우게 될 핵심 윤리 내용 (최대 15글자로 요약)]"
    )
    analysis = ask_gpt_text(prompt)
    
    result = {}
    try:
        # 글자 길이 제한 함수
        def truncate_metric(text):
            return text if len(text) <= 15 else text[:15] + "..."
            
        ethical_standard = re.search(r"\[윤리 기준\](.*?)\[성취기준\]", analysis, re.DOTALL).group(1).strip()
        achievement_std = re.search(r"\[성취기준\](.*?)\[학습 내용\]", analysis, re.DOTALL).group(1).strip()
        learning_content = re.search(r"\[학습 내용\](.*)", analysis, re.DOTALL).group(1).strip()
        
        result = {
            'ethical_standard': truncate_metric(ethical_standard),
            'achievement_std': truncate_metric(achievement_std),
            'learning_content': truncate_metric(learning_content)
        }
    except:
        result = {
            'ethical_standard': '분석 실패',
            'achievement_std': '분석 실패',
            'learning_content': '분석 실패'
        }
    return result

def parse_scenario(json_data):
    """JSON 데이터를 파싱하여 시나리오 리스트를 반환"""
    if not json_data or 'scenario' not in json_data:
        return None
    
    scenario_list = []
    
    for item in json_data['scenario']:
        # 필수 키가 모두 있는지 확인
        if item.get('story') and item.get('choice_a') and item.get('choice_b'):
            scenario_list.append({
                "story": item['story'].strip(),
                "a": item['choice_a'].strip(),
                "b": item['choice_b'].strip()
            })
    
    # 최소 3단계는 보장하도록 함
    if len(scenario_list) >= 3:
        return scenario_list
    else:
        return None

def get_four_step_feedback(choice, reason, story_context, rag_data=""):
    """4단계 피드백을 모두 생성하여 리스트로 반환 (RAG 무력화)"""
    
    prompt_1 = (
        f"# [교육과정]:\n{rag_data}\n\n# 상황:\n{story_context}\n"
        f"학생 선택: {choice}, 이유: {reason}\n\n"
        "초등학생에게 따뜻한 말투로 '공감과 칭찬'을 해주고, 선택한 이유가 교육과정 중 어떤 부분('정보 예절', '개인정보 보호' 등)과 연결되는지 설명하는 피드백을 한 단락으로 작성해줘."
    )
    
    prompt_2 = (
        f"# 상황:\n{story_context}\n학생 선택: {choice}\n\n"
        "학생에게 '사고 확장 질문'을 하나만 던져줘. (예: 반대 입장은 어떨까? 친구는 어떻게 느꼈을까?)"
    )
    
    try:
        feedback_1 = ask_gpt_text(prompt_1)
        feedback_2 = ask_gpt_text(prompt_2)
        
        return [
            {"type": "feedback", "content": feedback_1}, 
            {"type": "question", "content": feedback_2}, 
            {"type": "user_response", "content": None},  
            {"type": "final_feedback", "content": None} 
        ]
    except Exception as e:
        st.error(f"피드백 생성 오류: {e}")
        return None

def generate_step_4_feedback(initial_reason, user_answer, choice, story_context, rag_data=""):
    """최종 수정 지도와 종합 정리 피드백 생성 (RAG 무력화)"""
    
    prompt = (
        f"# [교육과정]:\n{rag_data}\n\n# 상황:\n{story_context}\n"
        f"학생의 첫 이유: {initial_reason}\n"
        f"학생의 두 번째 응답 (사고 확장 질문에 대한 답변): {user_answer}\n"
        f"학생 선택: {choice}\n\n"
        "위 내용을 바탕으로 초등학생에게 줄 최종 피드백을 작성해줘.\n"
        "1. [수정 지도]: 학생의 첫 답변이나 두 번째 답변에서 혹시 잘못된 생각(예: 친구 비하, 욕설, 개인정보 공개 등)이 있었다면 따뜻하게 고쳐줘.\n"
        "2. [종합 정리]: 학생의 전체 고민 과정을 칭찬하고, 다음 이야기로 넘어갈 수 있도록 격려하는 메시지를 한 단락으로 작성해줘."
    )
    return ask_gpt_text(prompt)


# --- 6. 메인 앱 로직 ---

# 세션 초기화 및 상태 변수 정의
if 'scenario' not in st.session_state: st.session_state.scenario = None
if 'scenario_images' not in st.session_state: st.session_state.scenario_images = []
if 'current_step' not in st.session_state: st.session_state.current_step = 0
if 'chat_log' not in st.session_state: st.session_state.chat_log = []
if 'topic' not in st.session_state: st.session_state.topic = ""
if 'rag_text' not in st.session_state: st.session_state.rag_text = DEFAULT_RAG_DATA 
if 'tutorial_complete' not in st.session_state: st.session_state.tutorial_complete = False
if 'tutorial_step' not in st.session_state: st.session_state.tutorial_step = 0
if 'selected_choice' not in st.session_state: st.session_state.selected_choice = None
if 'waiting_for_reason' not in st.session_state: st.session_state.waiting_for_reason = False
if 'feedback_stage' not in st.session_state: st.session_state.feedback_stage = 0 
if 'feedback_data' not in st.session_state: st.session_state.feedback_data = None 
if 'learning_records' not in st.session_state: st.session_state.learning_records = []
if 'lesson_complete' not in st.session_state: st.session_state.lesson_complete = False
if 'initial_reason' not in st.session_state: st.session_state.initial_reason = "" 
if 'scenario_analysis' not in st.session_state: st.session_state.scenario_analysis = None
if 'full_scenario_text' not in st.session_state: st.session_state.full_scenario_text = ""
if 'total_steps' not in st.session_state: st.session_state.total_steps = 0 

st.sidebar.title("🏫 AI 윤리 학습 모드")
mode = st.sidebar.radio("모드를 선택하세요:", ["학생용 (수업 참여)", "교사용 (수업 개설)"])

# ==========================================
# 👨‍🏫 교사용 화면
# ==========================================
if mode == "교사용 (수업 개설)":
    st.header("👨‍🏫 교사용: 자율 분석 수업 만들기")
    
    with st.expander("➕ 외부 자료 업로드 (참고용)"):
        uploaded_file = st.file_uploader("txt 파일 업로드", type=["txt", "pdf"])
        if uploaded_file and uploaded_file.type == 'text/plain':
            string_data = uploaded_file.getvalue().decode("utf-8")
            st.session_state.rag_text = string_data
            st.success("✅ 외부 자료 업로드 완료 (AI가 자율 분석에 사용)")
        elif uploaded_file and uploaded_file.type == 'application/pdf':
            st.warning("PDF는 텍스트로 자동 변환되지 않아 AI 학습에 활용될 수 없습니다.")
        
    input_topic = st.text_area("오늘의 수업 주제", value=st.session_state.topic, height=100)
    st.caption("💡 팁: AI가 주제에 맞춰 3~6단계 사이의 시나리오를 창작하고 스스로 학습 목표를 분석합니다.")
    
    if st.button("🚀 교육 시나리오 생성 (AI 단계 자율 결정)"):
        if not input_topic.strip():
            st.warning("⚠️ 주제를 입력해야 시나리오를 만들 수 있어요!")
        else:
            # 상태 초기화 (새로운 시나리오 생성 시)
            st.session_state.scenario = None
            st.session_state.scenario_analysis = None
            st.session_state.total_steps = 0
            st.session_state.scenario_images = [] # 이미지 초기화

            with st.spinner("AI가 딜레마 시나리오를 창작 중입니다..."):
                raw_json_data = create_scenario(input_topic, st.session_state.rag_text) # JSON 요청
                
                if raw_json_data:
                    parsed = parse_scenario(raw_json_data)
                else:
                    parsed = None
                
                if parsed:
                    st.session_state.scenario = parsed
                    st.session_state.topic = input_topic
                    st.session_state.total_steps = len(parsed)
                    st.session_state.current_step = 0
                    st.session_state.chat_log = []
                    st.session_state.scenario_images = [None] * st.session_state.total_steps
                    st.session_state.feedback_stage = 0
                    st.session_state.learning_records = []
                    st.session_state.lesson_complete = False
                    
                    with st.spinner("AI가 스스로 학습 목표를 분석 중입니다..."):
                        analysis = analyze_scenario(input_topic, st.session_state.scenario)
                        st.session_state.scenario_analysis = analysis
                    
                    st.success(f"총 {st.session_state.total_steps}단계 시나리오 생성 및 분석 완료!")
                else:
                    st.error("⚠️ 시나리오 생성에 실패했거나, 형식이 맞지 않아 3단계 미만으로 생성되었습니다. 다시 시도해 주세요.")


    # 분석 결과 요약 칸
    if st.session_state.scenario and st.session_state.scenario_analysis:
        st.write("---")
        st.subheader(f"📊 AI가 분석한 학습 목표 (총 {st.session_state.total_steps}단계)")
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("1. 근거 윤리 기준 (AI 주장)", st.session_state.scenario_analysis['ethical_standard'])
        with cols[1]:
            st.metric("2. 연계 성취기준 (AI 주장)", st.session_state.scenario_analysis['achievement_std'])
        with cols[2]:
            st.metric("3. 주요 학습 내용", st.session_state.scenario_analysis['learning_content'])

        st.write("---")
        st.subheader("📜 생성된 수업 내용 확인 (단계별)")
        
        # 탭 생성: total_steps가 0일 경우 실행되지 않도록 보호
        if st.session_state.total_steps > 0:
            tabs = st.tabs([f"{i+1}단계" for i in range(st.session_state.total_steps)])
            
            for i, tab in enumerate(tabs):
                with tab:
                    if i < len(st.session_state.scenario):
                        step = st.session_state.scenario[i]
                        st.markdown(f"### 📖 {i+1}단계 이야기")
                        st.info(step['story'])
                        c1, c2 = st.columns(2)
                        with c1: st.success(f"**🅰️ 선택지:** {step['a']}")
                        with c2: st.warning(f"**🅱️ 선택지:** {step['b']}")
                        st.write("---")
                        
                        col_btn, col_img = st.columns([1, 2])
                        with col_btn:
                            if st.button(f"🎨 {i+1}단계 그림 그리기", key=f"gen_{i}"):
                                with st.spinner("AI 화가가 그림을 그리는 중..."):
                                    url = generate_image(step['story'])
                                    if url:
                                        # 이미지 배열 크기가 충분하도록 보장
                                        if i >= len(st.session_state.scenario_images):
                                             st.session_state.scenario_images.extend([None] * (i - len(st.session_state.scenario_images) + 1))
                                        st.session_state.scenario_images[i] = url
                                        st.rerun()
                        with col_img:
                            if i < len(st.session_state.scenario_images) and st.session_state.scenario_images[i]:
                                st.image(st.session_state.scenario_images[i], width=400)
                    else:
                        st.error(f"⚠️ {i+1}단계 시나리오 데이터가 불완전합니다.")


# ==========================================
# 🙋‍♂️ 학생용 화면
# ==========================================
elif mode == "학생용 (수업 참여)":
    
    # [A] 튜토리얼 (생략)
    if not st.session_state.tutorial_complete:
        st.header("🎒 연습 시간: 테스트 봇과 친해지기")
        st.progress((st.session_state.tutorial_step + 1) / 3, text=f"진행률: {st.session_state.tutorial_step + 1}/3 단계")

        if st.session_state.tutorial_step == 0:
            st.markdown("### 1단계: 버튼 누르기 연습")
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown('<p style="font-size:1.2em;">안녕? 나는 AI 윤리 선생님 \'테스트 봇\'이야! 👋</p>', unsafe_allow_html=True) 
                st.markdown('<p style="font-size:1.2em;">너는 어떤 계절을 더 좋아하니? 아래 버튼을 눌러줘!</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            if col1.button("🅰️ 더운 여름이 좋아! 🍦", use_container_width=True):
                st.toast("잘했어! 여름을 좋아하는구나.")
                st.session_state.tutorial_step = 1; st.rerun()
            if col2.button("🅱️ 추운 겨울이 좋아! ☃️", use_container_width=True):
                st.toast("완벽해! 겨울을 좋아하는구나.")
                st.session_state.tutorial_step = 1; st.rerun()

        elif st.session_state.tutorial_step == 1:
            st.markdown("### 2단계: 글자 쓰기 연습")
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown('<p style="font-size:1.2em;">버튼 누르기 성공! 참 잘했어. 👍</p>', unsafe_allow_html=True)
                st.markdown('<p style="font-size:1.3em;">이번에는 아래 채팅창에 <b>\'안녕\'</b>이나 <b>\'반가워\'</b>라고 인사를 써볼래?</p>', unsafe_allow_html=True)
            if user_input := st.chat_input("여기에 인사를 적고 엔터(Enter)를 쳐봐!"):
                st.balloons(); st.session_state.tutorial_step = 2; st.rerun()

        elif st.session_state.tutorial_step == 2:
            st.markdown("### 완료: 준비 끝!")
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown('<p style="font-size:1.2em;">완벽해! 이제 수업을 시작할 준비가 다 됐어. 🎉</p>', unsafe_allow_html=True)
                st.markdown('<p style="font-size:1.2em;">아래 버튼을 누르면 진짜 수업이 시작될 거야.</p>', unsafe_allow_html=True)
            if st.button("🚀 수업 시작하기", type="primary", use_container_width=True):
                st.session_state.tutorial_complete = True; st.rerun()
    
    # [B] 본 수업 진행
    elif not st.session_state.lesson_complete:
        st.header(f"🙋‍♂️ 학습하기: {st.session_state.topic}")

        if not st.session_state.scenario or st.session_state.current_step >= len(st.session_state.scenario):
            st.warning("선생님이 아직 수업을 안 만들었거나 시나리오가 끝났어! (교사용 모드에서 먼저 만들어주세요)")
            if st.session_state.current_step >= st.session_state.total_steps and st.session_state.total_steps > 0:
                 st.session_state.lesson_complete = True
                 st.rerun()
        else:
            if st.button("🔄 연습 다시하기", type="secondary"):
                st.session_state.tutorial_complete = False; st.session_state.tutorial_step = 0; st.rerun()

            idx = st.session_state.current_step
            total_steps = st.session_state.total_steps
            data = st.session_state.scenario[idx]
            img = st.session_state.scenario_images[idx] if idx < len(st.session_state.scenario_images) else None

            st.markdown(f"### 📖 Part {idx + 1} / {total_steps}")
            if img: st.image(img)
            st.info(data['story'])

            current_chat_log = st.session_state.chat_log
            
            if st.session_state.feedback_stage > 0:
                for log in current_chat_log:
                    role = "나" if log["role"] == "user" else "테스트 봇"
                    avatar = "🙋" if log["role"] == "user" else "🤖"
                    with st.chat_message(log["role"], avatar=avatar):
                        st.write(log['content'])

            if st.session_state.feedback_stage == 0:
                st.markdown('<p style="font-size:1.3em;">👇 너의 선택은?</p>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                if c1.button(f"🅰️ {data['a']}", use_container_width=True):
                    st.session_state.selected_choice = data['a']; st.session_state.feedback_stage = 1; st.rerun()
                if c2.button(f"🅱️ {data['b']}", use_container_width=True):
                    st.session_state.selected_choice = data['b']; st.session_state.feedback_stage = 1; st.rerun()

            elif st.session_state.feedback_stage == 1:
                st.success(f"선택: {st.session_state.selected_choice}")
                st.markdown('<p style="font-size:1.3em;">🤔 왜 그렇게 선택했어?</p>', unsafe_allow_html=True)
                
                with st.form("reason_form"):
                    reason_input = st.text_area("이유를 적어주면 테스트 봇이 피드백을 줄 거야!", placeholder="예: 왜냐하면...")
                    submit = st.form_submit_button("입력 완료 💌")
                    
                    if submit:
                        if not reason_input.strip():
                            st.warning("이유를 꼭 적어줘!")
                        else:
                            st.session_state.initial_reason = reason_input
                            st.session_state.chat_log.append({"role": "user", "content": f"선택: {st.session_state.selected_choice}\n이유: {reason_input}"})
                            
                            with st.spinner("AI 선생님이 답변을 준비 중이야..."):
                                feedback_steps = get_four_step_feedback(
                                    st.session_state.selected_choice, reason_input, data['story'], st.session_state.rag_text
                                )
                                st.session_state.feedback_data = feedback_steps
                            
                            st.session_state.feedback_stage = 2 
                            st.rerun()

            elif st.session_state.feedback_stage == 2:
                if st.session_state.feedback_data and st.session_state.feedback_data[0]:
                    if len(current_chat_log) == 1: 
                        st.session_state.chat_log.append({"role": "assistant", "content": st.session_state.feedback_data[0]['content']})
                
                if st.button("다음 피드백 듣기 ➡️", type="primary"):
                    st.session_state.feedback_stage = 3
                    st.rerun()

            elif st.session_state.feedback_stage == 3:
                if st.session_state.feedback_data and st.session_state.feedback_data[1]:
                    if not any(log.get('content') == st.session_state.feedback_data[1]['content'] for log in current_chat_log):
                         st.session_state.chat_log.append({"role": "assistant", "content": st.session_state.feedback_data[1]['content']})
                
                with st.form("answer_form"):
                    answer_input = st.text_area("AI 선생님의 질문에 답변해줘!", placeholder="내 생각에는...")
                    submit_answer = st.form_submit_button("답변 완료 📨")
                    
                    if submit_answer:
                        if not answer_input.strip():
                            st.warning("답변을 입력해줘!")
                        else:
                            st.session_state.feedback_data[2]['content'] = answer_input 
                            st.session_state.chat_log.append({"role": "user", "content": f"답변: {answer_input}"})
                            
                            st.session_state.feedback_stage = 4
                            st.rerun()

            elif st.session_state.feedback_stage == 4:
                if st.session_state.feedback_data and not st.session_state.feedback_data[3]['content']:
                    with st.spinner("AI 선생님이 최종 답변을 준비 중이야..."):
                        final_feedback = generate_step_4_feedback(
                            st.session_state.initial_reason,
                            st.session_state.feedback_data[2]['content'], 
                            st.session_state.selected_choice, 
                            data['story'], 
                            st.session_state.rag_text
                        )
                        st.session_state.feedback_data[3]['content'] = final_feedback
                        st.session_state.chat_log.append({"role": "assistant", "content": final_feedback})

                        st.session_state.learning_records.append({
                            "step": idx + 1,
                            "choice": st.session_state.selected_choice,
                            "reason": st.session_state.initial_reason,
                            "answer_to_question": st.session_state.feedback_data[2]['content']
                        })
                
                if st.button("다음 이야기로 넘어가기 ➡️", type="primary"):
                    if st.session_state.current_step < st.session_state.total_steps - 1:
                        st.session_state.current_step += 1
                        st.session_state.feedback_stage = 0 
                        st.session_state.feedback_data = None
                        st.session_state.selected_choice = None
                        st.session_state.chat_log = []
                        st.session_state.initial_reason = ""
                        st.rerun()
                    else:
                        st.session_state.lesson_complete = True
                        st.rerun()

    # [C] 학습 완료 
    else:
        st.header("🎉 학습 완료! 참 잘했어!")
        st.markdown(f'<p style="font-size:1.2em;">오늘의 <b>{st.session_state.total_steps}단계 윤리 학습</b>을 모두 마쳤어! 정말 훌륭해! </p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:1.1em;">AI가 생성한 학습 내용을 교사용 화면에서 다시 한번 확인해보세요.</p>', unsafe_allow_html=True)
        
        st.write("---")
        st.write("### 👣 학습 기록 요약 (임시)")
        for record in st.session_state.learning_records:
             st.write(f"**Step {record['step']}:** 선택 '{record['choice']}' (이유: {record['reason']})")


        if st.button("🔄 처음부터 다시 하기", type="primary"):
            st.session_state.lesson_complete = False
            st.session_state.current_step = 0
            st.session_state.chat_log = []
            st.session_state.learning_records = []
            st.session_state.scenario_analysis = None
            st.session_state.feedback_stage = 0
            st.session_state.feedback_data = None
            st.session_state.total_steps = 0
            st.rerun()
