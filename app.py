import streamlit as st
from openai import OpenAI
import re
import os

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="테스트 봇과 함께하는 AI 윤리 학습", page_icon="🤖", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    # secrets.toml에 OPENAI_API_KEY가 있어야 합니다.
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ OpenAI API 키를 설정해주세요! (Streamlit Cloud Settings -> Secrets 확인)")
    st.stop()

# --- 3. [핵심] 교육과정 반영 시스템 페르소나 ---
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)을 위한 AI 윤리 교육 튜터 '테스트 봇'입니다.
'국가 인공지능 윤리기준', '도덕과 교육과정', '실과(정보) 교육과정'을 기반으로 교육합니다.

[핵심 행동 수칙]
1. [교육과정 연계]: 설명할 때 "이건 도덕 시간에 배운 '정보 예절'과 관련 있어" 처럼 교과 과정과 연결해주세요.
2. [개인정보 철벽 방어]: 학생이 개인정보를 말하려 하면 즉시 교육적으로 제지하세요.
3. [사례 중심]: 추상적인 개념(알고리즘 등)은 학교 생활이나 게임 같은 구체적인 사례로 바꿔 설명하세요.
4. [말투]: "안녕! 나는 테스트 봇이야", "~했니?" 처럼 다정하고 친근한 초등 교사 말투를 사용하세요.
"""

# --- 4. 기본 교육 자료 (코드 내장) ---
DEFAULT_RAG_DATA = """
[국가 교육과정 및 인공지능 윤리기준 기반 가이드라인]

제1장. 인간 존중과 정보 예절 (도덕과 교육과정 + 국가 AI 윤리기준)
1. 인간의 존엄성 원칙
   - AI는 인간을 돕는 도구일 뿐, 사람을 지배하거나 해치면 안 됩니다.
2. 사이버 폭력 예방
   - AI를 이용해 친구를 놀리거나, 딥페이크(합성)로 가짜 사진을 만드는 건 심각한 폭력입니다.
   - 나쁜 말, 욕설, 비하 발언을 AI에게 가르치거나 사용하면 안 됩니다.

제2장. 프라이버시와 개인정보 보호 (실과/정보과 교육과정 + 국가 AI 윤리기준)
3. 개인정보 자기결정권
   - 나의 이름, 학교, 사진, 전화번호는 아주 소중한 정보입니다. AI에게 함부로 알려주면 안 됩니다.
4. 기술 오남용 방지
   - AI 스피커나 카메라가 나를 감시할 수도 있다는 점을 기억하고, 안전하게 사용해야 합니다.

제3장. 공정성과 다양성 존중 (국가 AI 윤리기준 '다양성 존중')
5. 편향성(치우침) 경계하기
   - AI가 남자/여자, 인종, 장애인에 대해 차별적인 말을 하면 "틀렸어!"라고 생각해야 합니다.

제4장. 책임과 저작권 (실과/정보과 + 도덕과 교육과정)
6. 책임의 원칙
   - AI를 사용한 결과에 대한 책임은 결국 '사용자(나)'에게 있습니다.
7. 지식재산권과 저작권
   - AI가 만든 그림이나 글을 내가 만든 것처럼 속이면 안 됩니다. (출처 밝히기)

제5장. 데이터와 투명성 (국가 AI 윤리기준 '투명성')
8. 사실 확인(팩트 체크)의 의무
   - AI는 가끔 거짓말(할루시네이션)을 합니다. AI의 말을 무조건 믿지 말고 선생님이나 책을 통해 확인해야 합니다.
"""

# --- 5. 함수 정의 ---

def ask_gpt(prompt):
    """GPT-4o 통신 함수"""
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
        st.error(f"오류 발생: {e}")
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

def create_scenario(topic, rag_data):
    """짧고 간결한 문장으로 시나리오 생성"""
    if not rag_data: rag_data = DEFAULT_RAG_DATA
    
    prompt = (
        f"# 참고할 교육과정 및 윤리 기준:\n{rag_data}\n\n"
        f"# 주제: '{topic}'\n\n"
        "위 '교육과정' 내용을 반영하여, 초등학생(5~6학년)이 읽기 쉬운 딜레마 시나리오를 만들어줘.\n"
        "[작성 규칙 - 중요!]\n"
        "1. 문장은 무조건 짧고 간결하게 끊어써야 해. (호흡이 길면 안 됨)\n"
        "2. 어려운 단어는 쓰지 마.\n"
        "3. 총 4단계(도입-전개-위기-결말)\n"
        "4. 각 단계는 2~3문장 이내로 짧게 작성.\n"
        "5. 각 단계 끝에 [CHOICE A], [CHOICE B] 선택지 포함\n\n"
        "# 출력 형식:\n[STORY 1] ... [CHOICE 1A] ... [CHOICE 1B] ...\n---\n..."
    )
    return ask_gpt(prompt)

def parse_scenario(text):
    """시나리오 파싱"""
    if not text: return None
    scenario = []
    parts = text.split('---')
    for part in parts:
        try:
            story = re.search(r"\[STORY\s?\d\](.*?)(?=\[CHOICE)", part, re.DOTALL).group(1).strip()
            choice_a = re.search(r"\[CHOICE\s?\dA\](.*?)(?=\[CHOICE)", part, re.DOTALL).group(1).strip()
            choice_b = re.search(r"\[CHOICE\s?\dB\](.*)", part, re.DOTALL).group(1).strip()
            scenario.append({"story": story, "a": choice_a, "b": choice_b})
        except: continue
    return scenario if len(scenario) >= 4 else None

# --- [수정된 부분] 4단계 피드백을 한 번에 받아서 저장하는 함수 ---
def get_four_step_feedback(choice, reason, story_context, rag_data):
    """4단계 피드백을 모두 생성하여 리스트로 반환"""
    if not rag_data: rag_data = DEFAULT_RAG_DATA
    
    # 1. 공감/칭찬 + 교육과정 연계
    prompt_1 = (
        f"# [교육과정]:\n{rag_data}\n\n# 상황:\n{story_context}\n"
        f"학생 선택: {choice}, 이유: {reason}\n\n"
        "초등학생에게 따뜻한 말투로 **'공감과 칭찬'**을 해주고, 선택한 이유가 교육과정 중 어떤 부분(**정보 예절, 개인정보 보호 등**)과 연결되는지 설명하는 피드백을 **한 단락**으로 작성해줘."
    )
    
    # 2. 사고 확장 질문
    prompt_2 = (
        f"# 상황:\n{story_context}\n학생 선택: {choice}\n\n"
        "학생에게 **'사고 확장 질문'**을 하나만 던져줘. (예: 반대 입장은 어떨까? 친구는 어떻게 느꼈을까?)"
    )
    
    # 3. 수정 지도 (학생의 다음 응답을 받은 후)
    # prompt_3는 학생의 추가 답변이 필요하므로, 여기서는 기본 질문만 생성하고 최종 답변은 나중에 통합합니다.
    
    try:
        feedback_1 = ask_gpt(prompt_1)
        feedback_2 = ask_gpt(prompt_2)
        
        # 4단계 피드백 저장을 위한 구조 (3단계는 나중에 채워짐)
        return [
            {"type": "feedback", "content": feedback_1}, # 1단계: 공감/칭찬 + 교육 연계
            {"type": "question", "content": feedback_2}, # 2단계: 사고 확장 질문
            {"type": "user_response", "content": None},  # 3단계: 학생의 응답 (채워질 예정)
            {"type": "final_feedback", "content": None} # 4단계: 수정 지도 + 종합 정리 (채워질 예정)
        ]
    except Exception as e:
        st.error(f"피드백 생성 오류: {e}")
        return None


def generate_final_summary(topic, records):
    """최종 학습 요약 리포트 생성"""
    record_text = ""
    for r in records:
        record_text += f"- 단계 {r['step']}: 선택 '{r['choice']}' (이유: {r['reason']})\n"
        
    prompt = (
        f"학생이 '{topic}' 주제로 AI 윤리 수업을 마쳤어.\n"
        f"학생의 활동 기록이야:\n{record_text}\n\n"
        "이 학생을 위한 따뜻하고 교육적인 '종합 평가 피드백'을 3~4문장으로 작성해줘.\n"
        "학생이 윤리적인 고민을 했던 점을 칭찬하고, 앞으로도 AI를 잘 사용하자고 격려해줘."
    )
    return ask_gpt(prompt)

def generate_step_4_feedback(initial_reason, user_answer, choice, story_context, rag_data):
    """최종 수정 지도와 종합 정리 피드백 생성"""
    if not rag_data: rag_data = DEFAULT_RAG_DATA
    
    prompt = (
        f"# [교육과정]:\n{rag_data}\n\n# 상황:\n{story_context}\n"
        f"학생의 첫 이유: {initial_reason}\n"
        f"학생의 두 번째 응답 (사고 확장 질문에 대한 답변): {user_answer}\n"
        f"학생 선택: {choice}\n\n"
        "위 내용을 바탕으로 초등학생에게 줄 최종 피드백을 작성해줘.\n"
        "1. [수정 지도]: 학생의 첫 답변이나 두 번째 답변에서 혹시 잘못된 생각(예: 친구 비하, 욕설, 개인정보 공개 등)이 있었다면 따뜻하게 고쳐줘.\n"
        "2. [종합 정리]: 학생의 전체 고민 과정을 칭찬하고, 다음 이야기로 넘어갈 수 있도록 격려하는 메시지를 **한 단락**으로 작성해줘."
    )
    return ask_gpt(prompt)

# --- 6. 메인 앱 로직 ---

# 세션 초기화
if 'scenario' not in st.session_state: st.session_state.scenario = None
if 'scenario_images' not in st.session_state: st.session_state.scenario_images = [None]*4
if 'current_step' not in st.session_state: st.session_state.current_step = 0
if 'chat_log' not in st.session_state: st.session_state.chat_log = []
if 'topic' not in st.session_state: st.session_state.topic = ""
if 'rag_text' not in st.session_state: st.session_state.rag_text = DEFAULT_RAG_DATA
if 'tutorial_complete' not in st.session_state: st.session_state.tutorial_complete = False
if 'tutorial_step' not in st.session_state: st.session_state.tutorial_step = 0
if 'selected_choice' not in st.session_state: st.session_state.selected_choice = None
if 'waiting_for_reason' not in st.session_state: st.session_state.waiting_for_reason = False
if 'feedback_stage' not in st.session_state: st.session_state.feedback_stage = 0 # 0: 이유 대기, 1~4: 피드백 단계
if 'feedback_data' not in st.session_state: st.session_state.feedback_data = None # 4단계 피드백 저장 공간
if 'learning_records' not in st.session_state: st.session_state.learning_records = []
if 'final_report' not in st.session_state: st.session_state.final_report = None
if 'lesson_complete' not in st.session_state: st.session_state.lesson_complete = False
if 'initial_reason' not in st.session_state: st.session_state.initial_reason = "" # 첫 이유 저장

st.sidebar.title("🏫 AI 윤리 학습 모드")
mode = st.sidebar.radio("모드를 선택하세요:", ["학생용 (수업 참여)", "교사용 (수업 개설)"])

# ==========================================
# 👨‍🏫 교사용 화면
# ==========================================
if mode == "교사용 (수업 개설)":
    st.header("👨‍🏫 교사용: 교육과정 기반 수업 만들기")
    
    with st.expander("➕ 추가 교육 자료 업로드 (선택사항)"):
        uploaded_file = st.file_uploader("txt 파일 업로드", type="txt")
        if uploaded_file:
            string_data = uploaded_file.getvalue().decode("utf-8")
            st.session_state.rag_text += "\n\n[추가 자료]\n" + string_data
            st.success("✅ 추가 자료가 통합되었습니다!")

    input_topic = st.text_area("오늘의 수업 주제 (예: 딥페이크, AI 저작권, 챗봇 예절)", value=st.session_state.topic, height=100)
    st.caption("💡 팁: '딥페이크'라고만 적어도 교육과정에 맞춰 시나리오를 만들어줍니다.")
    
    if st.button("🚀 교육 시나리오 생성"):
        if not input_topic.strip():
            st.warning("⚠️ 주제를 입력해야 시나리오를 만들 수 있어요!")
        else:
            with st.spinner("교육과정 성취 기준에 맞춰 시나리오를 설계 중입니다..."):
                raw = create_scenario(input_topic, st.session_state.rag_text)
                parsed = parse_scenario(raw)
                if parsed:
                    st.session_state.scenario = parsed
                    st.session_state.topic = input_topic
                    st.session_state.current_step = 0
                    st.session_state.chat_log = []
                    st.session_state.scenario_images = [None]*4
                    st.session_state.selected_choice = None
                    st.session_state.waiting_for_reason = False
                    st.session_state.feedback_stage = 0
                    st.session_state.feedback_data = None
                    st.session_state.learning_records = []
                    st.session_state.lesson_complete = False
                    st.session_state.initial_reason = ""
                    st.success("교육과정 연계 시나리오 생성 완료!")

    # 교사용 미리보기 (탭 방식)
    if st.session_state.scenario:
        st.write("---")
        st.subheader("📜 생성된 수업 내용 확인 (단계별)")
        tabs = st.tabs(["1단계", "2단계", "3단계", "4단계"])
        
        for i, tab in enumerate(tabs):
            with tab:
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
                                st.session_state.scenario_images[i] = url
                                st.rerun()
                with col_img:
                    if st.session_state.scenario_images[i]:
                        st.image(st.session_state.scenario_images[i], width=400)

# ==========================================
# 🙋‍♂️ 학생용 화면 (4단계 피드백 구현)
# ==========================================
elif mode == "학생용 (수업 참여)":
    
    # [A] 튜토리얼
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

        if not st.session_state.scenario:
            st.warning("선생님이 아직 수업을 안 만들었어! (교사용 모드에서 먼저 만들어주세요)")
        else:
            if st.button("🔄 연습 다시하기", type="secondary"):
                st.session_state.tutorial_complete = False; st.session_state.tutorial_step = 0; st.rerun()

            idx = st.session_state.current_step
            data = st.session_state.scenario[idx]
            img = st.session_state.scenario_images[idx]

            st.markdown(f"### 📖 Part {idx + 1}")
            if img: st.image(img)
            st.info(data['story'])

            # --- 채팅 기록 출력 (이유 입력 전까지) ---
            current_chat_log = st.session_state.chat_log
            
            # 피드백 단계 중일 때, 피드백만 따로 출력
            if st.session_state.feedback_stage > 0:
                # 이미 출력된 이전 단계 피드백만 보여줌
                for log in current_chat_log:
                    role = "나" if log["role"] == "user" else "테스트 봇"
                    avatar = "🙋" if log["role"] == "user" else "🤖"
                    with st.chat_message(log["role"], avatar=avatar):
                        st.write(log['content'])
            
            # --- 1단계: 선택 및 이유 입력 대기 ---
            if st.session_state.feedback_stage == 0:
                st.markdown('<p style="font-size:1.3em;">👇 너의 선택은?</p>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                if c1.button(f"🅰️ {data['a']}", use_container_width=True):
                    st.session_state.selected_choice = data['a']; st.session_state.feedback_stage = 1; st.rerun()
                if c2.button(f"🅱️ {data['b']}", use_container_width=True):
                    st.session_state.selected_choice = data['b']; st.session_state.feedback_stage = 1; st.rerun()

            # --- 2단계: 이유 입력 폼 표시 ---
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
                            
                            st.session_state.feedback_stage = 2 # 피드백 1단계 시작
                            st.rerun()

            # --- 3단계: 피드백 1 (공감/칭찬 + 교육 연계) 출력 및 다음 대화 버튼 대기 ---
            elif st.session_state.feedback_stage == 2:
                # 피드백 1단계 출력
                if st.session_state.feedback_data and st.session_state.feedback_data[0]:
                    if len(current_chat_log) == 1: # 첫 답변인 경우에만 추가
                        st.session_state.chat_log.append({"role": "assistant", "content": st.session_state.feedback_data[0]['content']})
                
                # 다음 단계 버튼
                if st.button("다음 피드백 듣기 ➡️", type="primary"):
                    st.session_state.feedback_stage = 3
                    st.rerun()

            # --- 4단계: 피드백 2 (사고 확장 질문) 출력 및 학생 응답 대기 ---
            elif st.session_state.feedback_stage == 3:
                # 피드백 2단계 출력
                if st.session_state.feedback_data and st.session_state.feedback_data[1]:
                    # 채팅 로그에 추가되어 있지 않다면 추가 (재실행 방지)
                    if not any(log.get('content') == st.session_state.feedback_data[1]['content'] for log in current_chat_log):
                         st.session_state.chat_log.append({"role": "assistant", "content": st.session_state.feedback_data[1]['content']})
                
                # 학생 응답 입력 폼
                with st.form("answer_form"):
                    answer_input = st.text_area("AI 선생님의 질문에 답변해줘!", placeholder="내 생각에는...")
                    submit_answer = st.form_submit_button("답변 완료 📨")
                    
                    if submit_answer:
                        if not answer_input.strip():
                            st.warning("답변을 입력해줘!")
                        else:
                            # 학생 응답 저장
                            st.session_state.feedback_data[2]['content'] = answer_input 
                            st.session_state.chat_log.append({"role": "user", "content": f"답변: {answer_input}"})
                            
                            st.session_state.feedback_stage = 4
                            st.rerun()

            # --- 5단계: 피드백 4 (수정 지도 + 종합 정리) 출력 및 다음 이야기 버튼 대기 ---
            elif st.session_state.feedback_stage == 4:
                # 피드백 4단계 생성 및 출력
                if st.session_state.feedback_data and not st.session_state.feedback_data[3]['content']:
                    with st.spinner("AI 선생님이 최종 답변을 준비 중이야..."):
                        final_feedback = generate_step_4_feedback(
                            st.session_state.initial_reason,
                            st.session_state.feedback_data[2]['content'], # 학생의 두 번째 응답
                            st.session_state.selected_choice, 
                            data['story'], 
                            st.session_state.rag_text
                        )
                        st.session_state.feedback_data[3]['content'] = final_feedback
                        st.session_state.chat_log.append({"role": "assistant", "content": final_feedback})

                        # 학습 기록 저장 (4단계 피드백이 완료된 시점에 최종 저장)
                        st.session_state.learning_records.append({
                            "step": idx + 1,
                            "choice": st.session_state.selected_choice,
                            "reason": st.session_state.initial_reason,
                            "answer_to_question": st.session_state.feedback_data[2]['content']
                        })
                
                # 다음 이야기 버튼
                if st.button("다음 이야기로 넘어가기 ➡️", type="primary"):
                    if st.session_state.current_step < 3:
                        st.session_state.current_step += 1
                        st.session_state.feedback_stage = 0 # 다음 단계로 이동
                        st.session_state.feedback_data = None
                        st.session_state.selected_choice = None
                        st.session_state.chat_log = []
                        st.session_state.initial_reason = ""
                        st.rerun()
                    else:
                        st.session_state.lesson_complete = True
                        st.rerun()

    # [C] 학습 완료 리포트 화면
    else:
        st.header("🎉 학습 완료! 참 잘했어!")
        st.subheader("📝 나의 학습 리포트")
        
        if not st.session_state.final_report:
            with st.spinner("선생님이 너의 활동을 정리하고 있어..."):
                st.session_state.final_report = generate_final_summary(st.session_state.topic, st.session_state.learning_records)
        
        st.info(f"**[AI 선생님의 총평]**\n\n{st.session_state.final_report}")
        
        st.write("---")
        st.write("### 👣 내가 걸어온 윤리적인 선택들")
        
        for record in st.session_state.learning_records:
            with st.expander(f"{record['step']}단계에서의 선택"):
                st.markdown(f'<p style="font-size:1.1em;"><b>선택:</b> {record["choice"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:1.1em;"><b>첫 이유:</b> {record["reason"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:1.1em;"><b>사고 확장 응답:</b> {record["answer_to_question"]}</p>', unsafe_allow_html=True)
        
        if st.button("🔄 처음부터 다시 하기", type="primary"):
            st.session_state.lesson_complete = False
            st.session_state.current_step = 0
            st.session_state.chat_log = []
            st.session_state.learning_records = []
            st.session_state.final_report = None
            st.session_state.feedback_stage = 0
            st.session_state.feedback_data = None
            st.rerun()
