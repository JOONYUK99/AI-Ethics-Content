import streamlit as st
from openai import OpenAI
import re
import os

# --- 1. 페이지 설정 ---
# [변경] 쭈니봇 -> 테스트 봇
st.set_page_config(page_title="테스트 봇과 함께하는 AI 윤리 교실", page_icon="🤖", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ OpenAI API 키를 설정해주세요! (.streamlit/secrets.toml 파일 확인)")
    st.stop()

# --- 3. [핵심] 교육과정 반영 시스템 페르소나 ---
# [변경] 페르소나 이름 변경
SYSTEM_PERSONA = """
당신은 초등학생(5~6학년)을 위한 AI 윤리 교육 튜터 '테스트 봇'입니다.
'국가 인공지능 윤리기준', '도덕과 교육과정', '실과(정보) 교육과정'을 기반으로 교육합니다.

[핵심 행동 수칙]
1. [교육과정 연계]: 설명할 때 "이건 도덕 시간에 배운 '정보 예절'과 관련 있어" 처럼 교과 과정과 연결해주세요.
2. [개인정보 철벽 방어]: 학생이 개인정보를 말하려 하면 즉시 교육적으로 제지하세요.
3. [사례 중심]: 추상적인 개념(알고리즘 등)은 학교 생활이나 게임 같은 구체적인 사례로 바꿔 설명하세요.
4. [말투]: "안녕! 나는 테스트 봇이야", "~했니?" 처럼 다정하고 친근한 초등 교사 말투를 사용하세요.
"""

# --- 4. 함수 정의 ---

def load_reference_data():
    """reference.txt (통합 교육과정 자료) 읽기"""
    file_path = "reference.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return None

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
    """[핵심] 교육과정 및 사례 기반 시나리오 생성"""
    if not rag_data: rag_data = "기본 윤리: 남에게 피해 주지 않기"
    
    prompt = (
        f"# 참고할 교육과정 및 윤리 기준:\n{rag_data}\n\n"
        f"# 주제: '{topic}'\n\n"
        "위 '교육과정' 내용을 반영하여, 초등학생이 학교나 일상에서 겪을 법한 '구체적인 사례'로 딜레마 시나리오를 만들어줘.\n"
        "- 총 4단계(도입-전개-위기-결말)\n"
        "- 각 단계는 3~4문장\n"
        "- 각 단계 끝에 [CHOICE A], [CHOICE B] 선택지 포함\n"
        "- 내용이 너무 어렵지 않게, '친구 관계', '숙제', '게임' 같은 소재 활용\n\n"
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

def generate_educational_feedback(choice, reason, story_context, rag_data):
    """[핵심] 학생의 선택을 교육과정 성취기준과 연결하여 피드백"""
    if not rag_data: rag_data = "기본 윤리 원칙"

    prompt = (
        f"# [교육과정 및 국가 표준]:\n{rag_data}\n\n"
        f"# [현재 상황]:\n{story_context}\n\n"
        f"# [학생의 선택]: {choice}\n"
        f"# [학생의 이유]: {reason}\n\n"
        "위 내용을 바탕으로 초등학생에게 줄 교육적 피드백을 작성해줘.\n"
        "1. [공감과 칭찬]: 학생의 솔직한 생각에 먼저 공감해주고 칭찬해줘.\n"
        "2. [교육과정 연계]: 학생의 이유가 위 '교육과정'의 어떤 내용(예: 도덕과 정보예절, 실과 개인정보보호 등)과 연결되는지 구체적으로 설명해줘.\n"
        "3. [사고 확장 질문]: 반대 입장을 생각해보게 하는 질문을 하나 던져줘.\n"
        "4. [수정 지도]: 비속어나 개인정보가 포함되어 있다면 따뜻하게 고쳐줘."
    )
    return ask_gpt(prompt)

# --- 5. 메인 앱 로직 ---

# 세션 초기화
if 'scenario' not in st.session_state: st.session_state.scenario = None
if 'scenario_images' not in st.session_state: st.session_state.scenario_images = [None]*4
if 'current_step' not in st.session_state: st.session_state.current_step = 0
if 'chat_log' not in st.session_state: st.session_state.chat_log = []
if 'topic' not in st.session_state: st.session_state.topic = ""
if 'rag_text' not in st.session_state: st.session_state.rag_text = load_reference_data()
if 'tutorial_complete' not in st.session_state: st.session_state.tutorial_complete = False
if 'tutorial_step' not in st.session_state: st.session_state.tutorial_step = 0
if 'selected_choice' not in st.session_state: st.session_state.selected_choice = None
if 'waiting_for_reason' not in st.session_state: st.session_state.waiting_for_reason = False
if 'feedback_shown' not in st.session_state: st.session_state.feedback_shown = False

st.sidebar.title("🏫 AI 윤리 교실 모드")
mode = st.sidebar.radio("모드를 선택하세요:", ["학생용 (수업 참여)", "교사용 (수업 개설)"])

# ==========================================
# 👨‍🏫 교사용 화면
# ==========================================
if mode == "교사용 (수업 개설)":
    st.header("👨‍🏫 교사용: 교육과정 기반 수업 만들기")
    password = st.text_input("교사 인증 비밀번호 (1234)", type="password")
    
    if password == "1234":
        # RAG 데이터 확인
        with st.expander("📚 적용된 교육과정 및 윤리기준 확인"):
            if not st.session_state.rag_text:
                st.warning("⚠️ 'reference.txt' 파일이 없습니다. 기본 지식으로 작동합니다.")
            else:
                st.info("국가 인공지능 윤리기준, 도덕과/실과 교육과정이 통합 반영되었습니다.")
                st.text_area("내용 미리보기", st.session_state.rag_text, height=150, disabled=True)
            
            # 파일 업로드 기능 (추가 자료용)
            uploaded_file = st.file_uploader("추가 교육 자료 업로드 (txt)", type="txt")
            if uploaded_file:
                string_data = uploaded_file.getvalue().decode("utf-8")
                # 기존 자료에 덧붙이기
                st.session_state.rag_text += "\n\n[추가 자료]\n" + string_data
                st.success("✅ 추가 자료가 교육과정에 통합되었습니다!")

        input_topic = st.text_area("오늘의 수업 주제 (예: 딥페이크, AI 저작권, 챗봇 예절)", value=st.session_state.topic)
        st.caption("💡 팁: '딥페이크'라고만 적어도 교육과정에 맞춰 '친구 얼굴 합성 사례' 등을 만들어줍니다.")
        
        if st.button("🚀 교육 시나리오 생성"):
            with st.spinner("교육과정 성취 기준에 맞춰 시나리오를 설계 중입니다..."):
                # RAG 데이터를 함께 넘겨서 시나리오 생성
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
                    st.session_state.feedback_shown = False
                    st.success("교육과정 연계 시나리오 생성 완료!")
        
        if st.session_state.scenario:
            st.write("---")
            st.subheader("🖼️ 교육용 삽화 생성")
            cols = st.columns(4)
            for i in range(4):
                with cols[i]:
                    st.markdown(f"**단계 {i+1}**")
                    if st.session_state.scenario_images[i]:
                        st.image(st.session_state.scenario_images[i])
                    if st.button(f"그림 생성 {i+1}", key=f"gen_{i}"):
                        with st.spinner("그리는 중..."):
                            url = generate_image(st.session_state.scenario[i]['story'])
                            if url:
                                st.session_state.scenario_images[i] = url
                                st.rerun()

# ==========================================
# 🙋‍♂️ 학생용 화면
# ==========================================
elif mode == "학생용 (수업 참여)":
    
    # [A] 튜토리얼 (이름 변경)
    if not st.session_state.tutorial_complete:
        st.header("🎒 연습 시간: 테스트 봇과 친해지기")
        if st.session_state.tutorial_step == 0:
            st.info("안녕? 나는 테스트 봇이야! 버튼 누르는 연습을 해볼까?")
            c1, c2 = st.columns(2)
            if c1.button("🅰️ 여름이 좋아! 🍦"): st.toast("잘했어!"); st.session_state.tutorial_step = 1; st.rerun()
            if c2.button("🅱️ 겨울이 좋아! ☃️"): st.toast("완벽해!"); st.session_state.tutorial_step = 1; st.rerun()
        elif st.session_state.tutorial_step == 1:
            st.info("이번엔 채팅 연습이야. '안녕'이라고 인사해줄래?")
            if user_input := st.chat_input("여기에 입력해봐!"):
                st.balloons(); st.session_state.tutorial_step = 2; st.rerun()
        elif st.session_state.tutorial_step == 2:
            st.success("준비 끝! 이제 수업을 시작하자.")
            if st.button("🚀 수업 시작!"): st.session_state.tutorial_complete = True; st.rerun()

    # [B] 본 수업
    else:
        st.header(f"🙋‍♂️ 토론하기: {st.session_state.topic}")

        if not st.session_state.scenario:
            st.warning("선생님이 아직 수업을 안 만들었어!")
        else:
            if st.button("🔄 연습 다시하기", type="secondary"):
                st.session_state.tutorial_complete = False; st.session_state.tutorial_step = 0; st.rerun()

            idx = st.session_state.current_step
            data = st.session_state.scenario[idx]
            img = st.session_state.scenario_images[idx]

            st.markdown(f"### 📖 Part {idx + 1}")
            if img: st.image(img)
            st.info(data['story'])

            # 대화 기록 표시 (이름 변경)
            for msg in st.session_state.chat_log:
                role = "테스트 봇" if msg["role"] == "assistant" else "나"
                avatar = "🤖" if msg["role"] == "assistant" else "🙋"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg['content'])

            if not st.session_state.waiting_for_reason and not st.session_state.feedback_shown:
                st.write("### 👇 너의 선택은?")
                c1, c2 = st.columns(2)
                if c1.button(f"🅰️ {data['a']}", use_container_width=True):
                    st.session_state.selected_choice = data['a']; st.session_state.waiting_for_reason = True; st.rerun()
                if c2.button(f"🅱️ {data['b']}", use_container_width=True):
                    st.session_state.selected_choice = data['b']; st.session_state.waiting_for_reason = True; st.rerun()

            elif st.session_state.waiting_for_reason:
                st.success(f"**선택:** {st.session_state.selected_choice}")
                st.markdown("### 🤔 왜 그렇게 선택했어?")
                
                with st.form("reason_form"):
                    # [변경] 안내 문구 이름 변경
                    reason_input = st.text_area("이유를 적어주면 테스트 봇이 피드백을 줄 거야!", placeholder="예: 왜냐하면...")
                    submit = st.form_submit_button("입력 완료 💌")
                    
                    if submit:
                        if not reason_input.strip():
                            st.warning("이유를 꼭 적어줘!")
                        else:
                            st.session_state.chat_log.append({"role": "user", "content": f"선택: {st.session_state.selected_choice}\n이유: {reason_input}"})
                            with st.spinner("교육과정 성취기준 분석 중..."):
                                feedback = generate_educational_feedback(
                                    st.session_state.selected_choice, reason_input, data['story'], st.session_state.rag_text
                                )
                                st.session_state.chat_log.append({"role": "assistant", "content": feedback})
                            st.session_state.waiting_for_reason = False; st.session_state.feedback_shown = True; st.rerun()

            elif st.session_state.feedback_shown:
                if st.button("다음 이야기로 넘어가기 ➡️", type="primary"):
                    if st.session_state.current_step < 3:
                        st.session_state.current_step += 1; st.session_state.selected_choice = None; st.session_state.waiting_for_reason = False; st.session_state.feedback_shown = False; st.session_state.chat_log = []; st.rerun()
                    else:
                        st.balloons(); st.success("모든 토론이 끝났어! 훌륭해!")
