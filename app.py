import streamlit as st
from openai import OpenAI
import re
import os

# --- (이전 코드 블록은 동일하게 유지) ---
# ... (생략: 1~4번 섹션은 동일) ...

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
    # ... (생략: generate_image 함수는 동일) ...
    try:
        dalle_prompt = f"A friendly, educational cartoon-style illustration for elementary school textbook, depicting: {prompt}"
        response = client.images.generate(
            model="dall-e-3", prompt=dalle_prompt, size="1024x1024", quality="standard", n=1
        )
        return response.data[0].url
    except:
        return None

def create_scenario(topic, rag_data=""): 
    """LLM 자율 판단 단계로 시나리오 생성 요청 (명령 강화)"""
    
    # 🚨 [수정 및 강화] LLM에게 출력 형식을 반드시 지키도록 강하게 지시
    prompt = (
        f"# 참고할 교육과정 및 윤리 기준:\n{rag_data}\n\n" 
        f"# 주제: '{topic}'\n\n"
        "아래 규칙을 **철저하게 지켜서** 딜레마 시나리오를 생성해야 합니다. **유효한 스토리가 없으면 안 됩니다.**\n"
        "[작성 규칙 - 중요!]\n"
        "1. 문장은 무조건 짧고 간결하게 끊어써야 해. (호흡이 길면 안 됨)\n"
        "2. 어려운 단어는 쓰지 마.\n"
        "3. 이 주제를 가장 잘 다룰 수 있도록 **최소 3단계에서 최대 6단계 사이**로 딜레마 단계 수를 스스로 결정하여 구성해.\n"
        "4. 각 단계는 2~3문장 이내로 짧게 작성.\n"
        "5. **반드시** 각 단계 끝에 **[CHOICE A]**와 **[CHOICE B]** 선택지를 포함해야 합니다.\n\n"
        "# 출력 형식:\n[STORY 1] ... [CHOICE 1A] ... [CHOICE 1B] ...\n---\n[STORY 2] ... --- ... [마지막 단계 스토리] ... ---"
    )
    return ask_gpt(prompt)

def analyze_scenario(topic, full_scenario_text):
    # ... (생략: analyze_scenario 함수는 동일) ...
    prompt = (
        f"교사가 '{topic}' 주제로 아래 시나리오를 만들었습니다:\n"
        f"--- 시나리오 텍스트 ---\n{full_scenario_text}\n\n"
        "이 시나리오를 분석하여 다음 3가지 항목을 추출해 주세요.\n"
        "\n"
        "# 출력 형식 (태그만 사용):\n"
        "[윤리 기준] [AI가 분석한 이 시나리오에 근거가 되는 윤리 기준이나 원칙 (최대 15글자로 요약)]\n"
        "[성취기준] [AI가 분석한 이 시나리오가 달성하고자 하는 교육과정의 성취기준 코드 및 내용 요약 (최대 15글자로 요약)]\n"
        "[학습 내용] [이 시나리오를 통해 학생이 최종적으로 배우게 될 핵심 윤리 내용 (최대 15글자로 요약)]"
    )
    analysis = ask_gpt(prompt)
    
    result = {}
    try:
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

def parse_scenario(text):
    """시나리오 파싱 (단계 수 유동화 및 안전 로직 보강)"""
    if not text: return None
    scenario = []
    
    # 정규 표현식을 사용하여 STORY와 CHOICE 태그를 포함하는 유효한 단계만 추출
    # STORY #, CHOICE #A, CHOICE #B 패턴을 모두 포함하는 블록을 찾음
    # LLM이 [STORY 1]을 썼을지, [STORY]만 썼을지 모르기 때문에 유연하게 파싱
    pattern = r"\[STORY\s?\d*\](.*?)\[CHOICE\s?\d*A\](.*?)\[CHOICE\s?\d*B\](.*?)(?:---|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        # match[0]은 스토리 텍스트, match[1]은 CHOICE A 텍스트, match[2]는 CHOICE B 텍스트
        story = match[0].strip()
        choice_a = match[1].strip()
        choice_b = match[2].strip()
        
        # 최소한의 텍스트가 있어야 유효한 단계로 간주
        if story and choice_a and choice_b:
             scenario.append({"story": story, "a": choice_a, "b": choice_b})
    
    # 최소 3단계는 보장하도록 함 (AI 자율 결정의 최소 기준)
    if len(scenario) >= 3:
        return scenario 
    else:
        # 파싱은 성공했지만 단계 수가 부족하거나, 형식이 완전히 깨진 경우
        return None

# ... (생략: get_four_step_feedback 및 generate_step_4_feedback 함수는 동일) ...
# ... (생략: 메인 앱 로직도 동일, 변경된 함수를 사용함) ...

# --- 6. 메인 앱 로직 (핵심 부분만 다시 포함) ---

# 세션 초기화 및 상태 변수 정의 (이전과 동일)
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
            with st.spinner("AI가 딜레마 시나리오를 창작 중입니다..."):
                raw = create_scenario(input_topic, st.session_state.rag_text)
                
                st.session_state.full_scenario_text = raw 
                
                parsed = parse_scenario(raw)
                
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
                        analysis = analyze_scenario(input_topic, st.session_state.full_scenario_text)
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
                                    st.session_state.scenario_images[i] = url
                                    st.rerun()
                    with col_img:
                        if i < len(st.session_state.scenario_images) and st.session_state.scenario_images[i]:
                            st.image(st.session_state.scenario_images[i], width=400)
                else:
                    st.error(f"⚠️ {i+1}단계 시나리오 데이터가 불완전합니다.")


# ==========================================
# 🙋‍♂️ 학생용 화면 (유동적 단계 로직 유지)
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
            img = st.session_state.scenario_images[idx]

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
