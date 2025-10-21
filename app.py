import streamlit as st
import google.generativeai as genai
import re
import urllib.parse

# --- 1. AI 핵심 기능 함수 정의 ---
def get_model(model_name='gemini-pro'):
    return genai.GenerativeModel(model_name)

def generate_story_part(topic, history_summary=""):
    model = get_model()
    # 이야기를 '반드시 2문장'으로 만들도록 프롬프트 강화
    if not history_summary:
        prompt = f"'{topic}'라는 주제 혹은 아래의 실제 사례를 바탕으로, 초등학생 저학년도 이해할 수 있는 AI 윤리 동화의 '첫 부분'을 만들어줘. 이야기는 반드시 간결한 두 문장으로만 구성하고, 주인공이 중요한 결정을 내려야 하는 순간에서 끝나야 해. 절대 길게 쓰지 마."
    else:
        prompt = f"다음은 지금까지 진행된 이야기의 요약이야: '{history_summary}'. 이 이야기에 이어서, 학생의 선택으로 인해 벌어지는 '다음 사건'을 반드시 간결한 두 문장으로 만들어줘. 그리고 이야기가 또 다른 중요한 결정을 내려야 하는 순간에서 끝나도록 해줘."
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e: return f"이야기 생성 중 오류: {e}"

def generate_image_keywords(story_part):
    model = get_model()
    prompt = f"다음 한국어 문장의 핵심 내용을 대표하는 영어 단어 2개를 쉼표로 구분하여 짧게 요약해줘. 예: '슬픈 아이가 로봇과 함께 있다' -> 'sad child, robot'\n\n문장: {story_part}"
    try:
        response = model.generate_content(prompt)
        keywords = [keyword.strip() for keyword in response.text.strip().replace('*','').split(',')]
        return ",".join(keywords)
    except Exception:
        return "AI,robot"

def generate_choices_for_story(story_part):
    model = get_model()
    prompt = f"아래 이야기의 마지막 상황에서 주인공이 할 수 있는, 윤리적으로 상반된 두 가지 선택지를 초등학생 눈높이에 맞춰서 간결하게 만들어줘.\n[출력 형식]\nA: [A 선택지 내용]\nB: [B 선택지 내용]\n\n--- 이야기 ---\n{story_part}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e: return f"선택지 생성 중 오류: {e}"

def start_debate(current_story_log, choice):
    model = get_model()
    prompt = f"당신은 학생들을 아주 아끼는 다정한 AI 윤리 선생님입니다. 학생이 방금 내린 선택('{choice}')을 칭찬하고, 왜 그렇게 생각했는지 부드럽게 첫 질문을 던져주세요."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"토론 시작 중 오류: {e}"

def continue_debate(current_debate_history):
    model = get_model()
    prompt = f"당신은 다정한 AI 윤리 선생님입니다. 학생의 이전 답변에 공감하며 토론을 이어가주세요. '혹시 이런 점은 어떨까요?' 와 같이 부드러운 말투로 반대 관점이나 새로운 생각해볼 거리를 질문으로 제시해주세요.\n\n--- 지금까지의 토론 내용 ---\n{current_debate_history}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"토론 중 오류: {e}"

def generate_conclusion(final_history):
    model = get_model()
    prompt = (
        "다음은 한 학생이 AI 윤리 문제에 대해 총 4번의 선택과 토론을 거친 전체 기록입니다.\n\n"
        "# 당신의 역할:\n"
        "1. 학생의 고민 과정을 요약하고, 비판적 사고 능력을 따뜻하게 칭찬해주세요.\n"
        "2. 이 윤리적 딜레마를 마주했을 때, 초등학생이 현실에서 생각해 보거나 실천할 수 있는 구체적인 '대처 방법'이나 '마음가짐'을 한두 가지 제안해주세요.\n"
        "3. 정답은 없다는 점을 강조하며, 학생의 성장을 격려하는 메시지로 마무리해주세요.\n\n"
        f"--- 전체 기록 ---\n{final_history}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"결론 생성 중 오류: {e}"

# --- 2. Streamlit 앱 UI 및 로직 ---
st.set_page_config(page_title="AI 윤리 교육", page_icon="✨", layout="centered")
st.title("✨ 초등학생을 위한 AI 윤리 교육")

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("API 키를 설정해주세요!")
    st.stop()

if 'stage' not in st.session_state:
    st.session_state.stage = 'start'
    st.session_state.full_log = ""
    st.session_state.current_story_part_log = ""
    st.session_state.choice_count = 0
    st.session_state.debate_turns = 0
    st.session_state.MAX_CHOICES = 4

def restart_lesson():
    st.session_state.stage = 'start'
    st.session_state.full_log = ""
    st.session_state.current_story_part_log = ""
    st.session_state.choice_count = 0
    st.session_state.debate_turns = 0

if st.session_state.stage == 'start':
    # <--- 수정: 주제 선택(selectbox) 대신 시나리오 직접 입력(text_area)으로 변경
    st.info("AI 윤리 교육 콘텐츠로 만들고 싶은 실제 사례, 뉴스 기사 등을 아래에 입력해주세요.")
    teacher_text = st.text_area("시나리오 입력:", height=150, placeholder="예시: AI 그림 대회에서 인공지능으로 그린 그림이 1등을 차지해서 논란이 되었습니다...")
    if st.button("이 내용으로 교육 콘텐츠 생성하기"):
        if not teacher_text:
            st.warning("시나리오를 입력해주세요.")
        else:
            st.session_state.topic = teacher_text # 입력된 내용을 주제로 설정
            st.session_state.full_log = f"**입력 내용:** {st.session_state.topic[:70]}..."
            st.session_state.stage = 'story'
            st.rerun()

elif st.session_state.stage == 'story':
    history_summary = st.session_state.full_log[-300:] if st.session_state.choice_count > 0 else ""
    st.markdown(f"### ✨ 이야기 #{st.session_state.choice_count + 1} ✨")
    with st.spinner(f"AI가 이야기 #{st.session_state.choice_count + 1}을(를) 만들고 있어요..."):
        story_part = generate_story_part(st.session_state.topic, history_summary)
        keywords = generate_image_keywords(story_part)
        choices_text = generate_choices_for_story(story_part)
    
    try:
        encoded_keywords = urllib.parse.quote(keywords)
        st.image(f"https://placehold.co/600x300/E8E8E8/313131?text={encoded_keywords}", caption=f"AI가 상상한 이미지: {keywords}")
    except Exception:
        pass
        
    st.write(story_part)
    st.session_state.current_story_part_log = f"### 이야기 #{st.session_state.choice_count + 1}\n{story_part}"
    
    try:
        match_a = re.search(r"A:\s*(.*)", choices_text, re.DOTALL)
        match_b = re.search(r"B:\s*(.*)", choices_text, re.DOTALL)
        if not (match_a and match_b): raise ValueError("선택지 형식 오류")
        choice_a_text = match_a.group(1).strip()
        choice_b_text = match_b.group(1).strip()
        st.info("자, 이제 어떤 선택을 해볼까요?")
        col1, col2 = st.columns(2)
        if col1.button(f"A: {choice_a_text}", use_container_width=True, key=f"A_{st.session_state.choice_count}"):
            st.session_state.current_story_part_log += f"\n\n**>> 나의 선택:** {choice_a_text}"; st.session_state.stage = 'debate'; st.rerun()
        if col2.button(f"B: {choice_b_text}", use_container_width=True, key=f"B_{st.session_state.choice_count}"):
            st.session_state.current_story_part_log += f"\n\n**>> 나의 선택:** {choice_b_text}"; st.session_state.stage = 'debate'; st.rerun()
    except Exception as e:
        st.error("선택지를 만드는 데 실패했어요. AI의 답변 형식이 달랐을 수 있어요.")
        if st.button("이야기 다시 만들기"): st.rerun()

elif st.session_state.stage == 'debate':
    st.markdown(f"### 이야기 #{st.session_state.choice_count + 1} 토론")
    log_parts = st.session_state.current_story_part_log.split('\n\n')
    for p in log_parts:
        if p.startswith("**>> 나의 선택"): st.chat_message("user").write(p.replace("**>> 나의 선택:**",""))
        elif p.startswith("**AI 선생님:**"): st.chat_message("assistant").write(p.replace("**AI 선생님:**",""))
        elif p.startswith("**나 (의견"): st.chat_message("user").write(p)
        else: st.write(p)
    
    if st.session_state.debate_turns == 0:
        with st.chat_message("assistant"):
            with st.spinner("AI 선생님이 첫 질문을 준비하고 있어요..."):
                question = start_debate(st.session_state.current_story_part_log, st.session_state.current_story_part_log.split('>> 나의 선택:')[-1].strip())
                st.session_state.current_story_part_log += f"\n\n**AI 선생님:** {question}"
                st.session_state.debate_turns = 1; st.rerun()
    elif st.session_state.debate_turns == 1:
        if reply := st.chat_input("첫 번째 의견을 이야기해주세요:"):
            st.session_state.current_story_part_log += f"\n\n**나 (의견 1):** {reply}"; st.session_state.debate_turns = 2; st.rerun()
    elif st.session_state.debate_turns == 2:
        with st.chat_message("assistant"):
            with st.spinner("AI 선생님이 다음 질문을 생각 중이에요..."):
                question = continue_debate(st.session_state.current_story_part_log)
                st.session_state.current_story_part_log += f"\n\n**AI 선생님:** {question}"; st.session_state.debate_turns = 3; st.rerun()
    elif st.session_state.debate_turns == 3:
        if reply := st.chat_input("두 번째 의견을 이야기해주세요:"):
            st.session_state.current_story_part_log += f"\n\n**나 (의견 2):** {reply}"; st.session_state.debate_turns = 4; st.rerun()
    elif st.session_state.debate_turns == 4:
        st.info("토론이 완료되었어요. 아래 버튼을 눌러 다음으로 넘어가요!")
        st.session_state.full_log += f"\n\n---\n{st.session_state.current_story_part_log}"
        st.session_state.choice_count += 1
        if st.button("다음 이야기로" if st.session_state.choice_count < st.session_state.MAX_CHOICES else "최종 정리 보기"):
            st.session_state.debate_turns = 0
            st.session_state.current_story_part_log = ""
            if st.session_state.choice_count >= st.session_state.MAX_CHOICES:
                st.session_state.stage = 'conclusion'
            else:
                st.session_state.stage = 'story'
            st.rerun()

elif st.session_state.stage == 'conclusion':
    st.markdown("### 📚 우리의 전체 이야기와 고민 📚")
    st.markdown(st.session_state.full_log, unsafe_allow_html=True)
    st.markdown("---")
    with st.spinner("AI 선생님이 우리의 멋진 여정을 정리하고 있어요..."):
        conclusion = generate_conclusion(st.session_state.full_log)
        st.balloons(); st.success("모든 이야기가 끝났어요! 정말 수고 많았어요!")
        st.markdown("---")
        st.markdown("### ✨ AI 선생님의 최종 정리 및 대처법 ✨")
        st.write(conclusion)
    if st.button("새로운 주제로 다시 시작하기"):
        restart_lesson(); st.rerun()
