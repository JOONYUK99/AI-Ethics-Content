# ===================================================================
# 1. 필요한 모든 라이브러리 설치
# ===================================================================
!pip install -q streamlit pyngrok

# ===================================================================
# 2. Streamlit 앱 전체 코드를 app.py 파일로 저장
# ===================================================================
app_code = r"""
import streamlit as st
import json
import requests
import re
import urllib.parse

# --- 1. AI 핵심 기능 함수 정의 (최종 안정화 버전) ---
def call_gemini_api(prompt, api_key):
    # 안정성을 위해 v1 엔드포인트와 gemini-pro 모델을 직접 호출합니다.
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=120)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        # 오류 발생 시, 원본 메시지를 포함하여 반환
        return f"API 호출 중 오류가 발생했습니다: {e}\n응답: {response.text if 'response' in locals() else '없음'}"

def transform_scenario(teacher_input, api_key):
    prompt = (
        "당신은 초등학생 고학년을 위한 AI 윤리 교육용 인터랙티브 시나리오 작가입니다.\n"
        f"아래의 '입력 내용'을 바탕으로, 학생들이 몰입할 수 있고 총 4번의 선택을 하게 되는 완결된 이야기를 만들어주세요.\n"
        "각 이야기 파트는 반드시 3문장 이하의 짧은 길이로 구성해주세요.\n\n"
        "# 필수 출력 형식:\n"
        "[STORY 1] (이야기 내용) [CHOICE 1A] (A 선택지) [CHOICE 1B] (B 선택지)\n---\n"
        "[STORY 2] (이야기 내용) [CHOICE 2A] (A 선택지) [CHOICE 2B] (B 선택지)\n---\n"
        "[STORY 3] (이야기 내용) [CHOICE 3A] (A 선택지) [CHOICE 3B] (B 선택지)\n---\n"
        "[STORY 4] (이야기 내용) [CHOICE 4A] (A 선택지) [CHOICE 4B] (B 선택지)\n\n"
        f"--- 입력 내용 ---\n{teacher_input}"
    )
    return call_gemini_api(prompt, api_key)

def generate_image_keywords(story_part, api_key):
    prompt = f"다음 한국어 문장의 핵심 내용을 대표하는 영어 단어 2개를 쉼표로 구분하여 짧게 요약해줘. 예: '슬픈 아이가 로봇과 함께 있다' -> 'sad child, robot'\n\n문장: {story_part}"
    return call_gemini_api(prompt, api_key)

def start_debate(history, choice, api_key):
    prompt = (
        "당신은 다정한 AI 윤리 선생님입니다. 학생의 선택을 칭찬하고, 왜 그렇게 생각했는지 부드럽게 첫 질문을 던져주세요.\n\n"
        f"--- 이야기와 학생의 선택 ---\n{history}\n학생의 선택: {choice}\n\nAI 선생님의 따뜻한 첫 질문:"
    )
    return call_gemini_api(prompt, api_key)

def continue_debate(debate_history, api_key):
    prompt = (
        "당신은 다정한 AI 윤리 선생님입니다. 학생의 의견에 공감하며, '혹시 이런 점은 어떨까요?' 와 같이 부드러운 말투로 반대 관점을 제시하며 토론을 이어가주세요.\n\n"
        f"--- 지금까지의 토론 내용 ---\n{debate_history}\n\nAI 선생님의 다음 질문:"
    )
    return call_gemini_api(prompt, api_key)

def generate_conclusion(final_history, api_key):
    prompt = (
        "다음은 한 학생이 AI 윤리 문제에 대해 거친 전체 기록입니다. 이 기록을 바탕으로, 학생의 고민 과정을 요약하고 비판적 사고 능력을 칭찬해주세요. 그리고 이 딜레마에 대해 현실에서 생각해볼 '대처 방법'을 제안하며 따뜻하게 격려하는 메시지로 마무리해주세요.\n\n"
        f"--- 전체 기록 ---\n{final_history}"
    )
    return call_gemini_api(prompt, api_key)

# --- 2. Streamlit 앱 UI 및 로직 ---
st.set_page_config(page_title="AI 윤리 교육", page_icon="✨", layout="centered")
st.title("✨ 초등학생을 위한 AI 윤리 교육")

try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("API 키를 설정해주세요!")
    st.stop()

if 'stage' not in st.session_state:
    st.session_state.stage = 'teacher_input'
    st.session_state.full_scenario = []
    st.session_state.full_log = ""
    st.session_state.current_part = -1
    st.session_state.debate_turns = 0
    st.session_state.MAX_CHOICES = 4

def restart_lesson():
    st.session_state.stage = 'teacher_input'
    st.session_state.full_scenario = []
    st.session_state.full_log = ""
    st.session_state.current_part = -1
    st.session_state.debate_turns = 0

# 1. 교사 입력 단계
if st.session_state.stage == 'teacher_input':
    st.info("AI 윤리 교육 콘텐츠로 만들고 싶은 실제 사례, 뉴스 기사 등을 아래에 입력해주세요.")
    teacher_text = st.text_area("시나리오 입력:", height=150, placeholder="예시: AI 그림 대회에서 인공지능으로 그린 그림이 1등을 차지해서 논란이 되었습니다...")
    if st.button("이 내용으로 교육 콘텐츠 생성하기"):
        if not teacher_text:
            st.warning("시나리오를 입력해주세요.")
        else:
            st.session_state.teacher_input = teacher_text
            st.session_state.stage = 'story_generation'
            st.rerun()

# 2. AI 시나리오 생성 단계
elif st.session_state.stage == 'story_generation':
    with st.spinner("AI가 입력하신 내용을 바탕으로 시나리오를 만들고 있어요. 잠시만 기다려주세요..."):
        full_scenario_text = transform_scenario(st.session_state.teacher_input, API_KEY)
        st.session_state.full_scenario = []
        parts = full_scenario_text.split('---')
        for i, part in enumerate(parts):
            try:
                story = re.search(rf"\[STORY {i+1}\](.*?)(?=\[CHOICE {i+1}A\])", part, re.DOTALL).group(1).strip()
                choice_a = re.search(rf"\[CHOICE {i+1}A\](.*?)(?=\[CHOICE {i+1}B\])", part, re.DOTALL).group(1).strip()
                choice_b = re.search(rf"\[CHOICE {i+1}B\](.*)", part, re.DOTALL).group(1).strip()
                st.session_state.full_scenario.append({"story": story, "choice_a": choice_a, "choice_b": choice_b})
            except Exception: continue
        
        if len(st.session_state.full_scenario) >= 4:
            st.session_state.full_log = f"**원문 요약:** {st.session_state.teacher_input[:50]}..."
            st.session_state.current_part = 0
            st.session_state.stage = 'student_choice'
            st.rerun()
        else:
            st.error("AI가 시나리오 생성에 실패했어요. 입력 내용을 조금 더 구체적으로 작성한 후 다시 시도해주세요.")
            st.code(full_scenario_text)
            if st.button("처음으로 돌아가기"):
                restart_lesson(); st.rerun()

# 3. 학생 선택 단계
elif st.session_state.stage == 'student_choice':
    part = st.session_state.full_scenario[st.session_state.current_part]
    st.markdown(f"### ✨ 이야기 #{st.session_state.current_part + 1} ✨")
    keywords = generate_image_keywords(part['story'], API_KEY)
    encoded_keywords = urllib.parse.quote(keywords)
    st.image(f"https://placehold.co/600x300/E8E8E8/313131?text={encoded_keywords}", caption=f"AI가 생각한 이미지: {keywords}")
    st.write(part['story'])
    st.session_state.current_story_part_log = f"### 이야기 #{st.session_state.current_part + 1}\n{part['story']}"
    st.info("자, 이제 어떤 선택을 해볼까요?")
    col1, col2 = st.columns(2)
    if col1.button(f"A: {part['choice_a']}", use_container_width=True, key=f"A_{st.session_state.current_part}"):
        st.session_state.current_story_part_log += f"\n\n**>> 나의 선택:** {part['choice_a']}"; st.session_state.stage = 'debate'; st.rerun()
    if col2.button(f"B: {part['choice_b']}", use_container_width=True, key=f"B_{st.session_state.current_part}"):
        st.session_state.current_story_part_log += f"\n\n**>> 나의 선택:** {part['choice_b']}"; st.session_state.stage = 'debate'; st.rerun()

# 4. 토론 단계
elif st.session_state.stage == 'debate':
    st.markdown(f"### 이야기 #{st.session_state.current_part + 1} 토론")
    log_parts = st.session_state.current_story_part_log.split('\n\n')
    for p in log_parts:
        if p.startswith("**>> 나의 선택"): st.chat_message("user").write(p.replace("**>> 나의 선택:**",""))
        elif p.startswith("**AI 선생님:**"): st.chat_message("assistant").write(p.replace("**AI 선생님:**",""))
        elif p.startswith("**나 (의견"): st.chat_message("user").write(p)
        else: st.write(p)
    
    if st.session_state.debate_turns == 0:
        with st.chat_message("assistant"):
            with st.spinner("AI 선생님이 질문을 준비하고 있어요..."):
                question = start_debate(st.session_state.current_story_part_log, st.session_state.current_story_part_log.split('>> 나의 선택:')[-1].strip(), API_KEY)
                st.session_state.current_story_part_log += f"\n\n**AI 선생님:** {question}"; st.session_state.debate_turns = 1; st.rerun()
    elif st.session_state.debate_turns == 1:
        if reply := st.chat_input("첫 번째 의견을 이야기해주세요:"):
            st.session_state.current_story_part_log += f"\n\n**나 (의견 1):** {reply}"; st.session_state.debate_turns = 2; st.rerun()
    elif st.session_state.debate_turns == 2:
        with st.chat_message("assistant"):
            with st.spinner("AI 선생님이 다음 질문을 생각 중이에요..."):
                question = continue_debate(st.session_state.current_story_part_log, API_KEY)
                st.session_state.current_story_part_log += f"\n\n**AI 선생님:** {question}"; st.session_state.debate_turns = 3; st.rerun()
    elif st.session_state.debate_turns == 3:
        if reply := st.chat_input("두 번째 의견을 이야기해주세요:"):
            st.session_state.current_story_part_log += f"\n\n**나 (의견 2):** {reply}"; st.session_state.debate_turns = 4; st.rerun()
    elif st.session_state.debate_turns == 4:
        st.info("토론이 완료되었어요. 아래 버튼을 눌러 다음으로 넘어가요!")
        st.session_state.full_log += f"\n\n---\n{st.session_state.current_story_part_log}"
        st.session_state.current_part += 1
        if st.button("다음 이야기로" if st.session_state.current_part < st.session_state.MAX_CHOICES else "최종 정리 보기"):
            st.session_state.debate_turns = 0; st.session_state.current_story_part_log = ""
            if st.session_state.current_part >= st.session_state.MAX_CHOICES:
                st.session_state.stage = 'conclusion'
            else:
                st.session_state.stage = 'story'
            st.rerun()

# 5. 최종 결론 단계
elif st.session_state.stage == 'conclusion':
    st.markdown("### 📚 우리의 전체 이야기와 고민 📚")
    st.markdown(st.session_state.full_log, unsafe_allow_html=True)
    st.markdown("---")
    with st.spinner("AI 선생님이 우리의 멋진 여정을 정리하고 있어요..."):
        conclusion = generate_conclusion(st.session_state.full_log, API_KEY)
        st.balloons(); st.success("모든 이야기가 끝났어요! 정말 수고 많았어요!")
        st.markdown("---"); st.markdown("### ✨ AI 선생님의 최종 정리 및 대처법 ✨"); st.write(conclusion)
    if st.button("새로운 주제로 다시 시작하기"):
        restart_lesson(); st.rerun()
"""
with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

# ===================================================================
# 3. Colab에 API 키를 설정하고 Streamlit 앱 실행
# ===================================================================
from google.colab import userdata
from pyngrok import ngrok
import os

try:
    ngrok_token = userdata.get('NGROK_AUTH_TOKEN')
    api_key = userdata.get('GOOGLE_API_KEY')
    !ngrok authtoken {ngrok_token}
    secrets_dir = os.path.expanduser('~/.streamlit')
    os.makedirs(secrets_dir, exist_ok=True)
    with open(os.path.join(secrets_dir, "secrets.toml"), "w") as f:
        f.write(f'GOOGLE_API_KEY = "{api_key}"\n')
except (userdata.SecretNotFoundError, NameError):
    print("❗️ Colab Secrets에 'NGROK_AUTH_TOKEN'과 'GOOGLE_API_KEY'를 모두 추가해주세요.")
    raise SystemExit()

ngrok.kill()
try:
    public_url = ngrok.connect(8501)
    print("🎉 챗봇 앱이 준비되었습니다! 아래 링크를 클릭하여 접속하세요:")
    print(public_url)
except Exception as e:
    print(f"❗️ ngrok 연결 중 오류 발생: {e}")
    raise SystemExit()

!streamlit run app.py --logger.level=error
