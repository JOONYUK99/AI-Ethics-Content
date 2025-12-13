import streamlit as st
from openai import OpenAI
import re

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="쭈니봇과 함께 토론하기", page_icon="🤖", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ OpenAI API 키를 설정해주세요! (.streamlit/secrets.toml 파일 확인)")
    st.stop()

# --- 3. 함수 정의 (GPT/DALL-E 통신 및 로직) ---

def ask_gpt(prompt):
    """GPT-4o에게 질문하고 답을 받는 함수"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 초등학생을 위한 다정한 AI 윤리 선생님 '쭈니봇'입니다. 답변은 친절하고, 학생의 수준에 맞춰 쉬운 용어를 사용하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT 오류 발생: {e}")
        return None

def generate_image(prompt):
    """DALL-E 3로 이미지를 생성하는 함수 (NEW ✨)"""
    try:
        # 초등학생에게 적합한 동화풍 스타일로 프롬프트 수정
        dalle_prompt = f"A friendly, bright, cartoon-style illustration suitable for children's storybook, depicting the following scene: {prompt}"
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        st.error(f"이미지 생성 오류: {e}")
        return None

def create_scenario(topic):
    """교사가 입력한 주제로 시나리오 생성"""
    prompt = (
        f"주제: '{topic}'\n"
        "위 주제로 초등학생 고학년 대상의 딜레마 시나리오를 작성해줘.\n"
        "총 4단계(도입-전개-위기-결말) 구성이어야 해.\n"
        "각 단계는 3~4문장 정도로 간결하게 써줘.\n" # 이미지 생성을 위해 너무 길지 않게 요청
        "각 단계 끝에는 반드시 [CHOICE A]와 [CHOICE B] 형태의 선택지가 있어야 해.\n\n"
        "# 출력 형식 예시:\n"
        "[STORY 1] 이야기 내용...\n[CHOICE 1A] 선택지 A 내용\n[CHOICE 1B] 선택지 B 내용\n---\n"
        "[STORY 2] ...\n---\n... (4단계까지)"
    )
    return ask_gpt(prompt)

def parse_scenario(text):
    """생성된 텍스트를 구조화"""
    if not text: return None
    scenario = []
    parts = text.split('---')
    for part in parts:
        try:
            story = re.search(r"\[STORY\s?\d\](.*?)(?=\[CHOICE)", part, re.DOTALL).group(1).strip()
            choice_a = re.search(r"\[CHOICE\s?\dA\](.*?)(?=\[CHOICE)", part, re.DOTALL).group(1).strip()
            choice_b = re.search(r"\[CHOICE\s?\dB\](.*)", part, re.DOTALL).group(1).strip()
            scenario.append({"story": story, "a": choice_a, "b": choice_b})
        except:
            continue
    return scenario if len(scenario) >= 4 else None

def analyze_and_reply(history, user_input):
    """학생의 의견(자유 토론)을 분석하고 답변 생성"""
    prompt = (
        f"지금까지의 대화:\n{history}\n\n"
        f"학생의 의견: {user_input}\n\n"
        "학생의 의견에 공감해주고, 더 깊은 생각을 끌어내는 추가 질문을 하나 던져줘.\n"
        "말투는 '대단해!', '좋은 생각이야' 처럼 격려하는 말투로 해줘."
    )
    return ask_gpt(prompt)

# --- 4. 메인 앱 로직 ---

# 세션 상태 초기화
if 'scenario' not in st.session_state: st.session_state.scenario = None
if 'scenario_images' not in st.session_state: st.session_state.scenario_images = [None, None, None, None] # 이미지 URL 저장용
if 'current_step' not in st.session_state: st.session_state.current_step = 0
if 'chat_log' not in st.session_state: st.session_state.chat_log = []
if 'topic' not in st.session_state: st.session_state.topic = ""
if 'tutorial_complete' not in st.session_state: st.session_state.tutorial_complete = False
if 'tutorial_step' not in st.session_state: st.session_state.tutorial_step = 0

# --- 사이드바: 모드 선택 ---
st.sidebar.title("🏫 수업 모드 설정")
mode = st.sidebar.radio("모드를 선택하세요:", ["학생용 (수업 참여)", "교사용 (수업 개설)"])

# ==========================================
# 👨‍🏫 교사용 화면 (Teacher Mode)
# ==========================================
if mode == "교사용 (수업 개설)":
    st.header("👨‍🏫 교사용: 수업 만들기")
    st.info("이곳에서 수업 주제를 정하고 시나리오를 생성합니다.")

    password = st.text_input("교사 인증 비밀번호 (기본: 1234)", type="password")
    
    if password == "1234":
        input_topic = st.text_area("오늘의 토론 주제 입력", value=st.session_state.topic, height=100)
        
        # 1. 텍스트 시나리오 생성 버튼
        if st.button("🚀 시나리오(텍스트) 생성하기"):
            with st.spinner("쭈니봇이 이야기를 만들고 있어요..."):
                raw_text = create_scenario(input_topic)
                parsed_data = parse_scenario(raw_text)
                
                if parsed_data:
                    st.session_state.scenario = parsed_data
                    st.session_state.topic = input_topic
                    st.session_state.current_step = 0
                    st.session_state.chat_log = [] 
                    st.session_state.scenario_images = [None, None, None, None] # 이미지 초기화
                    st.success("✅ 텍스트 시나리오가 완성되었습니다! 아래에서 삽화를 추가해보세요.")
                else:
                    st.error("시나리오 생성 실패. 다시 시도해주세요.")

        # 2. 이미지 생성 섹션 (시나리오가 있을 때만 표시) (NEW ✨)
        if st.session_state.scenario:
            st.write("---")
            st.subheader("🖼️ (선택사항) 이야기 삽화 만들기")
            st.info("각 단계의 이야기를 바탕으로 AI가 그림을 그려줍니다. (장당 약 10~20초 소요)")

            cols = st.columns(4)
            for i in range(4):
                with cols[i]:
                    st.markdown(f"**Part {i+1} 삽화**")
                    # 이미 생성된 이미지가 있으면 보여주기
                    if st.session_state.scenario_images[i]:
                        st.image(st.session_state.scenario_images[i], use_column_width=True)
                        st.caption("✅ 생성 완료")
                    
                    # 이미지 생성 버튼
                    story_text = st.session_state.scenario[i]['story']
                    if st.button(f"✨ 그림 생성 ({i+1}단계)", key=f"gen_btn_{i}"):
                        with st.spinner("그림 그리는 중... 🎨"):
                            img_url = generate_image(story_text)
                            if img_url:
                                st.session_state.scenario_images[i] = img_url
                                st.rerun() # 이미지 표시를 위해 새로고침

            st.write("---")
            st.success("모든 준비가 끝나면 '학생용 모드'로 변경해주세요!")

    elif password:
        st.error("비밀번호가 틀렸습니다.")

# ==========================================
# 🙋‍♂️ 학생용 화면 (Student Mode)
# ==========================================
elif mode == "학생용 (수업 참여)":
    
    # [A] 튜토리얼 (연습 모드) - (기존과 동일)
    if not st.session_state.tutorial_complete:
        st.header("🎒 연습 시간: 쭈니봇과 친해지기")
        # ... (튜토리얼 코드는 길이상 생략, 기존 코드 그대로 유지됨) ...
        # 실제 실행시에는 이 부분에 이전 답변의 튜토리얼 코드가 그대로 들어갑니다.
        # 튜토리얼 1단계: 버튼 누르기 연습
        if st.session_state.tutorial_step == 0:
            st.markdown("""
            ### 1단계: 버튼 누르기 연습
            **쭈니봇:** 안녕? 나는 쭈니봇이야! 너는 어떤 계절을 더 좋아해? 
            아래 버튼 중 하나를 꾹 눌러봐!
            """)
            col1, col2 = st.columns(2)
            if col1.button("🅰️ 더운 여름이 좋아! 🍦"):
                st.toast("잘했어! 버튼은 이렇게 누르는 거야.")
                st.session_state.tutorial_step = 1
                st.rerun()
            if col2.button("🅱️ 추운 겨울이 좋아! ☃️"):
                st.toast("완벽해! 버튼은 이렇게 누르는 거야.")
                st.session_state.tutorial_step = 1
                st.rerun()

        # 튜토리얼 2단계: 채팅 입력 연습
        elif st.session_state.tutorial_step == 1:
            st.markdown("""
            ### 2단계: 채팅하기 연습
            **쭈니봇:** 버튼 누르기 성공! 👍
            이번에는 나한테 인사를 건네줄래?
            아래 입력창에 **"안녕"**이나 **"반가워"**라고 쓰고 엔터(Enter)를 쳐봐!
            """)
            
            if user_input := st.chat_input("여기에 인사를 적어봐!"):
                st.balloons()
                st.session_state.tutorial_step = 2
                st.rerun()

        # 튜토리얼 완료
        elif st.session_state.tutorial_step == 2:
            st.success("🎉 연습 끝! 이제 진짜 수업을 시작할 준비가 다 됐어.")
            if st.button("🚀 수업 시작하러 가기!"):
                st.session_state.tutorial_complete = True
                st.rerun()


    # [B] 본 수업 (Main Scenario)
    else:
        st.header(f"🙋‍♂️ 쭈니봇과 토론하기: {st.session_state.topic if st.session_state.topic else '주제 미정'}")

        if not st.session_state.scenario:
            st.warning("⚠️ 선생님이 아직 수업을 만들지 않았어요. 조금만 기다려주세요!")
        else:
            # 상단: 튜토리얼 다시하기 버튼
            if st.button("🔄 사용법 다시 연습하기", type="secondary"):
                st.session_state.tutorial_complete = False
                st.session_state.tutorial_step = 0
                st.rerun()

            # 현재 단계 데이터 가져오기
            current_idx = st.session_state.current_step
            step_data = st.session_state.scenario[current_idx]
            step_image = st.session_state.scenario_images[current_idx] # 해당 단계 이미지 URL
            
            st.markdown(f"### 📖 이야기 Part {current_idx + 1}")

            # [NEW ✨] 이미지가 있으면 표시
            if step_image:
                st.image(step_image, use_column_width=True)

            # 1. 이야기 텍스트 보여주기
            st.write(step_data['story'])
            
            # 2. 선택지 버튼 (A / B)
            col1, col2 = st.columns(2)
            if col1.button(f"🅰️ {step_data['a']}", use_container_width=True):
                st.session_state.chat_log.append({"role": "user", "content": f"선택: {step_data['a']}"})
                st.session_state.chat_log.append({"role": "assistant", "content": "그 선택을 했구나! 왜 그렇게 생각했는지 이유를 말해줄래?"})
                st.rerun()

            if col2.button(f"🅱️ {step_data['b']}", use_container_width=True):
                st.session_state.chat_log.append({"role": "user", "content": f"선택: {step_data['b']}"})
                st.session_state.chat_log.append({"role": "assistant", "content": "흥미로운 선택이야! 왜 그런 결정을 내렸어?"})
                st.rerun()

            st.markdown("---")

            # 3. 채팅창 (자유 의견 입력)
            # 이전 대화 기록 표시
            for msg in st.session_state.chat_log:
                role = "쭈니봇" if msg["role"] == "assistant" else "나"
                avatar = "🤖" if msg["role"] == "assistant" else "🙋"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(f"**{role}:** {msg['content']}")

            # 의견 입력창 활성화
            if user_input := st.chat_input("여기에 내 생각을 자유롭게 적어봐!"):
                # 내 의견 표시
                st.session_state.chat_log.append({"role": "user", "content": user_input})
                with st.chat_message("user", avatar="🙋"):
                    st.write(user_input)

                # 쭈니봇의 답변 생성 (GPT)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("쭈니봇이 생각 중..."):
                        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_log])
                        reply = analyze_and_reply(history_text, user_input)
                        
                        st.write(reply)
                        st.session_state.chat_log.append({"role": "assistant", "content": reply})

            # 4. 다음 단계로 넘어가기 버튼
            if len(st.session_state.chat_log) > 2: 
                if st.button("다음 이야기로 넘어가기 ➡️"):
                    if st.session_state.current_step < 3:
                        st.session_state.current_step += 1
                        st.session_state.chat_log = [] 
                        st.rerun()
                    else:
                        st.balloons()
                        st.success("모든 토론이 끝났어! 정말 멋진 생각들이었어. 👏👏")
