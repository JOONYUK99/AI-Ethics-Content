import streamlit as st
from openai import OpenAI
import re
import json

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="AI 토론 및 창작 시스템", page_icon="🎨", layout="wide")

# --- 2. OpenAI 클라이언트 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("⚠️ API 키 오류: .streamlit/secrets.toml 파일을 확인하세요.")
    st.stop()

# --- 3. 시스템 페르소나 (단답형/건조한 말투) ---
SYSTEM_PERSONA = """
당신은 AI 튜터입니다.
학생의 입력을 분석하고 피드백을 주세요.
말투 지침:
1. 감정을 배제하고 건조하게 말하세요.
2. '안녕', '반가워' 같은 인사말 금지.
3. '~단다', '~해요' 금지. '~다', '~가?', '~함' 등의 단답형 종결어미 사용.
4. 핵심만 1~2문장으로 짧게 요약하세요.
"""

# --- 4. 주요 함수 ---

def ask_gpt_json(prompt):
    """JSON 응답 요청 (오류 발생 시 빈 구조 반환)"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PERSONA},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        data = json.loads(response.choices[0].message.content.strip())
        # 반환된 데이터에 필수 키가 있는지 확인
        if "scenario" not in data:
            return {"scenario": []}
        return data
    except Exception:
        return {"scenario": []}

def ask_gpt_text(prompt):
    """텍스트 응답 요청"""
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
    except Exception:
        return "응답 불가."

def generate_image(prompt):
    """이미지 생성"""
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"Simple cartoon illustration, minimal style: {prompt}",
            size="1024x1024",
            n=1
        )
        return response.data[0].url
    except Exception:
        return None

# --- 5. 세션 상태 안전한 초기화 ---
if 'scenario' not in st.session_state: st.session_state.scenario = {"scenario": []}
if 'analysis' not in st.session_state: st.session_state.analysis = ""
if 'current_step' not in st.session_state: st.session_state.current_step = 0
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'topic' not in st.session_state: st.session_state.topic = ""
if 'tutorial_done' not in st.session_state: st.session_state.tutorial_done = False
if 'tutorial_step' not in st.session_state: st.session_state.tutorial_step = 1

# --- 6. 사이드바 ---
st.sidebar.title("🏫 AI 학습 시스템")
mode = st.sidebar.radio("모드", ["👨‍🏫 교사용", "🙋‍♂️ 학생용"])

# --- 7. 메인 로직 ---

# [교사용 모드]
if mode == "👨‍🏫 교사용":
    st.header("🛠️ 수업 생성")
    input_topic = st.text_input("주제", value=st.session_state.topic)
    
    if st.button("생성 시작"):
        if not input_topic:
            st.warning("주제를 입력하세요.")
        else:
            with st.spinner("생성 중..."):
                # 시나리오 생성
                s_prompt = f"주제 '{input_topic}'의 3단계 딜레마 시나리오. JSON 포맷: {{ 'scenario': [ {{ 'story': '...', 'choice_a': '...', 'choice_b': '...' }} ] }}"
                result = ask_gpt_json(s_prompt)
                
                # 데이터 유효성 검사 (빈 데이터 방지)
                if result and 'scenario' in result:
                    st.session_state.scenario = result
                else:
                    st.session_state.scenario = {"scenario": []}
                
                # 분석 생성
                a_prompt = f"주제 '{input_topic}'의 [핵심가치], [교과], [목표]를 단답형 명사로 요약."
                st.session_state.analysis = ask_gpt_text(a_prompt)
                
                # 상태 업데이트
                st.session_state.topic = input_topic
                st.session_state.current_step = 0
                
                # 기존 이미지 캐시 초기화
                keys_to_delete = [k for k in st.session_state.keys() if k.startswith("img_url_")]
                for k in keys_to_delete:
                    del st.session_state[k]
                    
                st.success("완료.")

    # 미리보기 (KeyError 방지 코드 적용됨)
    if st.session_state.analysis:
        st.divider()
        st.subheader("분석 결과")
        st.write(st.session_state.analysis)

    # [수정] 안전하게 접근: .get() 사용 및 리스트 여부 확인
    scenarios = st.session_state.scenario.get('scenario', [])
    if scenarios:
        with st.expander("시나리오 목록"):
            st.table(scenarios)

# [학생용 모드]
elif mode == "🙋‍♂️ 학생용":
    
    # 튜토리얼
    if not st.session_state.tutorial_done:
        st.header("🎒 연습 모드")
        st.progress(st.session_state.tutorial_step / 3)

        if st.session_state.tutorial_step == 1:
            st.subheader("1. 선택")
            if st.button("선택 A: 초콜릿"):
                st.toast("선택: 초콜릿")
                st.session_state.tutorial_step = 2
                st.rerun()
            if st.button("선택 B: 사탕"):
                st.toast("선택: 사탕")
                st.session_state.tutorial_step = 2
                st.rerun()

        elif st.session_state.tutorial_step == 2:
            st.subheader("2. 입력")
            t_input = st.text_input("입력창")
            if st.button("전송"):
                if t_input:
                    st.toast("전송 완료")
                    st.session_state.tutorial_step = 3
                    st.rerun()

        elif st.session_state.tutorial_step == 3:
            st.subheader("3. 생성")
            if st.button("이미지 생성 테스트"):
                with st.spinner("생성 중..."):
                    img = generate_image("Robot")
                    if img:
                        st.image(img)
                        if st.button("수업 입장"):
                            st.session_state.tutorial_done = True
                            st.rerun()

    # 실전 수업
    else:
        # [수정] 안전하게 접근: scenario 키가 없거나 비어있으면 빈 리스트 반환
        steps = st.session_state.scenario.get('scenario', [])
        
        if not steps:
            st.warning("수업 데이터가 없습니다. 교사용 탭에서 생성해주세요.")
            if st.button("새로고침"):
                st.rerun()
        
        else:
            idx = st.session_state.current_step
            total = len(steps)
            
            # 인덱스 초과 방지
            if idx >= total:
                st.balloons()
                st.success("수업 끝.")
                if st.button("처음으로"):
                    st.session_state.current_step = 0
                    st.session_state.tutorial_done = False
                    st.session_state.chat_history = []
                    st.rerun()
            else:
                data = steps[idx]
                st.progress((idx + 1) / total)
                st.subheader(f"단계 {idx+1}")

                # 이미지
                img_key = f"img_url_{idx}"
                if img_key not in st.session_state:
                    with st.spinner("이미지 로딩..."):
                        st.session_state[img_key] = generate_image(data.get('story', ''))
                
                if st.session_state.get(img_key):
                    st.image(st.session_state[img_key])

                st.info(data.get('story', ''))

                with st.form(f"form_{idx}"):
                    sel = st.radio("선택", [data.get('choice_a', 'A'), data.get('choice_b', 'B')])
                    reason = st.text_area("이유")
                    if st.form_submit_button("제출"):
                        if reason:
                            prompt = f"상황:{data['story']}, 선택:{sel}, 이유:{reason}. 단답형 피드백 및 질문."
                            with st.spinner("분석..."):
                                res = ask_gpt_text(prompt)
                                st.session_state.chat_history.append({"role": "user", "content": f"{sel}: {reason}"})
                                st.session_state.chat_history.append({"role": "assistant", "content": res})
                        else:
                            st.warning("이유 입력 필요.")

                # 채팅 기록
                if st.session_state.chat_history:
                    st.write("---")
                    for msg in st.session_state.chat_history:
                        role = "assistant" if msg["role"] == "assistant" else "user"
                        st.chat_message(role).write(msg["content"])
                    
                    if st.button("다음"):
                        st.session_state.current_step += 1
                        st.session_state.chat_history = []
                        st.rerun()
