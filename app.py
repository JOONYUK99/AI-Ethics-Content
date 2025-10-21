import streamlit as st
import google.generativeai as genai
import re
import urllib.parse

# --- 1. AI 핵심 기능 함수 정의 ---
def get_model():
    return genai.GenerativeModel('gemini-pro-latest')

# <--- RAG 기능을 위한 '지식 창고' (Knowledge Base) 정의 --->
KNOWLEDGE_BASE = """
사건명: 스캐터랩 '이루다' AI 챗봇 개인정보 유출 사건

요약: AI 챗봇 개발사인 '스캐터랩'은 '연애의 과학'이라는 다른 앱 사용자들의 카카오톡 대화 약 100억 건을 수집했습니다. 이 과정에서 사용자들에게 AI 챗봇 개발에 데이터가 사용된다는 사실을 명확히 알리지 않고 동의를 받지 않았습니다. 이렇게 모인 대화 내용은 개인정보를 제대로 지우지 않은 상태로 '이루다' 챗봇 모델 학습에 사용되었습니다. 그 결과, 챗봇 '이루다'가 대화 중에 실제 사람의 이름, 주소, 은행 이름 같은 개인정보를 그대로 말하는 심각한 문제가 발생했습니다.
"""

def transform_scenario(teacher_input):
    model = get_model()
    prompt = f"당신은 초등학생을 위한 AI 윤리 교육용 시나리오 작가입니다. 아래 '실제 사례'를 바탕으로, 학생들이 총 4번의 선택을 하게 되는 완결된 이야기를 만들어주세요. 각 파트는 3문장 이하로 짧게 구성하고, 끝에는 두 가지 선택지를 포함해주세요. # 필수 출력 형식: [STORY 1]...[CHOICE 1A]...[CHOICE 1B]---[STORY 2]...\n\n--- 실제 사례 ---\n{teacher_input}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e: return f"시나리오 생성 중 오류: {e}"

def start_debate(history, choice):
    # ... (이전과 동일)
    model = get_model()
    prompt = f"당신은 다정한 AI 선생님입니다. 학생이 방금 '{choice}'라고 선택했습니다. 그 선택을 칭찬하고, 왜 그렇게 생각했는지 부드럽게 첫 질문을 던져주세요."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"토론 시작 중 오류: {e}"

def continue_debate(debate_history):
    # ... (이전과 동일)
    model = get_model()
    prompt = f"당신은 다정한 AI 선생님입니다. 학생의 이전 답변에 공감하며, '혹시 이런 점은 어떨까요?' 와 같이 부드러운 말투로 반대 관점을 제시하는 질문을 던져주세요.\n\n--- 토론 내용 ---\n{debate_history}\n\nAI 선생님의 다음 질문:"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"토론 중 오류: {e}"

def generate_conclusion(final_history):
    # ... (이전과 동일)
    model = get_model()
    prompt = f"다음은 학생의 전체 토론 기록입니다. 과정을 칭찬하고, 현실적인 대처법을 제안하며 따뜻하게 격려하는 마무리 메시지를 작성해주세요.\n\n--- 전체 기록 ---\n{final_history}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"결론 생성 중 오류: {e}"


# <--- RAG 데모를 위한 새로운 AI 함수들 --->
def generate_normal_answer(question):
    """RAG 없이, AI의 일반 지식으로만 답변하는 함수 (환각 가능성)"""
    model = get_model()
    prompt = f"당신은 초등학생의 질문에 답하는 AI 선생님입니다. 다음 질문에 대해 아는 대로 최대한 자세히 설명해주세요.\n\n질문: {question}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e: return f"답변 생성 중 오류: {e}"

def generate_rag_answer(question, context):
    """RAG 기술을 적용하여, 주어진 '참고 자료'를 바탕으로만 답변하는 함수"""
    model = get_model()
    prompt = (
        "당신은 AI 선생님입니다. 아래 '참고 자료'의 내용만을 사용하여 학생의 '질문'에 대해 답변해주세요. "
        "참고 자료에 없는 내용은 절대 지어내면 안 됩니다.\n\n"
        f"--- 참고 자료 ---\n{context}\n\n"
        f"--- 질문 ---\n{question}\n\n"
        "답변:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e: return f"RAG 답변 생성 중 오류: {e}"

# --- 2. Streamlit 앱 UI 및 로직 ---
st.set_page_config(page_title="AI 윤리 교육 챗봇", page_icon="✨", layout="centered")
st.title("✨ 초등학생을 위한 AI 윤리 교육")

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("API 키를 설정해주세요!")
    st.stop()

if 'stage' not in st.session_state:
    st.session_state.stage = 'start'
    # ... (다른 상태 변수들)

def restart_lesson():
    # ... (이전과 동일)
    st.session_state.stage = 'start'
    # ...

# --- UI 렌더링 로직 (대부분 동일, RAG 데모 스테이지 추가) ---

if st.session_state.stage == 'start':
    # ... (교사 입력 부분은 이전과 동일)
    st.info("AI 윤리 교육 콘텐츠로 만들고 싶은 실제 사례, 뉴스 기사 등을 아래에 입력해주세요.")
    teacher_text = st.text_area("시나리오 입력:", height=150, placeholder="예시: 개발사가 이용자의 명시적 동의 없이 사적인 카카오톡 대화 데이터를 챗봇 학습에 무단으로 사용해 개인정보가 유출됐다.")
    if st.button("이 내용으로 교육 콘텐츠 생성하기"):
        if not teacher_text: st.warning("시나리오를 입력해주세요.")
        else:
            st.session_state.teacher_input = teacher_text
            st.session_state.stage = 'story_generation'
            st.rerun()

elif st.session_state.stage == 'story_generation':
    # ... (이전과 동일)
    pass
    
elif st.session_state.stage == 'story':
    # ... (이전과 동일)
    pass

elif st.session_state.stage == 'debate':
    # ... (이전과 동일)
    pass

elif st.session_state.stage == 'conclusion':
    # ... (이전과 동일, RAG 데모 버튼 추가)
    st.markdown(st.session_state.full_log, unsafe_allow_html=True)
    with st.spinner("AI 선생님이 우리의 멋진 여정을 정리하고 있어요..."):
        conclusion_text = generate_conclusion(st.session_state.full_log)
        st.balloons(); st.success("모든 이야기가 끝났어요! 정말 수고 많았어요!")
        st.markdown("---"); st.markdown("### 최종 정리"); st.write(conclusion_text)
    
    st.markdown("---")
    # <--- RAG 데모 시작 버튼 --->
    if st.button("🔬 RAG 기술 효과 확인하기"):
        st.session_state.stage = 'rag_demo'
        st.rerun()
        
    if st.button("새로운 주제로 다시 시작하기"):
        restart_lesson(); st.rerun()

# <--- RAG 데모를 위한 새로운 UI 스테이지 --->
elif st.session_state.stage == 'rag_demo':
    st.info("RAG(검색 증강 생성) 기술은 AI가 부정확한 정보를 지어내는 '환각' 현상을 방지하고, 검증된 사실만을 바탕으로 답변하도록 돕는 중요한 기술입니다.")
    
    st.markdown("#### RAG 효과 비교 테스트")
    rag_question = st.text_input("수업 내용과 관련된 사실에 대해 질문해보세요:", placeholder="예: 그 사건을 일으킨 회사 이름이 뭐예요?")

    if rag_question:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.warning("RAG 적용 전 (환각 가능성 O)")
            with st.spinner("AI가 자신의 지식으로 답변을 생성 중..."):
                normal_answer = generate_normal_answer(rag_question)
                st.write(normal_answer)
        
        with col2:
            st.success("RAG 적용 후 (사실 기반)")
            with st.spinner("AI가 '지식 창고'를 검색하여 답변을 생성 중..."):
                # 실제 구현에서는 질문과 관련된 부분을 검색하는 로직이 필요하지만,
                # 데모에서는 전체 지식 창고를 컨텍스트로 제공합니다.
                rag_answer = generate_rag_answer(rag_question, KNOWLEDGE_BASE)
                st.write(rag_answer)

    if st.button("처음으로 돌아가기"):
        restart_lesson(); st.rerun()
