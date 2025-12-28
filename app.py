elif mode == "🙋‍♂️ 학생용":
    # 튜토리얼 상태 초기화
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1

    # 튜토리얼이 아직 안 끝났다면 튜토리얼 화면 표시
    if not st.session_state.tutorial_done:
        st.header("🎒 수업 준비 운동 (튜토리얼)")
        st.write("본격적인 수업 전에 사용 방법을 먼저 익혀볼까요?")
        
        # 진행률 표시
        st.progress(st.session_state.tutorial_step / 3)

        # --- [1단계] 선택지 고르기 연습 ---
        if st.session_state.tutorial_step == 1:
            st.subheader("Mission 1. 선택하기 연습 👈")
            st.info("선생님이 질문을 하면, 너의 생각을 골라야 해. 아래에서 가장 좋아하는 간식을 골라볼까?")
            
            # 실제 수업과 똑같은 UI 사용 (Radio Button)
            snack = st.radio("가장 좋아하는 간식은?", ["달콤한 초콜릿 🍫", "바삭한 과자 🍪", "시원한 아이스크림 🍦"])
            
            if st.button("선택 완료! (다음으로)"):
                st.success(f"와! {snack}을(를) 좋아하는구나! 아주 잘 골랐어.")
                st.session_state.tutorial_step = 2
                st.rerun()

        # --- [2단계] 글쓰기 연습 ---
        elif st.session_state.tutorial_step == 2:
            st.subheader("Mission 2. 글쓰기 연습 ✍️")
            st.info("선택을 했으면 이유를 적어야겠지? 키보드로 너의 생각을 적는 연습을 해보자.")
            
            t_input = st.text_area("오늘 기분이 어떤지 적어주세요! (예: 날씨가 좋아서 신나!)")
            
            if st.button("다 썼어요! (다음으로)"):
                if len(t_input) > 2:  # 최소 2글자 이상 입력 확인
                    st.success("멋진 문장이야! 글쓰기 실력이 대단한걸?")
                    st.session_state.tutorial_step = 3
                    st.rerun()
                else:
                    st.warning("너무 짧아요! 조금만 더 길게 써볼까?")

        # --- [3단계] 프롬프트로 이미지 생성 연습 ---
        elif st.session_state.tutorial_step == 3:
            st.subheader("Mission 3. AI 화가와 그림 그리기 🎨")
            st.info("내가 상상한 장면을 글로 설명하면(프롬프트), AI가 그림을 그려줘. 한번 해볼까?")
            
            prompt_input = st.text_input("그리고 싶은 것을 설명해줘 (예: 우주복을 입은 귀여운 강아지)")
            
            if st.button("그림 생성하기 ✨"):
                if prompt_input:
                    with st.spinner("AI 화가가 붓을 들고 그림을 그리고 있어요..."):
                        # 실제 이미지 생성 함수 호출
                        img_url = generate_image(prompt_input)
                        
                        if img_url:
                            st.image(img_url, caption="네가 주문한 그림이야! 정말 멋진데?")
                            st.balloons()  # 축하 효과
                            st.success("모든 준비 운동 끝! 이제 진짜 수업으로 가보자.")
                            
                            # 튜토리얼 종료 버튼
                            if st.button("수업 입장하기 🚀"):
                                st.session_state.tutorial_done = True
                                st.rerun()
                        else:
                            st.error("앗, 그림을 그리는 도중에 실수가 있었어. 다시 한번 눌러볼래?")
                else:
                    st.warning("어떤 그림을 그릴지 먼저 적어줘야 해!")

    # 튜토리얼 완료 후 -> 실제 수업 화면 (이전 코드의 로직과 연결)
    else:
        # 선생님이 시나리오를 아직 안 짰을 경우 대기 화면
        if not st.session_state.scenario.get('scenario'):
            st.header("🏫 교실 대기 중...")
            st.image("https://media.giphy.com/media/l0HlBO7eyxdzTZtSS/giphy.gif", width=300) # 귀여운 대기 이미지(예시)
            st.info("선생님이 아직 수업 내용을 만들고 계셔! 잠시만 기다려줘. (새로고침하며 대기)")
            if st.button("새로고침 🔄"):
                st.rerun()
        else:
            # 여기에 아까 작성해드린 [실제 수업 로직]이 들어갑니다.
            # (이전에 작성해드린 코드의 'else' 블록 내부 내용을 여기에 붙여넣으세요)
            pass
