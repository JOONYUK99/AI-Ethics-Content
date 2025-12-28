else:
        if not st.session_state.scenario.get('scenario'):
            st.warning("선생님이 아직 수업을 만들지 않았습니다!")
        else:
            steps = st.session_state.scenario['scenario']
            idx = st.session_state.current_step
            total_steps = len(steps)

            # --- 1. 상단 진행 상태 바 (New) ---
            progress_value = (idx + 1) / total_steps
            st.progress(progress_value)
            st.caption(f"현재 진행률: {idx + 1} / {total_steps} 단계")

            if idx < total_steps:
                data = steps[idx]
                st.header(f"🗣️ {st.session_state.topic}")
                st.subheader(f"{idx+1}번째 이야기")

                # --- 2. 상황 이미지 자동 생성 및 표시 (New) ---
                # 세션에 현재 단계의 이미지가 있는지 확인 후 생성
                img_key = f"img_url_{idx}"
                if img_key not in st.session_state:
                    with st.spinner("AI 화가가 상황을 그림으로 설명해주고 있어요..."):
                        # DALL-E에게 상황에 맞는 구체적인 묘사 요청
                        img_url = generate_image(f"Scene for children: {data['story']}")
                        st.session_state[img_key] = img_url
                
                if st.session_state[img_key]:
                    st.image(st.session_state[img_key], use_container_width=True, caption=f"{idx+1}단계 상황 그림")

                # --- 3. 토론 내용 표시 ---
                st.info(data['story'])
                
                choice = st.radio("나의 선택은?", [data['choice_a'], data['choice_b']], key=f"radio_{idx}")
                reason = st.text_area("이유를 말해주세요!", placeholder="왜 그렇게 생각하는지 친구들에게 말하듯이 적어봐요.", key=f"reason_{idx}")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("주장 제출 📩", key=f"sub_{idx}"):
                        if not reason.strip():
                            st.warning("이유를 먼저 적어주세요!")
                        else:
                            f_prompt = f"상황: {data['story']}\n선택: {choice}\n이유: {reason}\n초등학생 수준에 맞춰 따뜻하게 격려하고 논리적인 질문을 던져줘."
                            with st.spinner("AI 튜터가 생각 중..."):
                                response = ask_gpt_text(f_prompt)
                                st.session_state.chat_history.append({"role": "bot", "content": response})
                
                # 대화 내역 출력
                for msg in st.session_state.chat_history:
                    if msg["role"] == "bot":
                        st.chat_message("assistant").write(msg["content"])

                # 다음 단계 버튼
                if st.button("다음 논제로 이동 ➡️", key=f"next_{idx}"):
                    st.session_state.current_step += 1
                    st.session_state.chat_history = []
                    st.rerun()
            else:
                st.balloons()
                st.success("학습을 모두 마쳤습니다! 🎉")
                if st.button("처음으로 돌아가기"):
                    st.session_state.current_step = 0
                    st.session_state.tutorial_done = False
                    # 생성된 이미지 키값들 초기화
                    for key in list(st.session_state.keys()):
                        if "img_url_" in key:
                            del st.session_state[key]
                    st.rerun()
