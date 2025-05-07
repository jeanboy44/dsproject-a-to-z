import asyncio

import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Agent 초기화
ollama_model = OpenAIModel(
    model_name="llama3.2", provider=OpenAIProvider(base_url="http://localhost:11434/v1")
)
agent = Agent(ollama_model)

st.title("Chat with Llama3.2")


async def run_agent_with_streaming(user_input: str):
    """Agent를 스트리밍 모드로 실행하고 결과를 표시합니다.

    Args:
        user_input (str): 사용자 입력
    """
    async with agent.run_stream(user_input) as result:
        partial_text = ""
        message_placeholder = st.empty()
        # 텍스트 스트리밍
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # 전체 응답을 세션 상태에 저장
        st.session_state.messages_llama.append(
            {"role": "assistant", "content": partial_text}
        )


async def main():
    """메인 애플리케이션 로직을 실행합니다."""
    # 채팅 기록 초기화
    if "messages_llama" not in st.session_state:
        st.session_state.messages_llama = []

    # 앱 재실행 시 채팅 기록에서 메시지 표시
    for message in st.session_state.messages_llama:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력에 반응
    if prompt := st.chat_input("Ask me anything!"):
        # 채팅 메시지 컨테이너에 사용자 메시지 표시
        st.chat_message("user").markdown(prompt)
        # 채팅 기록에 사용자 메시지 추가
        st.session_state.messages_llama.append({"role": "user", "content": prompt})

        # 어시스턴트 응답 받기
        with st.chat_message("assistant"):
            # 스트리밍으로 Agent 실행
            await run_agent_with_streaming(prompt)


if __name__ == "__main__":
    asyncio.run(main())
