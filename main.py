import streamlit as st
import asyncio
import nest_asyncio
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

nest_asyncio.apply()
set_tracing_disabled(True)
st.set_page_config(page_title="Gemini Agent Chat", page_icon="🤖")

st.title("🤖 Maryam Ai Chat ")

# Initialize session state variables
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar configuration panel
with st.sidebar:
    st.header("🔧 Configuration")
    api_key = st.text_input("GEMINI API Key", type="password")
    base_url = st.text_input(
        "Base URL",
        value="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model_name = st.text_input("Model", value="gemini-2.0-flash")

    if st.button("Initiate Agent"):
        if not api_key:
            st.error("❌ API key is required.")
        else:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            agent = Agent(
                name="Assistant",
                instructions="You are an expert of agentic AI.",
                model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
            )
            st.session_state.agent = agent
            st.success("✅ Agent initiated successfully!")

# Main Chat UI
if st.session_state.agent:
    # Show previous messages
    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender.lower()):
            st.markdown(message)

    # Input message
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Append user message
        st.session_state.chat_history.append(("User", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate assistant response
        async def run_agent():
            return await Runner.run(st.session_state.agent, user_input)

        try:
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(run_agent())
            st.session_state.chat_history.append(("Assistant", response.final_output))

            with st.chat_message("assistant"):
                st.markdown(response.final_output)

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Agent error: {e}")
else:
    st.warning("⚠️ Please enter your credentials and initiate the agent from the sidebar.")
