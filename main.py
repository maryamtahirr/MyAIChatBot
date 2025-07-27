import streamlit as st
import asyncio
import nest_asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

# Load environment variables from .env file
load_dotenv()

# Apply nested asyncio compatibility (required by Streamlit)
nest_asyncio.apply()
set_tracing_disabled(True)

# Streamlit UI config
st.set_page_config(page_title="Gemini Agent Chat", page_icon="🤖")
st.title("🤖 Maryam AI Chat")

# === Configuration ===
API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-2.0-flash"

# Check API key
if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# Initialize session state
if "agent" not in st.session_state:
    try:
        client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
        st.session_state.agent = Agent(
            name="Assistant",
            instructions="You are an expert of agentic AI.",
            model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        )
        st.success("✅ Agent initialized automatically.")
    except Exception as e:
        st.session_state.agent = None
        st.error(f"❌ Failed to initialize agent: {e}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Chat Interface ===
if st.session_state.agent:
    # Display chat history
    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender.lower()):
            st.markdown(message)

    # Chat input field
    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.chat_history.append(("User", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Async call to Gemini agent
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
    st.warning("⚠️ Agent is not initialized. Please check your API key or internet connection.")
