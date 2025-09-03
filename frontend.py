import streamlit as st
import asyncio
from marketagent import Runner, FinancialAgent, session  # import from your backend file

st.set_page_config(page_title="Deep Search Market Agent", page_icon="📈", layout="wide")


# ---------- STATE MANAGEMENT ----------
if "sessions" not in st.session_state:
    st.session_state.sessions = {"Session 1": []}   # {session_name: [(query, response), ...]}
if "active_session" not in st.session_state:
    st.session_state.active_session = "Session 1"


# ---------- SIDEBAR ----------
st.sidebar.title("💬 Conversations")

# New Chat button
if st.sidebar.button("➕ New Chat"):
    new_name = f"Session {len(st.session_state.sessions) + 1}"
    st.session_state.sessions[new_name] = []
    st.session_state.active_session = new_name

# Show sessions
for name in st.session_state.sessions.keys():
    if st.sidebar.button(name, key=name):
        st.session_state.active_session = name

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Chats"):
    st.session_state.sessions = {"Session 1": []}
    st.session_state.active_session = "Session 1"


# ---------- MAIN AREA ----------
st.title("📊 StoxFinlytics")
st.subheader("Deep Search Market Agent")
st.caption(f"Currently viewing: **{st.session_state.active_session}**")

# Display chat for active session
for q, r in st.session_state.sessions[st.session_state.active_session]:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(r)

# Input box (ENTER works automatically with st.chat_input)
user_query = st.chat_input("Ask about stocks, markets, or investing concepts...")

if user_query:
    # Show user msg immediately
    with st.chat_message("user"):
        st.write(user_query)

    # Run the backend agent
    with st.spinner("Fetching response... ⏳"):
        async def run_agent():
            result = await Runner.run(FinancialAgent, user_query, session=session)
            return result.final_output

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(run_agent())

    # Save into active session
    st.session_state.sessions[st.session_state.active_session].append((user_query, response))

    # Show assistant response
    with st.chat_message("assistant"):
        st.write(response)