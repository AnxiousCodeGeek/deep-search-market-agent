import streamlit as st
import asyncio
from marketagent import Runner, FinancialAgent, session  # import from your backend file

st.set_page_config(page_title="Deep Search Market Agent", page_icon="📈", layout="wide")

st.title("📊 Deep Search Market Agent")
st.write("Ask questions about stocks, markets, or investing concepts.")

# Input box
user_query = st.text_input("Enter your query:", placeholder="e.g., What is the latest on Nvidia stock?")

# Button
if st.button("Run Agent") and user_query:
    with st.spinner("Fetching response... ⏳"):
        async def run_agent():
            result = await Runner.run(FinancialAgent, user_query, session=session)
            return result.final_output

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(run_agent())

    st.subheader("💡 Agent Response")
    st.write(response)
