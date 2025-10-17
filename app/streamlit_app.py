import os
import textwrap
import streamlit as st
from agent import WebScraperCrewAgent

st.set_page_config(page_title="CrewAI Web Scraper", page_icon="🤖", layout="wide")

st.title("🤖 CrewAI Web Scraper & Summarizer")
st.caption("Enter a URL and a question. The agent will scrape and summarize.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    st.markdown("Note: API key is used only client-side to initialize the agent.")

query = st.text_area(
    "Your prompt (include at least one https:// URL)",
    height=120,
    placeholder="Summarize the key points from https://www.bbc.com/news/technology",
)

run = st.button("Run Agent", type="primary")

if run:
    if not api_key:
        st.error("Please provide your OpenAI API key.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_MODEL"] = model

    try:
        agent = WebScraperCrewAgent(model=model)
    except Exception as exc:
        st.error(f"Failed to init agent: {exc}")
        st.stop()

    with st.spinner("Scraping and summarizing..."):
        url, scraped, summary = agent.respond(query)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("URL")
        st.write(url or "(none)")
        st.subheader("Scraped Content (truncated)")
        st.code(scraped or "(no content)", language="markdown")

    with col2:
        st.subheader("Answer")
        st.write(summary)

st.markdown("---")
st.markdown(
    textwrap.dedent(
        """
        Tips:
        - Provide a full URL beginning with https://
        - Add a specific question to focus the summary
        - Content is truncated to avoid exceeding token limits
        """
    )
)
