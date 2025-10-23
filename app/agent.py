import os
import json
import re
import textwrap
from typing import Tuple

import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from langfuse import Langfuse


# Constants
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CAPTURE_COT = os.getenv("LANGFUSE_CAPTURE_COT", "false").lower() in ("1", "true", "yes", "on")


class WebScraperCrewAgent:
    """Scrapes public web pages and summarizes content using OpenAI, logs to Langfuse."""

    def __init__(self, model: str | None = None):
        api_key = os.getenv(OPENAI_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"Missing {OPENAI_API_KEY_ENV}. Set it in environment or sidebar.")

        self.client = OpenAI(api_key=api_key)
        self.model = model or DEFAULT_MODEL

        # Initialize Langfuse
        try:
            self.langfuse = Langfuse()
            print("✅ Langfuse initialized successfully")
            self.langfuse.trace(
                name="startup-test-trace",
                input="docker container startup",
                output="langfuse connection verified",
                tags=["init", "connectivity-check"],
            )
            print("✅ Langfuse startup test trace sent")
        except Exception as e:
            print(f"⚠️ Langfuse initialization failed: {e}")
            self.langfuse = None

    def _extract_first_url(self, text: str) -> str | None:
        urls = re.findall(r"https?://\S+", text)
        return urls[0] if urls else None

    def scrape_website(self, url: str, timeout_seconds: int = 15) -> str:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                )
            }
            resp = requests.get(url, timeout=timeout_seconds, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            paragraphs = [p.get_text(" ").strip() for p in soup.find_all("p")]
            text = " ".join(p for p in paragraphs if p)
            return text[:8000]
        except Exception as exc:
            return f"Error scraping website: {exc}"

    def summarize_content(self, text: str, query: str, trace=None, url: str | None = None) -> str:
        system_prompt = (
            "You are a precise web analyst. Use only the provided content. "
            "If information is missing, say so clearly. Keep answers concise but reasoning rich."
        )

        if CAPTURE_COT:
            user_prompt = (
                "Below is content scraped from a public website.\n\n"
                "Content:\n"
                f"{text}\n\n"
                "User question:\n"
                f"{query}\n\n"
                "Return a JSON object with the following keys:\n"
                "  - reasoning: a detailed, step-by-step internal chain-of-thought (several short paragraphs or numbered steps). "
                "    This is for internal telemetry only and should not be shown to the user. Avoid copying private data.\n"
                "  - intermediate_steps: an optional structured breakdown of reasoning steps as an array of objects, e.g.:\n"
                '    [ {"step": 1, "thought": "Identified key topics"}, {"step": 2, "thought": "Summarized events"} ]\n'
                "  - answer: the concise final answer to present to the user (1–3 short paragraphs).\n\n"
                "Make 'reasoning' thorough: include the steps you took to interpret the content, how you prioritized facts, "
                "which parts of the content you used, and any assumptions. Do NOT include credentials or PII.\n"
                "Return only valid JSON. Keep the object compact but allow substantial detail in 'reasoning'."
            )
        else:
            user_prompt = (
                f"Below is content scraped from a public website.\n\n"
                f"Content:\n{text}\n\n"
                f"User question:\n{query}\n\n"
                "Provide a clear, factual, carefully structured answer."
            )

        def _scrub_sensitive(s: str) -> str:
            s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", s)
            s = re.sub(r"\b(sk|pk)-[A-Za-z0-9_\-]+\b", "[REDACTED_KEY]", s)
            return s

        def _call_openai_and_parse() -> tuple[str, str | None, list | None]:
            if CAPTURE_COT:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=2000,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or ""
                try:
                    obj = json.loads(content)
                    answer = str(obj.get("answer", ""))
                    reasoning = obj.get("reasoning")
                    steps = obj.get("intermediate_steps")
                    if isinstance(reasoning, str):
                        reasoning = _scrub_sensitive(reasoning)
                    return answer, reasoning, steps
                except Exception:
                    return content, None, None
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=800,
                )
                return resp.choices[0].message.content, None, None

        # --- Langfuse logging ---
        if self.langfuse and trace is not None:
            try:
                gen = trace.generation(
                    name="summarize_content",
                    model=self.model,
                    input={"system": system_prompt, "user": user_prompt},
                    metadata={"temperature": 0.2, "url": url, "capture_cot": CAPTURE_COT},
                )
            except Exception:
                gen = None

            answer, reasoning, steps = _call_openai_and_parse()

            if gen is not None:
                gen.output = answer

            # Log reasoning (text) and structured steps (JSON array) in separate spans
            if CAPTURE_COT:
                if reasoning:
                    try:
                        trace.span(name="chain_of_thought").end(output={"reasoning": reasoning})
                    except Exception:
                        pass
                if steps:
                    try:
                        trace.span(name="intermediate_steps").end(output={"steps": steps})
                    except Exception:
                        pass

            return answer
        else:
            answer, _, _ = _call_openai_and_parse()
            return answer

    def respond(self, user_input: str) -> Tuple[str, str, str]:
        url = self._extract_first_url(user_input)
        if not url:
            return ("", "", "Please provide a valid website URL in your query.")

        trace = None
        if self.langfuse:
            try:
                trace = self.langfuse.trace(
                    name="web-scraper-respond",
                    input={"query": user_input, "url": url},
                    tags=["web-scraper", "crewai-agent"],
                )
                print("✅ Created Langfuse trace for web-scraper-respond")
            except Exception as e:
                print(f"⚠️ Failed to create Langfuse trace: {e}")
                trace = None

        scraped = self.scrape_website(url)
        if trace:
            try:
                trace.span(name="scrape_website").end(output={"characters": len(scraped)})
                print("✅ Logged scrape_website span to Langfuse")
            except Exception as e:
                print(f"⚠️ Failed to log scrape span: {e}")

        summary = self.summarize_content(scraped, user_input, trace=trace, url=url)
        return (url, scraped, summary)


# ---------------- STREAMLIT FRONTEND ----------------
st.title("🤖 CrewAI Web Scraper & Summarizer")
st.caption("Enter a URL and a question. The agent will scrape and summarize content.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    st.markdown("Note: API key is used only client-side to initialize the agent.")

query = st.text_area(
    "Your prompt (include at least one https:// URL)",
    height=120,
    placeholder="Example: Summarize the key points from https://www.bbc.com/news/technology",
)

if st.button("Run Agent", type="primary"):
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
