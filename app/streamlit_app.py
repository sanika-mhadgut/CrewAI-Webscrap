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
# Optional: capture model chain-of-thought in Langfuse only (never shown to users)
CAPTURE_COT = os.getenv("LANGFUSE_CAPTURE_COT", "false").lower() in ("1", "true", "yes", "on")

import os

print("Langfuse Host:", os.getenv("LANGFUSE_HOST"))
print("Public Key:", os.getenv("LANGFUSE_PUBLIC_KEY"))
print("Secret Key exists:", bool(os.getenv("LANGFUSE_SECRET_KEY")))

# --------------------- AGENT CLASS ---------------------
class WebScraperCrewAgent:
    """Scrapes public web pages and summarizes content using OpenAI, logs to Langfuse."""

    def __init__(self, model: str | None = None):
        api_key = os.getenv(OPENAI_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"Missing {OPENAI_API_KEY_ENV}. Set it in environment or sidebar.")

        # OpenAI setup
        self.client = OpenAI(api_key=api_key)
        self.model = model or DEFAULT_MODEL

        # Initialize Langfuse (new API)
        try:
            self.langfuse = Langfuse()
            print("✅ Langfuse initialized successfully")

            # Minimal test span to verify connectivity
            with self.langfuse.start_as_current_span(name="startup-test-span", metadata={"phase": "startup"}):
                print("🌐 Testing Langfuse connectivity...")

            self.langfuse.flush()
            print("✅ Langfuse startup test span sent")
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

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            paragraphs = [p.get_text(" ").strip() for p in soup.find_all("p")]
            text = " ".join(p for p in paragraphs if p)
            return text[:8000]
        except Exception as exc:
            return f"Error scraping website: {exc}"

    def summarize_content(self, text: str, query: str, url: str | None = None) -> str:
        """Summarize content using OpenAI and log the generation to Langfuse.

        If LANGFUSE_CAPTURE_COT is enabled, requests JSON output with
        fields {"reasoning", "answer"}. Only "answer" is shown to users;
        the "reasoning" is recorded in Langfuse under a dedicated span.
        """
        system_prompt = (
            "You are a precise web analyst. Use only the provided content. "
            "If information is missing, say so clearly. Keep answers concise."
        )

        if CAPTURE_COT:
            user_prompt = (
                f"Below is content scraped from a public website.\n\n"
                f"Content:\n{text}\n\n"
                f"User question:\n{query}\n\n"
                "Return a compact JSON object with keys 'reasoning' and 'answer'. "
                "Keep 'reasoning' brief and high-level; avoid sensitive data."
            )
        else:
            user_prompt = (
                f"Below is content scraped from a public website.\n\n"
                f"Content:\n{text}\n\n"
                f"User question:\n{query}\n\n"
                "Provide a clear, factual, carefully structured answer."
            )

        def _call_openai_and_parse() -> tuple[str, str | None]:
            """Returns (answer, reasoning_or_none)."""
            if CAPTURE_COT:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or ""
                try:
                    obj = json.loads(content)
                    return str(obj.get("answer", "")), obj.get("reasoning")
                except Exception:
                    # Fallback if JSON parsing fails: treat full content as answer
                    return content, None
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                return resp.choices[0].message.content, None

        if self.langfuse:
            with self.langfuse.start_as_current_generation(
                name="summarize_content",
                input={"system": system_prompt, "user": user_prompt},
                metadata={"url": url, "capture_cot": CAPTURE_COT},
                model=self.model,
            ) as gen:
                answer, reasoning = _call_openai_and_parse()
                gen.output = answer
                if CAPTURE_COT and reasoning:
                    # Store reasoning in a dedicated span to avoid mixing with user-visible output
                    with self.langfuse.start_as_current_span(
                        name="chain_of_thought",
                        metadata={"reasoning": reasoning},
                    ):
                        pass
                return answer
        else:
            answer, _ = _call_openai_and_parse()
            return answer

    def respond(self, user_input: str) -> Tuple[str, str, str]:
        """Main workflow: extract URL, scrape, summarize, and log to Langfuse."""
        url = self._extract_first_url(user_input)
        if not url:
            return ("", "", "Please provide a valid website URL in your query.")

        scraped, summary = "", ""
        if self.langfuse:
            # Root trace (span)
            with self.langfuse.start_as_current_span(
                name="web-scraper-respond",
                metadata={"query": user_input, "url": url},
            ):
                # Scrape website
                with self.langfuse.start_as_current_span(
                    name="scrape_website",
                    metadata={"url": url},
                ):
                    scraped = self.scrape_website(url)
                    print("✅ Logged scrape_website span to Langfuse")

                # Summarize
                summary = self.summarize_content(scraped, user_input, url=url)
                print("✅ Logged summarize_content generation to Langfuse")

            self.langfuse.flush()
        else:
            scraped = self.scrape_website(url)
            summary = self.summarize_content(scraped, user_input, url=url)

        return (url, scraped, summary)


# --------------------- STREAMLIT FRONTEND ---------------------
st.set_page_config(page_title="CrewAI Web Scraper", page_icon="🤖", layout="wide")
st.title("🤖 CrewAI Web Scraper & Summarizer")
st.caption("Enter a URL and a question. The agent will scrape and summarize content.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )
    model = st.text_input(
        "Model",
        value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
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
        **Tips:**
        - Provide a full URL beginning with https://  
        - Add a specific question to focus the summary  
        - Content is truncated to avoid exceeding token limits
        """
    )
)
