import os
import re
from typing import Tuple

import requests
from bs4 import BeautifulSoup
from openai import OpenAI


OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class WebScraperCrewAgent:
    """Scrapes public web pages and summarizes content using OpenAI."""

    def __init__(self, model: str | None = None):
        api_key = os.getenv(OPENAI_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Missing {OPENAI_API_KEY_ENV}. Set it in environment or .env file."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model or DEFAULT_MODEL

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

            # Remove script/style/nav/footer to reduce noise
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            paragraphs = [p.get_text(" ").strip() for p in soup.find_all("p")]
            text = " ".join(p for p in paragraphs if p)
            # Hard cap to avoid overlong prompts
            return text[:8000]
        except Exception as exc:
            return f"Error scraping website: {exc}"

    def summarize_content(self, text: str, query: str) -> str:
        system_prompt = (
            "You are a precise web analyst. Use only the provided content. "
            "If information is missing, say so clearly. Keep answers concise."
        )
        user_prompt = (
            "Below is content scraped from a public website.\n\n"
            f"Content:\n{text}\n\n"
            f"User question:\n{query}\n\n"
            "Provide a clear, factual, carefully structured answer."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def respond(self, user_input: str) -> Tuple[str, str, str]:
        """
        Process user input, scrape, and summarize.
        Returns: (url, scraped_text, summary)
        """
        url = self._extract_first_url(user_input)
        if not url:
            return ("", "", "Please provide a valid website URL in your query.")

        scraped = self.scrape_website(url)
        summary = self.summarize_content(scraped, user_input)
        return (url, scraped, summary)
