"""
Quick integration checks for LangSmith tracing and Tavily search.

Run from project root:
    python lessons/lesson03-agent/connection_checks.py
"""

from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load .env from project root
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)


def print_section(title: str) -> None:
    """Print a simple section header."""
    print(f"--- {title} ---")


def check_langsmith_tracing() -> None:
    """
    Run a tiny LLM call.
    If LangSmith tracing is configured correctly, the run should appear in LangSmith.
    """
    print_section("LangSmith tracing")

    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    api_key = os.getenv("LANGSMITH_API_KEY")
    project_name = os.getenv("LANGSMITH_PROJECT", "default")

    if not api_key or api_key == "your-langsmith-api-key":
        print("Skipped: no valid LANGSMITH_API_KEY found in .env\n")
        return

    if not tracing_enabled:
        print("Skipped: LANGSMITH_TRACING is not set to 'true'\n")
        return

    try:


        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

        response = llm.invoke([
            HumanMessage(content="Say 'tracing works' in exactly 3 words.")
        ])

        print(f"LLM replied: {response.content}")
        print(f"Check runs at: https://smith.langchain.com (project: {project_name})")
        print("OK\n")

    except Exception as exc:
        print(f"Error while testing LangSmith tracing: {exc}\n")


def check_tavily_search() -> None:
    """
    Run one Tavily search to verify TAVILY_API_KEY and the tool integration.
    """
    print_section("Tavily search")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your-tavily-api-key":
        print("Skipped: no valid TAVILY_API_KEY found in .env\n")
        return

    try:
        tool = TavilySearch(max_results=2)
        query = "What is LangChain in one sentence?"
        results = tool.invoke(query)

        print(f"Query: {query}")

        if isinstance(results, list):
            print(f"Got {len(results)} result(s).")
            if results:
                first = results[0]
                snippet = first.get("content") or first.get("raw_content") or str(first)
                print(f"First snippet: {snippet[:200]}...")
        else:
            print(f"Result: {str(results)[:200]}...")

        print("OK\n")

    except Exception as exc:
        print(f"Error while testing Tavily search: {exc}\n")


def main() -> None:
    """Run all connectivity checks."""
    check_langsmith_tracing()
    check_tavily_search()


if __name__ == "__main__":
    main()