"""
PDF Summarizer using LangChain and OpenAI/Ollama.

This script performs hierarchical summarization of a PDF document:
1. Summarize each page
2. Combine page summaries
3. Generate one final summary
"""

from __future__ import annotations

from typing import Literal
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


PDF_PATH = "lessons/Lesson02-langchain-basics/git_github_practical_guide.pdf"
MAX_PAGES = 5
MODEL_NAME = "gpt-4o-mini"

Provider = Literal["openai", "ollama"]


def get_llm(provider: Provider = "ollama"):
    """
    Return the configured LLM.
    """
    if provider == "openai":
        return ChatOpenAI(
            model=MODEL_NAME,
            temperature=0.2,
        )

    if provider == "ollama":
        return ChatOllama(
            model="gemma3:270m",
            temperature=0,
            num_predict=120,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def load_pdf_pages(pdf_path: str, max_pages: int):
    """
    Load PDF pages and limit them to max_pages.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    if not docs:
        raise ValueError("No pages found in the PDF.")

    return docs[:max_pages]


def build_page_summary_chain(llm):
    """
    Chain for summarizing a single page.
    """
    prompt = PromptTemplate.from_template(
        "You are summarizing one page from a PDF.\n"
        "Summarize this page in 3-5 short bullet points.\n\n"
        "Page text:\n{text}"
    )

    return prompt | llm | StrOutputParser()


def build_final_summary_chain(llm):
    """
    Chain for combining multiple page summaries into one final summary.
    """
    prompt = PromptTemplate.from_template(
        "You will receive summaries of multiple pages of a PDF.\n"
        "Combine them into ONE clear final summary in 8-12 bullet points.\n"
        "Keep the output concise and avoid repetition.\n\n"
        "Page summaries:\n{summaries}"
    )

    return prompt | llm | StrOutputParser()


def summarize_pdf(
    pdf_path: str,
    max_pages: int = 5,
    provider: Provider = "ollama",
) -> str:
    """
    Perform hierarchical PDF summarization.
    """
    llm = get_llm(provider=provider)
    docs = load_pdf_pages(pdf_path, max_pages=max_pages)

    page_chain = build_page_summary_chain(llm)
    final_chain = build_final_summary_chain(llm)

    page_summaries = []

    for i, doc in enumerate(docs, start=1):
        text = doc.page_content.strip()

        if not text:
            page_summaries.append(f"Page {i}:\n- This page appears to be empty.")
            continue

        summary = page_chain.invoke({"text": text})
        page_summaries.append(f"Page {i}:\n{summary}")

    combined_summaries = "\n\n".join(page_summaries)
    final_summary = final_chain.invoke({"summaries": combined_summaries})

    return final_summary


if __name__ == "__main__":
    try:
        final_summary = summarize_pdf(
            pdf_path=PDF_PATH,
            max_pages=MAX_PAGES,
            provider="ollama",
        )

        print("\n================ FINAL SUMMARY ================\n")
        print(final_summary)

    except Exception as e:
        print(f"Error: {e}")