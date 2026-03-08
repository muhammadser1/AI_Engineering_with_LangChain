"""
PDF Summarizer using LangChain and OpenAI.

This script performs hierarchical summarization of a PDF document.
"""

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


PDF_PATH = "lessons/Lesson02-langchain-basics/git_github_practical_guide.pdf"
MAX_PAGES = 5
MODEL_NAME = "gpt-4o-mini"


def summarize_pdf(pdf_path: str, max_pages: int = 5) -> str:
    # llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2)
    llm = ChatOllama(
        model="gemma3:270m",
        temperature=0,
        num_predict=80
    )
    page_summary_prompt = PromptTemplate.from_template(
        "Summarize this page in 3-5 bullet points:\n\n{text}"
    )

    final_summary_prompt = PromptTemplate.from_template(
        "You will receive summaries of multiple pages of a PDF.\n"
        "Combine them into ONE clear summary (8-12 bullet points).\n\n"
        "Page summaries:\n{summaries}"
    )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    docs = docs[:max_pages]

    page_summaries = []
    for i, doc in enumerate(docs, start=1):
        prompt = page_summary_prompt.invoke({"text": doc.page_content})
        resp = llm.invoke(prompt)
        page_summaries.append(f"Page {i}:\n{resp.content}")

    combined = "\n\n".join(page_summaries)
    final_prompt = final_summary_prompt.invoke({"summaries": combined})
    final_resp = llm.invoke(final_prompt)
    return final_resp.content


if __name__ == "__main__":
    final_summary = summarize_pdf(PDF_PATH, MAX_PAGES)
    print("\n================ FINAL SUMMARY ================\n")
    print(final_summary)
