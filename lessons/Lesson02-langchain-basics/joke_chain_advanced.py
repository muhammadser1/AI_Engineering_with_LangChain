from __future__ import annotations

from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama


Provider = Literal["openai", "ollama"]


def validate_inputs(topic: str, style: str) -> None:
    """Validate user inputs before sending them to the model."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")

    if not isinstance(style, str) or not style.strip():
        raise ValueError("style must be a non-empty string")


def build_prompt_template() -> PromptTemplate:
    """Create and return the reusable prompt template."""
    return PromptTemplate.from_template(
        "You are a joke generator.\n"
        "Create a {style} joke about {topic}.\n\n"
        "Rules:\n"
        "- Return EXACTLY two lines.\n"
        "- First line must start with: Question:\n"
        "- Second line must start with: Answer:\n"
        "- Keep it short and clean.\n"
        "- Do not add any extra explanation.\n"
    )


def get_llm(
    provider: Provider = "openai",
    temperature: float = 0.8,
):
    """
    Return the configured LLM based on the selected provider.
    """
    if provider == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
        )

    if provider == "ollama":
        return ChatOllama(
            model="gemma3:270m",
            temperature=temperature,
            num_predict=100,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def build_chain(
    provider: Provider = "openai",
    temperature: float = 0.8,
):
    """
    Build an LCEL chain:
    PromptTemplate -> LLM -> StrOutputParser
    """
    prompt_template = build_prompt_template()
    llm = get_llm(provider=provider, temperature=temperature)
    output_parser = StrOutputParser()

    chain = prompt_template | llm | output_parser
    return chain


def generate_joke(
    topic: str,
    style: str,
    provider: Provider = "openai",
    temperature: float = 0.8,
) -> str:
    """
    Generate a joke using an LCEL chain.
    """
    validate_inputs(topic, style)

    chain = build_chain(provider=provider, temperature=temperature)

    try:
        response = chain.invoke({
            "topic": topic.strip(),
            "style": style.strip(),
        })
        return response
    except Exception as exc:
        raise RuntimeError(f"Failed to generate joke: {exc}") from exc


def stream_joke(
    topic: str,
    style: str,
    provider: Provider = "openai",
    temperature: float = 0.8,
) -> None:
    """
    Stream the joke token by token.
    """
    validate_inputs(topic, style)

    chain = build_chain(provider=provider, temperature=temperature)

    try:
        for chunk in chain.stream({
            "topic": topic.strip(),
            "style": style.strip(),
        }):
            print(chunk, end="", flush=True)
        print()
    except Exception as exc:
        raise RuntimeError(f"Failed to stream joke: {exc}") from exc


if __name__ == "__main__":
    topic = "cat"
    style = "dad joke"

    try:
        print("=== Normal invoke ===")
        joke = generate_joke(
            topic=topic,
            style=style,
            provider="openai",
            temperature=0.8,
        )
        print(joke)

        print("\n=== Streaming ===")
        stream_joke(
            topic=topic,
            style=style,
            provider="openai",
            temperature=0.8,
        )

    except Exception as e:
        print(f"Error: {e}")