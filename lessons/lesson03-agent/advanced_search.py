from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def build_llm() -> ChatOpenAI:
    """Create and return the LLM."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.8
    )


def build_agent():
    """Create and return the agent with TavilySearch as the only tool."""
    llm = build_llm()
    tools = [TavilySearch(max_results=5)]
    return create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from langchain-course!")

    agent = build_agent()

    user_request = (
        "Search for 3 job postings for an AI engineer using LangChain "
        "in the Bay Area on LinkedIn and list their details."
    )

    result = agent.invoke({
        "messages": [HumanMessage(content=user_request)]
    })

    print("\nFinal result:\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
