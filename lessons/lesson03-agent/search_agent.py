from dotenv import load_dotenv
load_dotenv()

from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from pydantic import BaseModel
from typing import List

class Source(BaseModel):
    title: str
    url: str


class AgentResponse(BaseModel):
    # answer: str
    sources: List[Source]

def build_tavily_client() -> TavilyClient:
    """Create and return a Tavily client."""
    return TavilyClient()


def build_search_tool(tavily_client: TavilyClient):
    """Create and return a search tool using the given Tavily client."""

    @tool
    def search(query: str) -> str:
        """
        Search the internet using Tavily.

        Args:
            query: The search query.

        Returns:
            Search results as a string.
        """
        print(f"Searching for: {query}")
        return str(tavily_client.search(query=query))

    return search


def build_llm() -> ChatOpenAI:
    """Create and return the LLM (plain, so the agent can use tool calls)."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.8,
    )


def build_agent():
    """Create and return the agent with its tools."""
    tavily_client = build_tavily_client()
    search_tool = build_search_tool(tavily_client)
    llm = build_llm()

    tools = [search_tool]
    return create_agent(model=llm, tools=tools,response_format=AgentResponse)


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