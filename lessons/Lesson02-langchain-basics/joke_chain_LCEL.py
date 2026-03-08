from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def generate_joke(topic: str, style: str) -> str:
    """
    Generate a joke using LCEL chaining syntax.
    """

    # 1️⃣ Create the prompt template
    prompt_template = PromptTemplate.from_template(
        "Create a {style} joke about {topic}.\n"
        "Return EXACTLY two lines:\n"
        "1) Question: ...\n"
        "2) Answer: ...\n"
    )

    # 2️⃣ Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.8
    )

    # 3️⃣ Create LCEL chain
    chain = prompt_template | llm

    # 4️⃣ Invoke the chain
    response = chain.invoke({
        "topic": topic,
        "style": style,
    })

    # 5️⃣ Return the generated joke
    return response.content


if __name__ == "__main__":
    topic = "cat"
    style = "dad joke"

    joke = generate_joke(topic, style)

    print("\nGenerated Joke:\n")
    print(joke)