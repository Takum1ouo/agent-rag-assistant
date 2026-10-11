"""
Agent main loop (single-round version): the model decides on its own whether to call
a tool and which one.

Difference from query.py: query.py is a fixed pipeline (retrieve -> build prompt ->
generate). Here there is no hard-coded "if the question contains keyword X, take
branch Y" logic. Tool selection is left entirely to the model; the code only provides
the tool descriptions and executes the tool calls the model requests.

It relies on the Function Calling capability natively supported by the DeepSeek API
(OpenAI-compatible interface). Only one round of tool calls is supported; see
graph_agent.py for multi-step chained calls.

Usage (after `conda activate agent-rag`, from the project root):
    python src/agent.py "your question"
"""

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from tools import ALL_TOOLS

load_dotenv()

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

SYSTEM_HINT = (
    "You are an assistant with access to a few tools. Answer directly, without calling "
    "any tool, when the question needs none (greetings, casual chat). For questions that "
    "need a specific fact lookup, arithmetic, or the current date, choose the appropriate tool."
)


def build_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,  # routing decisions must be stable; no need for "creativity"
    )


def run_agent(question: str, verbose: bool = True):
    llm = build_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = [HumanMessage(content=f"{SYSTEM_HINT}\n\nUser question: {question}")]

    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    if not ai_message.tool_calls:
        # The model decided no tool is needed and answered directly.
        if verbose:
            print("[routing] The model decided no tool is needed and answered directly")
        return ai_message.content, []

    called_tools = []
    for call in ai_message.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        called_tools.append(tool_name)

        if verbose:
            print(f"[routing] The model chose tool: {tool_name}({tool_args})")

        tool_fn = TOOLS_BY_NAME.get(tool_name)
        if tool_fn is None:
            result = f"Unknown tool: {tool_name}"
        else:
            result = tool_fn.invoke(tool_args)

        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    final_message = llm_with_tools.invoke(messages)
    return final_message.content, called_tools


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "What is today's date?"
    answer, called_tools = run_agent(question)

    print(f"\nQuestion: {question}")
    print(f"Tools called: {called_tools if called_tools else '(no tool called)'}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
