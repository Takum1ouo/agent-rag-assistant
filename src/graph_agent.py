"""
Multi-step Agent built with LangGraph: it can loop and make decisions over several steps.

Difference from agent.py: agent.py supports only one round of tool calls (question ->
at most one round of tool calls -> answer), so it cannot handle questions where a later
step depends on the result of an earlier one ("look up A, decide whether to look up B
based on A, then compute"). Here the flow is built as a graph: after every step the model
decides again whether to answer directly or call another tool, until it stops requesting
tools. Tool selection is still done by the model's Function Calling ability; the graph
only manages, in a structured way, whether to keep looping.

Graph structure:
    START -> agent node -> (tool call requested?) -> yes -> tools node -> back to agent node (loop)
                                                   -> no  -> END (the current reply is the final answer)

Usage (after `conda activate agent-rag`, from the project root):
    python src/graph_agent.py "your question"
"""

import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import ALL_TOOLS

load_dotenv()

SYSTEM_HINT = (
    "You are an assistant with access to a few tools. Answer directly, without calling "
    "any tool, when the question needs none (greetings, casual chat). For questions that "
    "need a specific fact lookup, arithmetic, or the current date, choose the appropriate tool. "
    "If a question requires first looking up some information and then taking a further step "
    "based on it (for example, looking up the years of two events and then computing how many "
    "years apart they are), you may call tools in several consecutive steps; you do not need "
    "to call every tool at once."
)


class AgentState(TypedDict):
    # add_messages is a LangGraph reducer: when a node returns new messages they are
    # appended to the existing messages list instead of overwriting the conversation history.
    messages: Annotated[list, add_messages]


def build_graph():
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,  # routing decisions must be stable; no need for "creativity"
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def call_model(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.set_entry_point("agent")
    # tools_condition is a LangGraph built-in: it checks whether the message the agent node
    # just produced contains tool_calls. If so it routes to the "tools" node, otherwise to
    # END. This decides whether the loop continues; it is the same check as the `if` in
    # agent.py, expressed as a graph.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # after the tools run, return to the agent so the model decides the next step from the results

    return graph.compile()


_graph = None


def run_graph_agent(question: str, verbose: bool = True):
    global _graph
    if _graph is None:
        _graph = build_graph()

    messages = [HumanMessage(content=f"{SYSTEM_HINT}\n\nUser question: {question}")]
    final_state = _graph.invoke({"messages": messages}, config={"recursion_limit": 10})

    called_tools = []
    if verbose:
        for msg in final_state["messages"]:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for call in tool_calls:
                    print(f"[routing] Step {len(called_tools) + 1}: calling tool {call['name']}({call['args']})")
                    called_tools.append(call["name"])
    else:
        for msg in final_state["messages"]:
            for call in getattr(msg, "tool_calls", None) or []:
                called_tools.append(call["name"])

    answer = final_state["messages"][-1].content
    return answer, called_tools


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "What is today's date?"
    answer, called_tools = run_graph_agent(question)

    print(f"\nQuestion: {question}")
    print(f"Tool call order: {called_tools if called_tools else '(no tool called)'}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
