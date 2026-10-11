"""
Routing accuracy evaluation: does the model choose the right tool on its own?

22 questions in four categories: date lookup, arithmetic, knowledge-base lookup, and
small talk that needs no tool. Each question is labelled with the tool it is expected to
call (for small talk: no tool at all). Each question is run through graph_agent.py (the
LangGraph multi-step version), and the tools the model actually called are compared with
the expected ones to compute the accuracy.

Criterion: a question counts as correct if the expected tool was called and no other
type of tool was called; it does not have to be called exactly once (e.g. querying the
knowledge base twice with different wording is still correct routing). Not calling a tool
when one is needed, or calling the wrong tool, counts as wrong.

Usage (after `conda activate agent-rag`, from the project root): `python src/eval_agent_routing.py`
"""

from graph_agent import run_graph_agent

EVAL_SET = [
    # Date lookup
    {"question": "What is today's date?", "expected_tool": "get_current_date"},
    {"question": "What year is it now?", "expected_tool": "get_current_date"},
    {"question": "What's the date today?", "expected_tool": "get_current_date"},
    {"question": "What day of the week is it today?", "expected_tool": "get_current_date"},
    {"question": "Look up today's date for me.", "expected_tool": "get_current_date"},
    # Arithmetic
    {"question": "What is 138 times 47?", "expected_tool": "calculate"},
    {"question": "What is 999 divided by 3?", "expected_tool": "calculate"},
    {"question": "What is (25+75)*2?", "expected_tool": "calculate"},
    {"question": "What is 12 squared?", "expected_tool": "calculate"},
    {"question": "Calculate 256 minus 128.", "expected_tool": "calculate"},
    # Knowledge-base lookup
    {"question": "What is a black hole?", "expected_tool": "search_knowledge_base"},
    {"question": "Who invented the telephone?", "expected_tool": "search_knowledge_base"},
    {"question": "What theory did Charles Darwin propose?", "expected_tool": "search_knowledge_base"},
    {"question": "What is DNA made of?", "expected_tool": "search_knowledge_base"},
    {"question": "What was the Great Wall of China built for?", "expected_tool": "search_knowledge_base"},
    {"question": "How do vaccines work?", "expected_tool": "search_knowledge_base"},
    {"question": "Who painted the Mona Lisa?", "expected_tool": "search_knowledge_base"},
    # Small talk, no tool should be called
    {"question": "Hello, who are you?", "expected_tool": None},
    {"question": "What can you help me with?", "expected_tool": None},
    {"question": "Thanks for your help.", "expected_tool": None},
    {"question": "How are you?", "expected_tool": None},
    {"question": "Tell me a joke.", "expected_tool": None},
]


def main():
    correct = 0
    wrong = []

    for item in EVAL_SET:
        _, called_tools = run_graph_agent(item["question"], verbose=False)
        expected = item["expected_tool"]

        if expected is None:
            ok = len(called_tools) == 0
        else:
            ok = expected in called_tools and all(t == expected for t in called_tools)

        status = "OK" if ok else "WRONG"
        print(f"[{status}] {item['question']!r:46s} expected={expected}  actual={called_tools}")

        if ok:
            correct += 1
        else:
            wrong.append(item)

    total = len(EVAL_SET)
    print(f"\nRouting accuracy: {correct}/{total} = {correct/total:.1%}")


if __name__ == "__main__":
    main()
