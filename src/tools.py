"""
Tool definitions.

Three tools, wrapped with LangChain's @tool decorator:
- search_knowledge_base: query the knowledge base (Wikipedia corpus, hybrid retrieval)
- calculate: evaluate a math expression
- get_current_date: return today's date

The docstring of each tool is sent to the model as the tool description, and the model
uses it to decide whether to call the tool, so the descriptions need to be clear and accurate.
"""

import ast
import datetime
import operator
import threading

from langchain_core.tools import tool

from retrieval import Retriever

_retriever: Retriever | None = None
_retriever_lock = threading.Lock()


def _get_retriever() -> Retriever:
    # Lazy loading: the embedding model and the BM25 index are only loaded the first time
    # the knowledge-base tool is used, so questions unrelated to the knowledge base do not
    # have to wait for them.
    #
    # The lock is required: LangGraph's ToolNode runs the tool calls of one round in a thread
    # pool (e.g. when the model issues two knowledge-base queries at once). Two threads could
    # both see _retriever is None and both build a Retriever(), and chromadb's
    # PersistentClient is not thread-safe when the same path is created concurrently,
    # which raises an error.
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:  # double-checked: re-check after acquiring the lock to avoid creating it twice
                _retriever = Retriever()
    return _retriever


@tool
def search_knowledge_base(query: str) -> str:
    """Search the local knowledge base (a set of Wikipedia articles on geography, history, basic science, inventions and other general-knowledge topics).
    Use this tool when the user asks a specific factual question, e.g. "What is a black hole?" or "What theory is Charles Darwin known for?".
    """
    chunks = _get_retriever().hybrid_search(query, k=4)
    if not chunks:
        return "No relevant content was found in the knowledge base."
    return "\n\n".join(chunk.page_content for chunk in chunks)


# The calculator only lets these operators through and rejects everything else (a whitelist),
# which is safer than a blacklist that tries to list and block dangerous operations.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression content: {ast.dump(node)}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Supports addition, subtraction, multiplication, division,
    parentheses and exponentiation, e.g. "138 * 47", "(3 + 5) / 2", "12 ** 2"
    (use two asterisks for exponentiation, not ^, which is not exponentiation).
    Use this tool for any question involving concrete arithmetic instead of computing it yourself.
    """
    try:
        # The built-in eval() is deliberately not used: eval executes arbitrary code, which
        # is a code-injection risk. Instead the string is parsed into a syntax tree with
        # ast.parse and only whitelisted operators are evaluated node by node; any other
        # node (function call, variable name, ...) is rejected with an error.
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Calculation failed, could not parse the expression: {e}"


@tool
def get_current_date() -> str:
    """Return today's date. Use this tool when the user asks things like "what is today's date" or "what year is it"."""
    return datetime.date.today().isoformat()


ALL_TOOLS = [search_knowledge_base, calculate, get_current_date]
