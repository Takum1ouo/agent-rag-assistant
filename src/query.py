"""
Question-answering script.

Reads the existing index -> finds the most relevant chunks with hybrid retrieval
(vector + BM25, see retrieval.py) -> puts them into the prompt for the LLM -> prints
the answer. Prerequisite: src/ingest.py has been run once, so chroma_db/ exists.

Why hybrid retrieval instead of vector-only: in the comparison in eval_retrieval.py,
hybrid retrieval has a top-1 hit rate 5.9 percentage points higher than vector-only and
1.5 points higher than BM25-only, the best of the three, so this script uses it.

Usage (after `conda activate agent-rag`, from the project root):
    python src/query.py "your question"
Without an argument, the default question defined in this script is used.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from retrieval import Retriever

load_dotenv()

TOP_K = 4

DEFAULT_QUESTION = "What is a black hole?"


def build_prompt(question: str, chunks) -> str:
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    return f"""Answer the question using only the context below. If the context does not contain the relevant information, say clearly that you don't know. Do not make anything up.

Context:
{context}

Question: {question}
"""


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    retriever = Retriever()
    chunks = retriever.hybrid_search(question, k=TOP_K)

    print(f"--- Retrieved {len(chunks)} chunks ---")
    for i, chunk in enumerate(chunks, 1):
        source = Path(chunk.metadata.get("source", "?")).name
        page = chunk.metadata.get("page")
        location = f"page {page}" if page is not None else ""
        print(f"[{i}] from {source} {location}")

    prompt = build_prompt(question, chunks)

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    response = llm.invoke(prompt)

    print("\n--- Answer ---")
    print(response.content)


if __name__ == "__main__":
    main()
