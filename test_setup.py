"""
Environment check script: verifies that the Python environment, dependencies, network
and API key all work end to end.
Usage (after `conda activate agent-rag`, from the project root): `python test_setup.py`
Expected result: the model's one-sentence reply is printed with no errors.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key or "your_deepseek_api_key" in api_key:
    raise SystemExit(
        "DEEPSEEK_API_KEY is not set in .env. Get a key at https://platform.deepseek.com "
        "and put it in .env (see .env.example)."
    )

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
)

response = llm.invoke("Hello, what is information theory?")
print("Model reply:", response.content)
