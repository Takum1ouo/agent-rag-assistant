"""
Index-building script.

Reads all PDF/txt files under data/docs/ -> splits them into chunks -> generates
embeddings -> stores them in the local vector database (chroma_db/).
It only needs to run once; re-run it whenever the corpus changes.

Usage (after `conda activate agent-rag`, from the project root): `python src/ingest.py`
Prerequisite: run `python src/download_model.py` once to download the embedding model
locally. (In some network environments the built-in `huggingface_hub` download fails, so
the model is downloaded manually instead; see download_model.py. ingest.py and query.py
load the model straight from the local folder and do not go online.)
"""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from corpus import DOCS_DIR, load_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL = str(PROJECT_ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2")


def main():
    print(f"Reading and splitting documents from {DOCS_DIR}...")
    chunks = load_chunks()
    print(f"Split into {len(chunks)} chunks")

    if DB_DIR.exists():
        print(f"Found an old index at {DB_DIR}; deleting it before rebuilding (so old and new corpora do not mix)...")
        shutil.rmtree(DB_DIR)

    print("Loading the embedding model and generating vectors (this may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    print(f"Done. Index saved to {DB_DIR}")


if __name__ == "__main__":
    main()
