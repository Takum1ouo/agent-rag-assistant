"""
Shared module: the "read documents and split into chunks" logic lives here, so that
ingest.py (vector index) and retrieval.py (BM25) get exactly the same chunks. This
guarantees that vector retrieval and BM25 run over identical data, which makes the
comparison between them fair and meaningful.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def load_chunks():
    docs = []
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        docs.extend(PyPDFLoader(str(pdf_path)).load())
    for txt_path in sorted(DOCS_DIR.glob("*.txt")):
        docs.extend(TextLoader(str(txt_path), encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)
