"""
Retrieval module: three retrieval methods shared by eval_retrieval.py (comparison
experiment) and query.py (question answering).

- vector_search: vector-only retrieval (semantic similarity)
- bm25_search: keyword-only retrieval (lexical match)
- hybrid_search: fuses the two result lists with Reciprocal Rank Fusion (RRF)

Why RRF instead of a weighted sum of scores: the vector distance/similarity and the BM25
score are on different scales (BM25 is an arbitrary positive number that depends on
keyword hits and document length), so adding them with weights is not meaningful. RRF
ignores the raw scores and fuses by rank only, a common and well-validated approach in search.
"""

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from corpus import load_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL = str(PROJECT_ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2")

RRF_K = 60  # smoothing constant in the RRF formula (the commonly used value)

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str):
    # Capture runs of letters/digits as words. This is better than a plain .split(), because
    # "shading," and "BRDF." are then treated as the same words as "shading" and "BRDF";
    # otherwise BM25 would miss keyword matches that differ only by punctuation.
    return _WORD_RE.findall(text.lower())


class Retriever:
    """Builds the vector store and the BM25 index once and caches them, so they are not recomputed for every query."""

    def __init__(self):
        self.chunks = load_chunks()

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectordb = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

        tokenized_corpus = [_tokenize(chunk.page_content) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def vector_search(self, query: str, k: int = 4):
        return self.vectordb.similarity_search(query, k=k)

    def bm25_search(self, query: str, k: int = 4):
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked_indices[:k]]

    def hybrid_search(self, query: str, k: int = 4, fetch_k: int = 6, vector_weight: float = 1.0, bm25_weight: float = 1.0):
        # fetch_k=6 was chosen after comparing 4/6/10/20 in eval_retrieval.py:
        # a large value (e.g. 20) lets lower-ranked, mediocre candidates from both lists
        # take part in the fusion, which hurts the result.
        vector_results = self.vector_search(query, k=fetch_k)
        bm25_results = self.bm25_search(query, k=fetch_k)

        rrf_scores = {}
        doc_lookup = {}

        for rank, doc in enumerate(vector_results):
            key = _chunk_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0) + vector_weight / (RRF_K + rank + 1)
            doc_lookup[key] = doc

        for rank, doc in enumerate(bm25_results):
            key = _chunk_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0) + bm25_weight / (RRF_K + rank + 1)
            doc_lookup[key] = doc

        ranked_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
        return [doc_lookup[key] for key in ranked_keys[:k]]


def _chunk_key(doc):
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:50])
