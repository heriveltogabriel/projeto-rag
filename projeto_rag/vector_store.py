from dataclasses import dataclass
import re
from typing import List, Optional

import faiss
import numpy as np


@dataclass(frozen=True)
class TextChunk:
    text: str
    source: str
    chunk_id: int


@dataclass(frozen=True)
class SearchResult:
    chunk: TextChunk
    score: float


class VectorStore:
    def __init__(self):
        self._index: Optional[faiss.IndexFlatL2] = None
        self._chunks: List[TextChunk] = []

    @property
    def count(self) -> int:
        return len(self._chunks)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        normalized = np.asarray(vectors, dtype="float32").copy()
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        np.divide(normalized, norms, out=normalized, where=norms > 0)
        return normalized

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", text.lower())

    def build(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        if not chunks:
            raise ValueError("Nao ha chunks para indexar.")
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("Embeddings devem ser uma matriz 2D.")
        if vectors.shape[0] != len(chunks):
            raise ValueError("Quantidade de embeddings difere da quantidade de chunks.")

        normalized = self._normalize(vectors)
        index = faiss.IndexFlatIP(normalized.shape[1])
        index.add(normalized)
        self._index = index
        self._chunks = list(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        if self._index is None or not self._chunks:
            raise RuntimeError("Indice vazio. Rode a indexacao antes de pesquisar.")
        if top_k <= 0:
            raise ValueError("top_k deve ser maior que zero.")

        query = np.asarray(query_embedding, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = self._normalize(query)
        scores, indices = self._index.search(query, min(top_k, len(self._chunks)))

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(SearchResult(chunk=self._chunks[int(idx)], score=float(score)))
        return results

    def lexical_search(self, query: str, top_k: int) -> List[SearchResult]:
        if not self._chunks:
            raise RuntimeError("Indice vazio. Rode a indexacao antes de pesquisar.")
        if top_k <= 0:
            raise ValueError("top_k deve ser maior que zero.")

        query_tokens = set(self._tokens(query))
        if not query_tokens:
            return []

        query_text = query.strip().lower()
        scored: List[SearchResult] = []
        for chunk in self._chunks:
            chunk_text = chunk.text.lower()
            chunk_tokens = set(self._tokens(chunk.text))
            overlap = query_tokens.intersection(chunk_tokens)
            if not overlap and query_text not in chunk_text:
                continue

            token_score = len(overlap) / len(query_tokens)
            phrase_bonus = 0.5 if query_text and query_text in chunk_text else 0.0
            scored.append(SearchResult(chunk=chunk, score=token_score + phrase_bonus))

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[: min(top_k, len(scored))]
