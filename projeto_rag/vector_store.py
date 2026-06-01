from dataclasses import dataclass
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

    def build(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        if not chunks:
            raise ValueError("Nao ha chunks para indexar.")
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("Embeddings devem ser uma matriz 2D.")
        if vectors.shape[0] != len(chunks):
            raise ValueError("Quantidade de embeddings difere da quantidade de chunks.")

        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
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
        distances, indices = self._index.search(query, min(top_k, len(self._chunks)))

        results: List[SearchResult] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append(SearchResult(chunk=self._chunks[int(idx)], score=float(distance)))
        return results
