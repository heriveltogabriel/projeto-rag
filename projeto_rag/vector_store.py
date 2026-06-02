from dataclasses import dataclass
from collections import Counter
import math
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
    _STOPWORDS = {
        "a",
        "as",
        "ao",
        "aos",
        "com",
        "como",
        "da",
        "das",
        "de",
        "dela",
        "dele",
        "do",
        "dos",
        "e",
        "ela",
        "ele",
        "em",
        "era",
        "essa",
        "esse",
        "esta",
        "este",
        "foi",
        "na",
        "nas",
        "name",
        "nome",
        "no",
        "nos",
        "o",
        "os",
        "para",
        "por",
        "qual",
        "quais",
        "que",
        "quem",
        "se",
        "um",
        "uma",
        "the",
        "of",
        "and",
        "in",
        "on",
        "to",
        "who",
        "what",
        "which",
        "was",
        "is",
    }
    _RELATION_TERMS = {
        "esposa",
        "marido",
        "companheira",
        "companheiro",
        "pai",
        "mae",
        "mãe",
        "filho",
        "filha",
        "irmao",
        "irmão",
        "irma",
        "irmã",
        "wife",
        "husband",
        "father",
        "mother",
        "son",
        "daughter",
        "brother",
        "sister",
    }
    _CAPITALIZED_WORD = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+"

    def __init__(self):
        self._index: Optional[faiss.IndexFlatIP] = None
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

    @classmethod
    def _query_tokens(cls, text: str) -> List[str]:
        tokens = cls._tokens(text)
        content_tokens = [
            token
            for token in tokens
            if token not in cls._STOPWORDS and (len(token) > 1 or token.isdigit())
        ]
        return content_tokens or tokens

    @classmethod
    def _query_phrases(cls, text: str) -> List[str]:
        tokens = cls._query_tokens(text)
        phrases = []
        for size in (3, 2):
            for start in range(0, max(len(tokens) - size + 1, 0)):
                phrases.append(" ".join(tokens[start : start + size]))
        return phrases

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

        query_terms = list(dict.fromkeys(self._query_tokens(query)))
        query_tokens = set(query_terms)
        if not query_tokens:
            return []

        query_text = query.strip().lower()
        query_phrases = self._query_phrases(query)
        chunk_tokens = [self._tokens(chunk.text) for chunk in self._chunks]
        source_tokens = self._source_tokens()
        term_weights = {
            token: 0.15 if token in source_tokens else 1.0
            for token in query_terms
        }
        avg_doc_len = sum(len(tokens) for tokens in chunk_tokens) / max(len(chunk_tokens), 1)
        document_frequency = Counter()
        for tokens in chunk_tokens:
            token_set = set(tokens)
            for token in query_tokens:
                if token in token_set:
                    document_frequency[token] += 1

        total_documents = len(self._chunks)
        k1 = 1.5
        b = 0.75
        scored: List[SearchResult] = []
        for chunk, tokens in zip(self._chunks, chunk_tokens):
            chunk_text = chunk.text.lower()
            token_counts = Counter(tokens)
            overlap = query_tokens.intersection(token_counts)
            if not overlap and query_text not in chunk_text:
                continue

            bm25_score = 0.0
            doc_len = max(len(tokens), 1)
            for token in query_terms:
                term_frequency = token_counts.get(token, 0)
                if term_frequency == 0:
                    continue
                df = document_frequency[token]
                idf = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
                denominator = term_frequency + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
                bm25_score += term_weights[token] * idf * (term_frequency * (k1 + 1)) / denominator

            phrase_bonus = 0.0
            if query_text and query_text in chunk_text:
                phrase_bonus += 1.0
            for phrase in query_phrases:
                phrase_terms = self._tokens(phrase)
                if not phrase or phrase not in chunk_text:
                    continue
                phrase_weight = 0.15 if phrase_terms and all(term in source_tokens for term in phrase_terms) else 1.0
                phrase_bonus += 0.35 * phrase_weight

            overlap_weight = sum(term_weights[token] for token in overlap)
            query_weight = sum(term_weights[token] for token in query_tokens)
            coverage_bonus = overlap_weight / max(query_weight, 1)
            relation_bonus = self._relation_bonus(query_terms, chunk.text)
            scored.append(
                SearchResult(
                    chunk=chunk,
                    score=bm25_score + phrase_bonus + coverage_bonus + relation_bonus,
                )
            )

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[: min(top_k, len(scored))]

    def _source_tokens(self) -> set:
        sources = {chunk.source for chunk in self._chunks}
        if len(sources) != 1:
            return set()

        source = next(iter(sources))
        return set(self._tokens(source))

    @classmethod
    def _relation_bonus(cls, query_terms: List[str], chunk_text: str) -> float:
        bonus = 0.0
        chunk_lower = chunk_text.lower()
        for relation in set(query_terms).intersection(cls._RELATION_TERMS):
            relation_pattern = re.escape(relation)
            first_person_named = (
                rf"\b(?i:(?:minha|minhas|meu|meus|my)\s+{relation_pattern})"
                rf"\s+{cls._CAPITALIZED_WORD}"
            )
            first_person_plain = (
                f"minha {relation}",
                f"minhas {relation}",
                f"meu {relation}",
                f"meus {relation}",
                f"my {relation}",
            )
            if re.search(first_person_named, chunk_text):
                bonus += 2.5
            elif any(phrase in chunk_lower for phrase in first_person_plain):
                bonus += 0.75
        return bonus
