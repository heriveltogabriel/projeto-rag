import os
from typing import List, Protocol
import warnings

from sentence_transformers import CrossEncoder

from projeto_rag.config import Settings
from projeto_rag.model_cache import resolve_model_ref
from projeto_rag.vector_store import SearchResult


class Reranker(Protocol):
    def rerank(self, question: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        ...


class NoOpReranker:
    def rerank(self, question: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        return results[:top_k]


class CrossEncoderReranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._disabled = False
        if settings.hf_hub_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

    @property
    def model(self):
        if self._disabled:
            return None
        if self._model is None:
            model_ref = self.settings.reranker_model_path or self.settings.reranker_model
            resolved_ref = resolve_model_ref(model_ref, self.settings.hf_hub_offline)
            try:
                if resolved_ref is None:
                    raise FileNotFoundError(f"Reranker '{model_ref}' nao encontrado no cache local.")
                self._model = CrossEncoder(resolved_ref)
            except Exception as exc:
                self._disabled = True
                warnings.warn(
                    (
                        f"Nao foi possivel carregar o reranker '{model_ref}'. "
                        f"A busca continuara sem reranking. Motivo: {exc}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
        return self._model

    def rerank(self, question: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        if not results:
            return []

        pairs = [(question, result.chunk.text) for result in results]
        model = self.model
        if model is None:
            return results[:top_k]
        try:
            scores = model.predict(pairs, batch_size=4, convert_to_numpy=True)
        except Exception as exc:
            self._disabled = True
            warnings.warn(
                f"Erro ao executar reranker. A busca continuara sem reranking. Motivo: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return results[:top_k]
        reranked = [
            SearchResult(chunk=result.chunk, score=float(score))
            for result, score in zip(results, scores)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:top_k]
