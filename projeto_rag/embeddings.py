import os
from typing import List, Protocol
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer

from projeto_rag.config import Settings
from projeto_rag.model_cache import resolve_model_ref


class EmbeddingProvider(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbeddings:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        if settings.hf_hub_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

    @property
    def model(self):
        if self._model is None:
            model_ref = self.settings.embedding_model_path or self.settings.embedding_model
            resolved_ref = resolve_model_ref(model_ref, self.settings.hf_hub_offline)
            try:
                if resolved_ref is None:
                    raise FileNotFoundError(f"Modelo '{model_ref}' nao encontrado no cache local.")
                self._model = SentenceTransformer(resolved_ref)
            except Exception as exc:
                fallback_ref = self.settings.embedding_fallback_model
                if not fallback_ref or fallback_ref == model_ref:
                    raise
                resolved_fallback = resolve_model_ref(fallback_ref, self.settings.hf_hub_offline)
                if resolved_fallback is None:
                    raise
                warnings.warn(
                    (
                        f"Nao foi possivel carregar o modelo de embedding '{model_ref}'. "
                        f"Usando fallback '{fallback_ref}'. Motivo: {exc}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._model = SentenceTransformer(resolved_fallback)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        vectors = self.model.encode(
            texts,
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")
