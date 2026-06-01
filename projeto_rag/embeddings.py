import os
from typing import List, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from projeto_rag.config import Settings


class EmbeddingProvider(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbeddings:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        if settings.hf_hub_offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

    @property
    def model(self):
        if self._model is None:
            model_ref = self.settings.embedding_model_path or self.settings.embedding_model
            self._model = SentenceTransformer(model_ref)
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
