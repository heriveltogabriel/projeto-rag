from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    document_path: Path = BASE_DIR / os.getenv("RAG_DOCUMENT_PATH", "ManualCafe.pdf")
    chunk_size: int = _int_from_env("RAG_CHUNK_SIZE", 900)
    chunk_overlap: int = _int_from_env("RAG_CHUNK_OVERLAP", 180)
    top_k: int = _int_from_env("RAG_TOP_K", 6)
    retrieve_candidates: int = _int_from_env("RAG_RETRIEVE_CANDIDATES", 40)
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "BAAI/bge-m3",
    )
    embedding_model_path: str = os.getenv("RAG_EMBEDDING_MODEL_PATH", "")
    embedding_fallback_model: str = os.getenv(
        "RAG_EMBEDDING_FALLBACK_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    reranker_enabled: bool = _bool_from_env("RAG_RERANKER_ENABLED", True)
    reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_model_path: str = os.getenv("RAG_RERANKER_MODEL_PATH", "")
    hf_hub_offline: bool = _bool_from_env("RAG_HF_HUB_OFFLINE", False)
    ollama_model: str = os.getenv("RAG_OLLAMA_MODEL", "llama3:latest")


def get_settings() -> Settings:
    return Settings()
