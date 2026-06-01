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
    chunk_size: int = _int_from_env("RAG_CHUNK_SIZE", 500)
    chunk_overlap: int = _int_from_env("RAG_CHUNK_OVERLAP", 50)
    top_k: int = _int_from_env("RAG_TOP_K", 3)
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_model_path: str = os.getenv("RAG_EMBEDDING_MODEL_PATH", "")
    hf_hub_offline: bool = _bool_from_env("RAG_HF_HUB_OFFLINE", False)
    ollama_model: str = os.getenv("RAG_OLLAMA_MODEL", "llama3:latest")


def get_settings() -> Settings:
    return Settings()
