# Professional RAG Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the current local RAG prototype into a maintainable Python project with clear modules, documentation, tests, API, and CLI.

**Architecture:** The root scripts become compatibility wrappers while the real implementation moves into the `projeto_rag` package. Core modules handle configuration, document loading, chunking, embeddings, FAISS search, and RAG orchestration; FastAPI and CLI become thin adapters over that shared pipeline.

**Tech Stack:** Python 3.9, FastAPI, Uvicorn, pypdf, python-docx, sentence-transformers, FAISS, NumPy, Ollama Python client, standard-library `unittest`.

---

## File Map

- Create `.gitignore`: keep virtualenvs, caches, macOS files, and generated indexes out of Git.
- Create `requirements.txt`: document runtime dependencies already used by the project.
- Create `.env.example`: show supported configuration variables.
- Create `projeto_rag/__init__.py`: package marker and version.
- Create `projeto_rag/config.py`: environment-backed settings.
- Create `projeto_rag/document_loader.py`: supported file discovery and text extraction.
- Create `projeto_rag/chunker.py`: deterministic chunking with validation.
- Create `projeto_rag/embeddings.py`: embedding provider interface and SentenceTransformer implementation.
- Create `projeto_rag/vector_store.py`: FAISS-backed in-memory search.
- Create `projeto_rag/rag_pipeline.py`: indexing, retrieval, prompt creation, and optional Ollama generation.
- Create `projeto_rag/cli.py`: terminal entrypoint for smoke tests.
- Modify `app/main.py`: thin FastAPI layer over `RagPipeline`.
- Modify root prototype scripts: remove import-time execution and route users to package/CLI APIs.
- Create `tests/`: standard-library tests that run without network and without Ollama.
- Modify `Readme.md`: setup, commands, architecture, troubleshooting, and development workflow.

---

### Task 1: Repository Hygiene And Configuration Skeleton

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `projeto_rag/__init__.py`
- Create: `projeto_rag/config.py`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
.DS_Store
**/.DS_Store
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
env/
.env
.env.*
!.env.example
*.faiss
*.index
*.pkl
*.sqlite
*.log
```

- [ ] **Step 2: Create `requirements.txt`**

```text
fastapi
uvicorn
pydantic
pypdf
python-docx
sentence-transformers
faiss-cpu
numpy
ollama
streamlit
python-dotenv
```

- [ ] **Step 3: Create `.env.example`**

```dotenv
RAG_DOCUMENT_PATH=ManualCafe.pdf
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=3
RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_EMBEDDING_MODEL_PATH=
RAG_HF_HUB_OFFLINE=0
RAG_OLLAMA_MODEL=llama3:latest
```

- [ ] **Step 4: Create `projeto_rag/__init__.py`**

```python
"""Local-first RAG toolkit for technical manuals."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Create `projeto_rag/config.py`**

```python
from dataclasses import dataclass
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


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
```

- [ ] **Step 6: Verify Git ignores generated clutter**

Run: `git status --short`

Expected: `.venv/` and `.DS_Store` no longer appear once `.gitignore` exists; project source files still appear as untracked until committed.

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt .env.example projeto_rag/__init__.py projeto_rag/config.py
git commit -m "chore: add project configuration skeleton"
```

---

### Task 2: Chunking Module With Tests

**Files:**
- Create: `projeto_rag/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write `tests/test_chunker.py`**

```python
import unittest

from projeto_rag.chunker import chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_chunks_text_with_overlap(self):
        chunks = chunk_text("abcdefghij", chunk_size=4, overlap=1)
        self.assertEqual(chunks, ["abcd", "defg", "ghij", "j"])

    def test_ignores_empty_input(self):
        self.assertEqual(chunk_text("   ", chunk_size=4, overlap=1), [])

    def test_rejects_invalid_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            chunk_text("abc", chunk_size=0, overlap=0)

    def test_rejects_overlap_equal_to_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_text("abc", chunk_size=3, overlap=3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails before implementation**

Run: `.venv/bin/python -m unittest tests.test_chunker -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'projeto_rag.chunker'`.

- [ ] **Step 3: Create `projeto_rag/chunker.py`**

```python
from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero.")
    if overlap < 0:
        raise ValueError("overlap nao pode ser negativo.")
    if overlap >= chunk_size:
        raise ValueError("overlap deve ser menor que chunk_size.")

    normalized = text.strip()
    if not normalized:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(normalized):
        chunk = normalized[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m unittest tests.test_chunker -v`

Expected: PASS for 4 tests.

- [ ] **Step 5: Commit**

```bash
git add projeto_rag/chunker.py tests/test_chunker.py
git commit -m "feat: add tested text chunking"
```

---

### Task 3: Document Loader With Tests

**Files:**
- Create: `projeto_rag/document_loader.py`
- Create: `tests/test_document_loader.py`

- [ ] **Step 1: Write `tests/test_document_loader.py`**

```python
import tempfile
import unittest
from pathlib import Path

from projeto_rag.document_loader import (
    SUPPORTED_EXTENSIONS,
    extract_text,
    list_supported_files,
)


class DocumentLoaderTests(unittest.TestCase):
    def test_supported_extensions_include_project_formats(self):
        self.assertEqual(SUPPORTED_EXTENSIONS, {".pdf", ".txt", ".md", ".docx"})

    def test_lists_supported_files_and_ignores_hidden_runtime_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "notes.txt").write_text("hello", encoding="utf-8")
            (base / "image.png").write_text("ignored", encoding="utf-8")
            (base / ".venv").mkdir()
            (base / ".venv" / "ignored.txt").write_text("ignored", encoding="utf-8")

            files = list_supported_files(base, recursive=True)

        self.assertEqual([p.name for p in files], ["notes.txt"])

    def test_extracts_text_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual.md"
            path.write_text("# Titulo\nConteudo", encoding="utf-8")
            self.assertEqual(extract_text(path), "# Titulo\nConteudo")

    def test_rejects_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_text("ignored", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Extensao nao suportada"):
                extract_text(path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails before implementation**

Run: `.venv/bin/python -m unittest tests.test_document_loader -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'projeto_rag.document_loader'`.

- [ ] **Step 3: Create `projeto_rag/document_loader.py`**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
IGNORED_DIR_NAMES = {".venv", ".git", "__pycache__", "node_modules"}


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    text: str


def list_supported_files(base_path: Path, recursive: bool = True) -> List[Path]:
    base = Path(base_path).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Caminho nao encontrado: {base}")
    if base.is_file():
        return [base] if base.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not base.is_dir():
        raise NotADirectoryError(f"Caminho nao e uma pasta: {base}")

    candidates = base.rglob("*") if recursive else base.glob("*")
    files: List[Path] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if any(part in IGNORED_DIR_NAMES for part in candidate.parts):
            continue
        files.append(candidate)
    return sorted(files)


def extract_text(path: Path) -> str:
    file_path = Path(path).expanduser().resolve()
    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Extensao nao suportada: {extension}")

    if extension == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if extension == ".docx":
        document = DocxDocument(str(file_path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()

    return file_path.read_text(encoding="utf-8", errors="ignore").strip()


def load_documents(base_path: Path, recursive: bool = True) -> List[LoadedDocument]:
    documents: List[LoadedDocument] = []
    for file_path in list_supported_files(base_path, recursive=recursive):
        text = extract_text(file_path)
        if text:
            documents.append(LoadedDocument(path=file_path, text=text))
    return documents
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m unittest tests.test_document_loader -v`

Expected: PASS for 4 tests.

- [ ] **Step 5: Commit**

```bash
git add projeto_rag/document_loader.py tests/test_document_loader.py
git commit -m "feat: add document loading module"
```

---

### Task 4: Embeddings And Vector Store

**Files:**
- Create: `projeto_rag/embeddings.py`
- Create: `projeto_rag/vector_store.py`
- Create: `tests/test_vector_store.py`

- [ ] **Step 1: Write `tests/test_vector_store.py`**

```python
import unittest

import numpy as np

from projeto_rag.vector_store import TextChunk, VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_search_returns_nearest_chunks(self):
        chunks = [
            TextChunk(text="temperatura ideal espresso", source="a.pdf", chunk_id=1),
            TextChunk(text="limpeza do reservatorio", source="a.pdf", chunk_id=2),
        ]
        embeddings = np.array([[0.0, 0.0], [10.0, 10.0]], dtype="float32")
        store = VectorStore()
        store.build(chunks, embeddings)

        results = store.search(np.array([0.0, 0.1], dtype="float32"), top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.text, "temperatura ideal espresso")

    def test_search_before_build_raises_error(self):
        store = VectorStore()
        with self.assertRaisesRegex(RuntimeError, "Indice vazio"):
            store.search(np.array([0.0, 0.1], dtype="float32"), top_k=1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run vector store test to verify it fails before implementation**

Run: `.venv/bin/python -m unittest tests.test_vector_store -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'projeto_rag.vector_store'`.

- [ ] **Step 3: Create `projeto_rag/embeddings.py`**

```python
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
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")
```

- [ ] **Step 4: Create `projeto_rag/vector_store.py`**

```python
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
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m unittest tests.test_vector_store -v`

Expected: PASS for 2 tests.

- [ ] **Step 6: Commit**

```bash
git add projeto_rag/embeddings.py projeto_rag/vector_store.py tests/test_vector_store.py
git commit -m "feat: add embeddings and vector store abstractions"
```

---

### Task 5: RAG Pipeline With Deterministic Tests

**Files:**
- Create: `projeto_rag/rag_pipeline.py`
- Create: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Write `tests/test_rag_pipeline.py`**

```python
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np

from projeto_rag.config import Settings
from projeto_rag.rag_pipeline import RagPipeline


class FakeEmbeddings:
    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            text_lower = text.lower()
            vectors.append(
                [
                    1.0 if "espresso" in text_lower else 0.0,
                    1.0 if "limpeza" in text_lower else 0.0,
                ]
            )
        return np.asarray(vectors, dtype="float32")


class FakeGenerator:
    def generate(self, question: str, context: str) -> str:
        return f"Resposta para: {question} | Contexto: {context[:20]}"


class RagPipelineTests(unittest.TestCase):
    def test_retrieve_after_indexing(self):
        with tempfile.TemporaryDirectory() as tmp:
            manual = Path(tmp) / "manual.txt"
            manual.write_text("espresso temperatura ideal\nlimpeza do reservatorio", encoding="utf-8")
            settings = Settings(document_path=manual, chunk_size=40, chunk_overlap=0, top_k=1)
            pipeline = RagPipeline(settings, embeddings=FakeEmbeddings(), generator=FakeGenerator())

            summary = pipeline.index_path(manual, recursive=False)
            results = pipeline.retrieve("espresso", top_k=1)

        self.assertEqual(summary["documents"], 1)
        self.assertEqual(summary["chunks"], 1)
        self.assertEqual(results[0].chunk.source, str(manual))

    def test_answer_uses_generator_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            manual = Path(tmp) / "manual.txt"
            manual.write_text("espresso temperatura ideal", encoding="utf-8")
            settings = Settings(document_path=manual, chunk_size=50, chunk_overlap=0, top_k=1)
            pipeline = RagPipeline(settings, embeddings=FakeEmbeddings(), generator=FakeGenerator())
            pipeline.index_path(manual, recursive=False)

            response = pipeline.answer("espresso", top_k=1, generate=True)

        self.assertIn("Resposta para: espresso", response["answer"])
        self.assertEqual(len(response["sources"]), 1)

    def test_retrieve_before_indexing_raises_error(self):
        pipeline = RagPipeline(Settings(), embeddings=FakeEmbeddings(), generator=FakeGenerator())
        with self.assertRaisesRegex(RuntimeError, "Indice vazio"):
            pipeline.retrieve("espresso", top_k=1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run pipeline test to verify it fails before implementation**

Run: `.venv/bin/python -m unittest tests.test_rag_pipeline -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'projeto_rag.rag_pipeline'`.

- [ ] **Step 3: Create `projeto_rag/rag_pipeline.py`**

```python
from typing import Any, Dict, List, Optional, Protocol

import ollama

from projeto_rag.chunker import chunk_text
from projeto_rag.config import Settings, get_settings
from projeto_rag.document_loader import load_documents
from projeto_rag.embeddings import EmbeddingProvider, SentenceTransformerEmbeddings
from projeto_rag.vector_store import SearchResult, TextChunk, VectorStore


class TextGenerator(Protocol):
    def generate(self, question: str, context: str) -> str:
        ...


class OllamaGenerator:
    def __init__(self, model: str):
        self.model = model

    def generate(self, question: str, context: str) -> str:
        system_prompt = (
            "Voce e um assistente tecnico especialista no manual fornecido. "
            "Use apenas as informacoes do contexto para responder. "
            "Se a resposta nao estiver no contexto, diga que nao encontrou no manual.\n\n"
            f"Contexto:\n{context}"
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        return response["message"]["content"]


class RagPipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embeddings: Optional[EmbeddingProvider] = None,
        generator: Optional[TextGenerator] = None,
    ):
        self.settings = settings or get_settings()
        self.embeddings = embeddings or SentenceTransformerEmbeddings(self.settings)
        self.generator = generator or OllamaGenerator(self.settings.ollama_model)
        self.vector_store = VectorStore()

    def index_path(self, path=None, recursive: bool = True) -> Dict[str, Any]:
        target = path or self.settings.document_path
        documents = load_documents(target, recursive=recursive)
        chunks: List[TextChunk] = []
        for document in documents:
            for index, text in enumerate(
                chunk_text(document.text, self.settings.chunk_size, self.settings.chunk_overlap),
                start=1,
            ):
                chunks.append(TextChunk(text=text, source=str(document.path), chunk_id=index))

        if not chunks:
            raise ValueError("Nenhum texto foi extraido dos documentos suportados.")

        vectors = self.embeddings.embed([chunk.text for chunk in chunks])
        self.vector_store.build(chunks, vectors)
        return {"documents": len(documents), "chunks": len(chunks), "path": str(target)}

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[SearchResult]:
        query = question.strip()
        if not query:
            raise ValueError("Pergunta vazia.")
        vectors = self.embeddings.embed([query])
        return self.vector_store.search(vectors[0], top_k or self.settings.top_k)

    def answer(self, question: str, top_k: Optional[int] = None, generate: bool = True) -> Dict[str, Any]:
        results = self.retrieve(question, top_k=top_k)
        context = "\n---\n".join(result.chunk.text for result in results)
        answer = self.generator.generate(question, context) if generate else "Geracao desativada."
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "source": result.chunk.source,
                    "chunk_id": result.chunk.chunk_id,
                    "score": result.score,
                    "text": result.chunk.text[:500],
                }
                for result in results
            ],
        }
```

- [ ] **Step 4: Run pipeline tests**

Run: `.venv/bin/python -m unittest tests.test_rag_pipeline -v`

Expected: PASS for 3 tests.

- [ ] **Step 5: Run all tests so far**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: PASS for all tests.

- [ ] **Step 6: Commit**

```bash
git add projeto_rag/rag_pipeline.py tests/test_rag_pipeline.py
git commit -m "feat: add reusable rag pipeline"
```

---

### Task 6: FastAPI Adapter

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Replace `app/main.py`**

```python
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from projeto_rag.config import Settings, get_settings
from projeto_rag.document_loader import list_supported_files
from projeto_rag.rag_pipeline import RagPipeline


app = FastAPI(title="Projeto RAG", version="0.1.0")
settings = get_settings()
pipeline = RagPipeline(settings)


class IndexarRequest(BaseModel):
    caminho: Optional[str] = None
    recursivo: bool = True


class PerguntaRequest(BaseModel):
    pergunta: str
    top_k: Optional[int] = None
    gerar_resposta: bool = True


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/arquivos")
def arquivos(caminho: Optional[str] = None, recursivo: bool = True) -> Dict[str, Any]:
    target = Path(caminho).expanduser() if caminho else settings.base_dir
    try:
        files = list_supported_files(target, recursive=recursivo)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"caminho": str(target.resolve()), "quantidade": len(files), "arquivos": [str(p) for p in files]}


@app.post("/indexar")
def indexar(req: IndexarRequest) -> Dict[str, Any]:
    target = Path(req.caminho).expanduser() if req.caminho else settings.document_path
    try:
        return pipeline.index_path(target, recursive=req.recursivo)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/perguntar")
def perguntar(req: PerguntaRequest) -> Dict[str, Any]:
    try:
        return pipeline.answer(req.pergunta, top_k=req.top_k, generate=req.gerar_resposta)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
```

- [ ] **Step 2: Run import check**

Run: `.venv/bin/python -m compileall app projeto_rag`

Expected: command completes successfully.

- [ ] **Step 3: Run API smoke test**

Run: `.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8010`

Expected: server starts and logs `Uvicorn running on http://127.0.0.1:8010`.

In another terminal:

```bash
curl -s http://127.0.0.1:8010/health
```

Expected:

```json
{"status":"ok"}
```

- [ ] **Step 4: Stop Uvicorn**

Press `Ctrl+C` in the server terminal.

- [ ] **Step 5: Commit**

```bash
git add app/main.py
git commit -m "feat: expose rag pipeline through fastapi"
```

---

### Task 7: CLI And Legacy Script Cleanup

**Files:**
- Create: `projeto_rag/cli.py`
- Modify: `extrator.py`
- Modify: `fatiador.py`
- Modify: `vetorizador.py`
- Modify: `buscador.py`
- Modify: `app_rag.py`
- Modify: `app_web.py`

- [ ] **Step 1: Create `projeto_rag/cli.py`**

```python
import argparse
import json
from pathlib import Path

from projeto_rag.config import get_settings
from projeto_rag.rag_pipeline import RagPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Projeto RAG local para manuais tecnicos.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    index_parser = subcommands.add_parser("index", help="Indexa documentos suportados.")
    index_parser.add_argument("--path", default=None, help="Arquivo ou pasta a indexar.")
    index_parser.add_argument("--no-recursive", action="store_true", help="Nao buscar arquivos recursivamente.")

    ask_parser = subcommands.add_parser("ask", help="Indexa e pergunta em um unico comando.")
    ask_parser.add_argument("question", help="Pergunta para o manual.")
    ask_parser.add_argument("--path", default=None, help="Arquivo ou pasta a indexar antes da pergunta.")
    ask_parser.add_argument("--top-k", type=int, default=None, help="Quantidade de trechos recuperados.")
    ask_parser.add_argument("--no-generate", action="store_true", help="Nao chamar Ollama; retorna apenas fontes.")
    ask_parser.add_argument("--no-recursive", action="store_true", help="Nao buscar arquivos recursivamente.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    pipeline = RagPipeline(settings)
    target = Path(args.path).expanduser() if getattr(args, "path", None) else settings.document_path

    if args.command == "index":
        summary = pipeline.index_path(target, recursive=not args.no_recursive)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "ask":
        pipeline.index_path(target, recursive=not args.no_recursive)
        response = pipeline.answer(args.question, top_k=args.top_k, generate=not args.no_generate)
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Replace `extrator.py` with compatibility exports**

```python
from projeto_rag.document_loader import extract_text


def extrair_texto_do_pdf(caminho_do_pdf):
    return extract_text(caminho_do_pdf)


if __name__ == "__main__":
    print("Use: python -m projeto_rag.cli index --path ManualCafe.pdf")
```

- [ ] **Step 3: Replace `fatiador.py` with compatibility exports**

```python
from projeto_rag.chunker import chunk_text


if __name__ == "__main__":
    print("Use: python -m projeto_rag.cli index --path ManualCafe.pdf")
```

- [ ] **Step 4: Replace `vetorizador.py` with compatibility exports**

```python
from projeto_rag.embeddings import SentenceTransformerEmbeddings


if __name__ == "__main__":
    print("Use: python -m projeto_rag.cli ask \"sua pergunta\" --path ManualCafe.pdf")
```

- [ ] **Step 5: Replace `buscador.py` with compatibility guidance**

```python
from projeto_rag.rag_pipeline import RagPipeline


if __name__ == "__main__":
    print("Use: python -m projeto_rag.cli ask \"Qual e a temperatura ideal do espresso?\" --path ManualCafe.pdf")
```

- [ ] **Step 6: Replace `app_rag.py` with compatibility guidance**

```python
if __name__ == "__main__":
    print("Use: python -m projeto_rag.cli ask \"sua pergunta\" --path ManualCafe.pdf")
```

- [ ] **Step 7: Replace `app_web.py` with a minimal Streamlit adapter**

```python
import streamlit as st

from projeto_rag.config import get_settings
from projeto_rag.rag_pipeline import RagPipeline


st.set_page_config(page_title="Assistente de Manual", page_icon="🤖")
st.title("Assistente Tecnico do Manual")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RagPipeline(get_settings())
if "indexed" not in st.session_state:
    st.session_state.indexed = False

if st.button("Indexar manual"):
    with st.spinner("Indexando documentos..."):
        summary = st.session_state.pipeline.index_path()
        st.session_state.indexed = True
        st.success(f"Indexados {summary['chunks']} chunks de {summary['documents']} documento(s).")

question = st.chat_input("Pergunte algo sobre o manual")
if question:
    if not st.session_state.indexed:
        st.warning("Indexe o manual antes de perguntar.")
    else:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Consultando o RAG local..."):
                response = st.session_state.pipeline.answer(question)
                st.markdown(response["answer"])
                with st.expander("Fontes"):
                    for source in response["sources"]:
                        st.write(f"{source['source']} | chunk {source['chunk_id']} | score {source['score']:.4f}")
                        st.write(source["text"])
```

- [ ] **Step 8: Run CLI retrieval without Ollama generation**

Run: `.venv/bin/python -m projeto_rag.cli ask "temperatura ideal espresso" --path ManualCafe.pdf --no-generate`

Expected: JSON response with `"answer": "Geracao desativada."` and at least one source from `ManualCafe.pdf`.

- [ ] **Step 9: Commit**

```bash
git add projeto_rag/cli.py extrator.py fatiador.py vetorizador.py buscador.py app_rag.py app_web.py
git commit -m "feat: add cli and clean legacy scripts"
```

---

### Task 8: Documentation And Final Verification

**Files:**
- Modify: `Readme.md`

- [ ] **Step 1: Replace `Readme.md`**

```markdown
# Projeto RAG Local

Assistente local para consulta de manuais tecnicos usando RAG: extracao de documentos, chunking, embeddings, FAISS e geracao com Ollama.

## Objetivo

O projeto foi organizado para rodar localmente e manter documentos fora de APIs de terceiros. O fluxo principal indexa arquivos suportados, recupera os trechos mais relevantes e usa um modelo local no Ollama para gerar a resposta.

## Arquitetura

```text
Documento -> Loader -> Chunker -> Embeddings -> FAISS -> Prompt -> Ollama -> Resposta
```

- `projeto_rag/document_loader.py`: le PDF, TXT, MD e DOCX.
- `projeto_rag/chunker.py`: divide textos em blocos com overlap.
- `projeto_rag/embeddings.py`: cria vetores com SentenceTransformers.
- `projeto_rag/vector_store.py`: indexa e busca vetores com FAISS.
- `projeto_rag/rag_pipeline.py`: orquestra indexacao, busca e resposta.
- `app/main.py`: API FastAPI.
- `projeto_rag/cli.py`: comandos de terminal.
- `app_web.py`: interface Streamlit simples.

## Requisitos

- Python 3.9 ou superior
- Ollama instalado e em execucao
- Modelo de geracao local:

```bash
ollama pull llama3:latest
```

## Instalacao

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Configuracao

Copie `.env.example` para `.env` se quiser sobrescrever os defaults:

```bash
cp .env.example .env
```

Variaveis principais:

- `RAG_DOCUMENT_PATH`: arquivo ou pasta padrao.
- `RAG_CHUNK_SIZE`: tamanho dos chunks.
- `RAG_CHUNK_OVERLAP`: overlap entre chunks.
- `RAG_TOP_K`: quantidade de fontes recuperadas.
- `RAG_EMBEDDING_MODEL`: modelo SentenceTransformers.
- `RAG_EMBEDDING_MODEL_PATH`: caminho local para um snapshot ja baixado.
- `RAG_HF_HUB_OFFLINE`: use `1` para evitar chamadas ao Hugging Face.
- `RAG_OLLAMA_MODEL`: modelo local de geracao no Ollama.

## Uso Via CLI

Indexar o manual padrao:

```bash
.venv/bin/python -m projeto_rag.cli index --path ManualCafe.pdf
```

Perguntar sem chamar o Ollama, retornando apenas fontes:

```bash
.venv/bin/python -m projeto_rag.cli ask "Qual e a temperatura ideal do espresso?" --path ManualCafe.pdf --no-generate
```

Perguntar com geracao via Ollama:

```bash
.venv/bin/python -m projeto_rag.cli ask "Qual e a temperatura ideal do espresso?" --path ManualCafe.pdf
```

## Uso Via API

Subir a API:

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Health check:

```bash
curl -s http://127.0.0.1:8010/health
```

Indexar:

```bash
curl -s -X POST http://127.0.0.1:8010/indexar \
  -H 'Content-Type: application/json' \
  -d '{"caminho":"ManualCafe.pdf","recursivo":false}'
```

Perguntar:

```bash
curl -s -X POST http://127.0.0.1:8010/perguntar \
  -H 'Content-Type: application/json' \
  -d '{"pergunta":"Qual e a temperatura ideal do espresso?","top_k":3}'
```

## Uso Via Streamlit

```bash
.venv/bin/streamlit run app_web.py
```

## Testes

Os testes usam `unittest` e nao exigem rede nem Ollama:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Troubleshooting

### Hugging Face sem rede

Se o modelo ja estiver em cache local, configure:

```bash
export RAG_HF_HUB_OFFLINE=1
export RAG_EMBEDDING_MODEL_PATH="/caminho/para/snapshot/local"
```

### Ollama indisponivel

Verifique se o servidor local responde:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

Baixe o modelo de geracao:

```bash
ollama pull llama3:latest
```

### Scripts antigos

Os scripts na raiz foram mantidos como compatibilidade. Para novos usos, prefira:

```bash
.venv/bin/python -m projeto_rag.cli
```
```

- [ ] **Step 2: Run all tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: PASS for all tests.

- [ ] **Step 3: Run compile check**

Run: `.venv/bin/python -m compileall app projeto_rag`

Expected: command completes successfully.

- [ ] **Step 4: Run CLI smoke test without generation**

Run: `.venv/bin/python -m projeto_rag.cli ask "Qual e a temperatura ideal do espresso?" --path ManualCafe.pdf --no-generate`

Expected: JSON output includes sources from `ManualCafe.pdf`.

- [ ] **Step 5: Commit documentation**

```bash
git add Readme.md
git commit -m "docs: document professional rag workflow"
```

- [ ] **Step 6: Final commit status**

Run: `git status --short`

Expected: only intentionally untracked local files remain, such as `ManualCafe.pdf` if it is not committed. `.venv/` and `.DS_Store` are ignored.

---

## Self-Review

- Spec coverage: configuration, document loading, chunking, embeddings, vector search, pipeline, FastAPI, CLI, legacy cleanup, tests, and README are all covered by tasks.
- Placeholder scan: the plan contains no deferred implementation placeholders.
- Type consistency: `Settings`, `TextChunk`, `SearchResult`, `VectorStore`, `RagPipeline`, `EmbeddingProvider`, and `TextGenerator` are introduced before use in later tasks.
- Scope check: Docker, auth, persistent indexes, upload UI, and deployment remain out of scope.
