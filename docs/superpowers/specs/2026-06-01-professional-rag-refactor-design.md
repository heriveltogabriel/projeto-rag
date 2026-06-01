# Professional RAG Refactor Design

## Goal

Turn the current RAG prototype into a more professional, maintainable Python project while preserving its local-first purpose: ingest manuals, retrieve relevant context, and answer questions without sending documents to external APIs.

## Current State

The project has two overlapping implementations:

- A prototype chain in root-level scripts: `extrator.py`, `fatiador.py`, `vetorizador.py`, `buscador.py`, `app_rag.py`, and `app_web.py`.
- A FastAPI app in `app/main.py` with document extraction and simple token-based search.

The main issues are:

- Importing modules runs expensive work immediately.
- The embedding model load is fragile when Hugging Face connectivity is unavailable.
- Configuration is spread through scripts.
- There is no dependency manifest, test suite, CLI, or clear production-style structure.
- Documentation explains the concept but not a reliable developer workflow.

## Scope

This refactor will create a clean project layout with focused modules, configuration, tests, and documentation. It will avoid adding larger product features such as document upload UI, Docker, persistent background workers, authentication, or cloud deployment.

## Target Structure

```text
projeto_rag/
  app/
    main.py
  projeto_rag/
    __init__.py
    config.py
    document_loader.py
    chunker.py
    embeddings.py
    vector_store.py
    rag_pipeline.py
    cli.py
  tests/
  docs/
  requirements.txt
  .env.example
  Readme.md
```

## Components

### Configuration

`projeto_rag/config.py` will centralize defaults for document path, chunk size, overlap, retrieval count, embedding provider, embedding model, Ollama model, and local cache behavior. Values can be overridden with environment variables.

### Document Loading

`document_loader.py` will expose functions for listing supported files and extracting text from PDF, TXT, MD, and DOCX files. It will not print progress or execute extraction at import time.

### Chunking

`chunker.py` will expose a deterministic text chunker with validation for invalid chunk sizes and overlap. This keeps chunking testable without loading models.

### Embeddings

`embeddings.py` will provide a small interface for creating embeddings. The default implementation will use `sentence-transformers`, with clear support for offline cache usage. The design leaves room for an Ollama embedding provider later, but does not require it for this refactor.

### Vector Store

`vector_store.py` will wrap FAISS indexing and search. It will own vector normalization/type conversion and return source chunks with scores.

### RAG Pipeline

`rag_pipeline.py` will orchestrate document loading, chunking, embedding, FAISS retrieval, prompt creation, and Ollama generation. It will expose methods that FastAPI, Streamlit, and CLI can reuse.

### API

`app/main.py` will become a thin FastAPI layer around the pipeline. It will provide health, indexing, and question-answer endpoints. API responses will include answer text and source snippets.

### CLI

`cli.py` will support local smoke tests from the terminal, such as indexing a document and asking one question. This makes it easy to validate the system without Streamlit.

### Legacy Scripts

The root-level prototype scripts can remain temporarily as compatibility wrappers or be replaced with short messages that point to the new CLI/API. They should no longer contain the primary implementation.

## Data Flow

1. User indexes documents through CLI or API.
2. Loader extracts text from supported files.
3. Chunker splits text into overlapping chunks.
4. Embedding provider converts chunks to vectors.
5. FAISS stores vectors in memory.
6. User asks a question.
7. Question is embedded and searched against FAISS.
8. Top chunks are assembled into a grounded prompt.
9. Ollama generates the final answer.
10. API/CLI returns the answer and source snippets.

## Error Handling

The refactor will return clear errors for:

- Missing or invalid input folder/file.
- Unsupported file types.
- Empty extracted documents.
- Invalid chunk settings.
- Missing embedding model cache or failed model load.
- Ollama server unavailable.
- Asking before indexing.

## Testing

Focused tests will cover:

- Text chunk validation and overlap behavior.
- TXT/MD extraction and file filtering.
- Vector store indexing/search with deterministic fake embeddings.
- Pipeline behavior for empty indexes and basic retrieval.

Tests will avoid requiring Ollama, FAISS-heavy model downloads, or network access.

## Documentation

`Readme.md` will be rewritten to include:

- Project purpose and architecture.
- Local setup.
- Required Ollama model.
- Python dependencies.
- How to run API.
- How to run CLI.
- How to run tests.
- Troubleshooting for Hugging Face model cache and Ollama.

## Success Criteria

- The project has a clear Python package structure.
- Importing modules does not execute the RAG pipeline.
- API and CLI share the same core pipeline.
- Tests run without network access.
- README explains setup and operation from a clean checkout.
- The code remains local-first and does not send document contents to third-party APIs.

## Out Of Scope

- Docker packaging.
- Authentication.
- Persistent vector index files.
- Multi-user session storage.
- Web upload UI.
- Production deployment.
