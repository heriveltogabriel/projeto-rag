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

    @staticmethod
    def _build_context(results: List[SearchResult]) -> str:
        context_parts = []
        for index, result in enumerate(results, start=1):
            context_parts.append(
                "\n".join(
                    [
                        f"Fonte {index}",
                        f"Arquivo: {result.chunk.source}",
                        f"Chunk: {result.chunk.chunk_id}",
                        f"Score: {result.score:.4f}",
                        "Trecho:",
                        result.chunk.text,
                    ]
                )
            )
        return "\n\n---\n\n".join(context_parts)

    def answer(self, question: str, top_k: Optional[int] = None, generate: bool = True) -> Dict[str, Any]:
        results = self.retrieve(question, top_k=top_k)
        context = self._build_context(results)
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
