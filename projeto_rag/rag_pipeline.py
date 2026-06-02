from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import ollama

from projeto_rag.chunker import chunk_text
from projeto_rag.config import Settings, get_settings
from projeto_rag.document_loader import load_documents
from projeto_rag.embeddings import EmbeddingProvider, SentenceTransformerEmbeddings
from projeto_rag.reranker import CrossEncoderReranker, NoOpReranker, Reranker
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
            "Responda sempre em portugues do Brasil. "
            "Use apenas as informacoes do contexto para responder. "
            "Se a resposta nao estiver no contexto, responda exatamente: "
            "\"Nao encontrei essa informacao no PDF enviado.\"\n\n"
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
    _ANAPHORA_TERMS = {
        "ele",
        "ela",
        "dele",
        "dela",
        "autor",
        "autora",
        "biografado",
        "biografada",
        "personagem",
    }
    _MARITAL_TERMS = {
        "casado",
        "casada",
        "casamento",
        "casou",
        "casar",
        "solteiro",
        "solteira",
        "married",
        "wife",
        "husband",
        "spouse",
    }
    _MARITAL_EXPANSION = (
        "esposa marido companheira companheiro conjuge cônjuge casamento "
        "casado casada wife husband spouse married"
    )
    _DOCUMENT_TERM_STOPWORDS = {
        "a",
        "as",
        "biografia",
        "autobiografia",
        "de",
        "doc",
        "documento",
        "do",
        "dos",
        "ebook",
        "livro",
        "manual",
        "pdf",
        "the",
    }

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embeddings: Optional[EmbeddingProvider] = None,
        generator: Optional[TextGenerator] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.settings = settings or get_settings()
        self.embeddings = embeddings or SentenceTransformerEmbeddings(self.settings)
        self.generator = generator or OllamaGenerator(self.settings.ollama_model)
        if reranker is not None:
            self.reranker = reranker
        elif self.settings.reranker_enabled:
            self.reranker = CrossEncoderReranker(self.settings)
        else:
            self.reranker = NoOpReranker()
        self.vector_store = VectorStore()
        self.document_query_terms: List[str] = []

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
        self.document_query_terms = self._extract_document_query_terms([document.path for document in documents])
        return {"documents": len(documents), "chunks": len(chunks), "path": str(target)}

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[SearchResult]:
        query = question.strip()
        if not query:
            raise ValueError("Pergunta vazia.")
        result_count = top_k or self.settings.top_k
        candidate_count = max(result_count, self.settings.retrieve_candidates)
        expanded_query = self._expand_query(query)
        vectors = self.embeddings.embed([expanded_query])
        vector_results = self.vector_store.search(vectors[0], candidate_count)
        lexical_results = self.vector_store.lexical_search(expanded_query, candidate_count)
        candidates = self._merge_results(vector_results, lexical_results, candidate_count)
        return self.reranker.rerank(expanded_query, candidates, result_count)

    def _expand_query(self, query: str) -> str:
        query_terms = set(VectorStore._tokens(query))
        expansions = []

        if self.document_query_terms and self._should_add_document_terms(query_terms):
            expansions.append(" ".join(self.document_query_terms))

        if query_terms.intersection(self._MARITAL_TERMS):
            expansions.append(self._MARITAL_EXPANSION)

        if not expansions:
            return query

        return " ".join([query, *expansions])

    def _should_add_document_terms(self, query_terms: set) -> bool:
        if query_terms.intersection(self._ANAPHORA_TERMS):
            return True
        if query_terms.intersection(self._MARITAL_TERMS):
            return True
        return False

    @classmethod
    def _extract_document_query_terms(cls, paths: List[Path]) -> List[str]:
        if len(paths) != 1:
            return []

        stem = paths[0].stem
        stem = stem.replace("_", " ").replace("-", " ")
        terms = []
        for token in VectorStore._tokens(stem):
            if token in cls._DOCUMENT_TERM_STOPWORDS:
                continue
            if token.isdigit() or len(token) < 2:
                continue
            if len(token) >= 8 and all(char in "0123456789abcdef" for char in token):
                continue
            if token not in terms:
                terms.append(token)
        return terms[:6]

    @staticmethod
    def _merge_results(
        vector_results: List[SearchResult],
        lexical_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        combined = {}

        for result in vector_results:
            key = (result.chunk.source, result.chunk.chunk_id)
            combined[key] = SearchResult(chunk=result.chunk, score=result.score)

        for result in lexical_results:
            key = (result.chunk.source, result.chunk.chunk_id)
            lexical_boosted = SearchResult(chunk=result.chunk, score=result.score + 1.0)
            existing = combined.get(key)
            if existing is None or lexical_boosted.score > existing.score:
                combined[key] = lexical_boosted

        return sorted(combined.values(), key=lambda result: result.score, reverse=True)[:top_k]

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
        generation_error = None
        if generate:
            try:
                answer = self.generator.generate(question, context)
            except Exception as exc:
                generation_error = str(exc)
                answer = (
                    "Encontrei fontes relevantes, mas nao consegui gerar a resposta "
                    "porque o Ollama nao esta acessivel. Verifique se o Ollama esta rodando."
                )
        else:
            answer = "Geracao desativada."
        return {
            "question": question,
            "answer": answer,
            "error": generation_error,
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
