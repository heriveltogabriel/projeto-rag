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
                    1.0 if "irrelevante" in text_lower else 0.0,
                ]
            )
        return np.asarray(vectors, dtype="float32")


class FakeGenerator:
    def generate(self, question: str, context: str) -> str:
        return f"Resposta para: {question} | Contexto: {context[:20]}"


class FailingGenerator:
    def generate(self, question: str, context: str) -> str:
        raise RuntimeError("ollama offline")


class FakeReranker:
    def rerank(self, question, results, top_k):
        reranked = sorted(
            results,
            key=lambda result: 1 if question.lower() in result.chunk.text.lower() else 0,
            reverse=True,
        )
        return reranked[:top_k]


class RagPipelineTests(unittest.TestCase):
    def test_retrieve_after_indexing(self):
        with tempfile.TemporaryDirectory() as tmp:
            manual = Path(tmp) / "manual.txt"
            manual.write_text("espresso temperatura ideal\nlimpeza do reservatorio", encoding="utf-8")
            settings = Settings(document_path=manual, chunk_size=100, chunk_overlap=0, top_k=1)
            pipeline = RagPipeline(
                settings,
                embeddings=FakeEmbeddings(),
                generator=FakeGenerator(),
                reranker=FakeReranker(),
            )

            summary = pipeline.index_path(manual, recursive=False)
            results = pipeline.retrieve("espresso", top_k=1)

        self.assertEqual(summary["documents"], 1)
        self.assertEqual(summary["chunks"], 1)
        self.assertEqual(results[0].chunk.source, str(manual.resolve()))

    def test_answer_uses_generator_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            manual = Path(tmp) / "manual.txt"
            manual.write_text("espresso temperatura ideal", encoding="utf-8")
            settings = Settings(document_path=manual, chunk_size=50, chunk_overlap=0, top_k=1)
            pipeline = RagPipeline(
                settings,
                embeddings=FakeEmbeddings(),
                generator=FakeGenerator(),
                reranker=FakeReranker(),
            )
            pipeline.index_path(manual, recursive=False)

            response = pipeline.answer("espresso", top_k=1, generate=True)

        self.assertIn("Resposta para: espresso", response["answer"])
        self.assertEqual(len(response["sources"]), 1)
        self.assertIsNone(response["error"])

    def test_answer_keeps_sources_when_generator_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manual = Path(tmp) / "manual.txt"
            manual.write_text("espresso temperatura ideal", encoding="utf-8")
            settings = Settings(document_path=manual, chunk_size=50, chunk_overlap=0, top_k=1)
            pipeline = RagPipeline(
                settings,
                embeddings=FakeEmbeddings(),
                generator=FailingGenerator(),
                reranker=FakeReranker(),
            )
            pipeline.index_path(manual, recursive=False)

            response = pipeline.answer("espresso", top_k=1, generate=True)

        self.assertIn("Encontrei fontes relevantes", response["answer"])
        self.assertEqual(response["error"], "ollama offline")
        self.assertEqual(len(response["sources"]), 1)

    def test_retrieve_before_indexing_raises_error(self):
        pipeline = RagPipeline(
            Settings(),
            embeddings=FakeEmbeddings(),
            generator=FakeGenerator(),
            reranker=FakeReranker(),
        )
        with self.assertRaisesRegex(RuntimeError, "Indice vazio"):
            pipeline.retrieve("espresso", top_k=1)

    def test_retrieve_prioritizes_exact_terms_from_lexical_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            manual = Path(tmp) / "manual.txt"
            manual.write_text(
                "CSNY foi citado em uma secao curta.\n\n"
                "Este trecho irrelevante tem muitas outras palavras.",
                encoding="utf-8",
            )
            settings = Settings(document_path=manual, chunk_size=45, chunk_overlap=0, top_k=1)
            pipeline = RagPipeline(
                settings,
                embeddings=FakeEmbeddings(),
                generator=FakeGenerator(),
                reranker=FakeReranker(),
            )
            pipeline.index_path(manual, recursive=False)

            results = pipeline.retrieve("CSNY", top_k=1)

        self.assertEqual(results[0].chunk.chunk_id, 1)
        self.assertIn("CSNY", results[0].chunk.text)

    def test_retrieve_uses_more_candidates_before_reranking(self):
        class LastCandidateReranker:
            def rerank(self, question, results, top_k):
                return [results[-1]]

        with tempfile.TemporaryDirectory() as tmp:
            manual = Path(tmp) / "manual.txt"
            manual.write_text(
                "espresso primeiro\n\nlimpeza segundo\n\nirrelevante terceiro",
                encoding="utf-8",
            )
            settings = Settings(
                document_path=manual,
                chunk_size=18,
                chunk_overlap=0,
                top_k=1,
                retrieve_candidates=3,
            )
            pipeline = RagPipeline(
                settings,
                embeddings=FakeEmbeddings(),
                generator=FakeGenerator(),
                reranker=LastCandidateReranker(),
            )
            pipeline.index_path(manual, recursive=False)

            results = pipeline.retrieve("espresso", top_k=1)

        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
