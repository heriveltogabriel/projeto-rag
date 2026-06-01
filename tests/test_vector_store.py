import unittest

import numpy as np

from projeto_rag.vector_store import TextChunk, VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_search_returns_nearest_chunks(self):
        chunks = [
            TextChunk(text="temperatura ideal espresso", source="a.pdf", chunk_id=1),
            TextChunk(text="limpeza do reservatorio", source="a.pdf", chunk_id=2),
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        store = VectorStore()
        store.build(chunks, embeddings)

        results = store.search(np.array([0.9, 0.1], dtype="float32"), top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.text, "temperatura ideal espresso")
        self.assertGreater(results[0].score, 0.9)

    def test_search_before_build_raises_error(self):
        store = VectorStore()
        with self.assertRaisesRegex(RuntimeError, "Indice vazio"):
            store.search(np.array([0.0, 0.1], dtype="float32"), top_k=1)

    def test_lexical_search_finds_exact_acronym(self):
        chunks = [
            TextChunk(text="A banda CSNY aparece no documentario.", source="a.pdf", chunk_id=1),
            TextChunk(text="Outro trecho sobre cafe espresso.", source="a.pdf", chunk_id=2),
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        store = VectorStore()
        store.build(chunks, embeddings)

        results = store.lexical_search("CSNY", top_k=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, 1)
        self.assertGreaterEqual(results[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()
