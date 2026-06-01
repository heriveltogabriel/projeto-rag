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
