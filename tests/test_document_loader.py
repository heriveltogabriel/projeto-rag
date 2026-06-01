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
