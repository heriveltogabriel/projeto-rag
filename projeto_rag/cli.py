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
