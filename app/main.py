from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from projeto_rag.config import get_settings
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
    return {
        "caminho": str(target.resolve()),
        "quantidade": len(files),
        "arquivos": [str(path) for path in files],
    }


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
