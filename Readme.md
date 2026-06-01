# Projeto RAG Local

Assistente local para consulta de manuais tecnicos usando RAG: extracao de documentos,
chunking, embeddings, FAISS e geracao com Ollama.

## Objetivo

O projeto foi organizado para rodar localmente e manter documentos fora de APIs de
terceiros. O fluxo principal indexa arquivos suportados, recupera os trechos mais
relevantes e usa um modelo local no Ollama para gerar a resposta.

## Arquitetura

```text
Documento -> Loader -> Chunker -> Embeddings + Busca lexical -> Fusao de fontes -> Prompt -> Ollama -> Resposta
```

- `projeto_rag/document_loader.py`: le PDF, TXT, MD e DOCX.
- `projeto_rag/chunker.py`: divide textos em blocos com overlap.
- `projeto_rag/embeddings.py`: cria vetores com SentenceTransformers.
- `projeto_rag/vector_store.py`: indexa vetores com FAISS e tambem faz busca lexical.
- `projeto_rag/rag_pipeline.py`: combina busca vetorial e lexical, monta contexto e gera resposta.
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

A busca e hibrida: ela combina similaridade semantica por embeddings com busca por
termos exatos. Isso melhora consultas com siglas, codigos, nomes proprios e numeros.

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

A interface abre um chat local para PDFs enviados pelo usuario:

1. Clique em **Enviar PDF** e escolha um arquivo `.pdf`.
2. Clique em **Indexar PDF**.
3. Ajuste **Trechos recuperados** se quiser mandar mais contexto para a resposta.
4. Quando a indexacao terminar, faca perguntas no chat.

Os PDFs enviados pela interface sao salvos em `.rag_uploads/`, uma pasta local
ignorada pelo Git.

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
