# Projeto RAG Local

Assistente local para consulta de manuais tecnicos usando RAG: extracao de documentos,
chunking, embeddings, FAISS e geracao com Ollama.

## Objetivo

O projeto foi organizado para rodar localmente e manter documentos fora de APIs de
terceiros. O fluxo principal indexa arquivos suportados, recupera os trechos mais
relevantes e usa um modelo local no Ollama para gerar a resposta.

## Arquitetura

```text
Documento -> Loader -> Chunker -> Embeddings + BM25 -> Fusao -> Reranker opcional -> Prompt -> Ollama -> Resposta
```

- `projeto_rag/document_loader.py`: le PDF, TXT, MD e DOCX.
- `projeto_rag/chunker.py`: divide textos em blocos com overlap.
- `projeto_rag/embeddings.py`: cria vetores com SentenceTransformers e fallback local.
- `projeto_rag/vector_store.py`: indexa vetores com FAISS e faz busca lexical BM25.
- `projeto_rag/reranker.py`: reordena candidatos com CrossEncoder quando habilitado.
- `projeto_rag/rag_pipeline.py`: combina busca vetorial, BM25, reranking, contexto e resposta.
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
- `RAG_RETRIEVE_CANDIDATES`: quantidade interna de candidatos antes do reranking.
- `RAG_EMBEDDING_MODEL`: modelo SentenceTransformers.
- `RAG_EMBEDDING_MODEL_PATH`: caminho local para um snapshot ja baixado.
- `RAG_EMBEDDING_FALLBACK_MODEL`: modelo usado se o principal nao estiver disponivel.
- `RAG_RERANKER_ENABLED`: use `1` para habilitar reranking com CrossEncoder.
- `RAG_RERANKER_MODEL`: modelo CrossEncoder usado no reranking.
- `RAG_RERANKER_MODEL_PATH`: caminho local para um snapshot de reranker ja baixado.
- `RAG_HF_HUB_OFFLINE`: use `1` para evitar chamadas ao Hugging Face.
- `RAG_OLLAMA_MODEL`: modelo local de geracao no Ollama.

A busca e hibrida e em duas etapas:

1. O sistema recupera mais candidatos do que envia ao modelo (`RAG_RETRIEVE_CANDIDATES`).
2. A fusao combina embeddings FAISS com BM25 lexical.
3. Em PDFs unicos, termos que aparecem no nome do arquivo recebem peso menor para evitar
   que o titulo do documento esconda o trecho da resposta.
4. Perguntas factuais sobre relacoes, como "nome da esposa", ganham prioridade quando o
   trecho contem a relacao perto de um nome proprio.
5. Perguntas curtas com pronome ou estado civil, como "Ele e casado?", sao expandidas
   com o assunto provavel do documento e termos relacionados.
6. Se `RAG_RERANKER_ENABLED=1` e o modelo estiver disponivel, um CrossEncoder reordena
   os candidatos antes de montar o contexto final.

Isso melhora consultas com siglas, codigos, nomes proprios, numeros e perguntas factuais
em PDFs longos ou com muito ruido de indice/creditos.

Os defaults de `.env.example` apontam para `BAAI/bge-m3` e
`BAAI/bge-reranker-v2-m3`, que tendem a recuperar melhor texto em portugues. O primeiro
uso pode baixar arquivos grandes do Hugging Face. Se a rede estiver indisponivel, o
pipeline tenta usar `RAG_EMBEDDING_FALLBACK_MODEL` e continua sem reranker.

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

Para baixar e deixar os modelos recomendados em cache:

```bash
env HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120 \
  .venv/bin/python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

Se a rede estiver indisponivel, use modo offline. O pipeline pula modelos que nao
estiverem em cache e tenta o fallback:

```bash
export RAG_HF_HUB_OFFLINE=1
export RAG_RERANKER_ENABLED=0
```

Se o modelo ja estiver em cache local, voce tambem pode configurar o caminho exato:

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
