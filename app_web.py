import hashlib
import re
from pathlib import Path

import streamlit as st

from projeto_rag.config import get_settings
from projeto_rag.rag_pipeline import RagPipeline


st.set_page_config(page_title="Assistente de Manual")
st.title("Assistente Tecnico do Manual")


UPLOAD_DIR = Path(".rag_uploads")


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "documento.pdf"


def _save_uploaded_pdf(uploaded_file) -> tuple[Path, str]:
    content = uploaded_file.getvalue()
    digest = hashlib.sha256(content).hexdigest()[:12]
    safe_name = _safe_filename(uploaded_file.name)
    target = UPLOAD_DIR / f"{digest}_{safe_name}"
    UPLOAD_DIR.mkdir(exist_ok=True)
    if not target.exists():
        target.write_bytes(content)
    return target, digest


if "pipeline" not in st.session_state:
    st.session_state.pipeline = RagPipeline(get_settings())
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "uploaded_pdf_path" not in st.session_state:
    st.session_state.uploaded_pdf_path = None
if "uploaded_pdf_digest" not in st.session_state:
    st.session_state.uploaded_pdf_digest = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None

uploaded_pdf = st.file_uploader("Enviar PDF", type=["pdf"])

if uploaded_pdf is not None:
    uploaded_path, uploaded_digest = _save_uploaded_pdf(uploaded_pdf)
    if uploaded_digest != st.session_state.uploaded_pdf_digest:
        st.session_state.uploaded_pdf_path = uploaded_path
        st.session_state.uploaded_pdf_digest = uploaded_digest
        st.session_state.document_name = uploaded_pdf.name
        st.session_state.indexed = False
        st.session_state.pipeline = RagPipeline(get_settings())

    st.caption(f"Arquivo carregado: {st.session_state.document_name}")

if st.button("Indexar PDF", disabled=uploaded_pdf is None):
    if st.session_state.uploaded_pdf_path is None:
        st.warning("Envie um PDF antes de indexar.")
    else:
        with st.spinner("Indexando PDF..."):
            summary = st.session_state.pipeline.index_path(
                st.session_state.uploaded_pdf_path,
                recursive=False,
            )
            st.session_state.indexed = True
            st.success(
                f"Indexados {summary['chunks']} chunks de {summary['documents']} documento(s)."
            )

if uploaded_pdf is None:
    st.info("Envie um PDF para habilitar a indexacao e o chat.")

if st.session_state.indexed and st.session_state.document_name:
    st.success(f"Pronto para perguntas sobre: {st.session_state.document_name}")

question = st.chat_input("Pergunte algo sobre o PDF")
if question:
    if uploaded_pdf is None:
        st.warning("Envie um PDF antes de perguntar.")
    elif not st.session_state.indexed:
        st.warning("Indexe o PDF antes de perguntar.")
    else:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Consultando o RAG local..."):
                response = st.session_state.pipeline.answer(question)
                st.markdown(response["answer"])
                with st.expander("Fontes"):
                    for source in response["sources"]:
                        source_name = Path(source["source"]).name
                        st.write(
                            f"{source_name} | chunk {source['chunk_id']} | score {source['score']:.4f}"
                        )
                        st.write(source["text"])
