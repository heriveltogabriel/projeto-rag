import streamlit as st

from projeto_rag.config import get_settings
from projeto_rag.rag_pipeline import RagPipeline


st.set_page_config(page_title="Assistente de Manual")
st.title("Assistente Tecnico do Manual")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RagPipeline(get_settings())
if "indexed" not in st.session_state:
    st.session_state.indexed = False

if st.button("Indexar manual"):
    with st.spinner("Indexando documentos..."):
        summary = st.session_state.pipeline.index_path()
        st.session_state.indexed = True
        st.success(f"Indexados {summary['chunks']} chunks de {summary['documents']} documento(s).")

question = st.chat_input("Pergunte algo sobre o manual")
if question:
    if not st.session_state.indexed:
        st.warning("Indexe o manual antes de perguntar.")
    else:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Consultando o RAG local..."):
                response = st.session_state.pipeline.answer(question)
                st.markdown(response["answer"])
                with st.expander("Fontes"):
                    for source in response["sources"]:
                        st.write(
                            f"{source['source']} | chunk {source['chunk_id']} | score {source['score']:.4f}"
                        )
                        st.write(source["text"])
