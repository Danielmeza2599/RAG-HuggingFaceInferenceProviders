import os
import streamlit as st
from dotenv import load_dotenv
from rag_core import (EMBEDDING_MODEL, DEFAULT_LLM, extract_pdf, split_pages, build_index, retrieve, generate_answer)
from sentence_transformers import SentenceTransformer

load_dotenv()

st.set_page_config(page_title="RAG | MTW410", page_icon="📚", layout="wide")
st.title("📚 Chat con tus PDF · RAG + Hugging Face")
st.caption("MTW410 · PDF → fragmentos → embeddings → FAISS → modelo generativo")
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

with st.sidebar:
    st.header("Configuración")
    token = st.text_input("HF_TOKEN", value=os.environ.get("HF_TOKEN", ""), type="password", help="Token de Hugging Face con permisos de inferencia. No lo compartas.")
    model = st.text_input("Modelo generativo", value=DEFAULT_LLM, help="Debe admitir chat completions en Hugging Face Inference Providers")
    k = st.slider("Fragmentos recuperados", 1, 6, 3)
    st.info("Los PDF escaneados requieren OCR. No subas documentos confidenciales a proveedores externos.")

files = st.file_uploader("Carga uno o varios archivos PDF", type=["pdf"], accept_multiple_files=True)
if files:
    signatures = [(f.name, f.size, f.file_id) for f in files]
    if st.session_state.get("signatures") != signatures:
        with st.spinner("Extrayendo texto, calculando embeddings e indexando…"):
            try:
                pages = []
                for f in files:
                    pages.extend(extract_pdf(f.getvalue(), f.name))
                chunks = split_pages(pages)
                embedder = get_embedder()
                index = build_index(chunks, embedder)
                st.session_state.update(signatures=signatures, chunks=chunks, index=index, history=[])
            except Exception as exc:
                st.error(f"No fue posible procesar el PDF: {exc}")
                st.stop()
    st.success(f"Índice listo: {len(st.session_state['chunks'])} fragmentos de {len(files)} archivo(s).")
    for msg in st.session_state.get("history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("hits"):
                with st.expander("Ver fragmentos recuperados"):
                    for h in msg["hits"]:
                        st.markdown(f"**{h['source']} · página {h['page']} · similitud {h['score']:.3f}**")
                        st.write(h["text"])
    question = st.chat_input("Pregunta algo sobre tus documentos")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        hits = retrieve(question, st.session_state.index, st.session_state.chunks, get_embedder(), k=k)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Consultando el modelo generativo…"):
                    answer = generate_answer(question, hits, token, model)
            except Exception as exc:
                answer = f"No fue posible generar la respuesta: {exc}\n\nPuedes revisar los fragmentos recuperados a continuación y cambiar el modelo o verificar tu token."
            st.markdown(answer)
            with st.expander("Ver fragmentos recuperados"):
                for h in hits:
                    st.markdown(f"**{h['source']} · página {h['page']} · similitud {h['score']:.3f}**")
                    st.write(h["text"])
        st.session_state.history.append({"role": "assistant", "content": answer, "hits": hits})
else:
    st.info("Comienza cargando el PDF de ejemplo o uno de tus documentos.")
