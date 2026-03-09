import streamlit as st
from rag_core import ask_question

st.set_page_config(
    page_title="Biomedical RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Biomedical Research Assistant")
st.markdown("Local RAG System powered by MiniLM + ChromaDB + Ollama (llama3)")

# Sidebar
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider(
    "Top-K Retrieval Chunks",
    min_value=1,
    max_value=8,
    value=4
)

st.sidebar.markdown("""
**Embedding Model:** all-MiniLM-L6-v2  
**Vector DB:** ChromaDB  
**LLM:** llama3 (Ollama Local)  
**Chunk Size:** 800  
**Overlap:** 150  
""")

query = st.text_input("Enter your biomedical question:")

if query:

    with st.spinner("🔎 Retrieving and generating answer..."):
        answer, sources, chunks = ask_question(query, top_k)

    st.divider()

    st.subheader("🧠 Answer")
    st.write(answer)

    st.divider()

    st.subheader("📚 Sources")

    for i, source in enumerate(sources):
        with st.expander(f"Source {i+1}"):
            st.json(source)

    st.divider()

    st.subheader("🔍 Retrieved Context")

    for i, chunk in enumerate(chunks):
        with st.expander(f"Chunk {i+1}"):
            st.write(chunk)