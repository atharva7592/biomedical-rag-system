import streamlit as st
from rag_core import ask_question

import tempfile
from langchain_community.document_loaders import PyPDFLoader
from chunking import split_documents
from embeddings import load_embedding_model
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="Biomedical Research Assistant")

st.title("Biomedical Research Assistant")
st.write("Ask questions from biomedical research papers.")

# -----------------------------
# Upload PDF Section
# -----------------------------

st.subheader("Upload a Biomedical PDF")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

vectordb = None

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    chunks = split_documents(docs)

    embeddings = load_embedding_model()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=None
    )

    st.success("PDF uploaded and indexed successfully!")

# -----------------------------
# Question Input
# -----------------------------

query = st.text_input("Enter your question")

if st.button("Ask"):

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents..."):

            # If user uploaded a PDF, use it
            if vectordb is not None:

                docs = vectordb.similarity_search(query, k=5)

                context_chunks = [doc.page_content for doc in docs]
                context = "\n\n".join(context_chunks)

                answer = context_chunks[0] if context_chunks else "No relevant information found."

                sources = [doc.metadata for doc in docs]

                chunks = context_chunks

            # Otherwise use default RAG pipeline
            else:
                answer, sources, chunks = ask_question(query)

        # -----------------------------
        # Display Answer
        # -----------------------------

        st.subheader("Answer")
        st.write(answer)

        # -----------------------------
        # Debug Context
        # -----------------------------

        st.write("DEBUG chunks:", chunks)

        # -----------------------------
        # Sources
        # -----------------------------

        st.subheader("Sources")

        for i, s in enumerate(sources):

            source_file = s.get("source", "Unknown document")
            page = s.get("page", "Unknown page")

            st.write(f"Source {i+1}")
            st.write(f"Document: {source_file}")
            st.write(f"Page: {page}")
            st.write("---")

        # -----------------------------
        # Retrieved Context
        # -----------------------------

        with st.expander("Retrieved Context"):
            for c in chunks:
                st.write(c)