from embeddings import load_embedding_model
from vectorstore import load_vector_store
from langchain_groq import ChatGroq
import streamlit as st


# -------------------------------
# Initialize RAG System
# -------------------------------

def initialize_rag():

    print("🔄 Initializing RAG system...")

    embeddings = load_embedding_model()
    vectordb = load_vector_store(embeddings)

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama3-70b-8192",
        temperature=0
    )

    print("✅ RAG system ready.")

    return vectordb, llm


# -------------------------------
# RAG Function
# -------------------------------

def ask_question(query: str, k: int = 4):

    vectordb, llm = initialize_rag()

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(query)

    docs = docs[:5]

    context_chunks = [doc.page_content for doc in docs]

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a biomedical research assistant.

Use ONLY the information from the context to answer the question.

Rules:
- Do NOT repeat the entire context.
- Extract only relevant information.
- If information is missing say:
  "No relevant information found in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke([
    {"role": "user", "content": prompt}
])

    sources = [doc.metadata for doc in docs]

    return response.content, sources, context_chunks