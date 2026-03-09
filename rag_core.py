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
        model_name="llama-3.1-8b-instant",
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

    docs = docs[:3]

    context_chunks = [doc.page_content[:800] for doc in docs]

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

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        raise e

    sources = [doc.metadata for doc in docs]

    return response.content, sources, context_chunks