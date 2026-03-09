from embeddings import load_embedding_model
from vectorstore import load_vector_store
from langchain_community.llms import Ollama


# -------------------------------
# Initialize System
# -------------------------------

print("🔄 Initializing RAG system...")

embeddings = load_embedding_model()
vectordb = load_vector_store(embeddings)

llm = Ollama(
    model="llama3",
    temperature=0
)

print("✅ RAG system ready.")


# -------------------------------
# RAG Function
# -------------------------------

def ask_question(query: str, k: int = 4):

    # Create retriever dynamically
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # Retrieve documents
    docs = retriever.invoke(query)

    # Limit context to top 3 chunks to avoid overload
    docs = docs[:5]

    context_chunks = [doc.page_content for doc in docs]

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a biomedical research assistant.

Use ONLY the information from the context to answer the question.

Rules:
- Do NOT repeat the full context.
- Extract only the relevant information.
- Summarize the answer in 2–3 sentences.
- If the answer is not present in the context, respond exactly with:
No relevant information found in the provided documents.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    sources = [doc.metadata for doc in docs]

    return response.strip(), sources, context_chunks