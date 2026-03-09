from langchain_community.vectorstores import Chroma
from loader import load_documents
from chunking import split_documents


def load_vector_store(embeddings):

    # Load documents
    documents = load_documents()

    # Split into chunks
    chunks = split_documents(documents)

    # Create vector store IN MEMORY (no SQLite persistence)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=None
    )

    return vectordb