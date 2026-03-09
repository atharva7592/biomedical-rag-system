from langchain_community.vectorstores import Chroma
from loader import load_documents
from chunking import split_documents


def load_vector_store(embeddings):

    # Load PDFs from data folder
    documents = load_documents()

    # Split into chunks
    chunks = split_documents(documents)

    # Create vector database in memory
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectordb