from langchain_community.vectorstores import Chroma
import os


def create_vector_store(chunks, embeddings):

    os.makedirs("chroma_db", exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    vectordb.persist()

    return vectordb


def load_vector_store(embeddings):

    os.makedirs("chroma_db", exist_ok=True)

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    return vectordb