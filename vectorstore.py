from langchain_community.vectorstores import Chroma
import os

# Absolute path to project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")


def create_vector_store(chunks, embeddings):

    os.makedirs(DB_PATH, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    vectordb.persist()

    return vectordb


def load_vector_store(embeddings):

    os.makedirs(DB_PATH, exist_ok=True)

    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    return vectordb