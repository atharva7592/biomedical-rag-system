from loader import load_pdfs
from chunking import chunk_documents
from embeddings import load_embedding_model
from vectorstore import create_vector_store

print("Loading PDFs...")
docs = load_pdfs("data")
print(f"Loaded {len(docs)} documents")

print("Chunking documents...")
chunks = chunk_documents(docs)
print(f"Created {len(chunks)} chunks")

print("Loading embedding model...")
embeddings = load_embedding_model()

print("Creating vector store...")
vectordb = create_vector_store(chunks, embeddings)

print("✅ Vector database created successfully!")