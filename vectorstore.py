from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

def build_vectorstore(docs):
    embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embedder)
