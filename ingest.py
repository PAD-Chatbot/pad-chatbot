import os, pickle, faiss, numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

PDF_PATH     = "pad.pdf"
EMBED_MODEL  = "text-embedding-3-small"
INDEX_FILE   = "pad.index"
CHUNK_FILE   = "pad_chunks.pkl"


loader     = PyPDFLoader(PDF_PATH)
raw_docs   = loader.load()
splitter   = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
chunks     = splitter.split_documents(raw_docs)


embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vecs = [embedder.embed_query(c.page_content) for c in chunks]
dim  = len(vecs[0])


index = faiss.IndexFlatL2(dim)
index.add(np.array(vecs).astype("float32"))
faiss.write_index(index, INDEX_FILE)
pickle.dump(chunks, open(CHUNK_FILE, "wb"))

print(f"Saved {len(chunks)} chunks → {INDEX_FILE}")
