"""
ingest.py  – build the FAISS index
Author: Brandon Desbiens

Run this once (or whenever pad.pdf changes) to create:
* pad.index  – FAISS vectors for similarity search
* pad_chunks.pkl – list of text chunks with metadata

Steps
1. Read each PDF named in config (today just pad.pdf).
2. Split every page into ~300-char overlaps (RecursiveTextSplitter).
3. Embed each chunk with MiniLM.
4. Add vectors to a FAISS IndexFlatL2.
5. Persist index + chunks to disk.

You only need to re-run if you add the French PAD or change chunking.
"""

import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import config

# read PDF(s) and keep page numbers
reader_en = PdfReader(str(config.PDF_PATH))
docs: list[Document] = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
)

def add_pdf(path: Path, language: str) -> None:
# Extract every page, split to chunks, store page# + lang metadata.
    reader = PdfReader(str(path))
    for pg_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in splitter.split_text(text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"page": pg_num, "lang": language},
                )
            )

add_pdf(config.PDF_PATH, "en")

# If you later supply a French PAD (e.g., pad_fr.pdf) just drop it in
# the repo and uncomment / adapt these two lines:
# fr_path = config.BASE_DIR / "pad_fr.pdf"
# if fr_path.exists(): add_pdf(fr_path, "fr")

print(f"[ingest] produced {len(docs):,} text chunks")


# embed chunks with MiniLM (or whichever model is in config)
embedder = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
embeddings = np.array(
    embedder.embed_documents([d.page_content for d in docs]),
    dtype="float32",
)

# write FAISS index + pickle
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, str(config.INDEX_FILE))
pickle.dump(docs, open(config.CHUNK_FILE, "wb"))

print(f"[ingest] wrote {config.INDEX_FILE.name}  ({index.ntotal} vectors)")
print(f"[ingest] wrote {config.CHUNK_FILE.name}  ({len(docs)} chunks)")