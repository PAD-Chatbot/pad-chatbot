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

import pickle, faiss, numpy as np, re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import config

# load & tag pages
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
pages = []

for path in [config.PDF_PATH]: # extend list later for FR doc
    lang = "fr" if re.search(r"_fr\.pdf$", path.name, re.I) else "en"
    for page in PyPDFLoader(str(path)).load():
        page.metadata["lang"] = lang  # keep track of language
        pages.append(page)

print(f"[ingest] loaded {len(pages)} pages")

# chunk & embed
chunks  = splitter.split_documents(pages)
embed   = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
vectors = np.array(embed.embed_documents([c.page_content for c in chunks]),
                   dtype="float32")

# build FAISS index
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, str(config.INDEX_FILE))
pickle.dump(chunks, open(config.CHUNK_FILE, "wb"))
print(f"[ingest] wrote {config.INDEX_FILE.name} and {config.CHUNK_FILE.name}")