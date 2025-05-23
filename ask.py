import pickle, faiss, numpy as np
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_FILE  = "pad.index"
CHUNK_FILE  = "pad_chunks.pkl"

index   = faiss.read_index(INDEX_FILE)
chunks  = pickle.load(open(CHUNK_FILE, "rb"))
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = Ollama(model="gemma3:1b", base_url="http://localhost:11434", temperature=0.2)

SYSTEM = (
    "You are PAD Chat Bot, a specialized assistant trained exclusively on the Canadian Department of National Defence Project Approval Directive (PAD), April 2023 edition.\n\n"
    "You must answer the user's question **using only the information contained in the provided Context**.\n"
    "If the required information is **not explicitly present** in the Context, respond exactly with:\n"
    "\"I'm not certain the PAD addresses that.\"\n\n"
    "If the answer **is** found in the Context, provide a concise explanation in 3 to 6 clear sentences.\n"
    "After every factual claim, include the page number from the PAD in parentheses — for example: (p. 42).\n"
    "Do not guess, do not fabricate details, and do not rely on general knowledge outside the Context.\n"
    "Write formally and clearly. Default to English unless the question is asked in French.\n"
)


def retrieve(query, k=4):
    qvec = np.array(embedder.embed_query(query)).astype("float32")
    D, I = index.search(np.array([qvec]), k)
    return [chunks[i] for i in I[0]]

while True:
    user = input("\nYou > ").strip()
    if not user:
        break
    ctx = "\n\n".join(d.page_content for d in retrieve(user))
    prompt = f"{SYSTEM}\n\nContext:\n{ctx}\n\nQuestion: {user}"
    print("\nPAD Chat Bot >", llm.invoke(prompt))
