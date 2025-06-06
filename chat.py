"""
chat.py  – RAG + conversational memory
Author: Brandon Desbiens

Library that converts a question into an answer using
retrieval-augmented generation (RAG).

Flow:
- Retrieve TOP_K chunks with FAISS.
- Build a chat message list (system + history + user).
- Call ChatOllama (Mistral-7B-Instruct).
- Cache user/AI messages in RAM so follow-ups have context.

External entry-point:
    from chat import answer
    print(answer("What are the PAD phases?"))
"""

import pickle, faiss, numpy as np, os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
import config

# load artefacts once at import
index  = faiss.read_index(str(config.INDEX_FILE))
chunks = pickle.load(open(config.CHUNK_FILE, "rb"))
embed  = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
memory = ConversationBufferMemory(return_messages=True)

llm = ChatOllama(
    model       = config.OLLAMA_MODEL,
    base_url    = config.OLLAMA_URL,
    temperature = config.LLM_TEMPERATURE,
    lora_path   = os.getenv("OLLAMA_LORA_PATH"),  # set only if you train LoRA
)

_SYSTEM = (
    "You are PAD Chat Bot, trained solely on the 2023 Project Approval Directive.\n"
    "- Answer only from context.\n"
    '- If missing: "I\'m not certain the PAD addresses that. For further assistance, '
    'please contact your DDPC analyst or the PAD support help desk."\n'
    "- Keep replies 3–6 sentences, cite pages like (p. 42), match question language."
)

# helper: retrieve K most relevant chunks
def _retrieve(query: str, k: int = config.TOP_K):
    vec = np.array(embed.embed_query(query)).astype("float32")
    _, idx = index.search(vec[None, :], k)
    return [chunks[i] for i in idx[0]]

# public function
def answer(question: str) -> str:

    # Return a PAD-grounded answer; remembers chat history.
    ctx = "\n\n".join(c.page_content for c in _retrieve(question))
    messages = [
        {"role": "system", "content": _SYSTEM},
        *memory.buffer, # prior turns
        {"role": "user",
         "content": f"Context:\n{ctx}\n\nQuestion: {question}"}
    ]
    reply = llm.invoke(messages).content
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(reply)
    return reply