"""
chat.py  – RAG + conversational memory
Author: Brandon Desbiens

This module turns a raw question (English or French) into a context-grounded
answer drawn from the 2023 Project Approval Directive.

Processing steps
----------------
1) Load artefacts at import-time:
        FAISS vector index
        list of pre-split PAD chunks
        MiniLM embedding model

2) For a new question:
        Embed the query and retrieve TOP_K most similar chunks with FAISS.
        Assemble a *message list*:
            - system rules (how the bot must behave)
            - full chat history (conversation buffer)
            - user message that contains retrieved context + question
        Call the local Mistral-7B-Instruct model through Ollama.
        Store user + AI messages in RAM so follow-up questions carry context.

3) Return the model’s answer to the caller.

Public API
----------
from chat import answer
print(answer("What are the PAD phases?"))
"""

import pickle, faiss, numpy as np, os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
import config

# One-time startup: load vector index, text chunks and embedder
index  = faiss.read_index(str(config.INDEX_FILE))
chunks = pickle.load(open(config.CHUNK_FILE, "rb"))
embed  = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
# Keep running chat history in RAM (user <-> assistant turns)
memory = ConversationBufferMemory(return_messages=True)

# Handle to the local Ollama instance serving Mistral-7B-Instruct
llm = ChatOllama(
    model       = config.OLLAMA_MODEL,
    base_url    = config.OLLAMA_URL,
    temperature = config.LLM_TEMPERATURE,
    lora_path   = os.getenv("OLLAMA_LORA_PATH"),
)
# System-level rules injected at the start of every prompt
_SYSTEM = ("""
You are PAD Chat Bot. Your knowledge is limited **exclusively** to the 2023
Canadian Project Approval Directive (PAD).

- Answer **only** when the information is present in the context below.  
- If the answer is missing, reply exactly:  
  “I'm not certain the PAD addresses that. For further assistance, please
  contact your DDPC analyst or the PAD support help desk.”  
- Keep responses concise (≈ 3-6 sentences).  
- Cite pages inline like “(p. 42)”.  
- Respond in the same language (English / French) as the question.
""".strip()
)

# Helper – vector search
def _retrieve(query: str, k: int = config.TOP_K):
    vec = np.array(embed.embed_query(query)).astype("float32")
    _, idx = index.search(vec[None, :], k)
    return [chunks[i] for i in idx[0]]

# Public entry-point
def answer(question: str) -> str:

    # Build the chat message list expected by Ollama - this makes it remember chat history
    ctx = "\n\n".join(c.page_content for c in _retrieve(question))
    messages = [
        {"role": "system", "content": _SYSTEM},
        *memory.buffer, # prior turns
        {"role": "user",
         "content": f"Context:\n{ctx}\n\nQuestion: {question}"}
    ]
    reply = llm.invoke(messages).content
    # Persist the new dialogue turn in RAM
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(reply)
    return reply