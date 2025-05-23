from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ask import retrieve, llm, SYSTEM

app = FastAPI(title="PAD Chat Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_route(q: Query):
    ctx = "\n\n".join(d.page_content for d in retrieve(q.question))
    prompt = f"{SYSTEM}\n\nContext:\n{ctx}\n\nQuestion: {q.question}\n\nAnswer:"
    response = llm.invoke(prompt)
    return {"answer": response}