"""
app.py  – FastAPI wrapper exposing POST /ask
Author: Brandon Desbiens
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import answer

app = FastAPI(title="PAD Chat Bot API")

# configurable CORS
origins = os.getenv("ALLOWED_ORIGINS", "*") # comma-sep in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins.split(",")],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# request / response model
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_route(q: Query):
    return {"answer": answer(q.question)}