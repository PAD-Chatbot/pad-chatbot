"""
app.py – FastAPI wrapper exposing POST /ask
Author: Brandon Desbiens

FastAPI layers minimal HTTP on top of chat.answer():
* Adds CORS so a browser front-end can call the endpoint from http://localhost.
* Uses a plain (synchronous) route – chat.answer() is already blocking but
  short enough not to starve the event loop.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import answer

app = FastAPI(title="PAD Chat Bot API")

# CORS – in prod ALLOWED_ORIGINS can be "https://whateverLiveUrlWeWant.ca", these variables are set in config.py
origins = os.getenv("ALLOWED_ORIGINS", "*") # comma-sep in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins.split(",")],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# incoming JSON looks like {"question": "text"}
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_route(q: Query):
    return {"answer": answer(q.question)}